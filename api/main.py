"""
api/main.py
============
FastAPI application for the BI-101 Geospatial Profitability Predictor.

Endpoints
---------
POST /predict   – per-location prediction with SHAP drivers
POST /batch     – high-throughput batch prediction (vectorised)
POST /optimize  – BIP hub-placement optimisation
GET  /top       – top-N locations with SHAP drivers
"""

import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from fastapi import FastAPI, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from api.model_loader import load_all_models
import api.model_loader as ml
from api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionResult,
    SHAPDriver,
    OptimizeRequest,
    OptimizeResponse,
)
from src.lgbm_model import predict_with_ci
from src.explainer import get_top_drivers
from src.bip_optimizer import run_bip


# ──────────────────────────────────────────────────────────────────────
# Lifespan: load all models ONCE at startup
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_models()

    # ── Augment features_df with GWR-derived columns ──────────────
    # The LightGBM model was trained on 10 features (8 base + 2 GWR).
    # features.pkl only has the 8 base features, so we need to extract
    # gwr_intercept and gwr_local_r2 from the GWR results object and
    # merge them into features_df at startup.
    if ml.gwr_results is not None and ml.grid_gdf is not None:
        try:
            gwr_params = ml.gwr_results.params    # shape (n, p+1)
            gwr_r2 = ml.gwr_results.localR2       # shape (n, 1)

            # Build a mapping from grid_id → (intercept, local_r2)
            grid_ids = ml.grid_gdf["grid_id"].tolist()
            gwr_lookup = {}
            for i, gid in enumerate(grid_ids):
                gwr_lookup[gid] = (float(gwr_params[i, 0]), float(gwr_r2[i, 0]))

            # Add columns to features_df
            ml.features_df["gwr_intercept"] = ml.features_df["grid_id"].map(
                lambda gid: gwr_lookup.get(gid, (0.0, 0.0))[0]
            )
            ml.features_df["gwr_local_r2"] = ml.features_df["grid_id"].map(
                lambda gid: gwr_lookup.get(gid, (0.0, 0.0))[1]
            )
            print(
                f"[lifespan] Augmented features_df with GWR columns → "
                f"{ml.features_df.shape[1] - 1} features"
            )
        except Exception as exc:
            print(f"[lifespan] WARNING: GWR augmentation failed: {exc}")

    yield


app = FastAPI(
    title="BI-101 Geospatial Profitability Predictor",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────
# CORS middleware — allow everything for dev / dashboard access
# ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _snap_locations_to_grid(locations):
    """
    Create a GeoDataFrame from input locations and spatially join to the
    nearest grid-cell centroid.

    Returns
    -------
    snapped : GeoDataFrame
        One row per input location with the matched ``grid_id`` and
        grid-cell centroid coordinates appended.
    """
    try:
        points = [Point(loc.lon, loc.lat) for loc in locations]
        input_gdf = gpd.GeoDataFrame(
            {
                "input_lat": [loc.lat for loc in locations],
                "input_lon": [loc.lon for loc in locations],
                "input_grid_id": [loc.grid_id for loc in locations],
            },
            geometry=points,
            crs=ml.grid_gdf.crs,
        )

        # Build centroid layer from the loaded grid
        grid_centroids = ml.grid_gdf.copy()
        grid_centroids["geometry"] = grid_centroids.geometry.centroid

        snapped = gpd.sjoin_nearest(
            input_gdf,
            grid_centroids[["grid_id", "geometry"]],
            how="left",
            distance_col="snap_dist",
        )

        # If the caller supplied an explicit grid_id, override the snap
        if "input_grid_id" in snapped.columns:
            mask = snapped["input_grid_id"].notna()
            snapped.loc[mask, "grid_id"] = snapped.loc[mask, "input_grid_id"]

        return snapped

    except Exception as exc:
        raise RuntimeError(
            f"[main._snap_locations_to_grid] Failed: {exc}"
        ) from exc


def _recommendation_label(p):
    """Map probability → human-readable recommendation."""
    if np.isnan(p):
        return "skip"
    if p > 0.6:
        return "open"
    if p >= 0.4:
        return "monitor"
    return "skip"


def _feature_names():
    """Return feature column names from the loaded features DataFrame."""
    return [c for c in ml.features_df.columns if c != "grid_id"]


def _build_feature_row(grid_id, feature_names):
    """
    Look up a single grid cell's feature vector from features_df.

    Returns
    -------
    numpy.ndarray, shape (1, p)
    """
    try:
        row = ml.features_df.loc[
            ml.features_df["grid_id"] == grid_id, feature_names
        ]
        if row.empty:
            return None
        return row.values.astype(np.float64)
    except Exception as exc:
        raise RuntimeError(
            f"[main._build_feature_row] Failed for grid_id={grid_id}: {exc}"
        ) from exc


def _build_feature_matrix(grid_ids, feature_names):
    """
    Build the feature matrix for a list of grid_ids in one vectorised call.

    Returns
    -------
    X : numpy.ndarray, shape (m, p)
    valid_indices : list of int
        Positions within the input list that had matching features.
    valid_grid_ids : list of str
    """
    try:
        mask = ml.features_df["grid_id"].isin(grid_ids)
        matched = ml.features_df.loc[mask].copy()

        # Preserve ordering from the input grid_ids list
        matched = matched.set_index("grid_id").reindex(grid_ids).dropna(how="all")
        valid_grid_ids = matched.index.tolist()
        X = matched[feature_names].values.astype(np.float64)

        # Map back to positions in the original grid_ids list
        gid_to_pos = {gid: i for i, gid in enumerate(grid_ids)}
        valid_indices = [gid_to_pos[gid] for gid in valid_grid_ids]

        return X, valid_indices, valid_grid_ids

    except Exception as exc:
        raise RuntimeError(
            f"[main._build_feature_matrix] Failed: {exc}"
        ) from exc


def _get_training_data(feature_names):
    """
    Return training X and y arrays from features_df + grid_gdf for
    the bootstrap CI computation.
    """
    try:
        merged = ml.features_df.merge(
            ml.grid_gdf[["grid_id", "profitable"]],
            on="grid_id",
            how="inner",
        )
        has_label = merged["profitable"].notna()
        merged_labelled = merged[has_label]
        if merged_labelled.empty:
            return None, None
        X_train = merged_labelled[feature_names].values.astype(np.float64)
        y_train = merged_labelled["profitable"].values.astype(np.int32)
        return X_train, y_train
    except Exception as exc:
        raise RuntimeError(
            f"[main._get_training_data] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Per-location prediction: snap each input location to the nearest
    grid cell, predict profitability, and return SHAP top-3 drivers.
    """
    try:
        snapped = _snap_locations_to_grid(request.locations)
        feat_names = _feature_names()
        X_train, y_train = _get_training_data(feat_names)

        results = []
        for idx, row in snapped.iterrows():
            grid_id = str(row["grid_id"])
            lat = float(row["input_lat"])
            lon = float(row["input_lon"])

            # Cold-start: no feature data or no profitable label
            has_features = grid_id in set(
                ml.features_df["grid_id"].astype(str).tolist()
            )
            has_label = (
                "profitable" in ml.grid_gdf.columns
                and grid_id
                in set(
                    ml.grid_gdf.loc[
                        ml.grid_gdf["profitable"].notna(), "grid_id"
                    ]
                    .astype(str)
                    .tolist()
                )
            )
            cold = not (has_features and has_label)

            if cold:
                # Thompson Sampling fallback
                p_val = ml.thompson_sampler.get_probability_estimate(grid_id)
                ci_lo, ci_hi = None, None
            else:
                # LightGBM + bootstrap CI
                X_row = _build_feature_row(grid_id, feat_names)
                if X_row is None:
                    # Feature data missing — treat as cold start
                    p_val = ml.thompson_sampler.get_probability_estimate(grid_id)
                    ci_lo, ci_hi = None, None
                    cold = True
                else:
                    if X_train is not None and y_train is not None:
                        mean_p, ci_lower, ci_upper = predict_with_ci(
                            ml.lgbm_model, X_row, X_train, y_train,
                            n_bootstrap=20,
                        )
                        p_val = float(mean_p[0])
                        ci_lo = float(ci_lower[0])
                        ci_hi = float(ci_upper[0])
                    else:
                        proba = ml.lgbm_model.predict_proba(X_row)[:, 1]
                        p_val = float(proba[0])
                        ci_lo, ci_hi = None, None

            # SHAP top-3 drivers
            shap_drivers = []
            if not cold:
                X_shap = _build_feature_row(grid_id, feat_names)
                if X_shap is not None:
                    try:
                        drivers = get_top_drivers(
                            ml.shap_explainer, X_shap, feat_names, top_n=3
                        )
                        shap_drivers = [
                            SHAPDriver(feature=d["feature"], impact=d["impact"])
                            for d in drivers
                        ]
                    except Exception:
                        shap_drivers = []

            rec = _recommendation_label(p_val)

            results.append(
                PredictionResult(
                    grid_id=grid_id,
                    lat=lat,
                    lon=lon,
                    p_profit=round(p_val, 6),
                    ci_lower=round(ci_lo, 6) if ci_lo is not None else None,
                    ci_upper=round(ci_hi, 6) if ci_hi is not None else None,
                    recommendation=rec,
                    shap_drivers=shap_drivers,
                    is_cold_start=cold,
                )
            )

        return PredictResponse(predictions=results)

    except Exception as exc:
        raise RuntimeError(
            f"[main.predict] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# POST /batch
# ──────────────────────────────────────────────────────────────────────

@app.post("/batch", response_model=PredictResponse)
async def batch_predict(request: PredictRequest, response: Response):
    """
    High-throughput batch prediction.

    Vectorised: one sjoin_nearest, one LightGBM predict call for all
    non-cold-start cells, batch SHAP.  Target: 10 000 predictions
    in under 30 seconds.
    """
    try:
        t_start = time.perf_counter()
        n_total = len(request.locations)

        # ── 1. Snap ALL locations in one vectorised call ──────────────
        snapped = _snap_locations_to_grid(request.locations)
        feat_names = _feature_names()

        # ── 2. Classify cold-start vs historical ──────────────────────
        #    A cell has history if it exists in features_df AND has a
        #    non-NaN 'profitable' label in grid_gdf.  Thompson state is
        #    NOT a reliable indicator because no rewards are recorded in
        #    a synthetic / offline pipeline.
        grid_ids_all = snapped["grid_id"].astype(str).tolist()

        feature_grid_ids = set(ml.features_df["grid_id"].astype(str).tolist())

        # Also check the profitable column in grid_gdf
        if "profitable" in ml.grid_gdf.columns:
            profitable_map = dict(
                zip(
                    ml.grid_gdf["grid_id"].astype(str),
                    ml.grid_gdf["profitable"],
                )
            )
        else:
            profitable_map = {}

        cold_flags = []
        for gid in grid_ids_all:
            has_features = gid in feature_grid_ids
            has_label = (
                gid in profitable_map
                and pd.notna(profitable_map.get(gid))
            )
            cold_flags.append(not (has_features and has_label))

        cold_indices = [i for i, c in enumerate(cold_flags) if c]
        hist_indices = [i for i, c in enumerate(cold_flags) if not c]

        print(
            f"[batch] LightGBM cells: {len(hist_indices)}, "
            f"Cold start cells: {len(cold_indices)}"
        )

        hist_grid_ids = [grid_ids_all[i] for i in hist_indices]

        # ── 3. Pre-allocate result arrays ─────────────────────────────
        p_profits = np.full(n_total, np.nan, dtype=np.float64)
        ci_lowers = np.full(n_total, np.nan, dtype=np.float64)
        ci_uppers = np.full(n_total, np.nan, dtype=np.float64)

        # ── 4. Cold-start cells → Thompson Sampling ──────────────────
        for i in cold_indices:
            p_profits[i] = ml.thompson_sampler.get_probability_estimate(
                grid_ids_all[i]
            )

        # ── 5. Historical cells → ONE batch LightGBM call ────────────
        # Maps from hist position → original position
        shap_X_rows = {}  # grid_id → feature row for SHAP
        if hist_grid_ids:
            X_batch, valid_positions, valid_gids = _build_feature_matrix(
                hist_grid_ids, feat_names
            )

            if len(valid_gids) > 0:
                # Rapid point estimate without expensive 20x bootstrap retraining
                mean_p = ml.lgbm_model.predict_proba(X_batch)[:, 1]
                ci_lo = np.full(len(mean_p), np.nan)
                ci_hi = np.full(len(mean_p), np.nan)

                # Map results back to original positions
                for batch_idx, hist_pos in enumerate(valid_positions):
                    orig_idx = hist_indices[hist_pos]
                    p_profits[orig_idx] = mean_p[batch_idx]
                    ci_lowers[orig_idx] = ci_lo[batch_idx]
                    ci_uppers[orig_idx] = ci_hi[batch_idx]
                    shap_X_rows[grid_ids_all[orig_idx]] = X_batch[
                        batch_idx
                    ].reshape(1, -1)

            # Cells that were "historical" but had no feature data → cold-start
            missing_hist = set(range(len(hist_grid_ids))) - set(valid_positions)
            for pos in missing_hist:
                orig_idx = hist_indices[pos]
                gid = grid_ids_all[orig_idx]
                p_profits[orig_idx] = (
                    ml.thompson_sampler.get_probability_estimate(gid)
                )
                cold_flags[orig_idx] = True

        # ── 6. SHAP top-3 drivers per cell (batch-style) ─────────────
        # Computing SHAP for 13,000 cells takes too long for a single API call.
        # We skip it in batch and let the front-end call /predict for cell details.
        shap_results = {gid: [] for gid in grid_ids_all}

        # ── 7. Assemble results ───────────────────────────────────────
        results = []
        for i in range(n_total):
            gid = grid_ids_all[i]
            p_val = float(p_profits[i]) if not np.isnan(p_profits[i]) else 0.5
            ci_lo = (
                float(ci_lowers[i]) if not np.isnan(ci_lowers[i]) else None
            )
            ci_hi = (
                float(ci_uppers[i]) if not np.isnan(ci_uppers[i]) else None
            )
            rec = _recommendation_label(p_val)
            drivers = shap_results.get(gid, [])
            is_cold = cold_flags[i]

            results.append(
                PredictionResult(
                    grid_id=gid,
                    lat=float(snapped.iloc[i]["input_lat"]),
                    lon=float(snapped.iloc[i]["input_lon"]),
                    p_profit=round(p_val, 6),
                    ci_lower=round(ci_lo, 6) if ci_lo is not None else None,
                    ci_upper=round(ci_hi, 6) if ci_hi is not None else None,
                    recommendation=rec,
                    shap_drivers=drivers,
                    is_cold_start=is_cold,
                )
            )

        elapsed = time.perf_counter() - t_start

        # ── 8. Performance headers ───────────────────────────────────
        response.headers["X-Prediction-Count"] = str(n_total)
        response.headers["X-Processing-Time"] = f"{elapsed:.3f}"

        print(
            f"[batch] {n_total} predictions in {elapsed:.3f}s "
            f"({n_total / max(elapsed, 0.001):.0f} pred/s)"
        )

        return PredictResponse(predictions=results)

    except Exception as exc:
        raise RuntimeError(
            f"[main.batch_predict] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# POST /optimize
# ──────────────────────────────────────────────────────────────────────

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest):
    """
    Run BIP hub-placement optimisation over the current grid's
    profitability scores.
    """
    try:
        gdf = ml.grid_gdf.copy()

        # Ensure p_profit column exists; if not, default to Thompson estimates
        if "p_profit" not in gdf.columns:
            estimates = ml.thompson_sampler.get_all_estimates()
            gdf["p_profit"] = gdf["grid_id"].map(estimates).fillna(0.5)

        # Ensure centroid columns exist
        if "centroid_lat" not in gdf.columns:
            gdf["centroid_lat"] = gdf.geometry.centroid.y
        if "centroid_lon" not in gdf.columns:
            gdf["centroid_lon"] = gdf.geometry.centroid.x

        # ── Run BIP ──────────────────────────────────────────────────
        selected_ids, objective_value = run_bip(
            gdf=gdf,
            prob_col="p_profit",
            max_hubs=request.max_hubs,
            min_separation_km=request.min_separation_km,
            min_prob_threshold=request.min_prob_threshold,
        )

        # ── Verify separation constraint ─────────────────────────────
        from src.bip_optimizer import _haversine_km

        separation_met = True
        if len(selected_ids) > 1:
            selected_rows = gdf[gdf["grid_id"].isin(selected_ids)]
            lats = selected_rows["centroid_lat"].values.astype(float)
            lons = selected_rows["centroid_lon"].values.astype(float)

            for i in range(len(lats)):
                for j in range(i + 1, len(lats)):
                    dist = _haversine_km(lats[i], lons[i], lats[j], lons[j])
                    if dist < request.min_separation_km:
                        separation_met = False
                        print(
                            f"[optimize] WARNING: hubs {selected_ids[i]} and "
                            f"{selected_ids[j]} are only {dist:.2f} km apart "
                            f"(min={request.min_separation_km} km)"
                        )
                        break
                if not separation_met:
                    break

        return OptimizeResponse(
            selected_hubs=selected_ids,
            total_score=round(objective_value, 6),
            separation_constraint_met=separation_met,
        )

    except Exception as exc:
        raise RuntimeError(
            f"[main.optimize] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# GET /top
# ──────────────────────────────────────────────────────────────────────

@app.get("/top")
async def get_top_locations(
    n: int = Query(default=5, ge=1, le=50,
                   description="Number of top locations to return"),
    zone: Optional[str] = Query(default=None,
                   description="Filter by zone name e.g. Gurugram"),
    min_prob: float = Query(default=0.0,
                   description="Minimum p_profit threshold")
):
    # Use the already-loaded grid_gdf (loaded at startup, never reloaded)
    gdf = ml.grid_gdf.copy()

    # Filter by zone if provided
    if zone:
        gdf = gdf[gdf["zone_name"].str.lower() == zone.lower()]
        if len(gdf) == 0:
            return {"error": f"No cells found for zone: {zone}",
                    "valid_zones": ml.grid_gdf["zone_name"].unique().tolist()}

    # Filter by min probability
    gdf = gdf[gdf["p_profit"] >= min_prob]

    # Sort by p_profit descending, take top n
    top = gdf.nlargest(n, "p_profit")

    results = []
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        # Get SHAP drivers for this cell
        x_row = ml.features_df.loc[
            ml.features_df["grid_id"] == row["grid_id"]
        ].drop(columns=["grid_id"]).values

        drivers = []
        if len(x_row) > 0 and not row.get("is_cold_start", True):
            drivers = get_top_drivers(
                ml.shap_explainer,
                x_row[0],
                _feature_names(),
                top_n=3
            )

        results.append({
            "rank":           rank,
            "grid_id":        row["grid_id"],
            "zone":           row.get("zone_name", "unknown"),
            "lat":            row["centroid_lat"],
            "lon":            row["centroid_lon"],
            "p_profit":       round(float(row["p_profit"]), 4),
            "ci_lower":       round(float(row["ci_lower"]), 4) if row.get("ci_lower") else None,
            "ci_upper":       round(float(row["ci_upper"]), 4) if row.get("ci_upper") else None,
            "recommendation": row.get("recommendation", "monitor"),
            "top_drivers":    drivers
        })

    return {
        "top_locations":   results,
        "total_found":     len(results),
        "filters_applied": {
            "n":        n,
            "zone":     zone,
            "min_prob": min_prob
        }
    }
