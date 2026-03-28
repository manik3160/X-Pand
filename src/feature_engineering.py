"""
src/feature_engineering.py
===========================
Generates delivery-zone features, computes spatial lag variables,
and assembles the final feature matrix for the ML pipeline.

DATA SOURCES (after 02a + 02b notebooks):
  - pop_density         : REAL  — WorldPop 100m raster (02a_worldpop_population.ipynb)
  - pop_density_log     : REAL  — derived from WorldPop
  - competitor_count    : REAL  — OSM Overpass API GeoJSON (02b_competitor_features.ipynb)
  - competitor_density_1km : REAL — OSM, 1km radius count
  - nearest_competitor_km  : REAL — OSM, cKDTree distance
  - market_saturation   : REAL  — derived from OSM
  - opportunity_gap     : REAL  — WorldPop + OSM combined signal
  - income_index        : synthetic (no real source available)
  - road_density        : synthetic (replace with OSMnx if time permits)
  - transit_stops       : synthetic (replace with OSM if time permits)
  - warehouse_proximity_km : synthetic

All DataFrames maintain `grid_id` as the primary key.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from src.spatial_weights import compute_spatial_lag


# ──────────────────────────────────────────────────────────────────────
# Paths to real data outputs from 02a and 02b notebooks
# ──────────────────────────────────────────────────────────────────────
GRID_WITH_COMPETITORS_PATH = "data/processed/grid_with_competitors.pkl"
GRID_WITH_POPULATION_PATH  = "data/processed/grid_with_population.pkl"
GRID_BASE_PATH             = "data/processed/grid.geojson"


# ──────────────────────────────────────────────────────────────────────
# Feature columns used throughout the pipeline
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    # ── REAL features (from WorldPop + OSM) ───────────────────────────
    "pop_density",               # people per km² (WorldPop)
    "pop_density_log",           # log(pop_density+1) — less skewed for LGBM
    "competitor_count",          # restaurants inside this 500m cell (OSM)
    "competitor_density_1km",    # restaurants within 1km radius (OSM)
    "nearest_competitor_km",     # distance to nearest restaurant in km (OSM)
    "market_saturation",         # normalised competition score 0-1 (OSM)
    "opportunity_gap",           # high pop + low competition signal 0-1

    # ── SYNTHETIC features (no real source yet) ────────────────────────
    "income_index",              # proxy for area affluence
    "road_density",              # road network density (use OSMnx to make real)
    "transit_stops",             # public transport access
    "warehouse_proximity_km",    # distance to nearest dark kitchen / warehouse

    # ── COMPUTED spatial lag features (libpysal) ──────────────────────
    "lag_profitable",            # spatial lag of profitability label
    "lag_pop_density",           # spatial lag of population density
]

LABEL_COLUMN = "profitable"


# ──────────────────────────────────────────────────────────────────────
# Real data loader
# ──────────────────────────────────────────────────────────────────────

def load_grid_with_real_data():
    """
    Load the grid GeoDataFrame with real population and competitor
    features already attached.

    Tries paths in priority order:
      1. grid_with_competitors.pkl  (02b output — has both pop + competitor)
      2. grid_with_population.pkl   (02a output — has pop only)
      3. grid.geojson               (base grid — no real features)

    Returns
    -------
    geopandas.GeoDataFrame
        Grid with as many real feature columns as available.
    str
        Which source was loaded — useful for logging.
    """
    if os.path.exists(GRID_WITH_COMPETITORS_PATH):
        gdf = pd.read_pickle(GRID_WITH_COMPETITORS_PATH)
        source = "grid_with_competitors.pkl  (real pop + real competitors)"

    elif os.path.exists(GRID_WITH_POPULATION_PATH):
        gdf = pd.read_pickle(GRID_WITH_POPULATION_PATH)
        source = "grid_with_population.pkl  (real pop only — run 02b for competitors)"

    else:
        gdf = gpd.read_file(GRID_BASE_PATH)
        source = "grid.geojson  (base grid — no real features, all synthetic)"

    print(f"[feature_engineering] Loaded grid from: {source}")
    print(f"[feature_engineering] Grid shape: {gdf.shape}")
    return gdf, source


# ──────────────────────────────────────────────────────────────────────
# Synthetic feature generator (only for columns with no real source)
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_data(gdf):
    """
    Attach synthetic feature columns to the grid GeoDataFrame.

    IMPORTANT: This function now only generates synthetic values for
    features that do NOT have a real data source yet:
        - income_index
        - road_density
        - transit_stops
        - warehouse_proximity_km
        - profitable  (label — always synthetic for hackathon)

    Features that ARE real (if 02a/02b have been run) are NOT overwritten:
        - pop_density, pop_density_log       ← from WorldPop (02a)
        - competitor_count                   ← from OSM (02b)
        - competitor_density_1km             ← from OSM (02b)
        - nearest_competitor_km              ← from OSM (02b)
        - market_saturation                  ← from OSM (02b)
        - opportunity_gap                    ← from WorldPop + OSM (02b)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must already contain `grid_id`, `centroid_lat`, `centroid_lon`.
        Should be loaded via load_grid_with_real_data() first.

    Returns
    -------
    geopandas.GeoDataFrame
        Input GeoDataFrame with synthetic columns added.
    """
    try:
        required = {"grid_id", "centroid_lat", "centroid_lon"}
        missing = required - set(gdf.columns)
        if missing:
            raise ValueError(
                f"GeoDataFrame is missing required columns: {missing}"
            )

        n   = len(gdf)
        rng = np.random.RandomState(42)
        gdf = gdf.copy()

        # ── Population density — use real if available, else synthetic ─
        if "pop_density" not in gdf.columns or gdf["pop_density"].sum() == 0:
            print("[feature_engineering] pop_density not found — generating synthetic.")
            gdf["pop_density"] = np.clip(
                rng.normal(loc=5000, scale=2000, size=n), a_min=100, a_max=None
            )
        else:
            print(
                f"[feature_engineering] pop_density: using REAL WorldPop data "
                f"(mean={gdf['pop_density'].mean():.0f} people/km²)"
            )

        # ── Log population density — always derived from pop_density ──
        gdf["pop_density_log"] = np.log1p(gdf["pop_density"])

        # ── Competitor count — use real if available, else synthetic ───
        if "competitor_count" not in gdf.columns or gdf["competitor_count"].sum() == 0:
            print("[feature_engineering] competitor_count not found — generating synthetic.")
            gdf["competitor_count"] = rng.poisson(lam=3, size=n).astype(int)
        else:
            print(
                f"[feature_engineering] competitor_count: using REAL OSM data "
                f"(mean={gdf['competitor_count'].mean():.2f} per cell)"
            )

        # ── Competitor density 1km — use real if available ─────────────
        if "competitor_density_1km" not in gdf.columns:
            print("[feature_engineering] competitor_density_1km not found — generating synthetic.")
            gdf["competitor_density_1km"] = rng.poisson(lam=8, size=n).astype(int)
        else:
            print(
                f"[feature_engineering] competitor_density_1km: using REAL OSM data "
                f"(mean={gdf['competitor_density_1km'].mean():.2f})"
            )

        # ── Nearest competitor distance — use real if available ─────────
        if "nearest_competitor_km" not in gdf.columns:
            print("[feature_engineering] nearest_competitor_km not found — generating synthetic.")
            gdf["nearest_competitor_km"] = rng.uniform(low=0.05, high=3.0, size=n)
        else:
            print(
                f"[feature_engineering] nearest_competitor_km: using REAL OSM data "
                f"(mean={gdf['nearest_competitor_km'].mean():.3f} km)"
            )

        # ── Market saturation — use real if available ───────────────────
        if "market_saturation" not in gdf.columns:
            gdf["market_saturation"] = (
                gdf["competitor_density_1km"] /
                (gdf["competitor_density_1km"].quantile(0.95) + 1e-9)
            ).clip(0, 1)
            print("[feature_engineering] market_saturation: computed from competitor_density_1km.")
        else:
            print("[feature_engineering] market_saturation: using REAL OSM-derived data.")

        # ── Opportunity gap — use real if available ─────────────────────
        if "opportunity_gap" not in gdf.columns:
            p95_pop  = gdf["pop_density"].quantile(0.95)
            norm_pop = (gdf["pop_density"] / max(p95_pop, 1)).clip(0, 1)
            gdf["opportunity_gap"] = (norm_pop - gdf["market_saturation"]).clip(0, 1)
            print("[feature_engineering] opportunity_gap: computed from pop + market_saturation.")
        else:
            print("[feature_engineering] opportunity_gap: using REAL WorldPop+OSM data.")

        # ── PURELY SYNTHETIC features (no real data source) ────────────
        # Income index — no real data available, stays synthetic
        gdf["income_index"] = rng.uniform(low=0.3, high=1.0, size=n)

        # Road density — stays synthetic until OSMnx data is added
        # To make real: use OSMnx to compute road length per 500m cell
        gdf["road_density"] = rng.uniform(low=0.2, high=1.0, size=n)

        # Transit stops — stays synthetic until OSM transit data is added
        gdf["transit_stops"] = rng.poisson(lam=2, size=n).astype(int)

        # Warehouse proximity — fully synthetic
        gdf["warehouse_proximity_km"] = rng.uniform(low=0.5, high=15.0, size=n)

        # ── Binary label (profitability) ───────────────────────────────
        # The label uses a combination of real and synthetic features
        # so the model learns from real spatial patterns.
        noise = rng.normal(loc=0, scale=0.05, size=n)

        # Real signals drive the label:
        #   + high population          → more delivery demand
        #   + high road density        → riders can reach the zone
        #   + low competitor density   → less saturation = more opportunity
        #   + high opportunity gap     → underserved high-demand area
        score = (
            0.30 * (gdf["pop_density"].values / 10_000)
            + 0.20 * gdf["road_density"].values
            - 0.20 * gdf["market_saturation"].values
            + 0.20 * gdf["opportunity_gap"].values
            + 0.10 * gdf["income_index"].values
            + noise
        )

        # Top 10% by score → profitable = 1  (severe class imbalance preserved)
        threshold      = np.percentile(score, 90)
        gdf["profitable"] = (score >= threshold).astype(int)

        pos_count = int(gdf["profitable"].sum())
        pos_pct   = pos_count / n * 100
        print(
            f"[feature_engineering] Synthetic + real data ready for {n} cells | "
            f"profitable = {pos_count} ({pos_pct:.1f}%)"
        )

        return gdf

    except Exception as exc:
        raise RuntimeError(
            f"[feature_engineering.generate_synthetic_data] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# Spatial lag features (unchanged from original)
# ──────────────────────────────────────────────────────────────────────

def add_spatial_lag_features(gdf, weights):
    """
    Compute spatial lag features and append them to the GeoDataFrame.

    New columns added:
        - `lag_profitable`:  spatial lag of the `profitable` column
        - `lag_pop_density`: spatial lag of the `pop_density` column

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain `profitable` and `pop_density` columns.
    weights : libpysal.weights.W
        Row-standardized Queen contiguity weights matrix.

    Returns
    -------
    geopandas.GeoDataFrame
        Input GeoDataFrame with two new spatial lag columns.
    """
    try:
        for col in ["profitable", "pop_density"]:
            if col not in gdf.columns:
                raise ValueError(
                    f"Column '{col}' is required but missing from GeoDataFrame."
                )

        gdf = gdf.copy()
        gdf["lag_profitable"]  = compute_spatial_lag(gdf, weights, "profitable")
        gdf["lag_pop_density"] = compute_spatial_lag(gdf, weights, "pop_density")

        print(
            f"[feature_engineering] Added spatial lag features: "
            f"lag_profitable (mean={gdf['lag_profitable'].mean():.4f}), "
            f"lag_pop_density (mean={gdf['lag_pop_density'].mean():.1f})"
        )

        return gdf

    except Exception as exc:
        raise RuntimeError(
            f"[feature_engineering.add_spatial_lag_features] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# Feature matrix builder (updated with new columns)
# ──────────────────────────────────────────────────────────────────────

def build_feature_matrix(gdf):
    """
    Extract the feature matrix X and label vector y from the GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain all columns listed in FEATURE_COLUMNS and the
        `profitable` label column.

    Returns
    -------
    X : numpy.ndarray, shape (n_cells, n_features)
        Feature matrix.
    y : numpy.ndarray, shape (n_cells,)
        Binary label vector (0 or 1).
    feature_names : list of str
        Ordered list of feature column names.
    """
    try:
        required = set(FEATURE_COLUMNS) | {LABEL_COLUMN}
        missing  = required - set(gdf.columns)
        if missing:
            raise ValueError(
                f"GeoDataFrame is missing required columns for feature matrix: "
                f"{missing}. \nAvailable: {list(gdf.columns)}"
            )

        if "grid_id" not in gdf.columns:
            raise ValueError(
                "GeoDataFrame must contain 'grid_id' as primary key."
            )
        if gdf["grid_id"].duplicated().any():
            n_dup = gdf["grid_id"].duplicated().sum()
            raise ValueError(
                f"Found {n_dup} duplicate grid_id values. "
                "Each cell must be uniquely identified."
            )

        feature_names = list(FEATURE_COLUMNS)
        X = gdf[feature_names].values.astype(np.float64)
        y = gdf[LABEL_COLUMN].values.astype(np.int32)

        # ── Data quality check ────────────────────────────────────────
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(
                f"[feature_engineering] WARNING: X contains {nan_count} NaNs "
                f"and {inf_count} Infs — filling with 0."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Feature source summary ────────────────────────────────────
        real_cols = [
            "pop_density", "pop_density_log", "competitor_count",
            "competitor_density_1km", "nearest_competitor_km",
            "market_saturation", "opportunity_gap"
        ]
        synth_cols = [
            "income_index", "road_density",
            "transit_stops", "warehouse_proximity_km"
        ]
        lag_cols = ["lag_profitable", "lag_pop_density"]

        print(
            f"[feature_engineering] Built feature matrix:\n"
            f"  X.shape       = {X.shape}\n"
            f"  y.shape       = {y.shape}\n"
            f"  positive rate = {y.mean():.3f}\n"
            f"  REAL features ({len(real_cols)})    : {real_cols}\n"
            f"  SYNTHETIC features ({len(synth_cols)}) : {synth_cols}\n"
            f"  LAG features ({len(lag_cols)})      : {lag_cols}"
        )

        return X, y, feature_names

    except Exception as exc:
        raise RuntimeError(
            f"[feature_engineering.build_feature_matrix] Failed: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# Convenience function — full pipeline in one call
# ──────────────────────────────────────────────────────────────────────

def run_full_feature_pipeline(weights):
    """
    End-to-end feature pipeline.

    Loads the best available grid (real data if notebooks 02a/02b have
    been run, synthetic fallback otherwise), generates remaining synthetic
    features, adds spatial lag features, and returns the feature matrix.

    Parameters
    ----------
    weights : libpysal.weights.W
        Row-standardized Queen contiguity weights matrix.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Fully featured GeoDataFrame with all columns.
    X   : numpy.ndarray
        Feature matrix.
    y   : numpy.ndarray
        Label vector.
    feature_names : list of str
        Feature column names in the same order as X columns.
    """
    # Step 1: load grid (real data if available)
    gdf, _ = load_grid_with_real_data()

    # Step 2: add synthetic features for columns with no real source
    #         real columns (pop_density, competitor_count, etc.) are preserved
    gdf = generate_synthetic_data(gdf)

    # Step 3: spatial lag features (requires weights matrix)
    gdf = add_spatial_lag_features(gdf, weights)

    # Step 4: build numpy arrays
    X, y, feature_names = build_feature_matrix(gdf)

    return gdf, X, y, feature_names