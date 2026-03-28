"""
src/ensemble.py
================
Combines LightGBM predictions (for cells with history) with Thompson
Sampling estimates (for cold-start cells) into a single annotated
GeoDataFrame with profitability recommendations.

PERFORMANCE: LightGBM predictions are computed in a single batch call
(not row-by-row) and assigned back by index.
"""

import numpy as np
import pandas as pd


def generate_predictions(gdf, lgbm_model, thompson_sampler, X, feature_names):
    """
    Generate profitability predictions for every grid cell using a hybrid
    LightGBM + Thompson Sampling approach.

    Decision logic per cell:
        - If the cell has historical data (profitable is NOT NaN):
          → use LightGBM batch predictions with bootstrap 95% CI.
        - If the cell is cold-start (profitable IS NaN or never updated):
          → use Thompson Sampling posterior mean estimate.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain `grid_id` and `profitable` columns.
    lgbm_model : CalibratedClassifierCV
        Calibrated LightGBM model.
    thompson_sampler : ThompsonSampler
        Initialised Thompson Sampler with all grid_ids.
    X : numpy.ndarray, shape (n, p)
        Feature matrix aligned with gdf rows.
    feature_names : list of str
        Feature column names.

    Returns
    -------
    geopandas.GeoDataFrame
        With new columns: p_profit, ci_lower, ci_upper, is_cold_start,
        recommendation.
    """
    try:
        if "grid_id" not in gdf.columns:
            raise ValueError("GeoDataFrame must contain 'grid_id' column.")
        if "profitable" not in gdf.columns:
            raise ValueError("GeoDataFrame must contain 'profitable' column.")

        gdf = gdf.copy()
        n = len(gdf)

        # ── Initialise output columns ─────────────────────────────────
        p_profit = np.full(n, np.nan, dtype=np.float64)
        ci_lower = np.full(n, np.nan, dtype=np.float64)
        ci_upper = np.full(n, np.nan, dtype=np.float64)
        is_cold_start = np.zeros(n, dtype=bool)

        # ── Separate historical vs cold-start cells ───────────────────
        #    Cold start = profitable IS NaN (no historical label data)
        grid_ids = gdf["grid_id"].values
        has_history_mask = gdf["profitable"].notna().values
        cold_start_mask = ~has_history_mask

        n_historical = int(has_history_mask.sum())
        n_cold = int(cold_start_mask.sum())

        print(
            f"[ensemble] Cells: {n} total, {n_historical} with history, "
            f"{n_cold} cold-start"
        )
        print(
            f"LightGBM cells: {n_historical}, Cold start cells: {n_cold}"
        )

        # ── BATCH: LightGBM predictions for historical cells ─────────
        if n_historical > 0:
            from src.lgbm_model import predict_with_ci

            X_hist = X[has_history_mask]

            # Build training data from historical cells for bootstrap
            hist_indices = np.where(has_history_mask)[0]
            y_hist = gdf.loc[gdf.index[hist_indices], "profitable"].values.astype(
                np.int32
            )

            mean_probs, ci_lo, ci_hi = predict_with_ci(
                lgbm_model, X_hist, X_hist, y_hist, n_bootstrap=20
            )

            p_profit[has_history_mask] = mean_probs
            ci_lower[has_history_mask] = ci_lo
            ci_upper[has_history_mask] = ci_hi

        # ── Thompson Sampling for cold-start cells ────────────────────
        if n_cold > 0:
            cold_indices = np.where(cold_start_mask)[0]
            for idx in cold_indices:
                gid = grid_ids[idx]
                p_profit[idx] = thompson_sampler.get_probability_estimate(gid)
                is_cold_start[idx] = True

        # ── Assign columns ────────────────────────────────────────────
        gdf["p_profit"] = p_profit
        gdf["ci_lower"] = np.where(is_cold_start, np.nan, ci_lower)
        gdf["ci_upper"] = np.where(is_cold_start, np.nan, ci_upper)
        gdf["is_cold_start"] = is_cold_start

        # ── Generate recommendations ─────────────────────────────────
        recommendations = []
        for p in p_profit:
            if np.isnan(p):
                recommendations.append("skip")
            elif p > 0.6:
                recommendations.append("open")
            elif p >= 0.4:
                recommendations.append("monitor")
            else:
                recommendations.append("skip")

        gdf["recommendation"] = recommendations

        # ── Summary statistics ────────────────────────────────────────
        rec_counts = gdf["recommendation"].value_counts().to_dict()
        print(
            f"[ensemble] Predictions complete: "
            f"mean p_profit={gdf['p_profit'].mean():.4f} | "
            f"recommendations: {rec_counts}"
        )

        return gdf

    except Exception as exc:
        raise RuntimeError(
            f"[ensemble.generate_predictions] Failed: {exc}"
        ) from exc
