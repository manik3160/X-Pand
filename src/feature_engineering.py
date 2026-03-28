"""
src/feature_engineering.py
===========================
Generates synthetic delivery-zone features, computes spatial lag variables,
and assembles the final feature matrix for the ML pipeline.

All DataFrames maintain `grid_id` as the primary key.
"""

import numpy as np
import pandas as pd
from src.spatial_weights import compute_spatial_lag


# ──────────────────────────────────────────────────────────────────────
# Feature columns used throughout the pipeline
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "pop_density",
    "income_index",
    "competitor_count",
    "road_density",
    "transit_stops",
    "warehouse_proximity_km",
    "lag_profitable",
    "lag_pop_density",
]

LABEL_COLUMN = "profitable"


def generate_synthetic_data(gdf):
    """
    Attach realistic synthetic feature columns and a binary profitability
    label to the grid GeoDataFrame.

    The label is constructed so that ~10% of cells are profitable (severe
    class imbalance) and the label is meaningfully correlated with the
    generated features.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must already contain `grid_id`, `centroid_lat`, `centroid_lon`.

    Returns
    -------
    geopandas.GeoDataFrame
        The input GeoDataFrame augmented with 6 feature columns and
        the binary `profitable` label.
    """
    try:
        # ── Validate required columns ─────────────────────────────────
        required = {"grid_id", "centroid_lat", "centroid_lon"}
        missing = required - set(gdf.columns)
        if missing:
            raise ValueError(
                f"GeoDataFrame is missing required columns: {missing}"
            )

        n = len(gdf)
        rng = np.random.RandomState(42)

        # ── Generate feature columns ──────────────────────────────────
        gdf = gdf.copy()

        # Population density — normal distribution, clipped
        gdf["pop_density"] = np.clip(
            rng.normal(loc=5000, scale=2000, size=n), a_min=100, a_max=None
        )

        # Income index — uniform [0.3, 1.0]
        gdf["income_index"] = rng.uniform(low=0.3, high=1.0, size=n)

        # Competitor count — Poisson(λ=3)
        gdf["competitor_count"] = rng.poisson(lam=3, size=n).astype(int)

        # Road density — uniform [0.2, 1.0]
        gdf["road_density"] = rng.uniform(low=0.2, high=1.0, size=n)

        # Transit stops — Poisson(λ=2)
        gdf["transit_stops"] = rng.poisson(lam=2, size=n).astype(int)

        # Warehouse proximity — uniform [0.5, 15.0] km
        gdf["warehouse_proximity_km"] = rng.uniform(low=0.5, high=15.0, size=n)

        # ── Generate binary label ─────────────────────────────────────
        # Score combines features with controlled noise so the label is
        # genuinely correlated with pop_density, road_density, and
        # competitor_count.
        noise = rng.normal(loc=0, scale=0.05, size=n)
        score = (
            0.4 * (gdf["pop_density"].values / 10_000)
            + 0.3 * gdf["road_density"].values
            - 0.2 * (gdf["competitor_count"].values / 10)
            + noise
        )

        # Top 10% by score → profitable = 1
        threshold = np.percentile(score, 90)
        gdf["profitable"] = (score >= threshold).astype(int)

        pos_count = gdf["profitable"].sum()
        pos_pct = pos_count / n * 100
        print(
            f"[feature_engineering] Generated synthetic data for {n} cells | "
            f"profitable = {pos_count} ({pos_pct:.1f}%)"
        )

        return gdf

    except Exception as exc:
        raise RuntimeError(
            f"[feature_engineering.generate_synthetic_data] Failed: {exc}"
        ) from exc


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
        gdf["lag_profitable"] = compute_spatial_lag(gdf, weights, "profitable")
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
    X : numpy.ndarray, shape (n_cells, 8)
        Feature matrix.
    y : numpy.ndarray, shape (n_cells,)
        Binary label vector (0 or 1).
    feature_names : list of str
        Ordered list of the 8 feature column names.
    """
    try:
        # ── Validate columns ──────────────────────────────────────────
        required = set(FEATURE_COLUMNS) | {LABEL_COLUMN}
        missing = required - set(gdf.columns)
        if missing:
            raise ValueError(
                f"GeoDataFrame is missing required columns for feature matrix: "
                f"{missing}. Available: {list(gdf.columns)}"
            )

        # ── Check grid_id integrity ───────────────────────────────────
        if "grid_id" not in gdf.columns:
            raise ValueError(
                "GeoDataFrame must contain 'grid_id' as primary key."
            )
        if gdf["grid_id"].duplicated().any():
            n_dup = gdf["grid_id"].duplicated().sum()
            raise ValueError(
                f"Found {n_dup} duplicate grid_id values. Each cell must be "
                "uniquely identified."
            )

        feature_names = list(FEATURE_COLUMNS)
        X = gdf[feature_names].values.astype(np.float64)
        y = gdf[LABEL_COLUMN].values.astype(np.int32)

        print(
            f"[feature_engineering] Built feature matrix: "
            f"X.shape={X.shape}, y.shape={y.shape}, "
            f"positive rate={y.mean():.3f}"
        )

        return X, y, feature_names

    except Exception as exc:
        raise RuntimeError(
            f"[feature_engineering.build_feature_matrix] Failed: {exc}"
        ) from exc
