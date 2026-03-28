"""
src/spatial_weights.py
=======================
Builds Queen contiguity spatial weights matrices using libpysal
and computes spatial lag variables for GeoDataFrame columns.
"""

import numpy as np
import libpysal
from libpysal.weights.spatial_lag import lag_spatial
import joblib


def build_weights_matrix(gdf):
    """
    Compute a Queen contiguity spatial weights matrix from a grid GeoDataFrame.

    Queen contiguity defines neighbors as cells that share an edge OR a vertex
    (8-connectivity for regular grids).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain a valid 'geometry' column with Polygon geometries.

    Returns
    -------
    libpysal.weights.W
        Row-standardized Queen contiguity weights matrix.
    """
    try:
        if gdf is None or gdf.empty:
            raise ValueError("Input GeoDataFrame is None or empty.")
        if "geometry" not in gdf.columns:
            raise ValueError("GeoDataFrame must contain a 'geometry' column.")

        weights = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)

        # Row-standardize: each row of the weights matrix sums to 1
        weights.transform = "r"

        n_islands = len(weights.islands)
        print(
            f"[spatial_weights] Built Queen weights matrix: "
            f"{weights.n} observations, "
            f"mean neighbors = {weights.mean_neighbors:.2f}, "
            f"islands (no neighbors) = {n_islands}"
        )

        return weights

    except Exception as exc:
        raise RuntimeError(
            f"[spatial_weights.build_weights_matrix] Failed to build Queen "
            f"weights matrix: {exc}"
        ) from exc


def compute_spatial_lag(gdf, weights, col):
    """
    Compute the spatial lag of a column using the given weights matrix.

    The spatial lag for observation i is the weighted average of its
    neighbors' values:  lag_i = Σ_j w_ij * x_j

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain the column specified by `col`.
    weights : libpysal.weights.W
        Spatial weights matrix (should be row-standardized).
    col : str
        Name of the column to compute the spatial lag for.

    Returns
    -------
    numpy.ndarray
        Array of spatial lag values with the same length as gdf.
    """
    try:
        if col not in gdf.columns:
            raise ValueError(
                f"Column '{col}' not found in GeoDataFrame. "
                f"Available columns: {list(gdf.columns)}"
            )

        values = gdf[col].values.astype(np.float64)
        lag_values = lag_spatial(weights, values)

        print(
            f"[spatial_weights] Computed spatial lag for '{col}': "
            f"mean={lag_values.mean():.4f}, std={lag_values.std():.4f}"
        )

        return lag_values

    except Exception as exc:
        raise RuntimeError(
            f"[spatial_weights.compute_spatial_lag] Failed to compute spatial "
            f"lag for column '{col}': {exc}"
        ) from exc


def save_weights(weights, path):
    """
    Serialize the spatial weights object to disk using joblib.

    Parameters
    ----------
    weights : libpysal.weights.W
        The weights matrix to persist.
    path : str
        Destination file path (e.g., 'models/queen_weights.pkl').
    """
    try:
        if weights is None:
            raise ValueError("Cannot save a None weights object.")

        joblib.dump(weights, path)
        print(
            f"[spatial_weights] Saved weights matrix ({weights.n} obs) to {path}"
        )

    except Exception as exc:
        raise RuntimeError(
            f"[spatial_weights.save_weights] Failed to save weights to "
            f"{path}: {exc}"
        ) from exc
