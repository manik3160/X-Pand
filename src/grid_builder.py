"""
src/grid_builder.py
====================
Constructs a 500m × 500m geospatial grid over a bounding box.
Each cell is a Shapely Polygon stored in a GeoDataFrame with EPSG:4326 CRS.
"""

import math
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def build_grid(bbox, cell_size_meters=500):
    """
    Build a regular grid of square cells over a geographic bounding box.

    Parameters
    ----------
    bbox : tuple of float
        (min_lon, min_lat, max_lon, max_lat) in EPSG:4326 degrees.
    cell_size_meters : int, optional
        Side length of each grid cell in meters (default 500).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with columns: grid_id, geometry, centroid_lat, centroid_lon.
        CRS is set to EPSG:4326.
    """
    try:
        min_lon, min_lat, max_lon, max_lat = bbox

        # ── Validate bounding box ─────────────────────────────────────
        if min_lon >= max_lon:
            raise ValueError(
                f"min_lon ({min_lon}) must be less than max_lon ({max_lon})."
            )
        if min_lat >= max_lat:
            raise ValueError(
                f"min_lat ({min_lat}) must be less than max_lat ({max_lat})."
            )

        # ── Convert cell size from meters to degrees ──────────────────
        mean_lat = (min_lat + max_lat) / 2.0
        meters_per_deg_lat = 111_000.0
        meters_per_deg_lon = 111_000.0 * math.cos(math.radians(mean_lat))

        if meters_per_deg_lon == 0:
            raise ValueError(
                f"Cannot build grid at latitude {mean_lat}° — cosine is zero."
            )

        cell_height_deg = cell_size_meters / meters_per_deg_lat
        cell_width_deg = cell_size_meters / meters_per_deg_lon

        # ── Count rows and columns ────────────────────────────────────
        n_cols = int(math.ceil((max_lon - min_lon) / cell_width_deg))
        n_rows = int(math.ceil((max_lat - min_lat) / cell_height_deg))

        if n_cols <= 0 or n_rows <= 0:
            raise ValueError(
                f"Grid dimensions are non-positive (cols={n_cols}, rows={n_rows}). "
                "Check bbox and cell_size_meters."
            )

        # ── Build cells ───────────────────────────────────────────────
        grid_ids = []
        geometries = []
        centroid_lats = []
        centroid_lons = []
        cell_index = 0

        for row in range(n_rows):
            for col in range(n_cols):
                # Cell corner coordinates
                x_min = min_lon + col * cell_width_deg
                x_max = min_lon + (col + 1) * cell_width_deg
                y_min = min_lat + row * cell_height_deg
                y_max = min_lat + (row + 1) * cell_height_deg

                polygon = Polygon([
                    (x_min, y_min),
                    (x_max, y_min),
                    (x_max, y_max),
                    (x_min, y_max),
                    (x_min, y_min),
                ])

                centroid = polygon.centroid
                grid_ids.append(f"cell_{cell_index:04d}")
                geometries.append(polygon)
                centroid_lats.append(centroid.y)
                centroid_lons.append(centroid.x)
                cell_index += 1

        # ── Assemble GeoDataFrame ─────────────────────────────────────
        gdf = gpd.GeoDataFrame(
            {
                "grid_id": grid_ids,
                "centroid_lat": centroid_lats,
                "centroid_lon": centroid_lons,
            },
            geometry=geometries,
            crs="EPSG:4326",
        )

        print(
            f"[grid_builder] Built grid with {len(gdf)} cells "
            f"({n_rows} rows × {n_cols} cols) | "
            f"cell size = {cell_size_meters}m × {cell_size_meters}m"
        )

        return gdf

    except Exception as exc:
        raise RuntimeError(
            f"[grid_builder.build_grid] Failed to build grid for bbox={bbox}, "
            f"cell_size_meters={cell_size_meters}: {exc}"
        ) from exc


def save_grid(gdf, path):
    """
    Persist the grid GeoDataFrame as a GeoJSON file.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The grid to save.
    path : str
        Destination file path (should end in .geojson).
    """
    try:
        if gdf is None or gdf.empty:
            raise ValueError("Cannot save an empty or None GeoDataFrame.")

        gdf.to_file(path, driver="GeoJSON")
        print(
            f"[grid_builder] Saved {len(gdf)} grid cells to {path}"
        )

    except Exception as exc:
        raise RuntimeError(
            f"[grid_builder.save_grid] Failed to save grid to {path}: {exc}"
        ) from exc
