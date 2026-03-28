"""
src/city_grids.py
==================
Multi-city grid generation for the X-Pand.AI Geospatial Profitability
Predictor.

Defines bounding boxes for multiple Indian cities and generates grids
with synthetic features on-the-fly so that the LightGBM model can
score any city dynamically.

Delhi uses real WorldPop + OSM data (already processed).
Other cities get plausible synthetic features with city-specific
population & competition estimates.
"""

import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon


# ──────────────────────────────────────────────────────────────────────
# City definitions — (min_lon, min_lat, max_lon, max_lat)
# Plus metadata for realistic synthetic feature generation
# ──────────────────────────────────────────────────────────────────────

CITY_CONFIGS = {
    "delhi": {
        "name": "Delhi NCR",
        "bbox": (76.85, 28.40, 77.35, 28.85),
        "map_center": {"lat": 28.65, "lon": 77.10},
        "zoom": 10.2,
        "pop_mean": 12000,
        "pop_std": 5000,
        "competitor_lambda": 4,
        "income_range": (0.4, 1.0),
        "is_real_data": True,  # Delhi has real WorldPop + OSM data
    },
    "jaipur": {
        "name": "Jaipur",
        "bbox": (75.70, 26.82, 75.90, 26.98),
        "map_center": {"lat": 26.90, "lon": 75.80},
        "zoom": 11.5,
        "pop_mean": 8000,
        "pop_std": 3500,
        "competitor_lambda": 3,
        "income_range": (0.35, 0.85),
        "is_real_data": False,
    },
    "jalandhar": {
        "name": "Jalandhar",
        "bbox": (75.50, 31.28, 75.65, 31.40),
        "map_center": {"lat": 31.34, "lon": 75.575},
        "zoom": 12.0,
        "pop_mean": 6500,
        "pop_std": 2500,
        "competitor_lambda": 2,
        "income_range": (0.3, 0.80),
        "is_real_data": False,
    },
    "kolkata": {
        "name": "Kolkata",
        "bbox": (88.28, 22.45, 88.45, 22.65),
        "map_center": {"lat": 22.55, "lon": 88.365},
        "zoom": 11.5,
        "pop_mean": 15000,
        "pop_std": 6000,
        "competitor_lambda": 5,
        "income_range": (0.3, 0.85),
        "is_real_data": False,
    },
    "indore": {
        "name": "Indore",
        "bbox": (75.78, 22.65, 75.92, 22.78),
        "map_center": {"lat": 22.715, "lon": 75.85},
        "zoom": 12.0,
        "pop_mean": 7000,
        "pop_std": 3000,
        "competitor_lambda": 2,
        "income_range": (0.35, 0.85),
        "is_real_data": False,
    },
}


def get_city_list():
    """Return a list of (city_key, display_name) tuples."""
    return [(k, v["name"]) for k, v in CITY_CONFIGS.items()]


def get_city_config(city_key):
    """Return config dict for a city, or raise KeyError."""
    if city_key not in CITY_CONFIGS:
        raise KeyError(
            f"Unknown city '{city_key}'. "
            f"Available: {list(CITY_CONFIGS.keys())}"
        )
    return CITY_CONFIGS[city_key]


def build_city_grid(city_key, cell_size_meters=500):
    """
    Build a GeoDataFrame grid for the given city.

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Grid with columns: grid_id, centroid_lat, centroid_lon, geometry,
        coordinates (for PyDeck PolygonLayer).
    """
    cfg = get_city_config(city_key)
    bbox = cfg["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox

    mean_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lat = 111_000.0
    meters_per_deg_lon = 111_000.0 * math.cos(math.radians(mean_lat))

    cell_height_deg = cell_size_meters / meters_per_deg_lat
    cell_width_deg = cell_size_meters / meters_per_deg_lon

    n_cols = int(math.ceil((max_lon - min_lon) / cell_width_deg))
    n_rows = int(math.ceil((max_lat - min_lat) / cell_height_deg))

    grid_ids = []
    geometries = []
    centroid_lats = []
    centroid_lons = []
    idx = 0

    for row in range(n_rows):
        for col in range(n_cols):
            x_min = min_lon + col * cell_width_deg
            x_max = min_lon + (col + 1) * cell_width_deg
            y_min = min_lat + row * cell_height_deg
            y_max = min_lat + (row + 1) * cell_height_deg

            polygon = Polygon([
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max),
                (x_min, y_min),
            ])

            centroid = polygon.centroid
            grid_ids.append(f"{city_key}_{idx:05d}")
            geometries.append(polygon)
            centroid_lats.append(centroid.y)
            centroid_lons.append(centroid.x)
            idx += 1

    gdf = gpd.GeoDataFrame(
        {
            "grid_id": grid_ids,
            "centroid_lat": centroid_lats,
            "centroid_lon": centroid_lons,
        },
        geometry=geometries,
        crs="EPSG:4326",
    )

    # Pre-compute polygon coordinates for PyDeck PolygonLayer
    gdf["coordinates"] = gdf.geometry.apply(
        lambda geom: [list(c) for c in geom.exterior.coords]
    )

    print(
        f"[city_grids] Built grid for {cfg['name']}: "
        f"{len(gdf)} cells ({n_rows} rows x {n_cols} cols)"
    )
    return gdf


def generate_city_features(gdf, city_key, feature_names):
    """
    Generate synthetic feature values for a non-Delhi city grid so that
    the existing LightGBM model can score it.

    The feature columns are generated to match the training feature order
    exactly.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must have grid_id, centroid_lat, centroid_lon columns.
    city_key : str
        City identifier (e.g. 'jaipur').
    feature_names : list of str
        Ordered feature names the LightGBM model expects.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns = ['grid_id'] + feature_names.
    """
    import pandas as pd

    cfg = get_city_config(city_key)
    n = len(gdf)
    # Use a city-specific seed for reproducibility but variety
    seed = hash(city_key) % (2**31)
    rng = np.random.RandomState(seed)

    features = {"grid_id": gdf["grid_id"].values}

    # Latitude/longitude-based spatial variation for realism
    lats = gdf["centroid_lat"].values
    lons = gdf["centroid_lon"].values
    lat_norm = (lats - lats.min()) / max(lats.max() - lats.min(), 1e-9)
    lon_norm = (lons - lons.min()) / max(lons.max() - lons.min(), 1e-9)

    # Distance from city center creates urban-core-to-periphery gradient
    center_lat = cfg["map_center"]["lat"]
    center_lon = cfg["map_center"]["lon"]
    dist_from_center = np.sqrt(
        (lats - center_lat) ** 2 + (lons - center_lon) ** 2
    )
    # Normalise: 0 = center, 1 = edge
    dist_norm = dist_from_center / max(dist_from_center.max(), 1e-9)

    # Core urban areas (closer to center) have higher density
    urban_factor = 1.0 - dist_norm  # 1 at center, 0 at edge

    for feat in feature_names:
        if feat == "pop_density":
            base = cfg["pop_mean"] * (0.3 + 0.7 * urban_factor)
            noise = rng.normal(0, cfg["pop_std"] * 0.3, n)
            features[feat] = np.clip(base + noise, 100, None)

        elif feat == "pop_density_log":
            if "pop_density" in features:
                features[feat] = np.log1p(features["pop_density"])
            else:
                features[feat] = rng.uniform(5, 10, n)

        elif feat == "competitor_count":
            lam = cfg["competitor_lambda"] * (0.2 + 0.8 * urban_factor)
            features[feat] = rng.poisson(lam).astype(float)

        elif feat == "competitor_density_1km":
            lam = cfg["competitor_lambda"] * 2 * (0.3 + 0.7 * urban_factor)
            features[feat] = rng.poisson(lam).astype(float)

        elif feat == "nearest_competitor_km":
            # Denser areas → closer competitors
            base = 0.1 + 2.5 * dist_norm
            noise = rng.uniform(-0.05, 0.3, n)
            features[feat] = np.clip(base + noise, 0.01, 5.0)

        elif feat == "market_saturation":
            # Derived from competitor density
            if "competitor_density_1km" in features:
                cd = features["competitor_density_1km"]
                features[feat] = np.clip(cd / (np.percentile(cd, 95) + 1e-9), 0, 1)
            else:
                features[feat] = rng.uniform(0, 1, n)

        elif feat == "opportunity_gap":
            if "pop_density" in features and "market_saturation" in features:
                pop_norm = features["pop_density"] / max(
                    np.percentile(features["pop_density"], 95), 1
                )
                features[feat] = np.clip(
                    pop_norm - features["market_saturation"], 0, 1
                )
            else:
                features[feat] = rng.uniform(0, 1, n)

        elif feat == "income_index":
            lo, hi = cfg["income_range"]
            # Higher near center
            base = lo + (hi - lo) * (0.3 + 0.7 * urban_factor)
            noise = rng.uniform(-0.05, 0.05, n)
            features[feat] = np.clip(base + noise, lo, hi)

        elif feat == "road_density":
            base = 0.3 + 0.6 * urban_factor
            noise = rng.uniform(-0.1, 0.1, n)
            features[feat] = np.clip(base + noise, 0.1, 1.0)

        elif feat == "transit_stops":
            lam = 1 + 4 * urban_factor
            features[feat] = rng.poisson(lam).astype(float)

        elif feat == "warehouse_proximity_km":
            # Closer to center = shorter warehouse distance
            base = 1.0 + 12.0 * dist_norm
            noise = rng.uniform(-0.5, 1.0, n)
            features[feat] = np.clip(base + noise, 0.3, 20.0)

        elif feat == "lag_profitable":
            # Spatial lag — synthetic approximation using neighbor averaging
            features[feat] = 0.1 * urban_factor + rng.uniform(0, 0.05, n)

        elif feat == "lag_pop_density":
            if "pop_density" in features:
                # Smooth version of pop_density
                features[feat] = features["pop_density"] * (
                    0.8 + 0.2 * rng.uniform(0, 1, n)
                )
            else:
                features[feat] = rng.uniform(2000, 15000, n)

        elif feat in ("gwr_intercept",):
            features[feat] = rng.uniform(-0.1, 0.3, n)

        elif feat in ("gwr_local_r2",):
            features[feat] = rng.uniform(0.3, 0.9, n)

        else:
            # Unknown feature — fill with zeros
            print(f"[city_grids] WARNING: Unknown feature '{feat}' — filling with 0")
            features[feat] = np.zeros(n)

    df = pd.DataFrame(features)
    print(
        f"[city_grids] Generated {len(feature_names)} features for "
        f"{cfg['name']} ({n} cells)"
    )
    return df
