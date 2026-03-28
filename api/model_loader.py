"""
api/model_loader.py
====================
One-time loader for all ML artefacts.  Called once during FastAPI
lifespan startup -- never per request.

Now supports multi-city grids: Delhi loads from real data, other cities
are generated on-the-fly with synthetic features scored by the same
LightGBM model.
"""

import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd

from src.thompson_sampling import ThompsonSampler
from src.city_grids import (
    CITY_CONFIGS,
    build_city_grid,
    generate_city_features,
)


# ──────────────────────────────────────────────────────────────────────
# Module-level globals (initialised to None)
# ──────────────────────────────────────────────────────────────────────

lgbm_model = None
gwr_results = None
shap_explainer = None
weights_matrix = None

# Delhi-specific (real data)
grid_gdf = None
features_df = None
thompson_sampler = None

# Multi-city storage
# city_key -> { "grid_gdf": GeoDataFrame, "features_df": DataFrame,
#               "thompson_sampler": ThompsonSampler }
city_data = {}


# ──────────────────────────────────────────────────────────────────────
# Accessor helpers — used by main.py endpoints
# ──────────────────────────────────────────────────────────────────────

def get_city_grid(city_key):
    """Return the grid GeoDataFrame for a city."""
    if city_key in city_data:
        return city_data[city_key]["grid_gdf"]
    raise KeyError(f"City '{city_key}' not loaded.")


def get_city_features(city_key):
    """Return the features DataFrame for a city."""
    if city_key in city_data:
        return city_data[city_key]["features_df"]
    raise KeyError(f"City '{city_key}' not loaded.")


def get_city_thompson(city_key):
    """Return the ThompsonSampler for a city."""
    if city_key in city_data:
        return city_data[city_key]["thompson_sampler"]
    raise KeyError(f"City '{city_key}' not loaded.")


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────

def load_all_models():
    """
    Load every persisted artefact into module-level globals so that
    endpoint handlers can access them without disk I/O.

    Also generates grids + features for non-Delhi cities so they can
    be scored by the same LightGBM model.
    """
    global lgbm_model, gwr_results, shap_explainer, weights_matrix
    global grid_gdf, features_df, thompson_sampler
    global city_data

    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # ── Paths ────────────────────────────────────────────────────
        models_dir = os.path.join(base_dir, "models")
        data_dir = os.path.join(base_dir, "data", "processed")

        artefacts = {
            "lgbm_model": os.path.join(models_dir, "lgbm_model.pkl"),
            "gwr_coeffs": os.path.join(models_dir, "gwr_coeffs.pkl"),
            "shap_explainer": os.path.join(models_dir, "shap_explainer.pkl"),
            "weights_matrix": os.path.join(models_dir, "weights_matrix.pkl"),
            "grid_geojson": os.path.join(data_dir, "grid.geojson"),
            "features_pkl": os.path.join(data_dir, "features.pkl"),
        }

        # ── Verify all files exist before loading ────────────────────
        missing = [
            name for name, path in artefacts.items() if not os.path.isfile(path)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required artefacts: {missing}. "
                f"Expected paths: "
                + ", ".join(f"{n} -> {artefacts[n]}" for n in missing)
            )

        # ── Load joblib artefacts ────────────────────────────────────
        lgbm_model = joblib.load(artefacts["lgbm_model"])
        print("[model_loader] [OK] lgbm_model loaded")

        gwr_results = joblib.load(artefacts["gwr_coeffs"])
        print("[model_loader] [OK] gwr_results loaded")

        shap_explainer = joblib.load(artefacts["shap_explainer"])
        print("[model_loader] [OK] shap_explainer loaded")

        weights_matrix = joblib.load(artefacts["weights_matrix"])
        print("[model_loader] [OK] weights_matrix loaded")

        # ── Load Delhi spatial data ──────────────────────────────────
        grid_gdf = gpd.read_file(artefacts["grid_geojson"])
        print(f"[model_loader] [OK] grid_gdf loaded ({len(grid_gdf)} cells)")

        features_df = pd.read_pickle(artefacts["features_pkl"])
        print(f"[model_loader] [OK] features_df loaded (shape {features_df.shape})")

        # ── Initialise Delhi Thompson Sampler ────────────────────────
        if "grid_id" not in grid_gdf.columns:
            raise RuntimeError(
                "grid_gdf does not contain a 'grid_id' column. "
                f"Available columns: {list(grid_gdf.columns)}"
            )

        all_grid_ids = grid_gdf["grid_id"].tolist()
        thompson_sampler = ThompsonSampler(all_grid_ids)
        print(
            f"[model_loader] [OK] ThompsonSampler initialised with "
            f"{len(all_grid_ids)} grid cells"
        )

        # ── Store Delhi in city_data ─────────────────────────────────
        delhi_grid = grid_gdf.copy()
        if "centroid_lat" not in delhi_grid.columns:
            delhi_grid["centroid_lat"] = delhi_grid.geometry.centroid.y
        if "centroid_lon" not in delhi_grid.columns:
            delhi_grid["centroid_lon"] = delhi_grid.geometry.centroid.x
        if "coordinates" not in delhi_grid.columns:
            delhi_grid["coordinates"] = delhi_grid.geometry.apply(
                lambda geom: [list(c) for c in geom.exterior.coords]
            )

        city_data["delhi"] = {
            "grid_gdf": delhi_grid,
            "features_df": features_df.copy(),
            "thompson_sampler": thompson_sampler,
        }

        # ── Get feature names from Delhi (the training set) ──────────
        feature_names = [c for c in features_df.columns if c != "grid_id"]
        print(f"[model_loader] Feature names ({len(feature_names)}): {feature_names}")

        # ── Generate grids + features for other cities ───────────────
        for city_key, cfg in CITY_CONFIGS.items():
            if city_key == "delhi":
                continue  # Already loaded from real data

            print(f"[model_loader] Generating grid for {cfg['name']}...")
            city_grid = build_city_grid(city_key)
            city_feats = generate_city_features(
                city_grid, city_key, feature_names
            )
            city_ts = ThompsonSampler(city_grid["grid_id"].tolist())

            city_data[city_key] = {
                "grid_gdf": city_grid,
                "features_df": city_feats,
                "thompson_sampler": city_ts,
            }
            print(
                f"[model_loader] [OK] {cfg['name']}: "
                f"{len(city_grid)} cells, {len(feature_names)} features"
            )

        print(
            f"[model_loader] == All artefacts loaded successfully == "
            f"Cities: {list(city_data.keys())}"
        )

    except Exception as exc:
        raise RuntimeError(
            f"[model_loader.load_all_models] Failed to load artefacts: {exc}"
        ) from exc
