"""
api/model_loader.py
====================
One-time loader for all ML artefacts.  Called once during FastAPI
lifespan startup — never per request.
"""

import os
import joblib
import geopandas as gpd
import pandas as pd

from src.thompson_sampling import ThompsonSampler


# ──────────────────────────────────────────────────────────────────────
# Module-level globals (initialised to None)
# ──────────────────────────────────────────────────────────────────────

lgbm_model = None
gwr_results = None
shap_explainer = None
weights_matrix = None
grid_gdf = None
features_df = None
thompson_sampler = None


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────

def load_all_models():
    """
    Load every persisted artefact into module-level globals so that
    endpoint handlers can access them without disk I/O.

    Raises
    ------
    RuntimeError
        If any required file is missing or cannot be deserialised.
    """
    global lgbm_model, gwr_results, shap_explainer, weights_matrix
    global grid_gdf, features_df, thompson_sampler

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
                + ", ".join(f"{n} → {artefacts[n]}" for n in missing)
            )

        # ── Load joblib artefacts ────────────────────────────────────
        lgbm_model = joblib.load(artefacts["lgbm_model"])
        print(f"[model_loader] ✓ lgbm_model loaded from {artefacts['lgbm_model']}")

        gwr_results = joblib.load(artefacts["gwr_coeffs"])
        print(f"[model_loader] ✓ gwr_results loaded from {artefacts['gwr_coeffs']}")

        shap_explainer = joblib.load(artefacts["shap_explainer"])
        print(f"[model_loader] ✓ shap_explainer loaded from {artefacts['shap_explainer']}")

        weights_matrix = joblib.load(artefacts["weights_matrix"])
        print(f"[model_loader] ✓ weights_matrix loaded from {artefacts['weights_matrix']}")

        # ── Load spatial data ────────────────────────────────────────
        grid_gdf = gpd.read_file(artefacts["grid_geojson"])
        print(
            f"[model_loader] ✓ grid_gdf loaded from {artefacts['grid_geojson']} "
            f"({len(grid_gdf)} cells)"
        )

        features_df = pd.read_pickle(artefacts["features_pkl"])
        print(
            f"[model_loader] ✓ features_df loaded from {artefacts['features_pkl']} "
            f"(shape {features_df.shape})"
        )

        # ── Initialise Thompson Sampler ──────────────────────────────
        if "grid_id" not in grid_gdf.columns:
            raise RuntimeError(
                "grid_gdf does not contain a 'grid_id' column. "
                f"Available columns: {list(grid_gdf.columns)}"
            )

        all_grid_ids = grid_gdf["grid_id"].tolist()
        thompson_sampler = ThompsonSampler(all_grid_ids)
        print(
            f"[model_loader] ✓ ThompsonSampler initialised with "
            f"{len(all_grid_ids)} grid cells"
        )

        print("[model_loader] ══ All artefacts loaded successfully ══")

    except Exception as exc:
        raise RuntimeError(
            f"[model_loader.load_all_models] Failed to load artefacts: {exc}"
        ) from exc
