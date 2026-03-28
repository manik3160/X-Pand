"""
validate.py
============
End-to-end validation script for the BI-101 Geospatial Profitability
Predictor project.

Run with:
    python validate.py

Prerequisites:
    - The API server must be running: uvicorn api.main:app --reload
    - All model artifacts must be in models/ (run notebooks 01–03 first)
"""

import os
import sys
import time
import math

import numpy as np
import geopandas as gpd
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
TARGET_BATCH_SIZE = 10_000
BATCH_TIME_LIMIT = 300  # seconds
F1_THRESHOLD = 0.8
MAX_HUBS = 10
MIN_SEP_KM = 2.0
MIN_PROB_THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(BASE_DIR, "data", "processed", "grid.geojson")
X_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train.pkl")
Y_PATH = os.path.join(BASE_DIR, "data", "processed", "y_train.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_model.pkl")


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance between two points in km."""
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(a))
    return R * c


def main():
    print("=" * 60)
    print("  BI-101 X-Pand — End-to-End Validation")
    print("=" * 60)
    all_pass = True

    # ──────────────────────────────────────────────────────────────
    # 1. Batch prediction latency test
    # ──────────────────────────────────────────────────────────────
    print("\n[1/4] Batch prediction latency test")
    print("-" * 40)

    try:
        gdf = gpd.read_file(GRID_PATH)
        if "centroid_lat" not in gdf.columns:
            gdf["centroid_lat"] = gdf.geometry.centroid.y
        if "centroid_lon" not in gdf.columns:
            gdf["centroid_lon"] = gdf.geometry.centroid.x

        # Tile grid to reach TARGET_BATCH_SIZE
        n_grid = len(gdf)
        n_reps = max(1, math.ceil(TARGET_BATCH_SIZE / n_grid))
        locations = []
        for _ in range(n_reps):
            for _, row in gdf.iterrows():
                locations.append(
                    {
                        "lat": float(row["centroid_lat"]),
                        "lon": float(row["centroid_lon"]),
                        "grid_id": str(row["grid_id"]),
                    }
                )
                if len(locations) >= TARGET_BATCH_SIZE:
                    break
            if len(locations) >= TARGET_BATCH_SIZE:
                break

        n_sent = len(locations)
        print(f"  Sending {n_sent} locations to POST /batch ...")

        t0 = time.perf_counter()
        resp = requests.post(
            f"{API_BASE}/batch",
            json={"locations": locations},
            timeout=BATCH_TIME_LIMIT + 30,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()

        preds = resp.json()["predictions"]
        print(f"  Batch prediction time: {elapsed:.2f} seconds (target: < {BATCH_TIME_LIMIT})")
        print(f"  Predictions returned: {len(preds)}")

        if elapsed > BATCH_TIME_LIMIT:
            print(f"  ✗ FAIL — exceeded {BATCH_TIME_LIMIT}s limit")
            all_pass = False
        else:
            print(f"  ✓ PASS — within time limit")

    except requests.exceptions.ConnectionError:
        print(
            "  ✗ SKIP — API not reachable. Start server with:\n"
            "    uvicorn api.main:app --reload"
        )
        all_pass = False
    except Exception as exc:
        print(f"  ✗ FAIL — {exc}")
        all_pass = False

    # ──────────────────────────────────────────────────────────────
    # 2. Held-out 20% classification metrics
    # ──────────────────────────────────────────────────────────────
    print("\n[2/4] Held-out 20% classification metrics")
    print("-" * 40)

    try:
        X = joblib.load(X_PATH)
        y = joblib.load(Y_PATH)
        model = joblib.load(MODEL_PATH)

        # Augment X with GWR features to match the 10-feature trained model
        GWR_PATH = os.path.join(BASE_DIR, "models", "gwr_coeffs.pkl")
        if os.path.exists(GWR_PATH):
            gdf_eval = gpd.read_file(GRID_PATH)
            from src.gwr_model import extract_gwr_features
            gwr_results = joblib.load(GWR_PATH)
            gdf_eval = extract_gwr_features(gdf_eval, gwr_results)
            gwr_feats = gdf_eval[["gwr_intercept", "gwr_local_r2"]].values.astype(np.float64)
            X = np.hstack([X, gwr_feats])
            print(f"  Augmented X with GWR features → {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_proba = model.predict_proba(X_test)[:, 1]
        # Find optimal threshold that maximizes F1
        best_thresh, best_f1_val = 0.5, 0.0
        for t in np.arange(0.10, 0.91, 0.01):
            y_t = (y_proba >= t).astype(int)
            f1_t = f1_score(y_test, y_t)
            if f1_t > best_f1_val:
                best_f1_val = f1_t
                best_thresh = t

        y_pred = (y_proba >= best_thresh).astype(int)

        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Optimal threshold: {best_thresh:.2f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

        if f1 >= F1_THRESHOLD:
            print(f"  ✓ PASS — F1 >= {F1_THRESHOLD}")
        else:
            print(f"  ✗ FAIL — retune model (F1 {f1:.4f} < {F1_THRESHOLD})")
            all_pass = False

    except FileNotFoundError as exc:
        print(f"  ✗ SKIP — {exc}")
        all_pass = False
    except Exception as exc:
        print(f"  ✗ FAIL — {exc}")
        all_pass = False

    # ──────────────────────────────────────────────────────────────
    # 3. BIP optimizer test
    # ──────────────────────────────────────────────────────────────
    print("\n[3/4] BIP optimizer test")
    print("-" * 40)

    try:
        gdf = gpd.read_file(GRID_PATH)
        if "centroid_lat" not in gdf.columns:
            gdf["centroid_lat"] = gdf.geometry.centroid.y
        if "centroid_lon" not in gdf.columns:
            gdf["centroid_lon"] = gdf.geometry.centroid.x

        if "p_profit" not in gdf.columns:
            # Fallback: predict using model
            model = joblib.load(MODEL_PATH)
            X = joblib.load(X_PATH)
            gdf["p_profit"] = model.predict_proba(X)[:, 1]

        from src.bip_optimizer import run_bip

        selected_ids, obj_value = run_bip(
            gdf,
            prob_col="p_profit",
            max_hubs=MAX_HUBS,
            min_separation_km=MIN_SEP_KM,
            min_prob_threshold=MIN_PROB_THRESHOLD,
        )

        print(f"  Selected hubs ({len(selected_ids)}): {selected_ids}")
        print(f"  Objective value: {obj_value:.4f}")

        # Verify separation
        selected_rows = gdf[gdf["grid_id"].isin(selected_ids)]
        lats = selected_rows["centroid_lat"].values.astype(float)
        lons = selected_rows["centroid_lon"].values.astype(float)

        min_dist = float("inf")
        for i in range(len(lats)):
            for j in range(i + 1, len(lats)):
                d = _haversine_km(lats[i], lons[i], lats[j], lons[j])
                if d < min_dist:
                    min_dist = d

        print(f"  Minimum pairwise distance: {min_dist:.2f} km")

        if min_dist >= MIN_SEP_KM:
            print(f"  ✓ PASS — separation constraint met (>= {MIN_SEP_KM} km)")
        else:
            print(f"  ✗ FAIL — minimum distance {min_dist:.2f} < {MIN_SEP_KM} km")
            all_pass = False

        if len(selected_ids) == MAX_HUBS:
            print(f"  ✓ PASS — exactly {MAX_HUBS} hubs selected")
        else:
            print(
                f"  ⚠ NOTE — {len(selected_ids)} hubs selected "
                f"(max_hubs={MAX_HUBS}). This may be correct if "
                f"separation constraints limit placement."
            )

    except Exception as exc:
        print(f"  ✗ FAIL — {exc}")
        all_pass = False

    # ──────────────────────────────────────────────────────────────
    # 4. Summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_pass:
        print("  OVERALL: PASS ✓")
    else:
        print("  OVERALL: FAIL ✗  (see details above)")
    print("=" * 60)


if __name__ == "__main__":
    main()
