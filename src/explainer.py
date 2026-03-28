"""
src/explainer.py
=================
SHAP-based interpretability for the LightGBM profitability classifier.

Provides global feature importance, local per-cell explanations, and
top-driver extraction for stakeholder-facing reports.
"""

import numpy as np
import shap
import joblib


def build_shap_explainer(lgbm_model, X_train):
    """
    Build a SHAP TreeExplainer from the calibrated LightGBM model and
    compute SHAP values over the training set.

    Parameters
    ----------
    lgbm_model : CalibratedClassifierCV
        Calibrated classifier wrapping a LightGBM estimator.
    X_train : numpy.ndarray, shape (n, p)
        Training feature matrix used for background distribution.

    Returns
    -------
    explainer : shap.TreeExplainer
        SHAP explainer bound to the base LightGBM estimator.
    shap_values : numpy.ndarray, shape (n, p)
        SHAP values for the positive class across all training samples.
    """
    try:
        # ── Extract base LightGBM from calibration wrapper ────────────
        if not hasattr(lgbm_model, "calibrated_classifiers_"):
            raise ValueError(
                "Expected a CalibratedClassifierCV model with "
                "'calibrated_classifiers_' attribute. Got: "
                f"{type(lgbm_model).__name__}"
            )

        base_model = lgbm_model.calibrated_classifiers_[0].estimator

        print(
            f"[explainer] Extracted base LightGBM estimator: "
            f"{type(base_model).__name__}"
        )

        # ── Build TreeExplainer ───────────────────────────────────────
        explainer = shap.TreeExplainer(base_model)

        # ── Compute SHAP values ───────────────────────────────────────
        shap_values = explainer.shap_values(X_train)

        # Binary classification: shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values = shap_values[1]  # positive class
                print(
                    "[explainer] Binary classification detected — "
                    "using positive-class SHAP values."
                )
            else:
                raise ValueError(
                    f"Expected 2 classes in shap_values list, "
                    f"got {len(shap_values)}."
                )

        shap_values = np.asarray(shap_values, dtype=np.float64)

        print(
            f"[explainer] SHAP values computed: shape={shap_values.shape}, "
            f"mean abs SHAP = {np.abs(shap_values).mean():.6f}"
        )

        return explainer, shap_values

    except Exception as exc:
        raise RuntimeError(
            f"[explainer.build_shap_explainer] Failed: {exc}"
        ) from exc


def get_top_drivers(explainer, X_row, feature_names, top_n=3):
    """
    Identify the top contributing features for a single prediction.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        SHAP explainer (from ``build_shap_explainer``).
    X_row : numpy.ndarray, shape (p,) or (1, p)
        Feature vector for one grid cell.
    feature_names : list of str
        Ordered list of feature column names.
    top_n : int, optional
        Number of top drivers to return (default 3).

    Returns
    -------
    list of dict
        Each dict has keys ``"feature"`` (str) and ``"impact"`` (float).
        Sorted by absolute impact descending, truncated to ``top_n``.
    """
    try:
        # ── Reshape to (1, n_features) if needed ──────────────────────
        X_row = np.asarray(X_row, dtype=np.float64)
        if X_row.ndim == 1:
            X_row = X_row.reshape(1, -1)

        if X_row.shape[1] != len(feature_names):
            raise ValueError(
                f"X_row has {X_row.shape[1]} features but "
                f"feature_names has {len(feature_names)} entries."
            )

        # ── Compute SHAP values for this single row ──────────────────
        sv = explainer.shap_values(X_row)

        # Handle binary classification list output
        if isinstance(sv, list):
            if len(sv) == 2:
                sv = sv[1]  # positive class
            else:
                raise ValueError(
                    f"Expected 2 classes in shap_values list, got {len(sv)}."
                )

        sv = np.asarray(sv, dtype=np.float64).flatten()

        if len(sv) != len(feature_names):
            raise ValueError(
                f"SHAP values length ({len(sv)}) does not match "
                f"feature_names length ({len(feature_names)})."
            )

        # ── Build driver list sorted by |impact| ─────────────────────
        drivers = [
            {"feature": feature_names[i], "impact": float(sv[i])}
            for i in range(len(feature_names))
        ]
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)

        top_drivers = drivers[:top_n]

        return top_drivers

    except Exception as exc:
        raise RuntimeError(
            f"[explainer.get_top_drivers] Failed: {exc}"
        ) from exc


def save_explainer(explainer, path):
    """
    Persist the SHAP explainer object to disk.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        The SHAP explainer to save.
    path : str
        Destination file path.
    """
    try:
        if explainer is None:
            raise ValueError("Cannot save a None explainer.")

        joblib.dump(explainer, path)
        print(f"[explainer] Saved SHAP explainer to {path}")

    except Exception as exc:
        raise RuntimeError(
            f"[explainer.save_explainer] Failed to save to {path}: {exc}"
        ) from exc
