"""
src/lgbm_model.py
==================
LightGBM binary classifier with SMOTE oversampling, Optuna-compatible
hyperparameters, isotonic calibration, and bootstrap confidence intervals.

All predictions are parallelized via joblib.
"""

import warnings
import numpy as np
import joblib as jl
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed


# ──────────────────────────────────────────────────────────────────────
# Default LightGBM hyperparameters
# ──────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="binary",
    random_state=42,
    verbosity=-1,
)


def _make_lgbm(scale_pos_weight, extra_params=None):
    """Create a LGBMClassifier with default params + overrides."""
    params = {**DEFAULT_PARAMS, "scale_pos_weight": scale_pos_weight}
    if extra_params:
        params.update(extra_params)
    return LGBMClassifier(**params)


def train_lgbm(X_train, y_train, feature_names):
    """
    Train a calibrated LightGBM classifier with SMOTE oversampling.

    Steps:
        1. Apply SMOTE to balance the training set.
        2. Train LightGBM with 5-fold stratified CV (report F1 per fold).
        3. If mean F1 < 0.8, try class_weight='balanced' variant and
           pick whichever achieves higher mean F1.
        4. Wrap winner in CalibratedClassifierCV (isotonic).

    Parameters
    ----------
    X_train : numpy.ndarray, shape (n, p)
        Training feature matrix.
    y_train : numpy.ndarray, shape (n,)
        Training labels (0 or 1).
    feature_names : list of str
        Feature column names (for logging / SHAP).

    Returns
    -------
    calibrated_model : CalibratedClassifierCV
        Calibrated classifier ready for probability prediction.
    cv_scores : dict
        Mapping fold index → F1 score.
    """
    try:
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        if n_pos == 0:
            raise ValueError("No positive samples in y_train — cannot train.")

        scale_pos_weight = n_neg / n_pos

        print(
            f"[lgbm_model] Training data: {len(y_train)} samples, "
            f"{n_pos} positive ({n_pos / len(y_train) * 100:.1f}%), "
            f"scale_pos_weight={scale_pos_weight:.2f}"
        )

        # ── SMOTE oversampling ────────────────────────────────────────
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(
            f"[lgbm_model] After SMOTE: {len(y_resampled)} samples, "
            f"positive={int(y_resampled.sum())}"
        )

        # ── 5-fold stratified CV ──────────────────────────────────────
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lgbm_primary = _make_lgbm(scale_pos_weight)

        fold_f1_scores = {}
        fold_idx = 0
        for train_idx, val_idx in skf.split(X_resampled, y_resampled):
            X_fold_train = X_resampled[train_idx]
            y_fold_train = y_resampled[train_idx]
            X_fold_val = X_resampled[val_idx]
            y_fold_val = y_resampled[val_idx]

            fold_model = _make_lgbm(scale_pos_weight)
            fold_model.fit(X_fold_train, y_fold_train)
            y_pred = fold_model.predict(X_fold_val)
            f1 = f1_score(y_fold_val, y_pred)
            fold_f1_scores[fold_idx] = round(float(f1), 4)
            print(f"  Fold {fold_idx}: F1 = {f1:.4f}")
            fold_idx += 1

        mean_f1_primary = np.mean(list(fold_f1_scores.values()))
        print(f"[lgbm_model] Primary model mean CV F1: {mean_f1_primary:.4f}")

        best_lgbm = lgbm_primary
        best_f1 = mean_f1_primary
        best_cv_scores = fold_f1_scores

        # ── Fallback: try balanced class_weight if F1 < 0.8 ──────────
        if mean_f1_primary < 0.8:
            warnings.warn(
                f"[lgbm_model] Primary F1 ({mean_f1_primary:.4f}) < 0.8. "
                "Trying class_weight='balanced' variant.",
                RuntimeWarning,
            )
            balanced_f1_scores = {}
            fold_idx = 0
            for train_idx, val_idx in skf.split(X_resampled, y_resampled):
                X_fold_train = X_resampled[train_idx]
                y_fold_train = y_resampled[train_idx]
                X_fold_val = X_resampled[val_idx]
                y_fold_val = y_resampled[val_idx]

                bal_model = _make_lgbm(
                    scale_pos_weight,
                    extra_params={"class_weight": "balanced"},
                )
                bal_model.fit(X_fold_train, y_fold_train)
                y_pred = bal_model.predict(X_fold_val)
                f1 = f1_score(y_fold_val, y_pred)
                balanced_f1_scores[fold_idx] = round(float(f1), 4)
                fold_idx += 1

            mean_f1_balanced = np.mean(list(balanced_f1_scores.values()))
            print(
                f"[lgbm_model] Balanced variant mean CV F1: "
                f"{mean_f1_balanced:.4f}"
            )

            if mean_f1_balanced > mean_f1_primary:
                best_lgbm = _make_lgbm(
                    scale_pos_weight,
                    extra_params={"class_weight": "balanced"},
                )
                best_f1 = mean_f1_balanced
                best_cv_scores = balanced_f1_scores
                print("[lgbm_model] Using balanced variant (higher F1).")
            else:
                print("[lgbm_model] Keeping primary variant (higher F1).")

        # ── Train final model on full SMOTE data ──────────────────────
        best_lgbm.fit(X_resampled, y_resampled)

        # ── Isotonic calibration ──────────────────────────────────────
        calibrated_model = CalibratedClassifierCV(
            best_lgbm, method="isotonic", cv=5
        )
        calibrated_model.fit(X_resampled, y_resampled)

        print(
            f"[lgbm_model] Final calibrated model trained. "
            f"Best mean CV F1: {best_f1:.4f}"
        )

        return calibrated_model, best_cv_scores

    except Exception as exc:
        raise RuntimeError(
            f"[lgbm_model.train_lgbm] Failed: {exc}"
        ) from exc


def _train_and_predict_bootstrap(X_train, y_train, X_pred, seed,
                                 scale_pos_weight):
    """
    Single bootstrap iteration: subsample 80%, train LightGBM, predict
    probabilities on X_pred.
    """
    rng = np.random.RandomState(seed)
    n = len(X_train)
    sample_size = int(0.8 * n)
    indices = rng.choice(n, size=sample_size, replace=False)

    X_sub = X_train[indices]
    y_sub = y_train[indices]

    params = {**DEFAULT_PARAMS, "scale_pos_weight": scale_pos_weight,
              "random_state": seed}
    model = LGBMClassifier(**params)
    model.fit(X_sub, y_sub)

    probs = model.predict_proba(X_pred)[:, 1]
    return probs


def predict_with_ci(model, X, X_train, y_train, n_bootstrap=20):
    """
    Generate probability predictions with 95% confidence intervals via
    bootstrap ensemble.

    Parameters
    ----------
    model : CalibratedClassifierCV
        Calibrated primary model (used for point estimate cross-check).
    X : numpy.ndarray, shape (m, p)
        Feature matrix for prediction targets.
    X_train : numpy.ndarray, shape (n, p)
        Training features for bootstrap resampling.
    y_train : numpy.ndarray, shape (n,)
        Training labels.
    n_bootstrap : int
        Number of bootstrap iterations (default 20).

    Returns
    -------
    mean_prob : numpy.ndarray, shape (m,)
        Mean probability across bootstrap models.
    ci_lower : numpy.ndarray, shape (m,)
        2.5th percentile (lower bound of 95% CI).
    ci_upper : numpy.ndarray, shape (m,)
        97.5th percentile (upper bound of 95% CI).
    """
    try:
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        scale_pos_weight = n_neg / max(n_pos, 1)

        # ── Parallel bootstrap ────────────────────────────────────────
        all_probs = Parallel(n_jobs=-1, verbose=0)(
            delayed(_train_and_predict_bootstrap)(
                X_train, y_train, X, seed=42 + i, scale_pos_weight=scale_pos_weight
            )
            for i in range(n_bootstrap)
        )

        prob_matrix = np.stack(all_probs, axis=0)  # shape (n_bootstrap, m)

        mean_prob = prob_matrix.mean(axis=0)
        ci_lower = np.percentile(prob_matrix, 2.5, axis=0)
        ci_upper = np.percentile(prob_matrix, 97.5, axis=0)

        print(
            f"[lgbm_model] Bootstrap CI ({n_bootstrap} iterations): "
            f"mean_prob range [{mean_prob.min():.4f}, {mean_prob.max():.4f}], "
            f"avg CI width = {(ci_upper - ci_lower).mean():.4f}"
        )

        return mean_prob, ci_lower, ci_upper

    except Exception as exc:
        raise RuntimeError(
            f"[lgbm_model.predict_with_ci] Failed: {exc}"
        ) from exc


def save_model(model, path):
    """
    Persist a trained model to disk using joblib.

    Parameters
    ----------
    model : object
        The trained model to save.
    path : str
        Destination file path.
    """
    try:
        if model is None:
            raise ValueError("Cannot save a None model.")

        jl.dump(model, path)
        print(f"[lgbm_model] Saved model to {path}")

    except Exception as exc:
        raise RuntimeError(
            f"[lgbm_model.save_model] Failed to save to {path}: {exc}"
        ) from exc
