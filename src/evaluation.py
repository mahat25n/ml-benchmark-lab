import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


# ================================================================
# CLASSIFICATION EVALUATION
# ================================================================


def compute_confusion_components(y_true, y_pred):
    """Decompose a binary confusion matrix into TN, FP, FN, TP."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        raise ValueError(
            f"compute_confusion_components requires binary targets; "
            f"received confusion matrix of shape {cm.shape}."
        )
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _safe_metric(fn, *args, label, **kwargs):
    """Wrap a metric call and return NaN on failure rather than crashing the loop."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        warnings.warn(f"{label} could not be computed: {exc}")
        return np.nan


def _extract_proba(model, X):
    """
    Extract probability outputs from a fitted classifier.

    Returns
    -------
    y_prob  : ndarray or None
        Calibrated [0, 1] scores from predict_proba[:, 1].
        Used for Log Loss. None when predict_proba is unavailable.
    y_score : ndarray or None
        Ranking scores for ROC AUC and PR AUC.
        Uses predict_proba when available, falls back to decision_function.
        decision_function scores are valid for ranking metrics but NOT Log Loss.
    """
    y_prob, y_score = None, None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_score = y_prob
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    return y_prob, y_score


def compute_classification_metrics(
    y_true,
    y_pred,
    *,
    y_prob=None,
    y_score=None,
    average="weighted",
):
    """
    Compute a full suite of binary classification metrics.

    Parameters
    ----------
    y_true   : array-like  Ground-truth labels.
    y_pred   : array-like  Predicted class labels.
    y_prob   : array-like or None
        Calibrated [0, 1] probabilities (predict_proba[:, 1]).
        Required for Log Loss. Falls back to y_score for ROC/PR AUC
        when y_score is not explicitly supplied.
    y_score  : array-like or None
        Ranking scores for ROC AUC and PR AUC. Accepts decision_function
        output; need not be calibrated. Takes precedence over y_prob for
        rank-based metrics when both are supplied.
    average  : str
        sklearn averaging strategy for Precision, Recall, F1.

    Returns
    -------
    dict mapping metric name -> float (NaN when a metric cannot be computed)
    """
    # For rank-based metrics (ROC AUC, PR AUC): prefer explicit y_score,
    # fall back to y_prob so callers do not need to pass both redundantly.
    _score = y_score if y_score is not None else y_prob

    comp = compute_confusion_components(y_true, y_pred)
    tn, fp, fn, tp = comp["tn"], comp["fp"], comp["fn"], comp["tp"]

    # Sensitivity and Specificity derived directly from the confusion matrix.
    # zero_division guard mirrors the notebook's explicit conditional.
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy":          accuracy_score(y_true, y_pred),
        "Precision":         precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall":            recall_score(y_true, y_pred, average=average, zero_division=0),
        "Sensitivity":       sensitivity,
        "Specificity":       specificity,
        "F1 Score":          f1_score(y_true, y_pred, average=average, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC":               matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa":       cohen_kappa_score(y_true, y_pred),
        "ROC AUC": (
            _safe_metric(roc_auc_score, y_true, _score, label="ROC AUC")
            if _score is not None else np.nan
        ),
        "PR AUC": (
            _safe_metric(average_precision_score, y_true, _score, label="PR AUC")
            if _score is not None else np.nan
        ),
        "Log Loss": (
            _safe_metric(log_loss, y_true, y_prob, label="Log Loss")
            if y_prob is not None else np.nan
        ),
    }


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test, average="weighted"):
    """Fit a single classifier and return a complete metrics dict keyed by 'Model'."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob, y_score = _extract_proba(model, X_test)

    metrics = compute_classification_metrics(
        y_test, y_pred,
        y_prob=y_prob,
        y_score=y_score,
        average=average,
    )
    return {"Model": name, **metrics}


# ================================================================
# REGRESSION EVALUATION
# ================================================================


def compute_regression_metrics(y_true, y_pred):
    """
    Compute a standard regression metric suite.

    Parameters
    ----------
    y_true : array-like   Ground-truth continuous values.
    y_pred : array-like   Predicted continuous values.

    Returns
    -------
    dict with keys:
        "MAE"  : float   Mean Absolute Error
        "MSE"  : float   Mean Squared Error
        "RMSE" : float   Root Mean Squared Error
        "R2"   : float   Coefficient of Determination
        "MAPE" : float   Mean Absolute Percentage Error (%), NaN when
                         all y_true values are zero.

    Notes
    -----
    MAPE excludes samples where y_true == 0 to avoid division by zero.
    When all y_true values are zero, MAPE is returned as NaN.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true_arr, y_pred_arr)

    nonzero = y_true_arr != 0
    if nonzero.any():
        mape = float(
            np.mean(
                np.abs(
                    (y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]
                )
            ) * 100
        )
    else:
        warnings.warn(
            "compute_regression_metrics: all y_true values are zero — "
            "MAPE cannot be computed and is returned as NaN.",
            stacklevel=2,
        )
        mape = float("nan")

    return {
        "MAE":  float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "MSE":  float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2":   float(r2_score(y_true_arr, y_pred_arr)),
        "MAPE": mape,
    }


# ================================================================
# UNSUPERVISED EVALUATION
# ================================================================


def compute_clustering_metrics(X, labels):
    """
    Compute internal clustering quality metrics.

    DBSCAN noise points (label == -1) are excluded before computing metrics.
    When fewer than 2 non-noise clusters remain, all metrics are NaN.

    Parameters
    ----------
    X      : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)  Cluster label per sample.

    Returns
    -------
    dict with keys:
        "Silhouette"         : float  Higher is better (range -1 to 1).
        "Davies-Bouldin"     : float  Lower is better (>= 0).
        "Calinski-Harabasz"  : float  Higher is better (>= 0).
        "n_clusters"         : int    Number of non-noise clusters found.
        "n_noise"            : int    Number of noise points (label == -1).
    """
    X_arr    = np.asarray(X)
    labels   = np.asarray(labels)
    _nan     = float("nan")

    noise_mask   = labels == -1
    n_noise      = int(noise_mask.sum())
    X_clean      = X_arr[~noise_mask]
    labels_clean = labels[~noise_mask]
    n_clusters   = int(len(np.unique(labels_clean)))

    if n_clusters < 2:
        return {
            "Silhouette":        _nan,
            "Davies-Bouldin":    _nan,
            "Calinski-Harabasz": _nan,
            "n_clusters":        n_clusters,
            "n_noise":           n_noise,
        }

    return {
        "Silhouette":        float(_safe_metric(silhouette_score,        X_clean, labels_clean, label="Silhouette")),
        "Davies-Bouldin":    float(_safe_metric(davies_bouldin_score,    X_clean, labels_clean, label="Davies-Bouldin")),
        "Calinski-Harabasz": float(_safe_metric(calinski_harabasz_score, X_clean, labels_clean, label="Calinski-Harabasz")),
        "n_clusters":        n_clusters,
        "n_noise":           n_noise,
    }


# ================================================================
# FORECASTING EVALUATION
# ================================================================


def compute_forecast_metrics(y_true, y_pred):
    """
    Compute a forecasting metric suite for time-series evaluation.

    Parameters
    ----------
    y_true : array-like  Observed values.
    y_pred : array-like  Model forecasts.

    Returns
    -------
    dict with keys:
        "MAE"   : float   Mean Absolute Error
        "RMSE"  : float   Root Mean Squared Error
        "MAPE"  : float   Mean Absolute Percentage Error (%), NaN when
                          all y_true values are zero.
        "SMAPE" : float   Symmetric MAPE (%), NaN when all denominators
                          are zero (both y_true and y_pred are zero).

    Notes
    -----
    MAPE excludes samples where y_true == 0.
    SMAPE = 2 * mean(|y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    SMAPE excludes samples where both y_true and y_pred are zero.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true_arr, y_pred_arr))

    nonzero = y_true_arr != 0
    if nonzero.any():
        mape = float(
            np.mean(
                np.abs(
                    (y_true_arr[nonzero] - y_pred_arr[nonzero]) / y_true_arr[nonzero]
                )
            ) * 100
        )
    else:
        warnings.warn(
            "compute_forecast_metrics: all y_true values are zero — "
            "MAPE cannot be computed and is returned as NaN.",
            stacklevel=2,
        )
        mape = float("nan")

    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    nonzero_denom = denom != 0
    if nonzero_denom.any():
        smape = float(
            np.mean(
                2 * np.abs(y_true_arr[nonzero_denom] - y_pred_arr[nonzero_denom])
                / denom[nonzero_denom]
            ) * 100
        )
    else:
        warnings.warn(
            "compute_forecast_metrics: all denominators are zero — "
            "SMAPE cannot be computed and is returned as NaN.",
            stacklevel=2,
        )
        smape = float("nan")

    return {
        "MAE":   float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE":  float(np.sqrt(mse)),
        "MAPE":  mape,
        "SMAPE": smape,
    }


def compute_pca_metrics(pca_model):
    """
    Extract summary statistics from a fitted PCA model.

    Parameters
    ----------
    pca_model : fitted sklearn PCA instance

    Returns
    -------
    dict with keys:
        "n_components"              : int    Number of components retained.
        "cum_explained_variance_pct": float  Cumulative explained variance (0–100).
    """
    evr = np.asarray(pca_model.explained_variance_ratio_)
    return {
        "n_components":               int(len(evr)),
        "cum_explained_variance_pct": float(evr.sum() * 100),
    }
