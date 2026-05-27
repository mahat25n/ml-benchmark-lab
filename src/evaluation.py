import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
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
# REGRESSION EVALUATION  (not yet implemented)
# Future metrics: MAE, MSE, RMSE, R², Adjusted R², MAPE,
#                 Max Error, Explained Variance Score
# ================================================================

# def compute_regression_metrics(y_true, y_pred):
#     raise NotImplementedError

# def train_and_evaluate_regressor(name, model, X_train, X_test, y_train, y_test):
#     raise NotImplementedError


# ================================================================
# UNSUPERVISED EVALUATION  (not yet implemented)
# Future metrics: Silhouette Score, Davies-Bouldin Index,
#                 Calinski-Harabasz Index, Adjusted Rand Index
# ================================================================

# def compute_clustering_metrics(X, labels, y_true=None):
#     raise NotImplementedError
