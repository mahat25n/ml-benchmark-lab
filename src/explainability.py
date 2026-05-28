import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _validate_feature_names(feature_names, n_features):
    """
    Return a list of feature name strings of length n_features.

    If feature_names is None, generates generic labels ("feature_0", ...).
    Raises ValueError when the supplied list length does not match n_features.
    """
    if feature_names is None:
        return [f"feature_{i}" for i in range(n_features)]
    names = list(feature_names)
    if len(names) != n_features:
        raise ValueError(
            f"feature_names has {len(names)} entries but the model expects "
            f"{n_features} features."
        )
    return names


def _coef_to_importance(coef):
    """
    Convert a model's coef_ array to a 1-D importance vector.

    Binary classification   : coef_ is (1, n_features) or (n_features,).
                              Returns the raw signed values.
    Multiclass classification: coef_ is (n_classes, n_features).
                              Returns the mean absolute value across classes.
                              Sign is undefined for multiclass; per-class
                              decomposition is deferred to a future release.

    Returns
    -------
    importance : ndarray, shape (n_features,)
    source     : str describing the extraction path
    """
    arr = np.asarray(coef)
    if arr.ndim == 1:
        return arr, "coef_ (binary)"
    if arr.shape[0] == 1:
        return arr[0], "coef_ (binary)"
    # Multiclass: mean abs across classes for magnitude ranking
    mean_abs = np.abs(arr).mean(axis=0)
    return mean_abs, "coef_ (mean abs, multiclass)"


def _extract_model_importance(model):
    """
    Extract raw importance values from a fitted model.

    Priority order:
      1. feature_importances_  — tree-based models (RF, GBM, XGBoost, …)
      2. coef_                 — linear models (LogReg, SVM-linear, Ridge, …)

    Returns
    -------
    importance : ndarray, shape (n_features,)
    source     : str
    signed     : bool — True when importance values carry directional meaning

    Raises
    ------
    ValueError  model exposes neither attribute.
    """
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_)
        return vals, "feature_importances_", False

    if hasattr(model, "coef_"):
        vals, source = _coef_to_importance(model.coef_)
        # Signed only when binary (values retain positive/negative direction)
        signed = "multiclass" not in source
        return vals, source, signed

    raise ValueError(
        f"{type(model).__name__} exposes neither 'feature_importances_' nor "
        "'coef_'. Use compute_permutation_importance() for model-agnostic "
        "importance, or choose a tree-based or linear estimator."
    )


def _add_rank(df, by):
    """Insert a 1-based rank column computed from abs(by) descending."""
    df = df.copy()
    df.insert(0, "rank", df[by].abs().rank(ascending=False, method="min").astype(int))
    return df.sort_values("rank").reset_index(drop=True)


# ================================================================
# PUBLIC API
# ================================================================


def get_feature_importance(model, feature_names=None, *, normalize=False):
    """
    Extract built-in feature importance from a fitted model.

    Supports tree-based models (feature_importances_) and linear models
    (coef_). For coef_-based importance, the sign is preserved for binary
    classifiers — positive values push toward the positive class, negative
    values push toward the negative class — enabling downstream positive /
    negative contribution analysis.

    Parameters
    ----------
    model        : fitted sklearn-compatible estimator
        Must expose feature_importances_ or coef_. Call
        compute_permutation_importance() for model-agnostic importance.
    feature_names : list[str] or None
        Column names for the feature matrix. None generates generic labels.
        Must match the number of features the model was trained on.
    normalize    : bool
        When True, scales importance values to the [0, 1] range based on
        abs magnitude. The signed direction is preserved in the "importance"
        column; "importance_norm" reflects magnitude only.
        Default False.

    Returns
    -------
    pd.DataFrame with columns:
        "rank"           : int    1-based rank by absolute magnitude (1 = most important)
        "feature"        : str    feature name
        "importance"     : float  raw value; signed for coef_-based sources
        "abs_importance" : float  absolute value; use for magnitude comparison
        "source"         : str    attribute used ("feature_importances_",
                                   "coef_ (binary)", or "coef_ (mean abs, multiclass)")
        "importance_norm": float  [0,1]-scaled magnitude, only if normalize=True

    Sorted by rank ascending (most important feature first).

    Raises
    ------
    ValueError  model exposes neither feature_importances_ nor coef_.
    ValueError  feature_names length does not match model's feature count.
    """
    raw, source, signed = _extract_model_importance(model)
    n_features = len(raw)
    names = _validate_feature_names(feature_names, n_features)

    df = pd.DataFrame({
        "feature":        names,
        "importance":     raw,
        "abs_importance": np.abs(raw),
        "source":         source,
    })

    if not signed:
        # feature_importances_ and multiclass coef_ are already non-negative;
        # drop the redundant abs_importance column to keep output clean.
        df = df.drop(columns=["abs_importance"])
        sort_col = "importance"
    else:
        sort_col = "abs_importance"

    if normalize:
        max_abs = df[sort_col].max()
        df["importance_norm"] = df[sort_col] / max_abs if max_abs > 0 else 0.0

    df = _add_rank(df, sort_col)

    if not signed:
        # Warn if any feature_importances_ value is negative (should not happen,
        # but some custom estimators may violate the contract).
        if (df["importance"] < 0).any():
            warnings.warn(
                f"{source}: unexpected negative importance values. "
                "Results may be unreliable.",
                stacklevel=2,
            )

    return df


def compute_permutation_importance(
    model,
    X,
    y,
    feature_names=None,
    *,
    scoring=None,
    n_repeats=10,
    random_state=42,
    n_jobs=1,
):
    """
    Compute model-agnostic permutation feature importance.

    Measures the decrease in a performance metric when a single feature's
    values are randomly shuffled. A large drop indicates the model relies
    heavily on that feature. A near-zero or negative drop indicates the
    feature adds little or no value (and may even be noise).

    Works with any fitted estimator that exposes a predict or predict_proba
    method — no model-specific attributes required.

    Parameters
    ----------
    model        : fitted sklearn-compatible estimator
    X            : array-like, shape (n_samples, n_features)
        Evaluation set. Use X_test to measure generalisation importance;
        use X_train to measure fit importance (the two may differ).
    y            : array-like, shape (n_samples,)
        True labels or targets for X.
    feature_names : list[str] or None
        Column names. None generates generic labels. Must match n_features.
    scoring      : str, callable, or None
        Metric used to measure importance drop. None uses the estimator's
        default scorer. Common values: "roc_auc", "accuracy", "f1_weighted",
        "r2", "neg_mean_squared_error".
    n_repeats    : int
        Number of times each feature is shuffled. More repeats reduce
        variance of the importance estimate. Default 10.
    random_state : int
        Reproducibility seed. Default 42.
    n_jobs       : int
        Parallel jobs. -1 uses all available cores. Default 1.

    Returns
    -------
    pd.DataFrame with columns:
        "rank"             : int    1-based rank by mean importance (1 = most important)
        "feature"          : str    feature name
        "importance_mean"  : float  mean decrease in scoring metric across repeats
                             Positive = feature is useful.
                             Near-zero or negative = feature adds little value.
        "importance_std"   : float  standard deviation across repeats
        "importance_sem"   : float  standard error of the mean (std / sqrt(n_repeats))

    Sorted by rank ascending (most important feature first).

    Raises
    ------
    ValueError  feature_names length does not match X's column count.
    """
    X_arr = np.asarray(X)
    n_features = X_arr.shape[1]
    names = _validate_feature_names(feature_names, n_features)

    result = permutation_importance(
        model,
        X_arr,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    sem = result.importances_std / np.sqrt(n_repeats)

    df = pd.DataFrame({
        "feature":         names,
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
        "importance_sem":  sem,
    })

    n_negative = int((df["importance_mean"] < 0).sum())
    if n_negative > 0:
        warnings.warn(
            f"compute_permutation_importance: {n_negative} feature(s) show "
            "negative mean importance — shuffling them improves the score. "
            "Consider removing these features.",
            stacklevel=2,
        )

    df = _add_rank(df, "importance_mean")
    return df


# ================================================================
# SHAP-BASED EXPLAINABILITY
# ================================================================


def _require_shap():
    """Return the shap module, raising ImportError with an install hint if absent."""
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError(
            "SHAP is required for this function. "
            "Install it with: pip install shap"
        )


def _get_shap_explainer(shap, model, X_background, *, explainer="auto", task="classification"):
    """
    Instantiate the appropriate SHAP explainer.

    Selection priority when explainer="auto":
      1. TreeExplainer  — models with feature_importances_ (RF, GBM, XGBoost, …)
      2. LinearExplainer — models with coef_ (Ridge, Lasso, LinearRegression, …)
      3. KernelExplainer — model-agnostic fallback (slow; uses background data)

    Parameters
    ----------
    shap         : shap module (already imported)
    model        : fitted estimator
    X_background : ndarray  Background / reference data.
    explainer    : "auto" | "tree" | "linear" | "kernel"
    task         : "classification" | "regression"
                   Affects which prediction function KernelExplainer wraps.

    Returns
    -------
    (explainer_obj, kind_str)
    """
    if explainer == "auto":
        if hasattr(model, "feature_importances_"):
            kind = "tree"
        elif hasattr(model, "coef_"):
            kind = "linear"
        else:
            kind = "kernel"
    else:
        kind = str(explainer)

    if kind == "tree":
        return shap.TreeExplainer(model), "tree"

    elif kind == "linear":
        try:
            exp = shap.LinearExplainer(model, X_background)
        except TypeError:
            # Newer SHAP API requires a masker object
            exp = shap.LinearExplainer(
                model, shap.maskers.Independent(X_background)
            )
        return exp, "linear"

    elif kind == "kernel":
        # Wrap predict_proba[:, 1] for binary classifiers so SHAP values
        # represent the positive-class probability contribution.
        if task == "classification" and hasattr(model, "predict_proba"):
            fn = lambda x: model.predict_proba(x)[:, 1]  # noqa: E731
        else:
            fn = model.predict
        return shap.KernelExplainer(fn, X_background), "kernel"

    else:
        raise ValueError(
            f"Unknown explainer '{kind}'. "
            "Valid options: 'auto', 'tree', 'linear', 'kernel'."
        )


def _normalise_shap_output(raw, n_samples, n_features):
    """
    Coerce SHAP output to a 2-D float array of shape (n_samples, n_features).

    Handles:
    - SHAP Explanation objects (>= 0.40) — extracts .values
    - list of arrays (tree/linear classifiers): [class_0, class_1, ...]
    - 3-D arrays (n_samples, n_features, n_classes)
    - Already 2-D arrays

    For binary classification (2 classes), class-1 SHAP values are returned.
    For multiclass (>2 classes), the mean absolute SHAP across classes is returned.
    """
    # SHAP >= 0.40 Explanation objects
    if hasattr(raw, "values"):
        raw = raw.values

    if isinstance(raw, list):
        if len(raw) == 2:
            arr = np.asarray(raw[1], dtype=float)        # binary: class 1
        else:
            arr = np.mean(                               # multiclass: mean abs
                np.abs(np.stack(raw, axis=0)), axis=0
            ).astype(float)
    else:
        arr = np.asarray(raw, dtype=float)

    if arr.ndim == 3:                                    # (n_samples, n_features, n_classes)
        if arr.shape[2] == 2:
            arr = arr[:, :, 1]
        else:
            arr = np.mean(np.abs(arr), axis=2)

    if arr.ndim != 2:
        arr = arr.reshape(n_samples, n_features)

    return arr


def compute_shap_values(
    model,
    X,
    feature_names=None,
    *,
    explainer="auto",
    X_background=None,
    n_background=50,
    task="classification",
):
    """
    Compute SHAP values for a fitted model over a dataset.

    Automatically selects TreeExplainer, LinearExplainer, or KernelExplainer
    based on the model type (when explainer="auto"). KernelExplainer is the
    universal fallback but is significantly slower than the model-specific
    alternatives.

    Parameters
    ----------
    model         : fitted sklearn-compatible estimator
    X             : array-like, shape (n_samples, n_features)
        Samples to explain. Use X_test for generalisation analysis.
    feature_names : list[str] or None
        Column names. None generates generic labels.
    explainer     : "auto" | "tree" | "linear" | "kernel"
        Explainer type. "auto" selects based on model attributes.
    X_background  : array-like or None
        Background/reference data for LinearExplainer and KernelExplainer.
        When None, a random subsample of X (size n_background) is used.
    n_background  : int
        Number of background samples to draw from X when X_background is None.
        Default 50. Ignored when X_background is provided.
    task          : "classification" | "regression"
        Determines which prediction function KernelExplainer wraps.
        Ignored for TreeExplainer and LinearExplainer.

    Returns
    -------
    shap_values : ndarray, shape (n_samples, n_features)
        SHAP values — positive values increase the prediction, negative
        values decrease it. For binary classification, these are class-1
        contributions. For multiclass, mean absolute across classes.
    explainer_obj : fitted SHAP explainer instance
        Can be reused for additional calls or introspection.

    Raises
    ------
    ImportError  shap is not installed.
    ValueError   Unknown explainer type.
    ValueError   feature_names length mismatch.
    """
    shap = _require_shap()
    X_arr = np.asarray(X, dtype=float)
    n_samples, n_features = X_arr.shape
    _validate_feature_names(feature_names, n_features)

    if X_background is None:
        n_bg = min(n_background, n_samples)
        bg = shap.sample(X_arr, n_bg, random_state=42)
    else:
        bg = np.asarray(X_background, dtype=float)

    explainer_obj, _ = _get_shap_explainer(
        shap, model, bg, explainer=explainer, task=task
    )
    raw = explainer_obj.shap_values(X_arr)
    shap_arr = _normalise_shap_output(raw, n_samples, n_features)

    return shap_arr, explainer_obj


def summarise_shap_importance(shap_values, feature_names):
    """
    Aggregate raw SHAP values into a global feature importance DataFrame.

    Parameters
    ----------
    shap_values   : array-like, shape (n_samples, n_features)
        Raw SHAP values from compute_shap_values().
    feature_names : list[str]
        Feature column names. Length must equal n_features.

    Returns
    -------
    pd.DataFrame with columns:
        "rank"          : int    1-based rank by mean_abs_shap (1 = most important)
        "feature"       : str
        "mean_abs_shap" : float  Mean |SHAP| across all samples — global importance.
        "mean_shap"     : float  Mean signed SHAP — overall directional impact.
        "positive_mean" : float  Mean of positive SHAP values (where SHAP > 0).
        "negative_mean" : float  Mean of negative SHAP values (where SHAP < 0).

    Sorted by rank ascending (most important feature first).
    """
    arr = np.asarray(shap_values, dtype=float)
    names = list(feature_names)

    pos = np.where(arr > 0, arr, 0.0).mean(axis=0)
    neg = np.where(arr < 0, arr, 0.0).mean(axis=0)

    df = pd.DataFrame({
        "feature":       names,
        "mean_abs_shap": np.abs(arr).mean(axis=0),
        "mean_shap":     arr.mean(axis=0),
        "positive_mean": pos,
        "negative_mean": neg,
    })
    return _add_rank(df, "mean_abs_shap")


def explain_prediction(
    model,
    X_instance,
    feature_names=None,
    *,
    explainer="auto",
    X_background=None,
    n_background=50,
    task="classification",
):
    """
    Explain a single prediction using SHAP values (local explanation).

    Parameters
    ----------
    model        : fitted estimator
    X_instance   : array-like, shape (n_features,) or (1, n_features)
        The single sample to explain. Accepts both 1-D and 2-D input.
    feature_names : list[str] or None
    explainer    : "auto" | "tree" | "linear" | "kernel"
    X_background : array-like or None
    n_background : int
    task         : "classification" | "regression"

    Returns
    -------
    pd.DataFrame with columns:
        "rank"          : int    1-based rank by |shap_value|
        "feature"       : str
        "feature_value" : float  actual value of the feature for this instance
        "shap_value"    : float  SHAP contribution (signed)
        "abs_shap"      : float  magnitude of contribution

    Sorted by rank ascending (largest absolute contributor first).
    """
    X_arr = np.asarray(X_instance, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    n_features = X_arr.shape[1]
    names = _validate_feature_names(feature_names, n_features)

    shap_arr, _ = compute_shap_values(
        model, X_arr, names,
        explainer=explainer,
        X_background=X_background,
        n_background=n_background,
        task=task,
    )

    df = pd.DataFrame({
        "feature":       names,
        "feature_value": X_arr[0].tolist(),
        "shap_value":    shap_arr[0].tolist(),
        "abs_shap":      np.abs(shap_arr[0]).tolist(),
    })
    return _add_rank(df, "shap_value")


# ================================================================
# FUTURE: LIME  (not yet implemented)
# ================================================================
#
# compute_lime_explanation(model, X_train, instance, feature_names=None, *,
#                           mode="classification", n_features=10,
#                           num_samples=5000, random_state=42)
#   Fits a local linear surrogate around a single prediction instance.
#   Returns a DataFrame with columns: feature, weight, abs_weight.
#   Requires: pip install lime
#
# ================================================================
# FUTURE: PARTIAL DEPENDENCE AND ICE  (not yet implemented)
# ================================================================
#
# compute_partial_dependence(model, X, features, feature_names=None, *,
#                             grid_resolution=100, kind="average")
#   Wraps sklearn.inspection.partial_dependence.
#   kind="average"   → classic PDP
#   kind="individual" → ICE curves
#   kind="both"       → PDP overlaid on ICE
