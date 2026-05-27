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
# FUTURE: SHAP-BASED EXPLAINABILITY  (not yet implemented)
# ================================================================
#
# All planned SHAP functions operate on fitted models and return either
# a raw SHAP values array or a tidy DataFrame for downstream analysis.
# Plots are intentionally separated into plots.py (future extension).
# Requires: pip install shap
#
# compute_shap_values(model, X, feature_names=None, *, explainer="auto")
#   "auto" selects TreeExplainer for tree-based models, LinearExplainer
#   for linear models, and KernelExplainer as the model-agnostic fallback.
#   Returns a DataFrame (n_samples × n_features) of raw SHAP values.
#   Signed values encode direction: positive SHAP pushes prediction higher,
#   negative SHAP pushes prediction lower.
#   Enables downstream positive/negative contribution analysis per sample.
#
# summarise_shap_importance(shap_df, feature_names=None)
#   Aggregates raw SHAP values into a global feature importance table.
#   Returns DataFrame with columns: feature, mean_abs_shap, mean_shap,
#   positive_mean, negative_mean — supporting directional impact analysis.
#   positive_mean / negative_mean decompose each feature's average
#   contribution into the share that helps vs. hurts predictions.
#
# Planned plot hooks in plots.py (not implemented here):
#
#   plot_shap_summary(shap_df, X, feature_names)
#       Beeswarm plot: one dot per sample, x-axis = SHAP value,
#       colour = feature value. Shows direction, magnitude, and
#       feature-value correlation simultaneously.
#
#   plot_shap_beeswarm(shap_df, X, feature_names, *, max_display=20)
#       Ranked beeswarm sorted by mean abs SHAP. Industry standard for
#       communicating global feature impact to non-technical stakeholders.
#
#   plot_shap_dependence(shap_df, X, feature, interaction_feature=None)
#       SHAP dependence plot: scatter of feature value vs. SHAP value.
#       Reveals non-linear effects and interaction structure.
#
# ================================================================
# FUTURE: LIME  (not yet implemented)
# ================================================================
#
# compute_lime_explanation(model, X_train, instance, feature_names=None, *,
#                           mode="classification", n_features=10,
#                           num_samples=5000, random_state=42)
#   Fits a local linear surrogate around a single prediction instance.
#   Returns a DataFrame with columns: feature, weight, abs_weight.
#   Signed weights encode local contribution direction.
#   Use for per-instance "why did the model predict X?" analysis.
#   Requires: pip install lime
#
# ================================================================
# FUTURE: PARTIAL DEPENDENCE AND ICE  (not yet implemented)
# ================================================================
#
# compute_partial_dependence(model, X, features, feature_names=None, *,
#                             grid_resolution=100, kind="average")
#   Wraps sklearn.inspection.partial_dependence.
#   kind="average"   → classic PDP (marginalises over all other features)
#   kind="individual" → ICE curves (one line per sample, reveals heterogeneity)
#   kind="both"       → PDP overlaid on ICE
#   Returns a dict: {"grid_values": list[array], "pdp_values": array,
#                    "feature_names": list[str]}
#   Suitable for numeric and low-cardinality categorical features.
#   Plots delegated to plots.py (plot_partial_dependence, plot_ice_curves).
