"""
Feature selection utilities for supervised and unsupervised tasks.

All functions accept a numpy feature matrix X and a list of feature names,
and return a standardised result dict that can be fed directly into the
benchmark pipeline.

Result dict schema
------------------
{
  "method"            : str,
  "n_before"          : int,
  "n_after"           : int,
  "n_removed"         : int,
  "selected_features" : list[str],
  "removed_features"  : list[str],
  "selected_mask"     : np.ndarray  shape (n_before,) bool,
  "scores"            : dict[str, float] or None   # method-dependent ranking
}

X_selected is NOT stored in the result dict (it would double memory); callers
apply the mask themselves:  X_selected = X[:, result["selected_mask"]]
"""

import warnings

import numpy as np


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _validate_inputs(X, feature_names):
    X = np.asarray(X, dtype=float)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        feature_names = list(feature_names)
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X.shape[1] ({X.shape[1]})."
        )
    return X, feature_names


def _build_result(method, feature_names, selected_mask, scores=None):
    selected_mask = np.asarray(selected_mask, dtype=bool)
    selected = [f for f, m in zip(feature_names, selected_mask) if m]
    removed  = [f for f, m in zip(feature_names, selected_mask) if not m]
    return {
        "method":            method,
        "n_before":          len(feature_names),
        "n_after":           int(selected_mask.sum()),
        "n_removed":         int((~selected_mask).sum()),
        "selected_features": selected,
        "removed_features":  removed,
        "selected_mask":     selected_mask,
        "scores":            scores,
    }


def _resolve_n_features(n_features, n_total, minimum=1):
    """Return a safe n_features value, clamping to [minimum, n_total]."""
    if n_features is None:
        return n_total
    n = int(n_features)
    if n < minimum:
        warnings.warn(
            f"n_features={n} is below minimum {minimum}; "
            f"using {minimum}.", stacklevel=3
        )
        n = minimum
    if n > n_total:
        warnings.warn(
            f"n_features={n} exceeds available features {n_total}; "
            f"using {n_total}.", stacklevel=3
        )
        n = n_total
    return n


# ================================================================
# SELECTION METHODS
# ================================================================


def variance_threshold_selection(X, feature_names=None, *, threshold=0.0):
    """
    Remove features whose variance is at or below ``threshold``.

    Wraps sklearn.feature_selection.VarianceThreshold.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    feature_names: list[str] or None
    threshold    : float   Minimum variance to keep (default 0.0 removes
                           constant features).

    Returns
    -------
    dict   Standard selection result dict.
    """
    from sklearn.feature_selection import VarianceThreshold

    X, feature_names = _validate_inputs(X, feature_names)
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)

    variances = sel.variances_
    mask      = sel.get_support()
    scores    = {f: float(v) for f, v in zip(feature_names, variances)}

    return _build_result("variance_threshold", feature_names, mask, scores)


def correlation_selection(X, feature_names=None, *, threshold=0.95):
    """
    Remove features that are pairwise correlated above ``threshold``.

    For each correlated pair the feature with the **higher** mean absolute
    correlation is dropped, which is a greedy but deterministic strategy.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    feature_names: list[str] or None
    threshold    : float   Correlation magnitude above which one feature is
                           dropped. Default 0.95.

    Returns
    -------
    dict   Standard selection result dict.
           ``scores`` = {feature: mean absolute correlation with all others}
    """
    X, feature_names = _validate_inputs(X, feature_names)
    n = X.shape[1]

    if n < 2:
        return _build_result("correlation", feature_names, np.ones(n, dtype=bool))

    corr = np.abs(np.corrcoef(X, rowvar=False))
    np.fill_diagonal(corr, 0.0)

    mean_corr = corr.mean(axis=1)
    scores    = {f: float(v) for f, v in zip(feature_names, mean_corr)}

    # Greedy removal: iterate upper triangle
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            if corr[i, j] >= threshold:
                # Drop whichever has higher mean correlation
                if mean_corr[i] >= mean_corr[j]:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)

    mask = np.array([i not in to_remove for i in range(n)], dtype=bool)
    return _build_result("correlation", feature_names, mask, scores)


def mutual_information_selection(
    X, y, feature_names=None, *, n_features=10, task="classification"
):
    """
    Select the top ``n_features`` features ranked by mutual information with y.

    Uses sklearn.feature_selection.mutual_info_classif for classification and
    mutual_info_regression for regression.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray  (n_samples,)
    feature_names: list[str] or None
    n_features   : int or None   Number of features to keep. None keeps all.
    task         : "classification" | "regression"

    Returns
    -------
    dict   Standard selection result dict.
           ``scores`` = {feature: mutual information score}
    """
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )

    X, feature_names = _validate_inputs(X, feature_names)
    y = np.asarray(y)
    k = _resolve_n_features(n_features, X.shape[1])

    fn = mutual_info_classif if task == "classification" else mutual_info_regression
    mi_scores = fn(X, y, random_state=42)

    scores = {f: float(s) for f, s in zip(feature_names, mi_scores)}
    # Select indices of top-k by score (descending)
    top_indices = set(np.argsort(mi_scores)[::-1][:k])
    mask = np.array([i in top_indices for i in range(len(feature_names))], dtype=bool)

    return _build_result("mutual_information", feature_names, mask, scores)


def rfecv_selection(
    X, y, feature_names=None, *, estimator=None, task="classification", cv=5
):
    """
    Recursive feature elimination with cross-validation (RFECV).

    Automatically selects the optimal number of features by choosing the
    subset that maximises cross-validated score.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray  (n_samples,)
    feature_names: list[str] or None
    estimator    : sklearn estimator or None
        When None, uses LogisticRegression (classification) or
        LinearRegression (regression) as the base estimator — both are
        fast and provide coef_-based feature importances.
    task         : "classification" | "regression"
    cv           : int   Number of CV folds (default 5).

    Returns
    -------
    dict   Standard selection result dict.
           ``scores`` = {feature: mean CV score contribution (ranking)}
    """
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression, LogisticRegression

    X, feature_names = _validate_inputs(X, feature_names)
    y = np.asarray(y)

    if estimator is None:
        if task == "classification":
            estimator = LogisticRegression(
                max_iter=500, solver="lbfgs", C=1.0, random_state=42
            )
        else:
            estimator = LinearRegression()

    scoring = "accuracy" if task == "classification" else "r2"
    selector = RFECV(estimator, cv=cv, scoring=scoring, n_jobs=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X, y)

    mask   = selector.support_
    # ranking_: 1 = selected (lower is better)
    ranks  = selector.ranking_
    scores = {f: float(r) for f, r in zip(feature_names, ranks)}

    return _build_result("rfecv", feature_names, mask, scores)


def lasso_selection(
    X, y, feature_names=None, *, alpha=0.01, task="classification"
):
    """
    Select features with non-zero coefficients from a L1-regularised model.

    Uses Lasso (regression) or LogisticRegression with l1 penalty
    (classification). The alpha / C value controls sparsity.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray  (n_samples,)
    feature_names: list[str] or None
    alpha        : float
        Regularisation strength.  Lasso: larger = sparser.
        LogisticRegression: C = 1/alpha so larger alpha = stricter.
    task         : "classification" | "regression"

    Returns
    -------
    dict   Standard selection result dict.
           ``scores`` = {feature: |coefficient|}
    """
    from sklearn.linear_model import Lasso, LogisticRegression

    X, feature_names = _validate_inputs(X, feature_names)
    y = np.asarray(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == "classification":
            C = 1.0 / max(alpha, 1e-9)
            model = LogisticRegression(
                penalty="l1", C=C, solver="liblinear",
                max_iter=500, random_state=42
            )
            model.fit(X, y)
            coef = np.abs(model.coef_).max(axis=0)   # max over classes for multi-class
        else:
            model = Lasso(alpha=alpha, max_iter=5000, random_state=42)
            model.fit(X, y)
            coef = np.abs(model.coef_)

    scores = {f: float(c) for f, c in zip(feature_names, coef)}
    mask   = coef > 0

    if mask.sum() == 0:
        warnings.warn(
            f"lasso_selection: all coefficients are zero (alpha={alpha} too large). "
            "Returning all features.",
            stacklevel=2,
        )
        mask = np.ones(len(feature_names), dtype=bool)

    return _build_result("lasso", feature_names, mask, scores)


# ================================================================
# DISPATCHER
# ================================================================

_VALID_METHODS = {"variance", "correlation", "mutual_information", "rfecv", "lasso"}


def run_feature_selection(
    X, y=None, feature_names=None, *,
    method="variance",
    task="classification",
    n_features=None,
    **kwargs,
):
    """
    Unified entry point for all feature selection methods.

    Parameters
    ----------
    X            : np.ndarray  (n_samples, n_features)
    y            : np.ndarray or None   Required for supervised methods.
    feature_names: list[str] or None
    method       : str   One of: "variance", "correlation",
                         "mutual_information", "rfecv", "lasso".
    task         : "classification" | "regression"
    n_features   : int or None   Forwarded to methods that take it
                         (mutual_information; others use their own strategies).
    **kwargs     : Extra keyword arguments forwarded to the selected method.

    Returns
    -------
    dict   Standard selection result dict (see module docstring).
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method='{method}' is not recognised. "
            f"Valid methods: {sorted(_VALID_METHODS)}"
        )

    if method == "variance":
        return variance_threshold_selection(X, feature_names, **kwargs)

    if method == "correlation":
        return correlation_selection(X, feature_names, **kwargs)

    if y is None:
        raise ValueError(
            f"Supervised method '{method}' requires y (target array)."
        )

    if method == "mutual_information":
        kw = dict(kwargs)
        kw.setdefault("n_features", n_features if n_features is not None else 10)
        return mutual_information_selection(X, y, feature_names, task=task, **kw)

    if method == "rfecv":
        return rfecv_selection(X, y, feature_names, task=task, **kwargs)

    if method == "lasso":
        return lasso_selection(X, y, feature_names, task=task, **kwargs)
