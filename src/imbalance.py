import warnings

import numpy as np
import pandas as pd

try:
    from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC
    from imblearn.under_sampling import RandomUnderSampler
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False
    RandomOverSampler = RandomUnderSampler = SMOTE = SMOTENC = ADASYN = None


# ================================================================
# STRATEGY REGISTRY
# ================================================================

# Maps public strategy names to sampler classes.
# All samplers expose a .fit_resample(X, y) interface (imblearn API).
_REGISTRY = {
    "random_over":  RandomOverSampler,
    "random_under": RandomUnderSampler,
    "smote":        SMOTE,
    "smotenc":      SMOTENC,
    "adasyn":       ADASYN,
}

# Conservative defaults for each strategy.
# All are overridable via **kwargs in get_sampler() / apply_sampling().
_DEFAULTS = {
    "random_over": {
        "random_state": 42,
    },
    "random_under": {
        "random_state": 42,
    },
    "smote": {
        "random_state": 42,
        "k_neighbors":  5,
    },
    "smotenc": {
        # categorical_features is required; injected by get_sampler() from
        # its dedicated parameter rather than buried in **kwargs.
        "random_state": 42,
        "k_neighbors":  5,
    },
    "adasyn": {
        "random_state": 42,
        "n_neighbors":  5,
    },
}

# Exposed so callers can enumerate or validate strategy names without
# importing private module internals.
VALID_STRATEGIES = frozenset(_REGISTRY)


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _require_imblearn():
    if not _HAS_IMBLEARN:
        raise ImportError(
            "imbalanced-learn is required for imbalance handling. "
            "Install it with:  pip install imbalanced-learn"
        )


def _class_distribution(y):
    """Return a pd.Series of class counts sorted by class label."""
    unique, counts = np.unique(y, return_counts=True)
    return pd.Series(counts, index=unique, name="count")


def _check_minority_size(y, k_neighbors, strategy):
    """
    Warn if the minority class is too small for a k-NN-based sampler.

    SMOTE and its variants require each class to have at least
    k_neighbors + 1 samples to build a valid neighbourhood graph.
    """
    dist = _class_distribution(y)
    min_count = int(dist.min())
    if min_count <= k_neighbors:
        warnings.warn(
            f"{strategy}: minority class has only {min_count} sample(s), "
            f"but k_neighbors={k_neighbors} requires at least {k_neighbors + 1}. "
            "Reduce k_neighbors or use 'random_over' instead.",
            stacklevel=3,
        )


def _build_summary(strategy, dist_before, dist_after):
    """One-line resampling summary for logging and reporting."""
    n_before = int(dist_before.sum())
    n_after  = int(dist_after.sum())
    return (
        f"{strategy}: {n_before} -> {n_after} samples. "
        f"Before: {dist_before.to_dict()}  "
        f"After:  {dist_after.to_dict()}"
    )


# ================================================================
# PUBLIC API
# ================================================================


def get_sampler(strategy, *, categorical_features=None, **kwargs):
    """
    Return a configured imbalanced-learn sampler for the named strategy.

    Parameters
    ----------
    strategy : str
        Resampling strategy name. Valid values:

        "random_over"
            RandomOverSampler — duplicates minority-class samples at random.
            No assumption on feature types. Safe baseline for any dataset.

        "random_under"
            RandomUnderSampler — removes majority-class samples at random.
            Fast but discards potentially useful data.

        "smote"
            SMOTE — Synthetic Minority Oversampling Technique.
            Generates synthetic samples by interpolating between a minority
            sample and one of its k nearest neighbours. Requires all
            features to be numeric. Not suitable for categorical features.

        "smotenc"
            SMOTE for Nominal and Continuous features.
            Handles mixed numeric + categorical columns. The
            categorical_features parameter must identify which column
            indices are categorical.

        "adasyn"
            Adaptive Synthetic Sampling — like SMOTE but focuses
            synthetic generation on harder-to-learn boundary regions.
            Requires all features to be numeric.

    categorical_features : list[int] or None
        Column indices of categorical features. Required when
        strategy="smotenc"; ignored for all other strategies.

    **kwargs
        Override any default parameter for the underlying sampler class.
        Examples:
            get_sampler("smote", k_neighbors=3)
            get_sampler("random_over", sampling_strategy=0.8)
            get_sampler("smotenc", categorical_features=[2, 5], k_neighbors=3)

    Returns
    -------
    Configured sampler with a .fit_resample(X, y) interface.

    Raises
    ------
    ImportError   imbalanced-learn is not installed.
    ValueError    Unknown strategy name, or categorical_features missing
                  for "smotenc".
    """
    _require_imblearn()

    if strategy not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Valid options: {sorted(VALID_STRATEGIES)}"
        )

    if strategy == "smotenc":
        if categorical_features is None:
            raise ValueError(
                "strategy='smotenc' requires categorical_features: a list of "
                "integer column indices identifying categorical columns in X_train. "
                "Example: get_sampler('smotenc', categorical_features=[2, 5])"
            )
        kwargs["categorical_features"] = list(categorical_features)

    params = {**_DEFAULTS.get(strategy, {}), **kwargs}
    return _REGISTRY[strategy](**params)


def apply_sampling(
    X_train,
    y_train,
    strategy,
    *,
    categorical_features=None,
    **kwargs,
):
    """
    Apply a resampling strategy to training data and return enriched results.

    Only the training set should ever be resampled. Applying resampling to
    the test set would introduce data leakage and invalidate evaluation.

    Parameters
    ----------
    X_train             : array-like, shape (n_samples, n_features)
        Training feature matrix. Must not include the test set.
    y_train             : array-like, shape (n_samples,)
        Training labels.
    strategy            : str
        Resampling strategy name. See get_sampler() for valid values.
    categorical_features : list[int] or None
        Required when strategy="smotenc". Column indices of categorical
        features in X_train.
    **kwargs
        Forwarded to get_sampler(). Override any sampler default, e.g.
        k_neighbors, sampling_strategy, random_state.

    Returns
    -------
    dict with keys:
        "X_res"              : ndarray   Resampled feature matrix.
        "y_res"              : ndarray   Resampled labels.
        "sampler"            : fitted sampler object (for inspection/logging).
        "class_dist_before"  : pd.Series  Class counts before resampling.
        "class_dist_after"   : pd.Series  Class counts after resampling.
        "n_before"           : int        Total samples before resampling.
        "n_after"            : int        Total samples after resampling.
        "summary"            : str        One-line description for logging.
    """
    _require_imblearn()

    y_arr = np.asarray(y_train)

    # Warn if data appears already balanced (imbalance ratio < 1.5:1)
    dist_before = _class_distribution(y_arr)
    if len(dist_before) >= 2:
        ratio = dist_before.max() / dist_before.min()
        if ratio < 1.5:
            warnings.warn(
                f"apply_sampling: class imbalance ratio is only {ratio:.2f}:1. "
                "Resampling may not be necessary for near-balanced data.",
                stacklevel=2,
            )

    # For k-NN-based methods, verify minority class size before sampling
    k_based = {"smote", "smotenc", "adasyn"}
    if strategy in k_based:
        k = kwargs.get("k_neighbors", _DEFAULTS.get(strategy, {}).get("k_neighbors", 5))
        _check_minority_size(y_arr, k, strategy)

    sampler = get_sampler(strategy, categorical_features=categorical_features, **kwargs)
    X_res, y_res = sampler.fit_resample(X_train, y_arr)

    dist_after = _class_distribution(y_res)

    return {
        "X_res":             X_res,
        "y_res":             y_res,
        "sampler":           sampler,
        "class_dist_before": dist_before,
        "class_dist_after":  dist_after,
        "n_before":          int(dist_before.sum()),
        "n_after":           int(dist_after.sum()),
        "summary":           _build_summary(strategy, dist_before, dist_after),
    }


# ================================================================
# FUTURE: ADVANCED RESAMPLING  (not yet implemented)
# ================================================================
#
# All planned samplers follow the same imblearn API contract:
#   .fit_resample(X, y) -> (X_res, y_res)
# and will be registered in _REGISTRY / _DEFAULTS / VALID_STRATEGIES.
#
# BorderlineSMOTE:
#   strategy name: "borderline_smote"
#   Focuses synthetic generation on borderline minority samples — those
#   that lie near the class boundary and are hardest to classify correctly.
#   Two variants: "borderline-1" (uses minority neighbours only) and
#   "borderline-2" (uses both minority and majority neighbours).
#   Requires: imblearn.over_sampling.BorderlineSMOTE (already installed)
#   Key param: kind="borderline-1" | "borderline-2"
#
# KMeansSMOTE:
#   strategy name: "kmeans_smote"
#   Clusters the minority class with k-means before applying SMOTE within
#   each cluster. Avoids generating noise in sparse regions. Better than
#   vanilla SMOTE on datasets with irregular minority class structure.
#   Requires: imblearn.over_sampling.KMeansSMOTE (already installed)
#   Key param: kmeans_estimator (int or KMeans instance), cluster_balance_threshold
#
# SMOTEENN:
#   strategy name: "smoteenn"
#   Combination method: oversample minority with SMOTE, then clean noise
#   using Edited Nearest Neighbours (ENN removes samples misclassified
#   by their k nearest neighbours). Produces cleaner decision boundaries
#   than SMOTE alone.
#   Requires: imblearn.combine.SMOTEENN (already installed)
#   Key param: smote (SMOTE instance), enn (EditedNearestNeighbours instance)
#
# SMOTETomek:
#   strategy name: "smotetomek"
#   Combination method: oversample minority with SMOTE, then remove
#   Tomek links (borderline majority-minority pairs). Less aggressive
#   cleaning than SMOTEENN; retains more samples while tidying boundaries.
#   Requires: imblearn.combine.SMOTETomek (already installed)
#   Key param: smote (SMOTE instance), tomek (TomekLinks instance)
