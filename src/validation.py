from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)

from src.config import DEFAULT_VALIDATION, RANDOM_STATE, TEST_SIZE

# ================================================================
# STRATEGY REGISTRY
# ================================================================

# Maps public strategy names to sklearn splitter classes.
_REGISTRY = {
    "holdout":                   ShuffleSplit,
    "kfold":                     KFold,
    "stratified_kfold":          StratifiedKFold,
    "repeated_stratified_kfold": RepeatedStratifiedKFold,
    "group_kfold":               GroupKFold,
    "time_series_split":         TimeSeriesSplit,
}

# Conservative defaults for each strategy.
# All are overridable by passing **kwargs to get_cv_strategy().
_DEFAULTS = {
    "holdout": {
        "n_splits":     1,
        "test_size":    TEST_SIZE,
        "random_state": RANDOM_STATE,
    },
    "kfold": {
        "n_splits":     DEFAULT_VALIDATION["n_splits"],
        "shuffle":      True,
        "random_state": RANDOM_STATE,
    },
    "stratified_kfold": {
        "n_splits":     DEFAULT_VALIDATION["n_splits"],
        "shuffle":      True,
        "random_state": RANDOM_STATE,
    },
    "repeated_stratified_kfold": {
        "n_splits":     DEFAULT_VALIDATION["n_splits"],
        "n_repeats":    DEFAULT_VALIDATION["n_repeats"],
        "random_state": RANDOM_STATE,
    },
    "group_kfold": {
        # GroupKFold does not accept shuffle or random_state
        "n_splits":     DEFAULT_VALIDATION["n_splits"],
    },
    "time_series_split": {
        # gap and max_train_size default to None (sklearn default)
        "n_splits":     DEFAULT_VALIDATION["n_splits"],
    },
}

# Exposed so callers can enumerate or validate strategy names without
# importing private module internals.
VALID_STRATEGIES = frozenset(_REGISTRY)


# ================================================================
# PUBLIC API
# ================================================================


def get_cv_strategy(strategy, **kwargs):
    """
    Return a configured sklearn cross-validator for the named strategy.

    Parameters
    ----------
    strategy : str
        Strategy name. Valid values:

        "holdout"
            Single train/test split via ShuffleSplit(n_splits=1).
            Equivalent to train_test_split. Not stratified by default;
            use "stratified_kfold" with n_splits=1 for a stratified holdout.

        "kfold"
            K-fold cross-validation. Shuffled by default.
            Suitable for regression and balanced classification.

        "stratified_kfold"
            Stratified K-fold. Preserves class proportions in every fold.
            Preferred for classification, especially on imbalanced data.

        "repeated_stratified_kfold"
            Runs stratified K-fold multiple times with different shuffles.
            Reduces variance in the performance estimate.

        "group_kfold"
            Ensures no group (e.g., patient ID, company) appears in both
            train and test. Required for panel / longitudinal data.
            Caller must pass a groups array to .split().

        "time_series_split"
            Forward-chaining split: each fold trains on all past data and
            tests on the next block. Never looks ahead. No shuffle.
            Suitable for time series and sequential data.

    **kwargs
        Override any default parameter for the underlying sklearn class.
        Examples:
            get_cv_strategy("kfold", n_splits=10)
            get_cv_strategy("time_series_split", n_splits=8, gap=4)
            get_cv_strategy("repeated_stratified_kfold", n_repeats=5)

    Returns
    -------
    Configured sklearn splitter with a .split(X, y, groups) interface.

    Raises
    ------
    ValueError  Unknown strategy name.
    TypeError   Invalid kwargs for the underlying sklearn class.
    """
    if strategy not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Valid options: {sorted(VALID_STRATEGIES)}"
        )
    params = {**_DEFAULTS.get(strategy, {}), **kwargs}
    # sklearn raises ValueError when random_state is set alongside shuffle=False.
    # Drop it so caller overriding shuffle=False does not hit that constraint.
    if params.get("shuffle") is False and "random_state" in params:
        del params["random_state"]
    return _REGISTRY[strategy](**params)


def describe_cv_strategy(cv):
    """
    Return a concise, human-readable description of a CV strategy object.

    Designed for logging and report headers. Accepts any sklearn splitter;
    falls back to a parameter dump for unknown types.

    Parameters
    ----------
    cv : sklearn splitter

    Returns
    -------
    str
    """
    name = type(cv).__name__
    # Sklearn 1.x splitters do not implement get_params(); their constructor
    # parameters are stored as plain instance attributes accessible via vars().
    p = vars(cv)

    if name == "ShuffleSplit":
        ts     = p.get("test_size", "default")
        n      = p.get("n_splits", 1)
        suffix = f", {n} split(s)" if n != 1 else ""
        return f"Holdout split - test_size={ts}{suffix}"

    if name == "KFold":
        shuffled = ", shuffled" if p.get("shuffle") else ""
        return f"{p['n_splits']}-fold cross-validation{shuffled}"

    if name == "StratifiedKFold":
        shuffled = ", shuffled" if p.get("shuffle") else ""
        return f"Stratified {p['n_splits']}-fold cross-validation{shuffled}"

    if name == "RepeatedStratifiedKFold":
        # n_splits lives inside the cvargs dict, not as a top-level attribute.
        n_splits  = p.get("cvargs", {}).get("n_splits", "?")
        n_repeats = p.get("n_repeats", "?")
        return f"Repeated stratified K-fold - {n_splits} folds x {n_repeats} repeats"

    if name == "GroupKFold":
        return (
            f"Group {p['n_splits']}-fold - "
            "no group appears in both train and test folds"
        )

    if name == "TimeSeriesSplit":
        desc = f"Time-series split - {p['n_splits']} folds, forward-chaining"
        if p.get("max_train_size"):
            desc += f", max_train_size={p['max_train_size']}"
        if p.get("gap"):
            desc += f", gap={p['gap']}"
        return desc

    # Unknown type: emit a parameter dump, or just the class name if no attrs
    return f"{name}({p})" if p else name


# ================================================================
# FUTURE: TEMPORAL VALIDATION STRATEGIES  (not yet implemented)
# ================================================================
#
# Walk-forward validation:
#   walk_forward_split(X, y, n_splits, horizon, min_train_size=None)
#   Each fold trains on all past data and evaluates on the next `horizon`
#   steps. The training window expands with each fold (expanding window).
#   Equivalent to TimeSeriesSplit with a fixed test horizon per fold.
#
# Rolling window validation:
#   rolling_window_split(X, y, train_size, test_size, step=1)
#   Fixed-size training window slides forward `step` observations per fold.
#   Old data drops out as new data enters -- appropriate when the data
#   generating process is non-stationary and distant history is uninformative.
#
# Expanding window validation:
#   expanding_window_split(X, y, min_train_size, test_size, step=1)
#   Training window grows from min_train_size; test window stays fixed.
#   Equivalent to walk_forward_split without a max_train_size constraint.
#
# Nested cross-validation:
#   nested_cv(outer_cv, inner_cv)
#   Returns a configured (outer_cv, inner_cv) pair for unbiased evaluation
#   when hyperparameter search is part of the pipeline. The outer loop
#   estimates generalisation; the inner loop selects hyperparameters.
#   Delegates hyperparameter search to optimization.py once that exists.
