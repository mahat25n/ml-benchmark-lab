import time
import warnings

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import DEFAULT_OPTIMIZATION, RANDOM_STATE


# ================================================================
# DEFAULT SEARCH SPACES
# Lists are compatible with both GridSearchCV and RandomizedSearchCV.
# Call build_search_space() to get a per-model copy.
# ================================================================

_SPACES_CLASSIFICATION = {
    "Decision Tree": {
        "max_depth":         [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "criterion":         ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators":      [50, 100, 200, 300],
        "max_depth":         [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
    "AdaBoost": {
        "n_estimators":  [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "GradientBoost": {
        "n_estimators":  [50, 100, 200],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample":     [0.7, 0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators":    [50, 100, 200],
        "max_depth":       [3, 5, 7],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "SVM": {
        "C":      [0.1, 1.0, 10.0, 100.0],
        "gamma":  ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "poly", "sigmoid"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan", "minkowski"],
    },
    "ANN": {
        "hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
        "activation":         ["relu", "tanh"],
        "alpha":              [0.0001, 0.001, 0.01],
    },
}

_SPACES_REGRESSION = {
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "ElasticNet": {
        "alpha":    [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "Random Forest": {
        "n_estimators":      [50, 100, 200, 300],
        "max_depth":         [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
    "GradientBoost": {
        "n_estimators":  [50, 100, 200],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample":     [0.7, 0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators":    [50, 100, 200],
        "max_depth":       [3, 5, 7],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "SVR": {
        "C":       [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.1, 0.5],
        "gamma":   ["scale", "auto"],
        "kernel":  ["rbf", "poly"],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights":     ["uniform", "distance"],
    },
}

_SPACES_TIME_SERIES = {
    "Random Forest": {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators":  [50, 100, 200],
        "max_depth":     [3, 4, 6],
        "learning_rate": [0.01, 0.05, 0.1],
    },
}

# Public registry: task name → search-space dict.
# Keys must match model names used in CLASSIFICATION_MODELS, REGRESSION_MODELS,
# TIME_SERIES_MODELS registries in models.py.
DEFAULT_SEARCH_SPACES = {
    "classification": _SPACES_CLASSIFICATION,
    "regression":     _SPACES_REGRESSION,
    "time_series":    _SPACES_TIME_SERIES,
    "unsupervised":   {},   # no hyperparameter search for unsupervised models
}


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _format_cv_results(raw):
    """
    Convert sklearn's cv_results_ dict to a DataFrame sorted by rank.

    Sorting by rank_test_score (ascending) puts the best parameter
    combination at row 0, making the table human-readable without
    additional filtering.
    """
    df = pd.DataFrame(raw)
    if "rank_test_score" in df.columns:
        df = df.sort_values("rank_test_score").reset_index(drop=True)
    return df


def _package_results(search, refit):
    """
    Extract the four standard output fields from a fitted search object.

    best_estimator is None when refit=False because sklearn does not
    re-fit the model on the full training set in that case.
    """
    return {
        "best_estimator": getattr(search, "best_estimator_", None),
        "best_params":    search.best_params_,
        "best_score":     float(search.best_score_),
        "cv_results":     _format_cv_results(search.cv_results_),
    }


# ================================================================
# PUBLIC API — search-space utilities
# ================================================================


def build_search_space(model_name, task="classification"):
    """
    Return the default search space for a named model and task.

    Parameters
    ----------
    model_name : str
        Model registry key (e.g. "Random Forest", "XGBoost").
    task       : str
        One of "classification", "regression", "time_series", "unsupervised".

    Returns
    -------
    dict  Shallow copy of the default search space, or None when no
    default exists for this model/task combination.

    Raises
    ------
    ValueError  Unknown task name.
    """
    if task not in DEFAULT_SEARCH_SPACES:
        raise ValueError(
            f"build_search_space: unknown task '{task}'. "
            f"Valid: {sorted(DEFAULT_SEARCH_SPACES)}"
        )
    registry = DEFAULT_SEARCH_SPACES[task]
    space = registry.get(model_name)
    return dict(space) if space is not None else None


def validate_search_space(model, param_space):
    """
    Validate that all keys in param_space are valid parameters of model.

    Uses model.get_params() to determine valid parameter names.
    Silently passes when get_params() raises (e.g. exotic estimators).

    Parameters
    ----------
    model      : sklearn-compatible estimator
    param_space: dict   Parameter grid or distributions.

    Raises
    ------
    ValueError  If any key in param_space is not in model.get_params().
    """
    try:
        valid = set(model.get_params(deep=False).keys())
    except Exception:
        return  # can't validate — allow the search to proceed

    invalid = [k for k in param_space if k not in valid]
    if invalid:
        raise ValueError(
            f"validate_search_space: {type(model).__name__} does not "
            f"accept parameter(s): {invalid}. "
            f"Valid parameters: {sorted(valid)}"
        )


# ================================================================
# PUBLIC API — unified optimizer entry point
# ================================================================


def optimize_model(
    model,
    param_space,
    X_train,
    y_train,
    method="random",
    *,
    cv=DEFAULT_OPTIMIZATION["cv"],
    scoring=DEFAULT_OPTIMIZATION["scoring"],
    n_iter=20,
    n_jobs=DEFAULT_OPTIMIZATION["n_jobs"],
    random_state=RANDOM_STATE,
    refit=DEFAULT_OPTIMIZATION["refit"],
):
    """
    Unified entry point for hyperparameter optimization.

    Dispatches to grid or random search and augments the result with
    runtime metadata: search_duration, n_evaluations, and method.

    Parameters
    ----------
    model        : sklearn-compatible estimator (unfitted)
    param_space  : dict   Parameter grid (lists) or distributions.
    X_train      : array-like
    y_train      : array-like
    method       : "random" | "grid"
        "random" — RandomizedSearchCV (samples n_iter combinations).
        "grid"   — GridSearchCV (exhaustive enumeration).
    cv           : int or sklearn CV splitter   Inner cross-validation.
    scoring      : str, callable, or None       Optimisation metric.
    n_iter       : int   Combinations sampled (random search only).
    n_jobs       : int   Parallel jobs (-1 = all cores).
    random_state : int   Seed for reproducible random search.
    refit        : bool  Refit best estimator on full training set.

    Returns
    -------
    dict with keys:
        "best_estimator"  : fitted estimator or None (when refit=False)
        "best_params"     : dict, best hyperparameter combination
        "best_score"      : float, mean CV score for best_params
        "cv_results"      : pd.DataFrame, all evaluated combinations
        "search_duration" : float, wall-clock seconds for the search
        "n_evaluations"   : int, number of parameter combinations tried
        "method"          : str, "random" or "grid"

    Raises
    ------
    ValueError  Unknown method name, or invalid param_space keys.
    """
    validate_search_space(model, param_space)

    if method not in ("random", "grid"):
        raise ValueError(
            f"optimize_model: unknown method '{method}'. Valid: 'random', 'grid'."
        )

    t0 = time.perf_counter()

    if method == "grid":
        result = run_grid_search(
            model, param_space, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=n_jobs, refit=refit,
        )
    else:
        result = run_random_search(
            model, param_space, X_train, y_train,
            cv=cv, scoring=scoring, n_iter=n_iter,
            n_jobs=n_jobs, random_state=random_state, refit=refit,
        )

    result["search_duration"] = round(time.perf_counter() - t0, 3)
    result["n_evaluations"]   = len(result["cv_results"])
    result["method"]          = method
    return result


# ================================================================
# PUBLIC API — low-level search functions
# ================================================================


def run_grid_search(
    model,
    param_grid,
    X_train,
    y_train,
    *,
    cv=DEFAULT_OPTIMIZATION["cv"],
    scoring=DEFAULT_OPTIMIZATION["scoring"],
    n_jobs=DEFAULT_OPTIMIZATION["n_jobs"],
    refit=DEFAULT_OPTIMIZATION["refit"],
    verbose=DEFAULT_OPTIMIZATION["verbose"],
):
    """
    Run an exhaustive grid search over a fixed parameter grid.

    Evaluates every combination in param_grid using cross-validation.
    Use run_random_search() when the grid is large or continuous ranges
    are involved.

    Parameters
    ----------
    model            : sklearn-compatible estimator (unfitted)
    param_grid       : dict or list[dict]
        Parameter names (str) mapped to lists of values to try.
        Example: {"n_estimators": [100, 200], "max_depth": [3, 5, None]}
    X_train          : array-like, shape (n_samples, n_features)
    y_train          : array-like, shape (n_samples,)
    cv               : int or sklearn CV splitter
        Number of folds or a configured splitter from validation.py.
        Default 5.
    scoring          : str, callable, or None
        Metric name (e.g. "roc_auc", "f1_weighted", "r2") or sklearn
        scorer callable. None uses the estimator's default scorer.
    n_jobs           : int
        Number of parallel jobs. -1 uses all available cores.
    refit            : bool
        Refit the best estimator on the full training set. Required for
        best_estimator to be usable for prediction.
    verbose          : int
        Verbosity level forwarded to GridSearchCV (0 = silent).

    Returns
    -------
    dict with keys:
        "best_estimator" : fitted estimator or None (when refit=False)
        "best_params"    : dict of the best hyperparameter combination
        "best_score"     : float, mean CV score for best_params
        "cv_results"     : pd.DataFrame of all evaluated combinations,
                           sorted by rank (best first)
    """
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return _package_results(search, refit)


def run_random_search(
    model,
    param_distributions,
    X_train,
    y_train,
    *,
    cv=DEFAULT_OPTIMIZATION["cv"],
    scoring=DEFAULT_OPTIMIZATION["scoring"],
    n_iter=DEFAULT_OPTIMIZATION["n_iter"],
    n_jobs=DEFAULT_OPTIMIZATION["n_jobs"],
    random_state=RANDOM_STATE,
    refit=DEFAULT_OPTIMIZATION["refit"],
    verbose=DEFAULT_OPTIMIZATION["verbose"],
):
    """
    Run a randomised search over hyperparameter distributions.

    Samples n_iter combinations from param_distributions rather than
    exhaustively evaluating all combinations. Scales to large search
    spaces and continuous distributions (scipy.stats objects).

    Parameters
    ----------
    model                : sklearn-compatible estimator (unfitted)
    param_distributions  : dict
        Parameter names (str) mapped to distributions or lists of values.
        Example: {"n_estimators": [100, 200, 500],
                  "learning_rate": scipy.stats.loguniform(0.01, 0.3)}
    X_train              : array-like, shape (n_samples, n_features)
    y_train              : array-like, shape (n_samples,)
    cv                   : int or sklearn CV splitter
        Number of folds or a configured splitter from validation.py.
        Default 5.
    scoring              : str, callable, or None
        Metric name or scorer callable. None uses the estimator's default.
    n_iter               : int
        Number of parameter settings sampled. Default 50.
        Higher values improve coverage at the cost of compute time.
    n_jobs               : int
        Number of parallel jobs. -1 uses all available cores.
    random_state         : int
        Seed for reproducible sampling. Default 42.
    refit                : bool
        Refit the best estimator on the full training set.
    verbose              : int
        Verbosity level forwarded to RandomizedSearchCV (0 = silent).

    Returns
    -------
    dict with keys:
        "best_estimator" : fitted estimator or None (when refit=False)
        "best_params"    : dict of the best sampled hyperparameter combination
        "best_score"     : float, mean CV score for best_params
        "cv_results"     : pd.DataFrame of all sampled combinations,
                           sorted by rank (best first)
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=refit,
        verbose=verbose,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    return _package_results(search, refit)


# ================================================================
# FUTURE: ADVANCED OPTIMIZATION  (not yet implemented)
# ================================================================
#
# All planned search functions follow the same return contract as
# optimize_model() / run_grid_search / run_random_search:
#   best_estimator, best_params, best_score, cv_results,
#   search_duration, n_evaluations, method
#
# Bayesian Optimization via scikit-optimize:
#   run_bayesian_search(model, param_space, X_train, y_train, *,
#                       cv=5, scoring=None, n_calls=50, random_state=42)
#   param_space uses skopt.space objects (Real, Integer, Categorical).
#   Requires: pip install scikit-optimize
#
# Optuna:
#   run_optuna_search(model, param_space_fn, X_train, y_train, *,
#                     cv=5, scoring=None, n_trials=100,
#                     direction="maximize", timeout=None)
#   param_space_fn(trial) defines the search space using trial.suggest_*.
#   Pruning, parallelism, and study persistence handled by Optuna.
#   Requires: pip install optuna
#
# Nested cross-validation:
#   run_nested_cv(model, param_grid, X, y, *,
#                 outer_cv, inner_cv, scoring=None, search="grid")
#   Outer loop estimates generalisation error; inner loop selects params.
#   Returns per-outer-fold best_params, best_score, and test score.
