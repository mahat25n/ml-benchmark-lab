import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import DEFAULT_OPTIMIZATION, RANDOM_STATE


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
# PUBLIC API
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
# run_grid_search / run_random_search:
# (best_estimator, best_params, best_score, cv_results)
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
#   Pruning, parallelism, and study persistence handled by Optuna internally.
#   Requires: pip install optuna
#
# Hyperopt:
#   run_hyperopt_search(model, param_space, X_train, y_train, *,
#                       cv=5, scoring=None, max_evals=100, random_state=42)
#   param_space uses hp.choice / hp.uniform / hp.loguniform objects.
#   Requires: pip install hyperopt
#
# Nested cross-validation (when optimization.py connects to validation.py):
#   run_nested_cv(model, param_grid, X, y, *,
#                 outer_cv, inner_cv, scoring=None, search="grid")
#   Outer loop estimates generalisation error; inner loop selects params.
#   Returns per-outer-fold best_params, best_score, and test score.
