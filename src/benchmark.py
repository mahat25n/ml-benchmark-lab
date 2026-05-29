import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DEFAULT_OPTIMIZATION, DEFAULT_PATHS, DEFAULT_REPORTING, RANDOM_STATE, TEST_SIZE
from src.data import load_and_preprocess
from src.evaluation import (
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_forecast_metrics,
    compute_pca_metrics,
    compute_regression_metrics,
)
from src.models import get_models
from src.plots import (
    add_pr_curve,
    add_roc_curve,
    finalize_pr_plot,
    finalize_roc_plot,
    init_pr_figure,
    init_roc_figure,
    plot_confusion_matrix,
)
from src.reporting import export_results_csv, export_results_excel, export_results_word
from src.time_series import create_lag_features, create_rolling_features
from src.validation import walk_forward_split

# Tasks with a complete implementation. Raise NotImplementedError for others.
_SUPPORTED_TASKS = {"classification", "regression", "unsupervised", "time_series"}


# ================================================================
# LOW-LEVEL HELPERS  (score extraction, safe metric call)
# ================================================================


def _extract_scores(model, X):
    """
    Return (y_prob, y_score) for a fitted binary classifier.

    Mirrors evaluation._extract_proba. Kept here because benchmark.py needs
    y_score for plotting separately from the metric computation in evaluation.py.

    y_prob  : calibrated [0, 1] probabilities from predict_proba[:, 1]
    y_score : ranking scores; predict_proba or decision_function fallback
    """
    y_prob, y_score = None, None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        y_score = y_prob
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    return y_prob, y_score


def _extract_all_proba(model, X):
    """Return the full predict_proba matrix (all classes) or None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


# ================================================================
# TASK-SPECIFIC RUNNERS
# Each runner: fits one model, evaluates it, updates any plots, and
# returns (metrics_dict, y_pred) so the caller can handle common
# post-steps (confusion matrix, results list) without branching.
# ================================================================


def _run_binary_classification(name, model, X_train, X_test, y_train, y_test, ax_roc, ax_pr):
    """
    Fit, evaluate, and update ROC/PR axes for one binary classifier.

    ROC and PR curves are added to the caller-supplied axes so that all
    models accumulate onto the same combined figure managed by run_benchmark.

    Returns
    -------
    metrics : dict   Full binary classification metric suite.
    y_pred  : ndarray
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob, y_score = _extract_scores(model, X_test)

    metrics = compute_classification_metrics(
        y_test, y_pred, y_prob=y_prob, y_score=y_score
    )

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        pre, rec, _ = precision_recall_curve(y_test, y_score)
        add_roc_curve(ax_roc, fpr, tpr, metrics["ROC AUC"], name)
        add_pr_curve(ax_pr,  pre, rec,  metrics["PR AUC"],  name)

    return metrics, y_pred


def _run_multiclass_classification(name, model, X_train, X_test, y_train, y_test):
    """
    Fit and evaluate one multiclass classifier.

    Sensitivity, Specificity, and PR AUC are returned as NaN — they require
    per-class (OvR) decomposition deferred to a future release — so the
    results table schema stays identical to the binary case.

    ROC AUC uses one-vs-rest macro averaging when predict_proba is available.

    Returns
    -------
    metrics : dict   Multiclass-compatible metric subset (NaN for binary-only fields).
    y_pred  : ndarray
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob_all = _extract_all_proba(model, X_test)

    metrics = {
        "Accuracy":          accuracy_score(y_test, y_pred),
        "Precision":         precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall":            recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "Sensitivity":       float("nan"),
        "Specificity":       float("nan"),
        "F1 Score":          f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "MCC":               _safe(matthews_corrcoef, y_test, y_pred),
        "Cohen Kappa":       _safe(cohen_kappa_score, y_test, y_pred),
        "ROC AUC": (
            _safe(roc_auc_score, y_test, y_prob_all, multi_class="ovr", average="macro")
            if y_prob_all is not None else float("nan")
        ),
        "PR AUC":   float("nan"),
        "Log Loss": _safe(log_loss, y_test, y_prob_all) if y_prob_all is not None else float("nan"),
    }

    return metrics, y_pred


def _run_regression(name, model, X_train, X_test, y_train, y_test):
    """
    Fit and evaluate one regressor.

    Returns
    -------
    metrics : dict   Regression metric suite: MAE, MSE, RMSE, R2, MAPE.
    y_pred  : ndarray
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    return metrics, y_pred


def _run_unsupervised(name, model, X_train):
    """
    Fit one unsupervised model and return metrics.

    Clustering models (those with fit_predict) are evaluated with internal
    metrics. PCA/decomposition models return explained-variance statistics.
    In both cases the returned dict contains all keys from both schemas so
    every row in the results table shares the same columns.

    Returns
    -------
    metrics : dict   Clustering or PCA metrics with NaN padding.
    labels  : ndarray or None   Cluster labels; None for PCA.
    """
    _nan = float("nan")
    _CLUSTER_NAN = {
        "Silhouette": _nan, "Davies-Bouldin": _nan,
        "Calinski-Harabasz": _nan, "n_clusters": _nan, "n_noise": _nan,
    }
    _PCA_NAN = {"n_components": _nan, "cum_explained_variance_pct": _nan}

    is_clustering = hasattr(model, "fit_predict")
    if is_clustering:
        labels  = model.fit_predict(X_train)
        metrics = compute_clustering_metrics(X_train, labels)
        metrics.update(_PCA_NAN)
        return metrics, labels
    else:
        model.fit(X_train)
        metrics = compute_pca_metrics(model)
        metrics.update(_CLUSTER_NAN)
        return metrics, None


# ================================================================
# SHARED HELPERS
# ================================================================


def _export_results(results_df, out, export_formats, report_title, log, **word_kwargs):
    """
    Write results to each requested format.

    Parameters
    ----------
    results_df     : pd.DataFrame
    out            : Path  Output directory (must already exist).
    export_formats : list[str] or None  Subset of ["csv", "excel", "word"].
    report_title   : str
    log            : callable  Progress printer.
    **word_kwargs  : Forwarded to export_results_word (figure path dicts).
    """
    if not export_formats:
        return
    log(f"\n[export]  Writing report(s): {export_formats} ...")
    for fmt in export_formats:
        if fmt == "csv":
            p = out / "results.csv"
            export_results_csv(results_df, p)
            log(f"  csv   -> {p}")
        elif fmt == "excel":
            p = out / "results.xlsx"
            export_results_excel(results_df, p)
            log(f"  excel -> {p}")
        elif fmt == "word":
            p = out / "results.docx"
            export_results_word(results_df, p, title=report_title, **word_kwargs)
            log(f"  word  -> {p}")
        else:
            warnings.warn(
                f"run_benchmark: unknown export format '{fmt}', skipping.",
                stacklevel=3,
            )


# ================================================================
# TIME-SERIES RUNNERS
# ================================================================


def _run_time_series_walk_forward(name, model, X, y, splits):
    """
    Walk-forward evaluation of one model across all folds.

    For each (train_idx, test_idx) pair the model is deep-copied,
    fitted on the training slice, and evaluated on the test slice.
    No data from future folds is used during training.

    Parameters
    ----------
    name   : str
    model  : sklearn estimator (deep-copied per fold to prevent state bleed)
    X      : np.ndarray  Full lag-feature array (temporal order preserved).
    y      : np.ndarray  Full target array aligned with X.
    splits : list of (train_idx, test_idx) tuples

    Returns
    -------
    metrics     : dict  Forecast metrics averaged across folds + n_folds key.
    fold_results: list of (y_true_fold, y_pred_fold) ndarrays.
    """
    _nan_metrics = {"MAE": float("nan"), "RMSE": float("nan"),
                    "MAPE": float("nan"), "SMAPE": float("nan"), "n_folds": 0}

    if not splits:
        return _nan_metrics, []

    fold_metrics = []
    fold_results = []

    for train_idx, test_idx in splits:
        fold_model = copy.deepcopy(model)
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred_fold = fold_model.predict(X[test_idx])
        y_true_fold = y[test_idx]
        fold_results.append((y_true_fold, y_pred_fold))
        fold_metrics.append(compute_forecast_metrics(y_true_fold, y_pred_fold))

    keys = list(fold_metrics[0].keys())
    avg_metrics = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(m[k])]
        avg_metrics[k] = float(np.mean(vals)) if vals else float("nan")
    avg_metrics["n_folds"] = len(fold_metrics)

    return avg_metrics, fold_results


def _run_time_series_pipeline(
    csv_path, target_col, out, models_dict, *,
    lags, rolling_windows, ts_n_splits, ts_horizon,
    na_values, scaler, sort_by, round_digits,
    export_formats, report_title, log, verbose,
    tracker=None,
    optimize=False,
    optimization_method="random",
    search_space=None,
    n_iter=20,
    compute_stats=False,
):
    """
    Full time-series benchmark: load -> feature engineering ->
    walk-forward CV -> plots -> (export).

    Called by run_benchmark when task="time_series". Returns the same
    (results_df, preprocessor) tuple as the other task runners.
    """
    from src.data import load_data, preprocess_data
    from src.logging_utils import Timer as _Timer, run_diagnostics as _run_diagnostics
    from src.plots import plot_forecast, plot_rolling_forecast, plot_residuals_over_time

    _ts_total = _Timer()
    _ts_total.start()
    _ts_opt_elapsed = 0.0
    _ts_model_timings = {}

    log(f"[data]    Loading: {csv_path}")
    _ts_data = _Timer()
    _ts_data.start()
    df = load_data(csv_path, na_values=na_values)

    # Feature engineering — purely temporal, no future leakage
    if lags:
        df = create_lag_features(df, lags, target_col)
    if rolling_windows:
        df = create_rolling_features(df, rolling_windows, target_col)

    # Encode categoricals; skip scaling so each fold can scale independently
    X, y, preprocessor = preprocess_data(df, target_col, scaler=False)
    n = len(X)
    _ts_data.stop()
    log(f"[data]    n_samples={n}  features={X.shape[1]}")

    if tracker is not None:
        tracker.log_dataset(
            n_samples=n,
            n_features=X.shape[1],
            train_size=n,
            test_size=0,
            target_col=target_col,
        )

    # Default models registry for time series
    if models_dict is None:
        models_dict = dict(get_models("time_series"))
    else:
        models_dict = dict(models_dict)

    # Walk-forward splits
    horizon = ts_horizon if ts_horizon is not None else max(1, n // (ts_n_splits + 1))
    try:
        splits = list(walk_forward_split(X, y, ts_n_splits, horizon))
    except ValueError as exc:
        raise ValueError(f"run_benchmark(task='time_series'): {exc}") from exc

    if not splits:
        raise ValueError(
            f"run_benchmark(task='time_series'): no splits generated. "
            f"n_samples={n}, ts_n_splits={ts_n_splits}, horizon={horizon}."
        )
    log(f"[splits]  {len(splits)} fold(s), horizon={horizon}")

    if tracker is not None:
        tracker.log_config(ts_n_splits=len(splits), ts_horizon=horizon)

    # ── Diagnostics on first walk-forward split ────────────────────────────
    _ts_diag = {}
    try:
        _tr_idx, _te_idx = splits[0]
        _feat_names = list(preprocessor.get("feature_names") or []) or None
        _ts_diag = _run_diagnostics(
            X[_tr_idx], X[_te_idx], y[_tr_idx], y[_te_idx],
            feature_names=_feat_names,
            task="regression",
        )
        preprocessor["diagnostics"] = _ts_diag
        (out / "diagnostics_summary.json").write_text(
            json.dumps(_ts_diag, indent=2, default=str), encoding="utf-8"
        )
        if _ts_diag.get("has_issues"):
            log(f"[diag]    {len(_ts_diag['issue_summary'])} issue(s) detected:")
            for _issue in _ts_diag["issue_summary"]:
                log(f"[diag]      {_issue}")
    except Exception:
        pass

    # ── Optional: hyperparameter optimization (time series) ───────────────
    # Inner CV uses TimeSeriesSplit to respect temporal order.
    _do_optimize_ts = bool(optimize)
    if isinstance(optimize, dict) and optimize:
        _ts_effective_space = dict(optimize)
    elif _do_optimize_ts and search_space:
        _ts_effective_space = dict(search_space)
    else:
        _ts_effective_space = None

    ts_opt_results = {}

    if _do_optimize_ts:
        from sklearn.model_selection import TimeSeriesSplit
        from src.optimization import build_search_space, optimize_model
        _ts_opt_timer = _Timer()
        _ts_opt_timer.start()
        ts_inner_cv = TimeSeriesSplit(n_splits=min(3, max(2, ts_n_splits - 1)))
        log(f"[optim]   {optimization_method} search (TimeSeriesSplit inner CV) ...")
        for m_name, model in list(models_dict.items()):
            if _ts_effective_space is not None:
                space = _ts_effective_space.get(m_name)
                if space is None:
                    continue
            else:
                space = build_search_space(m_name, task="time_series")
                if space is None:
                    log(f"  {m_name:<30}  [no default search space, skipping]")
                    continue

            log(f"  {m_name:<30}")
            opt_result = optimize_model(
                model, space, X, y,
                method=optimization_method,
                cv=ts_inner_cv,
                n_iter=n_iter,
                random_state=42,
            )
            models_dict[m_name] = opt_result["best_estimator"]
            ts_opt_results[m_name] = opt_result
            log(
                f"    cv_score={opt_result['best_score']:.4f}  "
                f"n_evals={opt_result['n_evaluations']}  "
                f"duration={opt_result['search_duration']:.1f}s  "
                f"params={opt_result['best_params']}"
            )
        _ts_opt_timer.stop()
        _ts_opt_elapsed = _ts_opt_timer.elapsed

    # Per-model loop
    results        = []
    forecast_paths = {}
    ts_y_preds     = {}   # {name: (y_true_all, y_pred_all)} for compute_stats

    if tracker is not None:
        tracker.log_models(list(models_dict.keys()))

    log(f"[models]  Running {len(models_dict)} model(s) ...")
    for name, model in models_dict.items():
        log(f"  {name:<30}")
        _ts_m_timer = _Timer()
        _ts_m_timer.start()
        metrics, fold_results = _run_time_series_walk_forward(name, model, X, y, splits)
        _ts_m_timer.stop()
        _ts_model_timings[name] = round(_ts_m_timer.elapsed, 3)
        results.append({"Model": name, **metrics})

        if fold_results:
            y_true_all = np.concatenate([r[0] for r in fold_results])
            y_pred_all = np.concatenate([r[1] for r in fold_results])
            ts_y_preds[name] = (y_true_all, y_pred_all)
            safe = name.replace(" ", "_")

            fc_path = out / f"forecast_{safe}.png"
            plot_forecast(y_true_all, y_pred_all, name, fc_path)
            forecast_paths[name] = fc_path

            plot_rolling_forecast(fold_results, name, out / f"rolling_forecast_{safe}.png")
            plot_residuals_over_time(y_true_all, y_pred_all, name,
                                     out / f"res_time_{safe}.png")

            mae  = metrics.get("MAE",  float("nan"))
            rmse = metrics.get("RMSE", float("nan"))
            log(
                f"    mae={mae:.4f}  rmse={rmse:.4f}  "
                f"folds={metrics.get('n_folds', 0)}"
            )

    results_df = build_results_table(results, sort_by=sort_by, round_digits=round_digits)

    log("\n[results]")
    if verbose:
        print(results_df.to_string(index=False))

    # ── Optional: forecasting statistical comparison ───────────────────────
    ts_stats_summary = {}
    if compute_stats and len(ts_y_preds) >= 2:
        from src.stats import compare_forecast_models
        log("\n[stats]   Computing Diebold-Mariano pairwise tests ...")
        error_dict = {
            m: yt - yp for m, (yt, yp) in ts_y_preds.items()
        }
        for loss in ("mae", "mse"):
            try:
                dm_df = compare_forecast_models(
                    error_dict, h=horizon, loss=loss, alpha=0.05
                )
                ts_stats_summary[f"dm_{loss}"] = dm_df.to_dict(orient="records")
            except Exception as exc:
                warnings.warn(
                    f"run_benchmark: DM test ({loss}) failed: {exc}",
                    stacklevel=2,
                )
        if ts_stats_summary:
            preprocessor["stats_summary"] = ts_stats_summary
            log(f"[stats]   DM tests computed for {len(error_dict)} model(s).")

    _export_results(
        results_df, out, export_formats, report_title, log,
        forecast_paths=forecast_paths or None,
        optimization_results=ts_opt_results if ts_opt_results else None,
        stats_summary=ts_stats_summary if ts_stats_summary else None,
    )

    _ts_total.stop()
    preprocessor["timing"] = {
        "total_seconds":        round(_ts_total.elapsed, 3),
        "data_loading_seconds": round(_ts_data.elapsed, 3),
        "optimization_seconds": round(_ts_opt_elapsed, 3),
        "models":               _ts_model_timings,
    }

    if tracker is not None:
        tracker.log_metrics(results_df)
        if ts_opt_results:
            tracker.log_optimization(ts_opt_results)
        if ts_stats_summary:
            tracker.log_stats(ts_stats_summary)
        if _ts_diag:
            tracker.log_diagnostics(_ts_diag)
        tracker.save()
        preprocessor["experiment"] = {
            "run_id":  tracker.run_id,
            "run_dir": str(tracker.run_dir),
        }
        log(f"[track]   Run saved: {tracker.run_dir}")

    return results_df, preprocessor


# ================================================================
# PUBLIC API
# ================================================================


def build_results_table(
    results,
    sort_by=DEFAULT_REPORTING["sort_by"],
    round_digits=DEFAULT_REPORTING["round_digits"],
):
    """
    Convert a list of per-model metric dicts into a sorted, rounded DataFrame.

    Parameters
    ----------
    results     : list[dict]  Each dict must have a "Model" key plus metric keys.
    sort_by     : str         Column to sort descending. Silently ignored if absent.
    round_digits: int         Decimal places applied to all numeric columns.

    Returns
    -------
    pd.DataFrame  with reset integer index.
    """
    df = pd.DataFrame(results)
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False, na_position="last")
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(round_digits)
    return df.reset_index(drop=True)


def run_benchmark(
    csv_path,
    target_col,
    *,
    # ── Core pipeline ──────────────────────────────────────────────
    output_dir=DEFAULT_PATHS["output_dir"],
    models_dict=None,
    task="classification",
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=True,
    na_values=None,
    scaler=None,
    sort_by=DEFAULT_REPORTING["sort_by"],
    round_digits=DEFAULT_REPORTING["round_digits"],
    verbose=True,
    # ── Optional: imbalance handling ───────────────────────────────
    imbalance_strategy=None,
    imbalance_kwargs=None,
    categorical_features=None,
    # ── Optional: CV strategy (inner loop for optimization) ────────
    cv_strategy=None,
    # ── Optional: hyperparameter optimization ──────────────────────
    optimize=False,
    optimization_method="random",
    search_space=None,
    n_iter=20,
    # ── Optional: feature selection ────────────────────────────────
    feature_selection=None,
    n_features=None,
    feature_selection_kwargs=None,
    # ── Optional: feature importance ───────────────────────────────
    compute_importance=False,
    importance_method="model",
    # ── Optional: report export ────────────────────────────────────
    export_formats=None,
    report_title=DEFAULT_REPORTING["word_title"],
    # ── Optional: time-series configuration ───────────────────────
    lags=None,
    rolling_windows=None,
    ts_n_splits=5,
    ts_horizon=None,
    # ── Optional: experiment tracking ─────────────────────────────
    experiment_name=None,
    # ── Optional: statistical comparison ──────────────────────────
    compute_stats=False,
    # ── Optional: logging ──────────────────────────────────────────
    log_level="info",
):
    """
    Full benchmark pipeline: load → preprocess → (imbalance) → (optimize) →
    train/evaluate all models → plots → (importance) → (export) → results table.

    All integrations added since the original release are opt-in via keyword
    arguments that default to None / False. Omitting them exactly reproduces
    the original two-argument call behaviour.

    Parameters
    ----------
    csv_path        : str or Path    Path to the input CSV file.
    target_col      : str            Target column name. Never hardcoded.
    output_dir      : str or Path    Directory where all output files are saved.
    models_dict     : dict or None   {name: estimator}. Defaults to get_models(task).
    task            : str            One of "classification", "regression", "unsupervised", "time_series".
    test_size       : float          Held-out test fraction.
    random_state    : int
    stratify        : bool           Stratify split by y. Set False for regression.
    na_values       : list or None   Extra strings treated as NA.
    scaler          : scaler, None, or False
    sort_by         : str            Metric column to sort the results table descending.
    round_digits    : int            Decimal precision in the results table.
    verbose         : bool           Print progress to stdout.

    imbalance_strategy  : str or None
        Resampling strategy name recognised by imbalance.get_sampler().
        Applied to X_train / y_train before model training.
        Examples: "smote", "random_over", "adasyn", "smotenc".
    imbalance_kwargs    : dict or None
        Extra keyword arguments forwarded to apply_sampling() — e.g.
        {"k_neighbors": 3, "sampling_strategy": 0.8}.
    categorical_features : list[int] or None
        Column indices of categorical features. Required when
        imbalance_strategy="smotenc".

    cv_strategy     : str, sklearn splitter, or None
        When provided, used as the inner cross-validator for hyperparameter
        search. Accepts a strategy name from validation.VALID_STRATEGIES
        (e.g. "stratified_kfold") or a pre-configured sklearn splitter.
        When None, the optimization default (5-fold integer) is used.

    optimize        : bool or dict
        False / None → no optimization (default).
        True         → optimize every model in models_dict using built-in
                       default search spaces (or search_space if provided).
        dict         → {model_name: param_grid} — backward-compatible form;
                       treated as optimize=True with an explicit search_space.
    optimization_method : "random" | "grid"
        Search method. "random" uses RandomizedSearchCV; "grid" uses
        GridSearchCV. Default "random".
    search_space    : dict or None
        {model_name: param_grid_or_distributions} override. Used when
        optimize=True to specify custom spaces instead of the defaults.
        Ignored when optimize is a dict (backward compat).
    n_iter          : int
        Number of parameter combinations sampled by random search. Default 20.

    compute_importance : bool
        When True, compute feature importance for every model after the
        evaluation loop. Results stored under preprocessor["importance"].
    importance_method  : "model" | "permutation"
        "model"       — uses feature_importances_ or coef_ (fast, model-specific).
        "permutation" — uses permutation_importance on X_test (model-agnostic).

    export_formats  : list[str] or None
        Formats to export after evaluation. Any subset of:
        ["csv", "excel", "word"].
        Files are written to output_dir.
    report_title    : str
        Title heading for the Word report.

    experiment_name : str or None
        When provided, activates experiment tracking. A run directory is
        created at ``<output_dir>/experiments/<experiment_name>/<run_id>/``
        and four artifact files are written: config.json, metrics.csv,
        environment.txt, experiment_summary.json.
        When None (default), no tracking is performed.

    compute_stats : bool
        When True, computes bootstrap confidence intervals per model and (for
        classification) pairwise McNemar tests between all model pairs.
        Results stored under preprocessor["stats_summary"]. Default False.

    log_level : str
        Logging verbosity: "debug", "info", "warning", or "error".
        Controls both console output (when verbose=True) and the benchmark.log
        file written to output_dir. Default "info".

    Returns
    -------
    results_df   : pd.DataFrame
        Per-model metrics sorted by sort_by.
    preprocessor : dict
        Fitted encoders and scaler from preprocess_data.
        When compute_importance=True, also contains key "importance":
        {model_name: pd.DataFrame} with ranked feature importance.
        When experiment_name is set, also contains key "experiment":
        {"run_id": str, "run_dir": str}.
        Always contains key "timing": {total_seconds, data_loading_seconds,
        optimization_seconds, models: {name: seconds}}.
        For supervised tasks, contains key "diagnostics": data-quality report.
    """
    if task not in _SUPPORTED_TASKS:
        raise NotImplementedError(
            f"task='{task}' is not yet implemented. "
            f"Supported tasks: {sorted(_SUPPORTED_TASKS)}"
        )

    # ── Experiment tracker (opt-in) ────────────────────────────────────────
    tracker = None
    if experiment_name is not None:
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker(experiment_name, base_dir=output_dir)
        tracker.log_config(
            task=task,
            random_state=random_state,
            test_size=test_size,
            stratify=stratify,
            imbalance_strategy=imbalance_strategy,
            optimization_method=optimization_method if optimize else None,
            n_models_optimized=len(optimize) if isinstance(optimize, dict) else 0,
        )

    # These tasks don't support stratified splitting.
    if task in ("regression", "unsupervised", "time_series") and stratify:
        warnings.warn(
            f"run_benchmark: stratify=True is not valid for {task} tasks. "
            "Setting stratify=False automatically.",
            stacklevel=2,
        )
        stratify = False

    # ── Output directory ───────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Logger (file + optional console) ──────────────────────────────────
    from src.logging_utils import Timer, get_logger, run_diagnostics
    _logger = get_logger(
        "ml_benchmark",
        level=log_level,
        log_file=out / "benchmark.log",
        console=verbose,
    )
    _log = _logger.info

    # ── Time-series task — separate pipeline, returns early ────────────────
    if task == "time_series":
        return _run_time_series_pipeline(
            csv_path, target_col, out, models_dict,
            lags=lags,
            rolling_windows=rolling_windows,
            ts_n_splits=ts_n_splits,
            ts_horizon=ts_horizon,
            na_values=na_values,
            scaler=scaler,
            sort_by=sort_by,
            round_digits=round_digits,
            export_formats=export_formats,
            report_title=report_title,
            log=_log,
            verbose=verbose,
            tracker=tracker,
            optimize=optimize,
            optimization_method=optimization_method,
            search_space=search_space,
            n_iter=n_iter,
            compute_stats=compute_stats,
        )

    # ── Data ───────────────────────────────────────────────────────────────
    _total_timer = Timer()
    _total_timer.start()

    _log(f"[data]    Loading: {csv_path}")
    _data_timer = Timer()
    _data_timer.start()
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        csv_path,
        target_col,
        na_values=na_values,
        scaler=scaler,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    _data_timer.stop()
    n_classes    = preprocessor["n_classes"]
    task_is_reg  = (task == "regression")
    task_is_unsup = (task == "unsupervised")
    _log(
        f"[data]    train={X_train.shape}  test={X_test.shape}  "
        + (f"classes={n_classes}  " if task == "classification" else "")
        + f"features={len(preprocessor['feature_names'])}"
    )

    if tracker is not None:
        tracker.log_dataset(
            n_samples=X_train.shape[0] + X_test.shape[0],
            n_features=X_train.shape[1],
            train_size=X_train.shape[0],
            test_size=X_test.shape[0],
            target_col=target_col,
        )

    # ── Diagnostics (supervised tasks only) ───────────────────────────────
    _diag = {}
    if not task_is_unsup:
        try:
            _diag = run_diagnostics(
                X_train, X_test, y_train, y_test,
                feature_names=preprocessor.get("feature_names"),
                task=task,
            )
            preprocessor["diagnostics"] = _diag
            (out / "diagnostics_summary.json").write_text(
                json.dumps(_diag, indent=2, default=str), encoding="utf-8"
            )
            if _diag["has_issues"]:
                _log(f"[diag]    {len(_diag['issue_summary'])} issue(s) detected:")
                for _issue in _diag["issue_summary"]:
                    _log(f"[diag]      {_issue}")
        except Exception:
            pass

    # Combined ROC/PR curves are only valid for binary classification.
    # Multiclass requires OvR decomposition — deferred to a future release.
    is_binary = (n_classes == 2) and (task == "classification")
    if task == "classification" and not is_binary:
        warnings.warn(
            f"n_classes={n_classes}: combined ROC and PR curve plots require binary targets. "
            "Curve plots will be skipped. Per-model confusion matrices will still be saved. "
            "ROC AUC (OvR macro) is reported in the results table when predict_proba is available."
        )

    # ── Models ─────────────────────────────────────────────────────────────
    # Shallow-copy so replacing entries during optimization does not mutate
    # the module-level registry returned by get_models().
    if models_dict is None:
        models_dict = dict(get_models(task))
    else:
        models_dict = dict(models_dict)

    if tracker is not None:
        tracker.log_models(list(models_dict.keys()))

    # ── Optional: feature selection ────────────────────────────────────────
    _fs_result = None
    if feature_selection is not None and not task_is_unsup:
        from src.feature_selection import run_feature_selection
        _log(f"[featsel] Running '{feature_selection}' selection ...")
        _fs_kw = dict(feature_selection_kwargs or {})
        if n_features is not None:
            _fs_kw.setdefault("n_features", n_features)
        try:
            _fs_result = run_feature_selection(
                X_train, y_train,
                feature_names=preprocessor.get("feature_names"),
                method=feature_selection,
                task=task,
                **_fs_kw,
            )
            _mask = _fs_result["selected_mask"]
            X_train = X_train[:, _mask]
            X_test  = X_test[:, _mask]
            preprocessor["feature_names"] = _fs_result["selected_features"]
            preprocessor["feature_selection"] = {
                k: v for k, v in _fs_result.items() if k != "selected_mask"
            }
            _log(
                f"[featsel] {_fs_result['n_before']} → {_fs_result['n_after']} features "
                f"(removed {_fs_result['n_removed']})"
            )
            if tracker is not None:
                tracker.log_config(
                    feature_selection_method=feature_selection,
                    n_features_before=_fs_result["n_before"],
                    n_features_after=_fs_result["n_after"],
                )
        except Exception as exc:
            warnings.warn(
                f"run_benchmark: feature selection failed ({exc}); "
                "proceeding with all features.",
                stacklevel=2,
            )

    # ── Optional: imbalance handling ───────────────────────────────────────
    if imbalance_strategy is not None:
        from src.imbalance import apply_sampling
        _log(f"[balance] Applying '{imbalance_strategy}' resampling ...")
        sample_result = apply_sampling(
            X_train, y_train,
            imbalance_strategy,
            categorical_features=categorical_features,
            **(imbalance_kwargs or {}),
        )
        X_train = sample_result["X_res"]
        y_train = sample_result["y_res"]
        _log(f"[balance] {sample_result['summary']}")

    # ── Optional: CV strategy resolution ───────────────────────────────────
    # Resolved here so the same splitter is reused for every model's search.
    cv_for_search = DEFAULT_OPTIMIZATION["cv"]
    if cv_strategy is not None:
        if isinstance(cv_strategy, str):
            from src.validation import describe_cv_strategy, get_cv_strategy
            cv_for_search = get_cv_strategy(cv_strategy)
            _log(f"[cv]      {describe_cv_strategy(cv_for_search)}")
        else:
            cv_for_search = cv_strategy
            _log(f"[cv]      {type(cv_strategy).__name__} (caller-supplied)")

    # ── Optional: hyperparameter optimization ──────────────────────────────
    # optimize=False/None  → skip
    # optimize=True        → use build_search_space() or search_space override
    # optimize=<dict>      → backward compat; treat as per-model search space
    _do_optimize = bool(optimize)
    if isinstance(optimize, dict) and optimize:
        _effective_space = dict(optimize)          # old API: optimize was the space
    elif _do_optimize and search_space:
        _effective_space = dict(search_space)      # new API: explicit override
    else:
        _effective_space = None                    # None → build defaults per model

    opt_results   = {}   # {model_name: optimize_model() result}
    _opt_elapsed  = 0.0

    if _do_optimize:
        from src.optimization import build_search_space, optimize_model
        _opt_timer = Timer()
        _opt_timer.start()
        _log(
            f"[optim]   {optimization_method} search  cv={cv_for_search}  "
            f"n_iter={n_iter} ..."
        )
        for m_name, model in list(models_dict.items()):
            if _effective_space is not None:
                space = _effective_space.get(m_name)
                if space is None:
                    continue  # not in explicit dict — skip this model
            else:
                space = build_search_space(m_name, task=task)
                if space is None:
                    _log(f"  {m_name:<30}  [no default search space, skipping]")
                    continue

            _log(f"  {m_name:<30}")
            opt_result = optimize_model(
                model, space, X_train, y_train,
                method=optimization_method,
                cv=cv_for_search,
                n_iter=n_iter,
                random_state=random_state,
            )
            models_dict[m_name] = opt_result["best_estimator"]
            opt_results[m_name] = opt_result
            _log(
                f"    cv_score={opt_result['best_score']:.4f}  "
                f"n_evals={opt_result['n_evaluations']}  "
                f"duration={opt_result['search_duration']:.1f}s  "
                f"params={opt_result['best_params']}"
            )
        _opt_timer.stop()
        _opt_elapsed = _opt_timer.elapsed

    # ── Plot initialisation ────────────────────────────────────────────────
    # Always initialise to None; set only when binary so the finalize block
    # below can use a simple `if fig_roc` guard without tracking is_binary again.
    fig_roc = ax_roc = fig_pr = ax_pr = pr_baseline = None
    if is_binary:
        fig_roc, ax_roc = init_roc_figure()
        fig_pr,  ax_pr  = init_pr_figure()
        pr_baseline = float(y_test.mean())  # positive-class prevalence for PR baseline

    # Per-model figure paths; populated in the model loop.
    scatter_paths  = {}  # {name: Path}  actual vs predicted  (regression)
    residual_paths = {}  # {name: Path}  residuals            (regression)
    cluster_paths  = {}  # {name: Path}  cluster scatter      (unsupervised)
    pca_paths      = {}  # {name: Path}  PCA variance         (unsupervised)

    # ── Per-model loop ─────────────────────────────────────────────────────
    results       = []
    y_preds       = {}   # {name: ndarray} — populated for compute_stats
    _model_timings = {}

    _log(f"[models]  Running {len(models_dict)} model(s) ...")
    for name, model in models_dict.items():
        _log(f"  {name:<30}")
        _m_timer = Timer()
        _m_timer.start()

        if task_is_unsup:
            metrics, labels = _run_unsupervised(name, model, X_train)
        elif task_is_reg:
            metrics, y_pred = _run_regression(
                name, model, X_train, X_test, y_train, y_test
            )
            y_preds[name] = y_pred
        elif is_binary:
            metrics, y_pred = _run_binary_classification(
                name, model, X_train, X_test, y_train, y_test, ax_roc, ax_pr
            )
            y_preds[name] = y_pred
        else:
            metrics, y_pred = _run_multiclass_classification(
                name, model, X_train, X_test, y_train, y_test
            )
            y_preds[name] = y_pred

        _m_timer.stop()
        _model_timings[name] = round(_m_timer.elapsed, 3)
        results.append({"Model": name, **metrics})

        safe = name.replace(" ", "_")
        if task_is_unsup:
            from src.plots import plot_cluster_scatter, plot_pca_variance
            is_clustering = labels is not None
            if is_clustering:
                cs_path = out / f"cluster_{safe}.png"
                plot_cluster_scatter(
                    X_train, labels, name, cs_path,
                    feature_names=preprocessor["feature_names"],
                )
                cluster_paths[name] = cs_path
                _log(
                    f"    n_clusters={metrics.get('n_clusters')}  "
                    f"silhouette={metrics.get('Silhouette')}"
                )
            else:
                pv_path = out / f"pca_{safe}.png"
                plot_pca_variance(model, name, pv_path)
                pca_paths[name] = pv_path
                _log(
                    f"    n_components={metrics.get('n_components')}  "
                    f"cum_var%={metrics.get('cum_explained_variance_pct'):.2f}"
                )
        elif task_is_reg:
            from src.plots import plot_actual_vs_predicted, plot_residuals
            avp_path = out / f"avp_{safe}.png"
            res_path = out / f"res_{safe}.png"
            plot_actual_vs_predicted(y_test, y_pred, name, avp_path)
            plot_residuals(y_test, y_pred, name, res_path)
            scatter_paths[name]  = avp_path
            residual_paths[name] = res_path
            _log(
                f"    mae={metrics['MAE']:.4f}  "
                f"r2={metrics['R2']:.4f}  "
                f"rmse={metrics['RMSE']:.4f}"
            )
        else:
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, name, out / f"cm_{name.replace(' ', '_')}.png")
            _log(
                f"    acc={metrics['Accuracy']:.4f}  "
                f"auc={metrics['ROC AUC']:.4f}  "
                f"f1={metrics['F1 Score']:.4f}"
            )

    # ── Finalise combined plots ────────────────────────────────────────────
    if fig_roc:
        finalize_roc_plot(fig_roc, ax_roc, out / "roc_curve.png")
        finalize_pr_plot(fig_pr,  ax_pr,  out / "pr_curve.png", baseline=pr_baseline)
        _log(f"[plots]   Saved to: {out.resolve()}")

    # ── Results table ──────────────────────────────────────────────────────
    results_df = build_results_table(results, sort_by=sort_by, round_digits=round_digits)

    _log("\n[results]")
    if verbose:
        print(results_df.to_string(index=False))

    # ── Optional: statistical comparison ──────────────────────────────────
    stats_summary = {}
    if compute_stats and not task_is_unsup and y_preds:
        from src.stats import compute_bootstrap_ci, run_mcnemar_test
        _log("\n[stats]   Computing bootstrap CIs and pairwise tests ...")

        ci_results = {}
        for m_name, yp in y_preds.items():
            yp = np.asarray(yp)
            if task_is_reg:
                vals = np.abs(yp - np.asarray(y_test))
                ci  = compute_bootstrap_ci(vals, stat_fn=np.mean)
                ci_results[m_name] = {"metric": "MAE", **ci}
            else:
                vals = (yp == np.asarray(y_test)).astype(float)
                ci  = compute_bootstrap_ci(vals, stat_fn=np.mean)
                ci_results[m_name] = {"metric": "Accuracy", **ci}

        stats_summary["bootstrap_ci"] = ci_results

        # Pairwise McNemar for classification only
        if task == "classification" and len(y_preds) >= 2:
            model_names = list(y_preds.keys())
            pairs = {}
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    na, nb = model_names[i], model_names[j]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res = run_mcnemar_test(y_test, y_preds[na], y_preds[nb])
                        pairs[f"{na} vs {nb}"] = {
                            "statistic":    res["statistic"],
                            "p_value":      res["p_value"],
                            "n_discordant": res["n_discordant"],
                        }
                    except Exception:
                        pass
            stats_summary["mcnemar_pairs"] = pairs

        preprocessor["stats_summary"] = stats_summary
        _log(f"[stats]   Bootstrap CIs computed for {len(ci_results)} model(s).")

    # ── Timing summary ────────────────────────────────────────────────────
    _total_timer.stop()
    preprocessor["timing"] = {
        "total_seconds":        round(_total_timer.elapsed, 3),
        "data_loading_seconds": round(_data_timer.elapsed, 3),
        "optimization_seconds": round(_opt_elapsed, 3),
        "models":               _model_timings,
    }

    # ── Optional: experiment tracking ─────────────────────────────────────
    if tracker is not None:
        tracker.log_metrics(results_df)
        if opt_results:
            tracker.log_optimization(opt_results)
        if stats_summary:
            tracker.log_stats(stats_summary)
        if _diag:
            tracker.log_diagnostics(_diag)
        tracker.save()
        preprocessor["experiment"] = {
            "run_id":  tracker.run_id,
            "run_dir": str(tracker.run_dir),
        }
        _log(f"[track]   Run saved: {tracker.run_dir}")

    # ── Optional: feature importance ───────────────────────────────────────
    if compute_importance:
        from src.explainability import compute_permutation_importance, get_feature_importance
        feature_names = preprocessor["feature_names"]
        importance_results = {}
        _log(f"\n[explain] Computing {importance_method} importance ...")
        for m_name, model in models_dict.items():
            try:
                if importance_method == "permutation":
                    imp_df = compute_permutation_importance(
                        model, X_test, y_test, feature_names,
                        random_state=random_state,
                    )
                else:
                    imp_df = get_feature_importance(model, feature_names)
                importance_results[m_name] = imp_df
                _log(f"  {m_name:<30}  top: {imp_df['feature'].iloc[0]}")
            except Exception as exc:
                warnings.warn(
                    f"run_benchmark: importance skipped for '{m_name}' ({exc})",
                    stacklevel=2,
                )
        if importance_results:
            preprocessor["importance"] = importance_results

    # ── Optional: feature selection plots ─────────────────────────────────
    if _fs_result is not None:
        try:
            from src.plots import (
                plot_feature_importance_ranking,
                plot_selected_features_summary,
            )
            if _fs_result.get("scores"):
                plot_feature_importance_ranking(
                    _fs_result["scores"],
                    f"Feature Scores — {feature_selection}",
                    out / "fs_feature_scores.png",
                )
            plot_selected_features_summary(
                _fs_result,
                "Feature Selection Summary",
                out / "fs_selection_summary.png",
            )
            _log("[featsel] Plots saved.")
        except Exception as exc:
            warnings.warn(
                f"run_benchmark: feature selection plots failed ({exc})",
                stacklevel=2,
            )

    # ── Optional: report export ────────────────────────────────────────────
    roc_path = out / "roc_curve.png"
    pr_path  = out / "pr_curve.png"
    cm_paths_word = {
        m: out / f"cm_{m.replace(' ', '_')}.png"
        for m in models_dict
    }
    _export_results(
        results_df, out, export_formats, report_title, _log,
        roc_path=roc_path          if is_binary and roc_path.exists() else None,
        pr_path=pr_path            if is_binary and pr_path.exists()  else None,
        cm_paths=cm_paths_word     if task == "classification"         else None,
        scatter_paths=scatter_paths    if task_is_reg   else None,
        residual_paths=residual_paths  if task_is_reg   else None,
        cluster_paths=cluster_paths    if task_is_unsup else None,
        pca_paths=pca_paths            if task_is_unsup else None,
        optimization_results=opt_results if opt_results else None,
        stats_summary=stats_summary if stats_summary else None,
    )

    return results_df, preprocessor


# ================================================================
# FUTURE ORCHESTRATION
# ================================================================
#
# Cross-validation mode:
#   run_cv_benchmark(csv_path, target_col, cv_strategy, ...) →
#       iterates over folds, aggregates per-fold results, returns
#       mean ± std metric table (no single-split ROC/PR curves)
#
# Statistical testing:
#   run_statistical_comparison(results_df, test="friedman") →
#       post-hoc tables, critical difference diagram
