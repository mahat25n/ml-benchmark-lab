import copy
import warnings
from pathlib import Path

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
from src.evaluation import compute_classification_metrics
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

# Tasks with a complete implementation. Raise NotImplementedError for others
# until regression and unsupervised evaluation modules are ready.
_SUPPORTED_TASKS = {"classification"}


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
    optimize=None,
    optimize_method="random",
    # ── Optional: feature importance ───────────────────────────────
    compute_importance=False,
    importance_method="model",
    # ── Optional: report export ────────────────────────────────────
    export_formats=None,
    report_title=DEFAULT_REPORTING["word_title"],
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
    task            : str            "classification" (regression/unsupervised: future).
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

    optimize        : dict or None
        {model_name: param_grid_or_distributions} mapping.
        For each entry, runs the chosen search method on that model before
        the main evaluation loop. The best estimator replaces the original
        model in models_dict for the rest of the pipeline.
        Example: {"Random Forest": {"n_estimators": [100, 300], "max_depth": [5, 10]}}
    optimize_method : "random" | "grid"
        Search method applied to every entry in optimize.
        "random" uses RandomizedSearchCV; "grid" uses GridSearchCV.

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

    Returns
    -------
    results_df   : pd.DataFrame
        Per-model metrics sorted by sort_by.
    preprocessor : dict
        Fitted encoders and scaler from preprocess_data.
        When compute_importance=True, also contains key "importance":
        {model_name: pd.DataFrame} with ranked feature importance.
    """
    if task not in _SUPPORTED_TASKS:
        raise NotImplementedError(
            f"task='{task}' is not yet implemented. "
            f"Supported tasks: {sorted(_SUPPORTED_TASKS)}"
        )

    def _log(msg):
        if verbose:
            print(msg)

    # ── Output directory ───────────────────────────────────────────────────
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    _log(f"[data]    Loading: {csv_path}")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess(
        csv_path,
        target_col,
        na_values=na_values,
        scaler=scaler,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    n_classes = preprocessor["n_classes"]
    _log(
        f"[data]    train={X_train.shape}  test={X_test.shape}  "
        f"classes={n_classes}  features={len(preprocessor['feature_names'])}"
    )

    # Combined ROC/PR curves are only valid for binary classification.
    # Multiclass requires OvR decomposition — deferred to a future release.
    is_binary = n_classes == 2
    if not is_binary:
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
    if optimize is not None:
        from src.optimization import run_grid_search, run_random_search
        opt_fn = run_random_search if optimize_method == "random" else run_grid_search
        _log(
            f"[optim]   Optimizing {len(optimize)} model(s) "
            f"via {optimize_method} search (cv={cv_for_search}) ..."
        )
        for m_name, param_space in optimize.items():
            if m_name not in models_dict:
                warnings.warn(
                    f"run_benchmark: optimize key '{m_name}' not found in "
                    "models_dict — skipping.",
                    stacklevel=2,
                )
                continue
            _log(f"  {m_name:<30}")
            opt_kwargs = {"cv": cv_for_search}
            if optimize_method == "random":
                opt_kwargs["random_state"] = random_state
            opt_result = opt_fn(
                models_dict[m_name], param_space, X_train, y_train,
                **opt_kwargs,
            )
            models_dict[m_name] = opt_result["best_estimator"]
            _log(
                f"    cv_score={opt_result['best_score']:.4f}  "
                f"params={opt_result['best_params']}"
            )

    # ── Plot initialisation ────────────────────────────────────────────────
    # Always initialise to None; set only when binary so the finalize block
    # below can use a simple `if fig_roc` guard without tracking is_binary again.
    fig_roc = ax_roc = fig_pr = ax_pr = pr_baseline = None
    if is_binary:
        fig_roc, ax_roc = init_roc_figure()
        fig_pr,  ax_pr  = init_pr_figure()
        pr_baseline = float(y_test.mean())  # positive-class prevalence for PR baseline

    # ── Per-model loop ─────────────────────────────────────────────────────
    results = []

    _log(f"[models]  Running {len(models_dict)} model(s) ...")
    for name, model in models_dict.items():
        _log(f"  {name:<30}")

        if is_binary:
            metrics, y_pred = _run_binary_classification(
                name, model, X_train, X_test, y_train, y_test, ax_roc, ax_pr
            )
        else:
            metrics, y_pred = _run_multiclass_classification(
                name, model, X_train, X_test, y_train, y_test
            )
        # Future task routing slots in here:
        # elif task == "regression":
        #     metrics, y_pred = _run_regression(name, model, X_train, X_test, y_train, y_test)
        # elif task == "unsupervised":
        #     metrics = _run_unsupervised(name, model, X_train, X_test)
        #     y_pred = None

        results.append({"Model": name, **metrics})

        # Confusion matrix — common to all supervised tasks
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

    # ── Optional: report export ────────────────────────────────────────────
    if export_formats:
        from src.reporting import export_results_csv, export_results_excel, export_results_word
        _log(f"\n[export]  Writing report(s): {export_formats} ...")
        for fmt in export_formats:
            if fmt == "csv":
                p = out / "results.csv"
                export_results_csv(results_df, p)
                _log(f"  csv   -> {p}")
            elif fmt == "excel":
                p = out / "results.xlsx"
                export_results_excel(results_df, p)
                _log(f"  excel -> {p}")
            elif fmt == "word":
                p        = out / "results.docx"
                roc_path = out / "roc_curve.png"
                pr_path  = out / "pr_curve.png"
                cm_paths = {
                    n: out / f"cm_{n.replace(' ', '_')}.png"
                    for n in models_dict
                }
                export_results_word(
                    results_df, p,
                    title=report_title,
                    roc_path=roc_path if is_binary and roc_path.exists() else None,
                    pr_path=pr_path   if is_binary and pr_path.exists()  else None,
                    cm_paths=cm_paths,
                )
                _log(f"  word  -> {p}")
            else:
                warnings.warn(
                    f"run_benchmark: unknown export format '{fmt}', skipping.",
                    stacklevel=2,
                )

    return results_df, preprocessor


# ================================================================
# FUTURE TASK RUNNERS  (not yet implemented)
# All runners follow the same contract: (name, model, X_train, X_test,
# y_train, y_test, **task_kwargs) → (metrics_dict, y_pred or None)
# ================================================================
#
# def _run_regression(name, model, X_train, X_test, y_train, y_test):
#     → evaluation.compute_regression_metrics()
#     → plots.plot_residuals(), plots.plot_predicted_vs_actual()
#
# def _run_unsupervised(name, model, X_train, X_test):
#     → evaluation.compute_clustering_metrics()
#     → plots.plot_silhouette(), plots.plot_cluster_projection()
#     → returns (metrics, None)   # no y_pred
#
# def _run_time_series(name, model, X_train, X_test, y_train, y_test):
#     → validation.walk_forward_split() before fit
#     → evaluation.compute_regression_metrics() or classification metrics
#
# ================================================================
# FUTURE ORCHESTRATION  (not yet implemented)
# ================================================================
#
# Cross-validation mode (future):
#   run_cv_benchmark(csv_path, target_col, cv_strategy, ...) →
#       iterates over folds, aggregates per-fold results, returns
#       mean ± std metric table (no single-split ROC/PR curves)
#
# Statistical testing (future stats.py):
#   run_statistical_comparison(results_df, test="friedman") →
#       post-hoc tables, critical difference diagram
