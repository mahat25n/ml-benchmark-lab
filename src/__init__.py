"""
ml-benchmark-lab
================
Research-grade ML benchmarking framework.

Supports classification, regression, unsupervised, and time-series tasks
with modular evaluation, SHAP explainability, statistical testing, and
publication-ready export.
"""

__version__ = "0.6.0"
__author__  = "Mahat Ibrahim"
__email__   = "mahatibrahim@ihu.edu.tr"
__license__ = "MIT"

# ── Core pipeline ─────────────────────────────────────────────────
from src.benchmark import build_results_table, run_benchmark

# ── Configuration ─────────────────────────────────────────────────
from src.config import (
    DEFAULT_OPTIMIZATION,
    DEFAULT_PATHS,
    DEFAULT_PLOTTING,
    DEFAULT_REPORTING,
    DEFAULT_VALIDATION,
    N_JOBS,
    RANDOM_STATE,
    TEST_SIZE,
    VALID_SECTIONS,
    get_config,
)

# ── Data ──────────────────────────────────────────────────────────
from src.data import load_and_preprocess, load_data, preprocess_data, split_data

# ── Time series ───────────────────────────────────────────────────
from src.time_series import create_lag_features, create_rolling_features

# ── Models ────────────────────────────────────────────────────────
from src.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    TIME_SERIES_MODELS,
    UNSUPERVISED_MODELS,
    get_models,
)

# ── Evaluation ────────────────────────────────────────────────────
from src.evaluation import (
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_confusion_components,
    compute_forecast_metrics,
    compute_pca_metrics,
    compute_regression_metrics,
)

# ── Validation ────────────────────────────────────────────────────
from src.validation import (
    VALID_STRATEGIES,
    describe_cv_strategy,
    expanding_window_split,
    get_cv_strategy,
    rolling_window_split,
    walk_forward_split,
)

# ── Optimization ──────────────────────────────────────────────────
from src.optimization import (
    DEFAULT_SEARCH_SPACES,
    build_search_space,
    optimize_model,
    run_grid_search,
    run_random_search,
    validate_search_space,
)

# ── Imbalance ─────────────────────────────────────────────────────
from src.imbalance import VALID_STRATEGIES as VALID_SAMPLERS
from src.imbalance import apply_sampling, get_sampler

# ── Explainability ────────────────────────────────────────────────
from src.explainability import (
    compute_permutation_importance,
    explain_prediction,
    get_feature_importance,
    summarise_shap_importance,
)

# compute_shap_values requires shap — imported lazily to avoid hard dependency
def compute_shap_values(*args, **kwargs):
    """Compute SHAP values. Requires: pip install shap."""
    from src.explainability import compute_shap_values as _fn
    return _fn(*args, **kwargs)

# ── Statistics ────────────────────────────────────────────────────
from src.stats import (
    run_friedman_test,
    run_mcnemar_test,
    run_wilcoxon_test,
    run_paired_ttest,
    run_corrected_kfold_ttest,
    compute_confidence_interval,
    compute_bootstrap_ci,
    compute_average_ranks,
    compute_metric_leaderboard,
    compute_pairwise_comparisons,
    compute_significance_summary,
    compute_cd_nemenyi,
    prepare_cd_diagram_data,
    compute_forecast_error_series,
    run_diebold_mariano_test,
    compare_forecast_models,
)

# ── Plots ─────────────────────────────────────────────────────────
from src.plots import (
    # Classification
    plot_confusion_matrix,
    init_roc_figure, add_roc_curve, finalize_roc_plot,
    init_pr_figure,  add_pr_curve,  finalize_pr_plot,
    # Regression
    plot_actual_vs_predicted,
    plot_residuals,
    plot_error_distribution,
    # Unsupervised
    plot_cluster_scatter,
    plot_pca_variance,
    # Time series
    plot_forecast,
    plot_rolling_forecast,
    plot_residuals_over_time,
    # SHAP (requires pip install shap for plot_shap_summary)
    plot_shap_bar,
    plot_shap_dependence,
    plot_shap_summary,
    # Statistical comparison
    plot_ranking_bar,
    plot_confidence_intervals,
)

# ── Reporting ─────────────────────────────────────────────────────
from src.reporting import export_results_csv, export_results_excel, export_results_word

# ── Experiment tracking ────────────────────────────────────────────
from src.experiment import (
    ExperimentTracker,
    capture_environment,
    generate_run_id,
    set_global_seed,
)


__all__ = [
    # Version
    "__version__", "__author__",
    # Core pipeline
    "run_benchmark", "build_results_table",
    # Configuration
    "get_config",
    "RANDOM_STATE", "N_JOBS", "TEST_SIZE",
    "DEFAULT_PATHS", "DEFAULT_PLOTTING",
    "DEFAULT_VALIDATION", "DEFAULT_OPTIMIZATION", "DEFAULT_REPORTING",
    "VALID_SECTIONS",
    # Data
    "load_data", "preprocess_data", "split_data", "load_and_preprocess",
    # Time series
    "create_lag_features", "create_rolling_features",
    # Models
    "get_models",
    "CLASSIFICATION_MODELS", "REGRESSION_MODELS",
    "UNSUPERVISED_MODELS", "TIME_SERIES_MODELS",
    # Evaluation
    "compute_classification_metrics", "compute_confusion_components",
    "compute_regression_metrics",
    "compute_clustering_metrics", "compute_pca_metrics",
    "compute_forecast_metrics",
    # Validation
    "get_cv_strategy", "describe_cv_strategy", "VALID_STRATEGIES",
    "walk_forward_split", "rolling_window_split", "expanding_window_split",
    # Optimization
    "optimize_model", "build_search_space", "validate_search_space",
    "DEFAULT_SEARCH_SPACES",
    "run_grid_search", "run_random_search",
    # Imbalance
    "get_sampler", "apply_sampling", "VALID_SAMPLERS",
    # Explainability
    "get_feature_importance", "compute_permutation_importance",
    "compute_shap_values", "summarise_shap_importance", "explain_prediction",
    # Statistics — basic
    "run_mcnemar_test", "run_wilcoxon_test", "run_friedman_test",
    # Statistics — advanced
    "run_paired_ttest", "run_corrected_kfold_ttest",
    "compute_confidence_interval", "compute_bootstrap_ci",
    "compute_average_ranks", "compute_metric_leaderboard",
    "compute_pairwise_comparisons", "compute_significance_summary",
    "compute_cd_nemenyi", "prepare_cd_diagram_data",
    # Statistics — forecasting
    "compute_forecast_error_series", "run_diebold_mariano_test",
    "compare_forecast_models",
    # Plots — classification
    "plot_confusion_matrix",
    "init_roc_figure", "add_roc_curve", "finalize_roc_plot",
    "init_pr_figure",  "add_pr_curve",  "finalize_pr_plot",
    # Plots — regression
    "plot_actual_vs_predicted", "plot_residuals", "plot_error_distribution",
    # Plots — unsupervised
    "plot_cluster_scatter", "plot_pca_variance",
    # Plots — time series
    "plot_forecast", "plot_rolling_forecast", "plot_residuals_over_time",
    # Plots — SHAP
    "plot_shap_summary", "plot_shap_bar", "plot_shap_dependence",
    # Plots — statistical comparison
    "plot_ranking_bar", "plot_confidence_intervals",
    # Reporting
    "export_results_csv", "export_results_excel", "export_results_word",
    # Experiment tracking
    "ExperimentTracker", "generate_run_id", "set_global_seed", "capture_environment",
]
