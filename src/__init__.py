"""
ml-benchmark-lab
================
Research-grade machine learning benchmarking and comparison framework.
"""

from src.benchmark import build_results_table, run_benchmark
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
from src.data import load_and_preprocess, load_data, preprocess_data, split_data
from src.evaluation import compute_classification_metrics, compute_confusion_components
from src.explainability import compute_permutation_importance, get_feature_importance
from src.imbalance import VALID_STRATEGIES as VALID_SAMPLERS
from src.imbalance import apply_sampling, get_sampler
from src.models import get_models
from src.optimization import run_grid_search, run_random_search
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
from src.stats import run_friedman_test, run_mcnemar_test, run_wilcoxon_test
from src.validation import VALID_STRATEGIES, describe_cv_strategy, get_cv_strategy

__version__ = "0.1.0"
__author__  = "Mahat Ibrahim"

__all__ = [
    # Core pipeline
    "run_benchmark",
    "build_results_table",
    # Config
    "get_config",
    "RANDOM_STATE", "N_JOBS", "TEST_SIZE",
    "DEFAULT_PATHS", "DEFAULT_PLOTTING",
    "DEFAULT_VALIDATION", "DEFAULT_OPTIMIZATION", "DEFAULT_REPORTING",
    "VALID_SECTIONS",
    # Data
    "load_data", "preprocess_data", "split_data", "load_and_preprocess",
    # Models
    "get_models",
    # Evaluation
    "compute_classification_metrics", "compute_confusion_components",
    # Validation
    "get_cv_strategy", "describe_cv_strategy", "VALID_STRATEGIES",
    # Optimization
    "run_grid_search", "run_random_search",
    # Imbalance
    "get_sampler", "apply_sampling", "VALID_SAMPLERS",
    # Explainability
    "get_feature_importance", "compute_permutation_importance",
    # Statistics
    "run_mcnemar_test", "run_wilcoxon_test", "run_friedman_test",
    # Plots
    "plot_confusion_matrix",
    "init_roc_figure", "add_roc_curve", "finalize_roc_plot",
    "init_pr_figure",  "add_pr_curve",  "finalize_pr_plot",
    # Reporting
    "export_results_csv", "export_results_excel", "export_results_word",
]
