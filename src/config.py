"""
Centralized configuration for ML Benchmark Lab.

All default values used across modules are defined here. Individual
modules consume these values directly; callers can override any key
by passing explicit arguments to the relevant function.

Usage examples
--------------
    from src.config import DEFAULT_VALIDATION, DEFAULT_OPTIMIZATION
    cv = get_cv_strategy(DEFAULT_VALIDATION["strategy"],
                         n_splits=DEFAULT_VALIDATION["n_splits"])

    from src.config import get_config
    cfg = get_config("plotting")   # safe mutable copy
"""

import copy


# ================================================================
# GLOBAL SEEDS AND PARALLELISM
# Defined as module-level constants so other config sections can
# reference them, ensuring a single source of truth.
# ================================================================

RANDOM_STATE = 42
N_JOBS       = -1   # -1 = use all available CPU cores
TEST_SIZE    = 0.2  # default held-out fraction across all splitting operations


# ================================================================
# DEFAULT_PATHS
# Root directories for all file-based outputs. All modules that
# write files resolve their paths relative to these roots.
# ================================================================

DEFAULT_PATHS = {
    # Primary output root: plots, confusion matrices, result tables
    "output_dir":  "outputs",

    # Dedicated subdirectory for Word/Excel/CSV/PDF/LaTeX reports
    "reports_dir": "outputs/reports",

    # Subdirectory for serialised model artefacts (future: joblib dumps)
    "models_dir":  "outputs/models",
}


# ================================================================
# DEFAULT_PLOTTING
# Shared visual defaults consumed by plots.py. All values map
# directly onto matplotlib / seaborn parameters.
# ================================================================

DEFAULT_PLOTTING = {
    # File output
    "dpi":              300,       # publication-quality raster resolution
    "format":           "png",     # saved figure format: "png" | "pdf" | "svg"

    # Figure dimensions (width, height) in inches
    "figsize_roc":      (9, 7),    # combined ROC curve figure
    "figsize_pr":       (9, 7),    # combined precision-recall curve figure
    "figsize_cm":       (5, 4),    # individual confusion matrix figure
    "figsize_default":  (8, 6),    # fallback for other plot types

    # Line and marker styling
    "line_width":       1.8,       # curve line width
    "marker_size":      5,

    # Typography
    "font_size":        12,        # axis label font size
    "title_size":       13,        # figure title font size
    "tick_size":        10,        # axis tick label font size
    "legend_font_size": 9,         # legend entry font size
    "annot_size":       12,        # heatmap annotation font size (confusion matrix)

    # Grid and layout
    "grid_alpha":       0.6,       # grid line transparency
    "grid_linestyle":   "--",
}


# ================================================================
# DEFAULT_VALIDATION
# Defaults for cross-validation strategy construction in validation.py.
# ================================================================

DEFAULT_VALIDATION = {
    # Strategy name recognised by validation.get_cv_strategy()
    "strategy":     "stratified_kfold",

    # Number of folds (n_splits for KFold / StratifiedKFold families)
    "n_splits":     5,

    # Number of repetitions for RepeatedStratifiedKFold
    "n_repeats":    10,

    # Holdout fraction when strategy="holdout"
    "test_size":    TEST_SIZE,

    "random_state": RANDOM_STATE,
}


# ================================================================
# DEFAULT_OPTIMIZATION
# Defaults for hyperparameter search in optimization.py.
# ================================================================

DEFAULT_OPTIMIZATION = {
    # Number of parameter combinations sampled in RandomizedSearchCV
    "n_iter":       50,

    # Inner CV folds used during search (int or sklearn splitter)
    "cv":           5,

    # Scoring metric: None defers to the estimator's default scorer.
    # Common overrides: "roc_auc", "f1_weighted", "accuracy", "r2"
    "scoring":      None,

    "n_jobs":       N_JOBS,
    "random_state": RANDOM_STATE,

    # Refit the best estimator on the full training set after search
    "refit":        True,

    # Verbosity forwarded to GridSearchCV / RandomizedSearchCV (0 = silent)
    "verbose":      0,
}


# ================================================================
# DEFAULT_REPORTING
# Defaults for export functions in reporting.py.
# ================================================================

DEFAULT_REPORTING = {
    # Results table formatting
    "sort_by":      "ROC AUC",   # metric column to sort descending
    "round_digits": 4,           # decimal places in exported tables

    # Export formats produced by default when all formats are requested
    # Valid values: "csv", "excel", "word", "latex", "pdf", "html"
    "export_formats": ["csv", "excel", "word"],

    # Excel workbook
    "sheet_name":   "Results",

    # Word / PDF figure sizes in inches
    "figure_width": 6.0,         # full-width figures (ROC, PR)
    "cm_width":     3.0,         # per-confusion-matrix thumbnail width

    # Default report heading used by export_results_word()
    "word_title":   "Benchmark Report",
}


# ================================================================
# PUBLIC API
# ================================================================

# Registry mapping section names to their config dicts.
# Add new sections here as additional config groups are introduced.
_SECTIONS = {
    "paths":        DEFAULT_PATHS,
    "plotting":     DEFAULT_PLOTTING,
    "validation":   DEFAULT_VALIDATION,
    "optimization": DEFAULT_OPTIMIZATION,
    "reporting":    DEFAULT_REPORTING,
}

VALID_SECTIONS = frozenset(_SECTIONS)


def get_config(section=None):
    """
    Return a deep copy of one or all configuration sections.

    Deep-copying prevents callers from accidentally mutating module-level
    defaults, which would silently affect all subsequent calls in the same
    process.

    Parameters
    ----------
    section : str or None
        Name of the configuration section to return. Valid values:
        "paths", "plotting", "validation", "optimization", "reporting".
        None returns all sections merged into a single flat dict.

    Returns
    -------
    dict  A mutable copy of the requested configuration.

    Raises
    ------
    ValueError  Unknown section name.

    Examples
    --------
    >>> cfg = get_config("validation")
    >>> cfg["n_splits"] = 10          # safe: does not affect DEFAULT_VALIDATION
    >>> all_cfg = get_config()        # flat dict of every key
    """
    if section is None:
        merged = {}
        for d in _SECTIONS.values():
            merged.update(d)
        return copy.deepcopy(merged)

    if section not in _SECTIONS:
        raise ValueError(
            f"Unknown config section '{section}'. "
            f"Valid options: {sorted(VALID_SECTIONS)}"
        )

    return copy.deepcopy(_SECTIONS[section])


# ================================================================
# FUTURE: ADDITIONAL CONFIGURATION SECTIONS  (not yet implemented)
# ================================================================
#
# As new modules are added, register their defaults here following
# the same pattern: a module-level dict + an entry in _SECTIONS.
#
# DEFAULT_IMBALANCE — imbalance.py
#   {
#       "strategy":            "smote",
#       "k_neighbors":         5,
#       "random_state":        RANDOM_STATE,
#       "imbalance_threshold": 1.5,   # ratio below which resampling is skipped
#   }
#
# DEFAULT_STATS — stats.py
#   {
#       "alpha":       0.05,
#       "correction":  True,          # Yates' continuity correction in McNemar
#       "alternative": "two-sided",   # Wilcoxon alternative hypothesis
#       "zero_method": "wilcox",      # Wilcoxon zero-difference treatment
#   }
#
# DEFAULT_EXPLAINABILITY — explainability.py
#   {
#       "n_repeats":    10,           # permutation importance repeats
#       "scoring":      None,         # None defers to estimator default
#       "random_state": RANDOM_STATE,
#       "n_jobs":       N_JOBS,
#       "normalize":    False,
#       "shap_explainer": "auto",     # "auto" | "tree" | "linear" | "kernel"
#   }
#
# DEFAULT_DASHBOARD — dashboard.py
#   {
#       "host":       "127.0.0.1",
#       "port":       8050,
#       "debug":      False,
#       "theme":      "flatly",       # Bootstrap theme for Dash
#   }
