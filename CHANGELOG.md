# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] - 2026-05-29

### Added

**Feature selection** (`src/feature_selection.py`)
- `variance_threshold_selection` — removes constant / low-variance features
- `correlation_selection` — greedy removal of highly correlated feature pairs
- `mutual_information_selection` — top-k features by mutual information score
- `rfecv_selection` — recursive feature elimination with cross-validation
- `lasso_selection` — non-zero coefficients from L1-regularised model
- `run_feature_selection` — unified dispatcher for all five methods
- Standard result dict: `selected_features`, `removed_features`, `selected_mask`, `scores`
- `run_benchmark` parameters: `feature_selection=`, `n_features=`, `feature_selection_kwargs=`
- Saves `fs_feature_scores.png` and `fs_selection_summary.png` to output directory
- 56 tests in `tests/test_feature_selection.py`

**Model persistence and inference** (`src/model_io.py`)
- `save_model` / `load_model` — joblib serialisation with `model_metadata.json`
- `save_pipeline` / `load_pipeline` — bundled estimator + preprocessor dict
- `predict_from_csv` — load CSV → apply preprocessing → run inference → optional CSV output
- `batch_predict` — chunked inference on large numpy arrays; accepts path or live estimator
- `save_model_for_run` — writes into an experiment run directory
- Metadata tracks: model type, training timestamp, feature names, framework versions
- CLI: `ml-benchmark save-model`, `ml-benchmark predict`
- 53 tests in `tests/test_model_io.py`
- `examples/model_persistence_example.py`

**Dataset profiling** (`src/data_profile.py`)
- `profile_dataset` — accepts CSV path or DataFrame; exports JSON + CSV
- `summarize_columns` — per-column stats: kind, missing, cardinality, mean, skewness, kurtosis, percentiles
- `summarize_target` — binary / multiclass / continuous classification + imbalance ratio
- `summarize_missingness` — per-column pattern: complete / partial / mostly_missing
- `plot_missing_heatmap`, `plot_class_distribution`, `plot_numeric_distributions`
- CLI: `ml-benchmark profile`
- `ExperimentTracker.log_profile` integration
- 83 tests in `tests/test_data_profile.py`
- `examples/data_profile_example.py`

**Experiment aggregation and analytics** (`src/experiment_analysis.py`)
- `load_experiments` — scans persisted run directories
- `aggregate_experiments` — mean, std, win rate, average rank per model
- `compare_experiments` — wide pivot table by experiment group
- `summarize_experiment_history` — full pipeline with optional CSV / JSON / Word export
- `plot_model_win_frequency`, `plot_average_rank`, `plot_metric_distribution`, `plot_experiment_timeline`
- `export_aggregate_csv`, `export_aggregate_json`
- CLI: `ml-benchmark analyze`
- `export_analysis_word` in `reporting.py`
- 55 tests in `tests/test_experiment_analysis.py`
- `examples/experiment_analysis_example.py`

**Feature selection plots** (`src/plots.py`)
- `plot_feature_importance_ranking` — horizontal bar chart of feature scores
- `plot_selected_features_summary` — selected vs. removed count bar chart

### Changed
- `pyproject.toml`: version `1.0.0`; Development Status `5 - Production/Stable`; description and keywords updated
- `src/__init__.py`: version `1.0.0`; docstring updated; all new public symbols exported

---

## [0.6.0] - 2026-05-28

### Added

**Advanced logging and diagnostics** (`src/logging_utils.py`)
- `get_logger` — file + console logger with configurable level
- `Timer` — context-manager and manual start/stop elapsed timing
- `capture_warnings` — redirect Python warnings to the logger
- `run_diagnostics` — data-quality checks: NaN/Inf, class imbalance, feature scale, train/test distribution shift
- `run_benchmark` parameters: `log_level=`
- `ExperimentTracker.log_diagnostics` integration
- CLI: `--log-level debug/info/warning/error` for `ml-benchmark run`
- 52 tests in `tests/test_logging_utils.py`
- `examples/logging_diagnostics_example.py`

---

## [0.5.0] - 2026-05-27

### Added

**Experiment aggregation and analytics** (shipped as part of the 1.0.0 block above)

---

## [0.4.0] - 2026-05-26

### Added

**Experiment tracking** (`src/experiment.py`)
- `ExperimentTracker` — local filesystem run tracking
- `generate_run_id`, `set_global_seed`, `capture_environment`
- Writes: `config.json`, `metrics.csv`, `environment.txt`, `experiment_summary.json`
- `run_benchmark` parameter: `experiment_name=`
- 42 tests in `tests/test_experiment.py`
- `examples/experiment_tracking_example.py`

**Advanced statistical testing** (`src/stats.py` extensions)
- `run_paired_ttest`, `run_corrected_kfold_ttest`
- `compute_confidence_interval`, `compute_bootstrap_ci`
- `compute_average_ranks`, `compute_metric_leaderboard`
- `compute_pairwise_comparisons`, `compute_significance_summary`
- `compute_cd_nemenyi`, `prepare_cd_diagram_data`
- `compute_forecast_error_series`, `run_diebold_mariano_test`, `compare_forecast_models`
- `plot_ranking_bar`, `plot_confidence_intervals`
- `export_results_word` extended with optimization and statistical analysis sections
- 89 tests in `tests/test_stats_advanced.py` and `tests/test_forecasting_stats.py`
- `examples/statistical_comparison_example.py`, `examples/diebold_mariano_example.py`

---

## [0.3.0] - 2026-05-25

### Added

**Time-series support**
- `create_lag_features`, `create_rolling_features` in `time_series.py`
- Walk-forward CV: `walk_forward_split`, `rolling_window_split`, `expanding_window_split`
- `_run_time_series_walk_forward` and `_run_time_series_pipeline` in `benchmark.py`
- 3 time-series models: LinearRegression, RandomForest, XGBoost
- `compute_forecast_metrics` — fold-averaged MAE, RMSE, MAPE, SMAPE
- `plot_forecast`, `plot_rolling_forecast`, `plot_residuals_over_time`
- `run_benchmark` parameters: `lags=`, `rolling_windows=`, `ts_n_splits=`, `ts_horizon=`
- CLI: `--lags`, `--ts-n-splits`, `--ts-horizon` flags for `ml-benchmark run`
- 60 tests in `tests/test_time_series.py`
- `examples/time_series_example.py`

**Hyperparameter optimization** (`src/optimization.py`)
- `run_grid_search`, `run_random_search`, `optimize_model`
- `build_search_space`, `DEFAULT_SEARCH_SPACES`, `validate_search_space`
- `run_benchmark` parameters: `optimize=`, `optimization_method=`, `n_iter=`, `cv_strategy=`
- CLI: `--optimize`, `--optimization-method`, `--n-iter`, `--cv-strategy`
- 49 tests in `tests/test_optimization.py`
- `examples/optimization_example.py`

---

## [0.2.0] - 2025-05-28

### Added

**Regression support**
- `compute_regression_metrics` — MAE, MSE, RMSE, R2, MAPE (MAPE returns NaN when all targets are zero)
- `_run_regression` runner in `benchmark.py` — fits regressor, evaluates, returns metrics
- `plot_actual_vs_predicted`, `plot_residuals`, `plot_error_distribution` in `plots.py`
- `stratify=True` auto-corrected to `False` for regression tasks with `UserWarning`
- 9 regression models in `REGRESSION_MODELS`: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, GradientBoost, XGBoost, SVR, KNN
- Word report extended with `scatter_paths` and `residual_paths` sections

**Unsupervised learning support**
- `compute_clustering_metrics` — Silhouette, Davies-Bouldin, Calinski-Harabasz; excludes DBSCAN noise (label=-1); returns NaN when fewer than 2 clusters found
- `compute_pca_metrics` — n_components and cumulative explained variance %
- `_run_unsupervised` runner — dispatches clustering vs PCA via `hasattr(model, "fit_predict")`
- `plot_cluster_scatter` — 2-D scatter with automatic PCA projection for >2 features; noise points rendered as gray "x"
- `plot_pca_variance` — bar chart with cumulative explained variance red line overlay
- 5 unsupervised models in `UNSUPERVISED_MODELS`: KMeans, Agglomerative, DBSCAN, Gaussian Mixture, PCA
- Word report extended with `cluster_paths` and `pca_paths` sections

**SHAP explainability**
- `compute_shap_values` — auto-selects TreeExplainer, LinearExplainer, or KernelExplainer; always returns `(n_samples, n_features)` float array; handles SHAP Explanation objects (>= 0.40), list outputs, and 3-D arrays
- `summarise_shap_importance` — global importance DataFrame: `mean_abs_shap`, `mean_shap`, `positive_mean`, `negative_mean`
- `explain_prediction` — local explanation for a single instance: `feature_value`, `shap_value`, `abs_shap`, ranked
- `plot_shap_summary` — SHAP beeswarm via `shap.summary_plot` with `show=False`
- `plot_shap_bar` — horizontal bar chart of mean |SHAP| per feature
- `plot_shap_dependence` — feature value vs SHAP scatter; optional interaction-feature colour encoding

**Reporting improvements**
- `_add_cm_pairs` generalised with `heading=` parameter — reused for scatter, residual, cluster, and PCA figure sections
- All Word report section additions are fully optional (None-guarded)

**Statistical testing**
- `run_mcnemar_test`, `run_wilcoxon_test`, `run_friedman_test` in `stats.py`
- Bug fix: `run_friedman_test` now raises `ValueError` (not `IndexError`) when passed a 1-D array

**Example scripts**
- `examples/classification_example.py`
- `examples/regression_example.py`
- `examples/unsupervised_example.py`
- `examples/explainability_example.py`

**Test suite** (256 tests total)
- `tests/test_regression.py` — 20 tests
- `tests/test_unsupervised.py` — 28 tests
- `tests/test_explainability_shap.py` — 34 tests (skipped when shap not installed)
- `tests/test_reporting.py` — 22 tests
- `tests/test_stats.py` — 39 tests

### Changed
- `pyproject.toml`: version bumped to `0.2.0`; added `docs` extra; `setuptools>=68`; `[project.urls]` added; `[tool.pytest.ini_options]` and `[tool.mypy]` sections added
- `src/__init__.py`: exports extended to cover all new public API symbols
- `_SUPPORTED_TASKS` in `benchmark.py` now includes `"unsupervised"`
- `is_binary` now scoped to `task == "classification"` (previously `not task_is_reg`)

---

## [0.1.0] - 2025-05-01

### Added

**Core framework**
- `run_benchmark` — full classification pipeline: load → preprocess → train → evaluate → plot → export
- `build_results_table` — sorted, rounded DataFrame from per-model metric dicts
- `load_and_preprocess` — CSV loading, categorical encoding, standard scaling, train/test split
- 8 classification models in `CLASSIFICATION_MODELS`
- Full binary classification metric suite: Accuracy, Precision, Recall, Sensitivity, Specificity, F1, Balanced Accuracy, MCC, Cohen Kappa, ROC AUC, PR AUC, Log Loss
- Multiclass support (OvR ROC AUC, NaN for binary-only fields)
- Combined ROC and PR curve figures (binary only)
- Per-model confusion matrix PNGs
- `get_feature_importance` — tree and linear model importance with signed coef\_ support
- `compute_permutation_importance` — model-agnostic importance via sklearn
- `run_grid_search`, `run_random_search` — hyperparameter search wrappers
- `get_cv_strategy`, `describe_cv_strategy` — 6 cross-validation strategy types
- `get_sampler`, `apply_sampling` — SMOTE, ADASYN, random over/under, SMOTENC
- `export_results_csv`, `export_results_excel`, `export_results_word` — publication-ready exports
- CLI entry point `ml-benchmark`
- `.github/workflows/ci.yml` — Python 3.10 / 3.11 / 3.12 matrix CI
