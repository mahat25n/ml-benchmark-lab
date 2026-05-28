# Changelog

All notable changes to this project will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

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
