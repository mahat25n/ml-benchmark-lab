# ML Benchmark Lab

Research-grade machine learning benchmarking framework.
Supports **classification**, **regression**, **unsupervised**, and **time-series** tasks with modular
evaluation, SHAP explainability, statistical significance testing, hyperparameter optimization,
feature selection, model persistence, experiment tracking, dataset profiling, and
publication-ready export.

---

## Features

- One-call benchmark across all models with a consistent metric suite
- Classification (binary + multiclass), regression, unsupervised (clustering + PCA), and time series
- ROC curves, PR curves, confusion matrices, residual plots, cluster scatter, PCA variance, forecast plots
- SHAP explainability: TreeExplainer, LinearExplainer, KernelExplainer with auto-selection
- Global and local SHAP explanations with beeswarm, bar, and dependence plots
- Permutation importance as a model-agnostic alternative
- **Feature selection**: variance threshold, correlation filtering, mutual information, RFECV, Lasso — all sklearn-only
- Hyperparameter optimization: grid search and randomized search with built-in default search spaces
- Imbalance handling: SMOTE, ADASYN, SMOTENC, random over/under-sampling
- Statistical significance testing: McNemar, Wilcoxon, Friedman, paired t-test, corrected k-fold t-test
- Advanced ranking: average ranks, metric leaderboard, pairwise comparisons, Nemenyi CD
- Diebold-Mariano test for forecasting accuracy comparison
- Bootstrap confidence intervals for all tasks
- **Experiment tracking**: local filesystem runs with config, metrics, environment, and summary JSON
- **Experiment analytics**: aggregate results across runs — win rates, average ranks, timelines
- **Dataset profiling**: per-column stats, missingness analysis, class distribution, skewness/kurtosis
- **Model persistence**: `save_model`, `load_model`, `save_pipeline`, `load_pipeline`, `predict_from_csv`, `batch_predict`
- **Logging and diagnostics**: structured logging with `get_logger`, `Timer`, `run_diagnostics`
- Export: CSV, formatted Excel, Word report with embedded figures and statistical tables
- CLI: `ml-benchmark run`, `profile`, `analyze`, `save-model`, `predict`, `info`, `examples`, `version`
- Fully opt-in integrations — the simplest call is two arguments

---

## Installation

### From source

```bash
git clone https://github.com/mahatibrahim/ml-benchmark-lab
cd ml-benchmark-lab
pip install -e .
```

### With optional extras

```bash
pip install -e ".[shap]"       # SHAP explainability (TreeExplainer, KernelExplainer, ...)
pip install -e ".[dev]"        # pytest, pytest-cov, ruff, mypy
pip install -e ".[docs]"       # Sphinx documentation build
pip install -e ".[all]"        # shap + stats extras
```

### Requirements

Python >= 3.10. Core runtime dependencies are installed automatically:
`numpy`, `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `scipy`,
`matplotlib`, `seaborn`, `openpyxl`, `python-docx`.

---

## Command-line interface

After installation (`pip install -e .`) the `ml-benchmark` command is available.

### Commands

| Command | Description |
|---|---|
| `ml-benchmark run <csv> <target> [options]` | Run a full benchmark pipeline |
| `ml-benchmark profile <csv> [options]` | Profile a dataset and produce statistical reports |
| `ml-benchmark analyze [options]` | Aggregate and compare results across experiment runs |
| `ml-benchmark save-model <model.pkl> [options]` | Save a trained model to a directory |
| `ml-benchmark predict <csv> <model_path> [options]` | Run batch inference on a CSV |
| `ml-benchmark info` | Print version, models, and capabilities |
| `ml-benchmark examples` | List available example scripts |
| `ml-benchmark version` | Print the package version |

### `run` — full option reference

```
ml-benchmark run <csv_path> <target_col> [OPTIONS]

Positional:
  csv_path            Path to the input CSV file
  target_col          Name of the target column

Core:
  --task              classification | regression | unsupervised | time_series
                        (default: classification)
  --output-dir DIR    Directory for all output files  (default: outputs)
  --test-size FLOAT   Held-out test fraction          (default: 0.2)
  --random-state INT  Global random seed              (default: 42)

Optional features:
  --imbalance STR     Resampling strategy: smote, adasyn, random_over, random_under, smotenc
  --cv-strategy STR   Inner CV strategy: stratified_kfold, kfold, time_series_split, ...
  --export FMT        Comma-separated: csv, excel, word  (e.g. --export csv,word)
  --report-title STR  Title for the Word report  (default: "Benchmark Report")

Experiment tracking:
  --experiment-name NAME
                      Enable tracking; writes artifacts under
                      <output-dir>/experiments/<name>/<run-id>/

Optimization:
  --optimize          Optimize every model with built-in default search spaces
  --optimization-method  random | grid  (default: random)
  --n-iter INT        Parameter combinations for random search  (default: 20)

Statistical comparison:
  --compute-stats     Bootstrap CIs + McNemar (classification) or
                      Diebold-Mariano (time_series) pairwise tests

Time-series extras:
  --lags 1,2,3        Comma-separated lag values for feature engineering
  --ts-n-splits INT   Walk-forward folds  (default: 5)
  --ts-horizon INT    Test-window size per fold  (default: auto)

Output:
  --log-level         debug | info | warning | error  (default: info)
  --quiet             Suppress all progress output
```

### CLI examples

```bash
# Basic classification
ml-benchmark run churn.csv Churn

# Regression with CSV + Word export
ml-benchmark run prices.csv price --task regression --export csv,word

# Time series with lag features and DM statistical tests
ml-benchmark run ts.csv value --task time_series --lags 1,2,3 --compute-stats

# Optimize + track experiment
ml-benchmark run churn.csv Churn --optimize --experiment-name churn_v1

# Quiet mode (no progress, just results) with Excel export
ml-benchmark run data.csv target --export excel --quiet

# Profile a dataset before benchmarking
ml-benchmark profile data.csv --target label --plots

# Aggregate results from multiple experiment runs
ml-benchmark analyze --base-dir outputs --experiment-name churn_v1

# Save a trained model from a run directory
ml-benchmark save-model outputs/model.pkl --output-dir saved/

# Batch predict on new data
ml-benchmark predict new_data.csv saved/ --output-path predictions.csv

# Show all capabilities
ml-benchmark info

# List available example scripts
ml-benchmark examples
```

---

## Python API quick start

### Classification

```python
from src.benchmark import run_benchmark

results_df, preprocessor = run_benchmark(
    "data.csv", "target",
    output_dir="outputs/",
    verbose=True,
)
print(results_df[["Model", "Accuracy", "ROC AUC", "F1 Score"]])
```

### Regression

```python
results_df, preprocessor = run_benchmark(
    "data.csv", "price",
    task="regression",
    stratify=False,
    output_dir="outputs/",
)
print(results_df[["Model", "MAE", "RMSE", "R2"]])
```

### Unsupervised (clustering + PCA)

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

results_df, preprocessor = run_benchmark(
    "data.csv", "label",          # target_col required but ignored internally
    task="unsupervised",
    stratify=False,
    models_dict={
        "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
        "PCA":    PCA(n_components=2, random_state=42),
    },
    output_dir="outputs/",
)
print(results_df[["Model", "Silhouette", "n_clusters", "cum_explained_variance_pct"]])
```

### Time series (walk-forward forecasting)

```python
results_df, preprocessor = run_benchmark(
    "ts_data.csv", "value",
    task="time_series",
    lags=[1, 2, 3, 5],        # lag features: value_lag_1 ... value_lag_5
    ts_n_splits=4,             # walk-forward folds
    ts_horizon=20,             # steps in each test window
    compute_stats=True,        # Diebold-Mariano pairwise tests
    output_dir="outputs/",
)
print(results_df[["Model", "MAE", "RMSE", "MAPE", "SMAPE", "n_folds"]])
# preprocessor["stats_summary"]["dm_mae"] — DM test results
```

### With feature selection

```python
results_df, preprocessor = run_benchmark(
    "data.csv", "target",
    feature_selection="mutual_information",
    n_features=10,
)
fs = preprocessor["feature_selection"]
print(f"Selected {fs['n_after']} of {fs['n_before']} features")
print("Kept:", fs["selected_features"])
```

### With hyperparameter optimization

```python
results_df, preprocessor = run_benchmark(
    "churn.csv", "Churn",
    optimize=True,                       # use built-in default search spaces
    optimization_method="random",
    n_iter=20,
    export_formats=["csv", "excel", "word"],
)
```

### With imbalance handling and export

```python
results_df, preprocessor = run_benchmark(
    "churn.csv", "Churn",
    imbalance_strategy="smote",
    export_formats=["csv", "excel", "word"],
    report_title="Churn Prediction Benchmark",
)
```

### With experiment tracking

```python
results_df, preprocessor = run_benchmark(
    "churn.csv", "Churn",
    experiment_name="churn_v1",
    output_dir="outputs/",
)
# Artifacts written to: outputs/experiments/churn_v1/<run_id>/
#   config.json, metrics.csv, environment.txt, experiment_summary.json
run_info = preprocessor["experiment"]
print(run_info["run_id"], run_info["run_dir"])
```

### With bootstrap CIs and pairwise statistical tests

```python
results_df, preprocessor = run_benchmark(
    "churn.csv", "Churn",
    compute_stats=True,
)
ss = preprocessor["stats_summary"]
# ss["bootstrap_ci"]   — per-model 95% bootstrap confidence intervals
# ss["mcnemar_pairs"]  — pairwise McNemar test results (classification)
```

### Dataset profiling

```python
from src.data_profile import profile_dataset

profile = profile_dataset(
    "data.csv", target_col="label",
    output_dir="outputs/profile",
    plots_dir="outputs/profile/plots",
)
print(f"Rows: {profile['n_rows']}, Completeness: {profile['dataset_completeness_pct']:.1f}%")
print(profile["columns"][["column", "kind", "n_missing", "skewness"]])
```

### Model persistence

```python
from src.model_io import save_pipeline, load_pipeline, predict_from_csv

# Save model + preprocessor together
save_pipeline(fitted_model, preprocessor, "saved_model/")

# Load and predict on new data
model, prep = load_pipeline("saved_model/")
predictions = predict_from_csv("new_data.csv", "saved_model/", target_col="label")
```

### Experiment aggregation

```python
from src.experiment_analysis import summarize_experiment_history

summary = summarize_experiment_history(
    "outputs/", experiment_name="churn_v1",
    export_dir="outputs/analysis", plots_dir="outputs/analysis/plots",
)
print(summary["aggregate"][["Model", "mean_Accuracy", "win_pct", "avg_rank"]])
```

### SHAP explainability

```python
from src.explainability import compute_shap_values, summarise_shap_importance, explain_prediction
from src.plots import plot_shap_summary, plot_shap_bar, plot_shap_dependence

# Requires: pip install shap
shap_values, explainer = compute_shap_values(
    model, X_test, feature_names=feature_names,
)

# Global importance table
importance_df = summarise_shap_importance(shap_values, feature_names)
print(importance_df[["rank", "feature", "mean_abs_shap"]])

# Local explanation for one prediction
local_df = explain_prediction(model, X_test[0], feature_names=feature_names)
print(local_df[["feature", "feature_value", "shap_value"]])

# Plots
plot_shap_summary(shap_values, X_test, feature_names, "My Model", "shap_summary.png")
plot_shap_bar(importance_df, "My Model", "shap_bar.png")
```

### Statistical comparison across datasets

```python
import pandas as pd
from src.stats import (
    run_friedman_test, run_wilcoxon_test,
    compute_average_ranks, compute_metric_leaderboard,
    compute_pairwise_comparisons, prepare_cd_diagram_data,
    run_diebold_mariano_test, compare_forecast_models,
)

# Friedman test over multiple datasets
scores = pd.DataFrame({
    "Random Forest": [0.91, 0.88, 0.93, 0.87, 0.92],
    "XGBoost":       [0.93, 0.90, 0.95, 0.89, 0.94],
    "SVM":           [0.85, 0.82, 0.87, 0.81, 0.86],
})
result = run_friedman_test(scores)
print(result["interpretation"])

# Average ranks + Nemenyi CD
avg_ranks = compute_average_ranks(scores)    # rank 1 = best
cd_data   = prepare_cd_diagram_data(scores)  # cd, significant_pairs

# Forecast comparison (Diebold-Mariano)
errors = {"RF": y_true - y_pred_rf, "LR": y_true - y_pred_lr}
dm_df  = compare_forecast_models(errors, h=1, loss="mae")
print(dm_df[["Model_A", "Model_B", "DM_Statistic", "p_value", "Favored"]])
```

---

## Framework architecture

```
CSV input
    |
    v
data.py          load_and_preprocess()
    |              - read CSV, handle NA
    |              - encode categoricals (LabelEncoder)
    |              - scale features (StandardScaler, optional)
    |              - train/test split (stratified or not)
    |
    v
data_profile.py  profile_dataset()             [optional]
    |              - per-column stats, missingness, target analysis
    |
    v
feature_selection.py  run_feature_selection()  [optional]
    |              - variance, correlation, MI, RFECV, Lasso
    |
    v
imbalance.py     apply_sampling()              [optional]
    |              - SMOTE / ADASYN / random over/under
    |
    v
optimization.py  optimize_model()              [optional]
    |              - per-model param search before evaluation
    |              - built-in default search spaces
    |
    v
time_series.py   create_lag_features()         [time_series, optional]
    |              create_rolling_features()
    |
    v
models.py        get_models(task)
    |              - classification (8 models)
    |              - regression     (9 models)
    |              - unsupervised   (5 models: KMeans, DBSCAN, ...)
    |              - time_series    (3 models: LinearRegression, RF, XGBoost)
    |
    v
evaluation.py    compute_*_metrics()
    |              - classification:  12 metrics incl. ROC AUC, PR AUC
    |              - regression:      MAE, MSE, RMSE, R2, MAPE
    |              - clustering:      Silhouette, Davies-Bouldin, Calinski-Harabasz
    |              - pca:             n_components, cum_explained_variance_pct
    |              - forecasting:     MAE, RMSE, MAPE, SMAPE (fold-averaged)
    |
    v
plots.py         plot_confusion_matrix()        [classification]
    |            plot_actual_vs_predicted()      [regression]
    |            plot_residuals()                [regression]
    |            plot_cluster_scatter()          [unsupervised]
    |            plot_pca_variance()             [unsupervised]
    |            plot_forecast()                 [time_series]
    |            plot_rolling_forecast()         [time_series]
    |            plot_residuals_over_time()      [time_series]
    |            plot_ranking_bar()              [stats]
    |            plot_confidence_intervals()     [stats]
    |            plot_shap_*()                   [explainability]
    |            plot_missing_heatmap()          [data profiling]
    |            plot_class_distribution()       [data profiling]
    |            plot_numeric_distributions()    [data profiling]
    |            plot_feature_importance_ranking() [feature selection]
    |            plot_selected_features_summary()  [feature selection]
    |            plot_model_win_frequency()      [experiment analysis]
    |            plot_average_rank()             [experiment analysis]
    |            plot_metric_distribution()      [experiment analysis]
    |            plot_experiment_timeline()      [experiment analysis]
    |
    v
explainability.py compute_shap_values()         [optional, requires shap]
    |              summarise_shap_importance()
    |              explain_prediction()
    |              compute_permutation_importance()
    |
    v
stats.py         run_mcnemar_test()             [optional]
    |            run_wilcoxon_test()
    |            run_friedman_test()
    |            run_paired_ttest()
    |            run_corrected_kfold_ttest()
    |            compute_confidence_interval()
    |            compute_bootstrap_ci()
    |            compute_average_ranks()
    |            compute_metric_leaderboard()
    |            compute_pairwise_comparisons()
    |            compute_cd_nemenyi()
    |            run_diebold_mariano_test()
    |            compare_forecast_models()
    |
    v
reporting.py     export_results_csv()
    |            export_results_excel()         [formatted headers, frozen pane]
    |            export_results_word()           [title, table, figures, stats]
    |            export_analysis_word()          [experiment analysis report]
    |
    v
experiment.py    ExperimentTracker              [optional]
    |              - config.json, metrics.csv
    |              - environment.txt, experiment_summary.json
    |              - diagnostics_summary.json, profile_summary.json
    |
    v
experiment_analysis.py  summarize_experiment_history()  [optional]
    |              - aggregate across runs, compare experiments
    |
    v
model_io.py      save_model / load_model        [optional]
    |            save_pipeline / load_pipeline
    |            predict_from_csv / batch_predict
    |
    v
logging_utils.py get_logger / Timer             [always active]
    |            run_diagnostics
    |
    v
benchmark.py     run_benchmark()                [orchestrates all of the above]
    |
    v
cli.py           ml-benchmark run / profile / analyze /
                 save-model / predict / info / examples / version
```

---

## Supported models

### Classification (8 models)

| Model | Class | Key defaults |
|---|---|---|
| Decision Tree | `DecisionTreeClassifier` | `max_depth=5` |
| Random Forest | `RandomForestClassifier` | `n_estimators=200, class_weight="balanced"` |
| AdaBoost | `AdaBoostClassifier` | `n_estimators=200, learning_rate=0.5` |
| GradientBoost | `GradientBoostingClassifier` | `n_estimators=200, learning_rate=0.05` |
| XGBoost | `XGBClassifier` | `n_estimators=200, learning_rate=0.05` |
| SVM | `SVC` | `kernel="rbf", class_weight="balanced"` |
| KNN | `KNeighborsClassifier` | `n_neighbors=5` |
| ANN | `MLPClassifier` | `hidden_layer_sizes=(32, 16)` |

### Regression (9 models)

| Model | Class |
|---|---|
| Linear Regression | `LinearRegression` |
| Ridge | `Ridge(alpha=1.0)` |
| Lasso | `Lasso(alpha=0.1)` |
| ElasticNet | `ElasticNet(alpha=0.1, l1_ratio=0.5)` |
| Random Forest | `RandomForestRegressor` |
| GradientBoost | `GradientBoostingRegressor` |
| XGBoost | `XGBRegressor` |
| SVR | `SVR(kernel="rbf")` |
| KNN | `KNeighborsRegressor(n_neighbors=5)` |

### Unsupervised (5 models)

| Model | Type | Class |
|---|---|---|
| KMeans | Clustering | `KMeans(n_clusters=3)` |
| Agglomerative | Clustering | `AgglomerativeClustering(n_clusters=3)` |
| DBSCAN | Clustering | `DBSCAN(eps=0.5, min_samples=5)` |
| Gaussian Mixture | Clustering | `GaussianMixture(n_components=3)` |
| PCA | Decomposition | `PCA(n_components=2)` |

### Time Series (3 models)

| Model | Class |
|---|---|
| Linear Regression | `LinearRegression` |
| Random Forest | `RandomForestRegressor(n_estimators=100, max_depth=5)` |
| XGBoost | `XGBRegressor(n_estimators=100, max_depth=4)` |

Pass a custom `models_dict` to override any of the above.

---

## Supported metrics

### Classification

| Metric | Binary | Multiclass |
|---|---|---|
| Accuracy | Yes | Yes |
| Precision / Recall / F1 | Yes | Yes (weighted) |
| Sensitivity / Specificity | Yes | NaN |
| Balanced Accuracy | Yes | Yes |
| MCC / Cohen Kappa | Yes | Yes |
| ROC AUC | Yes | Yes (OvR macro) |
| PR AUC | Yes | NaN |
| Log Loss | Yes | Yes |

### Regression

MAE, MSE, RMSE, R2, MAPE (NaN when all ground-truth values are zero).

### Clustering

Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score.
DBSCAN noise points (label = -1) excluded from all metric computations.
All metrics return NaN when fewer than 2 non-noise clusters are found.

### Forecasting (time series)

MAE, RMSE, MAPE, SMAPE — reported as fold-averaged values.
MAPE returns NaN when all ground-truth values are zero.
SMAPE uses symmetric denominator `|y_true| + |y_pred|`; returns NaN only when all denominators are zero.

---

## Feature selection methods

| Method | Function | Description |
|---|---|---|
| `variance` | `variance_threshold_selection` | Removes constant / low-variance features |
| `correlation` | `correlation_selection` | Greedy removal of highly correlated pairs |
| `mutual_information` | `mutual_information_selection` | Top-k features by MI score |
| `rfecv` | `rfecv_selection` | Optimal count via recursive elimination + CV |
| `lasso` | `lasso_selection` | Non-zero coefficients from L1-regularised model |

---

## Supported CV strategies

| Name | Class |
|---|---|
| `stratified_kfold` | `StratifiedKFold` |
| `kfold` | `KFold` |
| `repeated_stratified_kfold` | `RepeatedStratifiedKFold` |
| `group_kfold` | `GroupKFold` |
| `time_series_split` | `TimeSeriesSplit` |
| `holdout` | `ShuffleSplit` |

### Temporal split generators (time series)

| Function | Description |
|---|---|
| `walk_forward_split` | Expanding training window; fixed test horizon per fold |
| `rolling_window_split` | Fixed-size sliding training window |
| `expanding_window_split` | Growing training window; explicit test_size |

---

## Supported imbalance strategies

| Name | Class |
|---|---|
| `smote` | `SMOTE` |
| `smotenc` | `SMOTENC` |
| `adasyn` | `ADASYN` |
| `random_over` | `RandomOverSampler` |
| `random_under` | `RandomUnderSampler` |

---

## Supported export formats

| Format | Function | Notes |
|---|---|---|
| CSV | `export_results_csv` | Plain table, no formatting |
| Excel | `export_results_excel` | Bold headers, alternating rows, frozen pane |
| Word | `export_results_word` | Title, results table, figures, stats sections |

---

## Project structure

```
ml-benchmark-lab/
├── pyproject.toml          # Package metadata, extras, tool config
├── LICENSE                 # MIT
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── src/
│   ├── __init__.py         # Public API — all symbols re-exported here  (v1.0.0)
│   ├── cli.py              # ml-benchmark CLI entry point
│   ├── config.py           # Centralized defaults (plotting, reporting, ...)
│   ├── data.py             # Loading, encoding, scaling, splitting
│   ├── models.py           # Model registries (classification / regression / unsupervised / time_series)
│   ├── time_series.py      # Lag and rolling feature utilities
│   ├── evaluation.py       # Metric computation — no plotting, no training
│   ├── plots.py            # Figure creation — no metrics, no training
│   ├── validation.py       # Cross-validation and temporal split strategies
│   ├── optimization.py     # Grid search and randomized search
│   ├── imbalance.py        # Resampling strategies
│   ├── feature_selection.py# Feature selection (variance, correlation, MI, RFECV, Lasso)
│   ├── explainability.py   # Feature importance, permutation importance, SHAP
│   ├── stats.py            # Statistical significance tests and ranking utilities
│   ├── experiment.py       # Experiment tracking (local filesystem)
│   ├── experiment_analysis.py  # Aggregate and compare experiment runs
│   ├── data_profile.py     # Dataset profiling and automated reports
│   ├── model_io.py         # Model persistence and batch inference
│   ├── logging_utils.py    # Structured logging, timing, data-quality diagnostics
│   ├── reporting.py        # Export utilities (CSV, Excel, Word)
│   └── benchmark.py        # End-to-end orchestration pipeline
├── tests/
│   ├── conftest.py
│   ├── test_benchmark.py
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_data_profile.py
│   ├── test_experiment.py
│   ├── test_experiment_analysis.py
│   ├── test_explainability.py
│   ├── test_explainability_shap.py   # skipped when shap not installed
│   ├── test_feature_selection.py
│   ├── test_forecasting_stats.py
│   ├── test_imbalance.py
│   ├── test_logging_utils.py
│   ├── test_model_io.py
│   ├── test_optimization.py
│   ├── test_regression.py
│   ├── test_reporting.py
│   ├── test_stats.py
│   ├── test_stats_advanced.py
│   ├── test_time_series.py
│   ├── test_unsupervised.py
│   └── test_validation.py
└── examples/
    ├── classification_example.py
    ├── regression_example.py
    ├── unsupervised_example.py
    ├── time_series_example.py
    ├── optimization_example.py
    ├── explainability_example.py        # requires: pip install shap
    ├── statistical_comparison_example.py
    ├── diebold_mariano_example.py
    ├── logging_diagnostics_example.py
    ├── experiment_tracking_example.py
    ├── experiment_analysis_example.py
    ├── data_profile_example.py
    ├── model_persistence_example.py
    └── feature_selection_example.py
```

---

## Running the examples

```bash
python examples/classification_example.py
python examples/regression_example.py
python examples/unsupervised_example.py
python examples/time_series_example.py
python examples/optimization_example.py
python examples/statistical_comparison_example.py
python examples/diebold_mariano_example.py
python examples/logging_diagnostics_example.py
python examples/experiment_tracking_example.py
python examples/experiment_analysis_example.py
python examples/data_profile_example.py
python examples/model_persistence_example.py
python examples/feature_selection_example.py
python examples/explainability_example.py   # requires: pip install shap
```

All outputs are written to `examples/outputs/` (or `outputs/`) and excluded from version control.

---

## Running tests

```bash
pip install -e ".[dev,shap]"
pytest                              # all tests
pytest tests/test_cli.py           # CLI tests only
pytest -m "not integration"        # fast tests only
pytest --cov=src --cov-report=term-missing
```

---

## Roadmap

### Completed
- [x] Classification, regression, unsupervised, time-series benchmarking
- [x] Hyperparameter optimization (grid search, randomized search, default search spaces)
- [x] Imbalance handling (SMOTE, ADASYN, SMOTENC, random over/under)
- [x] Feature selection (variance, correlation, mutual information, RFECV, Lasso)
- [x] SHAP explainability (TreeExplainer, LinearExplainer, KernelExplainer)
- [x] Statistical significance testing (McNemar, Wilcoxon, Friedman)
- [x] Advanced statistical comparison (paired t-test, corrected k-fold t-test, bootstrap CI, average ranks, pairwise comparisons, Nemenyi CD)
- [x] Diebold-Mariano test for forecast accuracy comparison
- [x] Experiment tracking (local filesystem, config + metrics + environment + summary)
- [x] Experiment aggregation and analytics (win rates, average ranks, timelines)
- [x] Dataset profiling (per-column stats, missingness, class distribution, skewness/kurtosis)
- [x] Model persistence (save/load model and pipeline, batch inference, predict_from_csv)
- [x] Logging and diagnostics (structured logging, timing, data-quality checks)
- [x] CLI: `run`, `profile`, `analyze`, `save-model`, `predict`, `info`, `examples`, `version`
- [x] Word report with embedded figures, optimization summary, statistical analysis tables

### Planned — explainability
- [ ] LIME local explanations
- [ ] Partial dependence plots (PDP) and individual conditional expectation (ICE) curves
- [ ] SHAP waterfall plots

### Planned — optimization
- [ ] Bayesian optimization via `scikit-optimize`
- [ ] Optuna integration with pruning and study persistence
- [ ] Nested cross-validation

### Planned — statistical testing
- [ ] Bayesian signed-rank and correlated t-tests

### Planned — imbalance
- [ ] BorderlineSMOTE, KMeansSMOTE, SMOTEENN, SMOTETomek

### Planned — export
- [ ] LaTeX table export
- [ ] PDF report generation
- [ ] HTML interactive report

---

## License

MIT License. See [LICENSE](LICENSE) for details.
