# ML Benchmark Lab

Research-grade machine learning benchmarking and comparison framework.
Designed for reproducible model evaluation, publication-ready outputs, and
modular extension across every stage of the ML pipeline.

---

## Overview

ML Benchmark Lab automates the full evaluation workflow:

```
CSV data  ->  preprocess  ->  (imbalance)  ->  (optimize)
          ->  train / evaluate all models
          ->  ROC curves, PR curves, confusion matrices
          ->  (feature importance)  ->  results table  ->  (export)
```

Every integration is **opt-in**. The simplest call is two arguments:

```python
from src.benchmark import run_benchmark

results_df, preprocessor = run_benchmark("data.csv", "target")
```

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/your-org/ml-benchmark-lab
cd ml-benchmark-lab
pip install -e .
```

### Install optional extras

```bash
pip install -e ".[shap]"        # SHAP explainability
pip install -e ".[optuna]"      # Optuna hyperparameter search
pip install -e ".[stats]"       # Nemenyi post-hoc, Bayesian comparison tests
pip install -e ".[dev]"         # pytest, ruff
pip install -e ".[all]"         # all optional extras
```

### Requirements

- Python >= 3.10
- See `requirements.txt` for the full dependency list

---

## Quick Start

### Run a benchmark

```python
from src.benchmark import run_benchmark

results_df, preprocessor = run_benchmark(
    "churn.csv",
    "Churn",
    output_dir="outputs",
    sort_by="ROC AUC",
    verbose=True,
)
print(results_df)
```

### With imbalance handling and export

```python
results_df, preprocessor = run_benchmark(
    "churn.csv",
    "Churn",
    imbalance_strategy="smote",
    export_formats=["csv", "excel", "word"],
    report_title="Churn Prediction Benchmark",
)
```

### With hyperparameter optimization

```python
results_df, preprocessor = run_benchmark(
    "churn.csv",
    "Churn",
    cv_strategy="stratified_kfold",
    optimize={
        "Random Forest": {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 10, None],
        },
        "XGBoost": {
            "n_estimators": [100, 300],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6],
        },
    },
    optimize_method="random",
)
```

### With feature importance

```python
results_df, preprocessor = run_benchmark(
    "churn.csv",
    "Churn",
    compute_importance=True,
    importance_method="permutation",
)

for model_name, imp_df in preprocessor["importance"].items():
    print(f"\n{model_name}")
    print(imp_df.head(5))
```

### Individual module usage

```python
from src.validation import get_cv_strategy, describe_cv_strategy
from src.optimization import run_random_search
from src.imbalance import apply_sampling
from src.stats import run_friedman_test
from src.explainability import get_feature_importance

# Cross-validation strategy
cv = get_cv_strategy("stratified_kfold", n_splits=10)
print(describe_cv_strategy(cv))

# Hyperparameter search
result = run_random_search(model, param_grid, X_train, y_train, cv=cv)
print(result["best_params"], result["best_score"])

# Imbalance handling
resampled = apply_sampling(X_train, y_train, "smote")
X_res, y_res = resampled["X_res"], resampled["y_res"]

# Statistical comparison across datasets
import pandas as pd
scores = pd.DataFrame({
    "RF":  [0.91, 0.88, 0.93, 0.87, 0.92],
    "XGB": [0.93, 0.90, 0.95, 0.89, 0.94],
    "SVM": [0.85, 0.82, 0.87, 0.81, 0.86],
})
result = run_friedman_test(scores)
print(result["interpretation"])
```

### Command-line interface

```bash
ml-benchmark run churn.csv Churn
ml-benchmark run churn.csv Churn --imbalance smote --export csv,excel,word
ml-benchmark run churn.csv Churn --cv-strategy stratified_kfold --quiet
ml-benchmark info
```

---

## Supported Models

Default classification models (`src/models.py`):

| Model | Class | Key Hyperparameters |
|---|---|---|
| Decision Tree | `DecisionTreeClassifier` | `max_depth=5` |
| Random Forest | `RandomForestClassifier` | `n_estimators=200, max_depth=8, class_weight="balanced"` |
| AdaBoost | `AdaBoostClassifier` | `n_estimators=200, learning_rate=0.5` |
| GradientBoost | `GradientBoostingClassifier` | `n_estimators=200, learning_rate=0.05, max_depth=3` |
| XGBoost | `XGBClassifier` | `n_estimators=200, learning_rate=0.05, max_depth=5` |
| SVM | `SVC` | `kernel="rbf", class_weight="balanced"` |
| KNN | `KNeighborsClassifier` | `n_neighbors=5` |
| ANN | `MLPClassifier` | `hidden_layer_sizes=(32, 16), activation="relu"` |

Pass a custom `models_dict` to `run_benchmark()` to override entirely.

---

## Supported Optimization Methods

| Method | Function | Backend |
|---|---|---|
| Grid Search | `run_grid_search()` | `sklearn.GridSearchCV` |
| Randomized Search | `run_random_search()` | `sklearn.RandomizedSearchCV` |
| Bayesian (planned) | `run_bayesian_search()` | `scikit-optimize` |
| Optuna (planned) | `run_optuna_search()` | `optuna` |
| Hyperopt (planned) | `run_hyperopt_search()` | `hyperopt` |

---

## Supported Validation Strategies

| Strategy Name | Class | Description |
|---|---|---|
| `stratified_kfold` | `StratifiedKFold` | Preserves class proportions per fold |
| `kfold` | `KFold` | Standard k-fold, shuffled |
| `repeated_stratified_kfold` | `RepeatedStratifiedKFold` | Reduces variance across repeats |
| `group_kfold` | `GroupKFold` | No group leaks between folds |
| `time_series_split` | `TimeSeriesSplit` | Forward-chaining, no lookahead |
| `holdout` | `ShuffleSplit` | Single train/test split |

---

## Supported Imbalance Methods

| Strategy Name | Class | Description |
|---|---|---|
| `random_over` | `RandomOverSampler` | Duplicate minority samples at random |
| `random_under` | `RandomUnderSampler` | Remove majority samples at random |
| `smote` | `SMOTE` | Synthetic interpolation (numeric features only) |
| `smotenc` | `SMOTENC` | SMOTE for mixed numeric + categorical |
| `adasyn` | `ADASYN` | Adaptive synthetic, boundary-focused |
| `borderline_smote` (planned) | `BorderlineSMOTE` | Focus on borderline minority samples |
| `kmeans_smote` (planned) | `KMeansSMOTE` | Cluster-based synthetic generation |
| `smoteenn` (planned) | `SMOTEENN` | SMOTE + Edited Nearest Neighbours |
| `smotetomek` (planned) | `SMOTETomek` | SMOTE + Tomek link removal |

---

## Supported Export Formats

| Format | Function | Status |
|---|---|---|
| CSV | `export_results_csv()` | Available |
| Excel (.xlsx) | `export_results_excel()` | Available — formatted headers, alternating rows, frozen pane |
| Word (.docx) | `export_results_word()` | Available — title, results table, ROC, PR, confusion matrices |
| LaTeX | `export_results_latex()` | Planned |
| PDF | `export_results_pdf()` | Planned |
| HTML | `export_results_html()` | Planned |

---

## Supported Statistical Tests

| Test | Function | Use Case |
|---|---|---|
| McNemar | `run_mcnemar_test()` | Two classifiers, one dataset, raw predictions |
| Wilcoxon signed-rank | `run_wilcoxon_test()` | Two classifiers, multiple datasets or folds |
| Friedman | `run_friedman_test()` | Three or more classifiers, multiple datasets |
| Nemenyi (planned) | `run_nemenyi_test()` | Post-hoc after a significant Friedman result |
| Diebold-Mariano (planned) | `run_diebold_mariano_test()` | Pairwise forecast accuracy (time series) |

---

## Project Structure

```
ml-benchmark-lab/
├── pyproject.toml          # Package metadata and build config
├── requirements.txt        # Runtime dependencies
├── README.md
├── .gitignore
└── src/
    ├── __init__.py         # Public API surface
    ├── cli.py              # ml-benchmark command-line interface
    ├── config.py           # Centralized defaults (DEFAULT_PLOTTING, etc.)
    ├── data.py             # CSV loading, encoding, scaling, splitting
    ├── models.py           # Model registries (classification / regression / unsupervised)
    ├── evaluation.py       # Metric computation (no plotting)
    ├── plots.py            # ROC, PR, confusion matrix visualisation (no training)
    ├── validation.py       # Cross-validation strategy utilities
    ├── optimization.py     # Grid search and randomized search wrappers
    ├── imbalance.py        # Resampling strategies (SMOTE, ADASYN, etc.)
    ├── explainability.py   # Feature importance and permutation importance
    ├── stats.py            # Statistical significance tests
    ├── reporting.py        # CSV / Excel / Word export utilities
    ├── benchmark.py        # End-to-end orchestration pipeline
    └── ROADMAP.md          # Project rules and planned features
```

---

## Roadmap

### In progress
- [ ] Regression task support (`_run_regression`, `compute_regression_metrics`)
- [ ] Unsupervised task support (`_run_unsupervised`, `compute_clustering_metrics`)

### Planned — explainability
- [ ] SHAP integration (`compute_shap_values`, `summarise_shap_importance`)
- [ ] SHAP beeswarm and dependence plots
- [ ] LIME local explanations
- [ ] Partial dependence plots and ICE curves

### Planned — optimization
- [ ] Bayesian optimization via `scikit-optimize`
- [ ] Optuna integration with pruning and study persistence
- [ ] Nested cross-validation (outer evaluation + inner search)

### Planned — validation
- [ ] Walk-forward and rolling window validation for time series
- [ ] Expanding window validation

### Planned — statistics
- [ ] Nemenyi post-hoc test (follows significant Friedman result)
- [ ] Diebold-Mariano test for forecast accuracy
- [ ] Bayesian signed-rank and correlated t-tests (`baycomp`)

### Planned — imbalance
- [ ] BorderlineSMOTE, KMeansSMOTE, SMOTEENN, SMOTETomek

### Planned — export
- [ ] LaTeX table export (publication-ready)
- [ ] PDF report generation
- [ ] HTML interactive report with sortable DataTables

### Planned — infrastructure
- [ ] `dashboard.py` — interactive results browser (Dash / Streamlit)
- [ ] Full CLI (`ml-benchmark compare`, `ml-benchmark report`)
- [ ] Test suite (`tests/`)

---

## License

MIT License. See `LICENSE` for details.
