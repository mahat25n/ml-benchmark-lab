# ML Benchmark Lab — Rules and Roadmap

---

## 1. PROJECT RULES

### Definitions

| Term | Definition |
|---|---|
| **Feature matrix X** | 2-D numeric array `(n_samples, n_features)`. Never contains the target column. |
| **Target variable y** | 1-D array `(n_samples,)`. Extracted from the DataFrame before any feature transformation. |
| **Feature column** | Any column in the source DataFrame that is not `target_col`. |
| **Target column** | The single column identified by the caller-supplied `target_col` string. |
| **Categorical feature** | Feature column with dtype `object` or `category`. Requires encoding before model input. |
| **Numeric feature** | Feature column with a numeric dtype. May require scaling depending on model family. |
| **Binary classification** | Target with exactly 2 distinct values. `y` encoded to integers 0 / 1. |
| **Multiclass classification** | Target with 3 or more distinct values. `y` encoded to integers 0 … K-1. |
| **Regression** | Continuous numeric target. No encoding applied. `y` preserved as float. |
| **Unsupervised learning** | No target column. Only `X` exists. `y` is absent. |

### Data Rules

- `target_col` must always be passed by the caller. It is never inferred or hardcoded inside any module.
- `X` must exclude `y`. The target column is dropped from the feature matrix before any transformation.
- Non-numeric `y` (dtype `object` or `category`) must be encoded to integers via `LabelEncoder`. The fitted encoder is stored in `preprocessor["target_encoder"]`.
- Numeric `y` must be preserved as-is. No implicit re-encoding or casting.
- **Data leakage (current prototype limitation):** `preprocess_data()` currently fits the scaler and feature encoders on the full dataset before splitting. Future versions must fit all preprocessing objects on `X_train` only and apply `.transform()` to `X_test` and inference data.

---

## 2. ROADMAP

### Planned Modules

| Module | Responsibility |
|---|---|
| `validation.py` | Cross-validation strategies: `StratifiedKFold`, `RepeatedStratifiedKFold`, nested CV, time-series splits |
| `optimization.py` | Hyperparameter search: `GridSearchCV`, `RandomizedSearchCV`, Optuna integration |
| `imbalance.py` | Imbalance handling methods and imbalance-aware metric reporting |
| `stats.py` | Statistical significance tests across benchmark results |
| `reporting.py` | Structured export of benchmark outputs to multiple formats |
| `dashboard.py` | Interactive results browser for policy and industry users |
| `explainability.py` | Model explanation and interpretability plots |
| `config.py` | Centralized configuration for paths, defaults, and experiment settings |

---

### Planned Features by Area

#### Regression Metrics
`MAE`, `MSE`, `RMSE`, `R²`, `Adjusted R²`, `MAPE`, `Max Error`, `Explained Variance Score`

#### Unsupervised Metrics
`Silhouette Score`, `Davies-Bouldin Index`, `Calinski-Harabasz Index`, `Adjusted Rand Index`

#### Imbalance Methods (`imbalance.py`)
`SMOTE`, `SMOTE-NC` (mixed numeric/categorical), `SMOTE-N` (nominal only), `ADASYN`, `Borderline-SMOTE`, `KMeans-SMOTE`

#### Time-Series and Panel Validation (`validation.py`)
- Walk-forward validation
- Rolling window validation
- Expanding window validation
- Nested cross-validation
- `GroupKFold` for panel / longitudinal data

#### Statistical Tests (`stats.py`)
- `McNemar` — pairwise classifier comparison on a single dataset
- `Wilcoxon signed-rank` — pairwise comparison across multiple datasets
- `Friedman` — multi-classifier comparison across multiple datasets
- `Nemenyi` — post-hoc pairwise test following a significant Friedman result
- `Diebold-Mariano` — pairwise forecast accuracy comparison (time series)

#### Explainability and Diagnostics (`explainability.py`)
- SHAP summary plots
- SHAP waterfall and dependence plots
- Calibration curves
- Feature importance plots (impurity-based and permutation)
- Critical difference diagrams (paired with `stats.py` output)

#### Export Formats (`reporting.py`)
- LaTeX table (publication-ready)
- Word document (`.docx`)
- Excel workbook (`.xlsx`)
- PDF report
- HTML report

#### Dashboard (`dashboard.py`)
- Interactive metric comparison across models and datasets
- Filter and sort by any metric column
- Designed for policy users and industry stakeholders who do not run code directly
