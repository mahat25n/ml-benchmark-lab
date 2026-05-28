# Contributing to ML Benchmark Lab

Thank you for considering a contribution. This document covers how to set up
the development environment, run tests, and submit a pull request.

---

## Development setup

```bash
git clone https://github.com/mahatibrahim/ml-benchmark-lab
cd ml-benchmark-lab
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,shap]"
```

## Running tests

```bash
# Fast unit tests only
pytest

# Include slow integration tests
pytest -m integration

# With coverage report
pytest --cov=src --cov-report=term-missing

# A specific file
pytest tests/test_benchmark.py -v
```

## Code style

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check src/           # lint
ruff check src/ --fix     # auto-fix safe issues
```

Rules enforced: `E`, `F`, `W`, `I` (isort). Line length is 100. All new code
must pass `ruff check` before a PR is merged.

## Module boundaries

The architecture enforces strict separation between concerns. Before adding
code, confirm which module it belongs in:

| Concern | Module |
|---|---|
| CSV loading, encoding, scaling, splitting | `data.py` |
| Model registry | `models.py` |
| Metric computation | `evaluation.py` |
| Figure creation and saving | `plots.py` |
| Cross-validation strategy construction | `validation.py` |
| Hyperparameter search | `optimization.py` |
| Resampling | `imbalance.py` |
| Feature importance, SHAP | `explainability.py` |
| Statistical tests | `stats.py` |
| File export (CSV, Excel, Word) | `reporting.py` |
| End-to-end orchestration | `benchmark.py` |
| Centralized defaults | `config.py` |

**Hard rules** (enforced in code review):
- No model training inside `data.py`
- No plotting inside `evaluation.py`
- No preprocessing inside `models.py`
- No metric computation inside `plots.py`
- No mixed classification and regression metrics in the same function

## Adding a new model

1. Add the estimator to the appropriate registry dict in `models.py`
   (`CLASSIFICATION_MODELS`, `REGRESSION_MODELS`, or `UNSUPERVISED_MODELS`).
2. Add a test in the relevant `test_*.py` file that confirms the new key
   is present and the model has `fit` / `predict`.

## Adding a new metric

1. Add the computation to the relevant function in `evaluation.py`
   (`compute_classification_metrics`, `compute_regression_metrics`, etc.).
2. Return `float("nan")` instead of raising when the metric is undefined
   (e.g. ROC AUC with only one class in the test set).
3. Add tests for normal input, edge cases, and NaN conditions.

## Adding a new export format

1. Add `export_results_<format>(results_df, output_path, **kwargs)` to
   `reporting.py` following the existing pattern.
2. Add the format name to `run_benchmark`'s `export_formats` handling block.
3. Add tests that verify the file is created and non-empty.

## Pull request checklist

- [ ] Tests added or updated for the changed code
- [ ] All 256 existing tests still pass (`pytest`)
- [ ] `ruff check src/` passes with no errors
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No new hard dependencies added without discussion

## Reporting a bug

Open an issue at https://github.com/mahatibrahim/ml-benchmark-lab/issues with:
- Python version and OS
- Minimal reproducible example
- Full traceback
