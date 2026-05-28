"""
Hyperparameter optimization example.

Demonstrates three optimization modes:
  1. optimize=True  — use built-in default search spaces
  2. search_space   — provide custom per-model search spaces
  3. Standalone optimize_model() for fine-grained control

All examples use lightweight models and small n_iter to run quickly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.benchmark import run_benchmark
from src.optimization import (
    DEFAULT_SEARCH_SPACES,
    build_search_space,
    optimize_model,
    validate_search_space,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Inspect default search spaces ──────────────────────────────────────────

print("=== Default Search Spaces ===")
rf_space = build_search_space("Random Forest", task="classification")
print(f"Random Forest (classification): {list(rf_space.keys())}")

xgb_reg = build_search_space("XGBoost", task="regression")
print(f"XGBoost (regression):           {list(xgb_reg.keys())}")

ts_rf = build_search_space("Random Forest", task="time_series")
print(f"Random Forest (time_series):    {list(ts_rf.keys())}")

unknown = build_search_space("UnknownModel", task="classification")
print(f"Unknown model returns:          {unknown}")
print()


# ── 2. validate_search_space standalone ───────────────────────────────────────

print("=== Validate Search Space ===")
model = DecisionTreeClassifier()
try:
    validate_search_space(model, {"max_depth": [3, 5], "min_samples_split": [2, 4]})
    print("Valid space: OK")
except ValueError as e:
    print(f"Error: {e}")

try:
    validate_search_space(model, {"nonexistent_param": [1, 2]})
except ValueError as e:
    print(f"Invalid space caught: {e}")
print()


# ── 3. optimize_model standalone ──────────────────────────────────────────────

print("=== optimize_model standalone ===")
X, y = make_classification(n_samples=300, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

result = optimize_model(
    DecisionTreeClassifier(random_state=42),
    {"max_depth": [3, 5, 7], "min_samples_split": [2, 5]},
    X_train, y_train,
    method="random",
    cv=3,
    n_iter=6,
    random_state=42,
)
print(f"Method:         {result['method']}")
print(f"Best params:    {result['best_params']}")
print(f"Best CV score:  {result['best_score']:.4f}")
print(f"# Evaluations:  {result['n_evaluations']}")
print(f"Duration:       {result['search_duration']:.2f}s")
preds = result["best_estimator"].predict(X_test)
acc = (preds == y_test).mean()
print(f"Test accuracy:  {acc:.4f}")
print()


# ── 4. run_benchmark with optimize=True (use defaults) ────────────────────────

print("=== run_benchmark  optimize=True (default search spaces) ===")

np.random.seed(42)
X_arr, y_arr = make_classification(
    n_samples=300, n_features=8, n_informative=5, random_state=42
)
df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(8)])
df["target"] = y_arr
csv_path = OUTPUT_DIR / "opt_data.csv"
df.to_csv(csv_path, index=False)

results_df, preprocessor = run_benchmark(
    str(csv_path),
    "target",
    task="classification",
    models_dict={
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=20, random_state=42),
    },
    output_dir=str(OUTPUT_DIR),
    optimize=True,
    optimization_method="random",
    n_iter=5,
    verbose=True,
)
print("\nResults:")
print(results_df[["Model", "Accuracy", "ROC AUC"]].to_string(index=False))
print()


# ── 5. run_benchmark with custom search_space ─────────────────────────────────

print("=== run_benchmark  optimize=True + custom search_space ===")

results_df2, _ = run_benchmark(
    str(csv_path),
    "target",
    task="classification",
    models_dict={
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    },
    output_dir=str(OUTPUT_DIR / "custom_space"),
    optimize=True,
    search_space={
        "Decision Tree": {
            "max_depth":   [3, 5, 7, None],
            "criterion":   ["gini", "entropy"],
        }
    },
    optimization_method="grid",
    verbose=True,
)
print("\nResults:")
print(results_df2[["Model", "Accuracy"]].to_string(index=False))
print()


# ── 6. run_benchmark with backward-compat dict API ────────────────────────────

print("=== run_benchmark  optimize=dict (backward compat) ===")

results_df3, _ = run_benchmark(
    str(csv_path),
    "target",
    task="classification",
    models_dict={
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    },
    output_dir=str(OUTPUT_DIR / "compat_space"),
    optimize={
        "Decision Tree": {"max_depth": [3, 5], "min_samples_split": [2, 4]}
    },
    optimization_method="random",
    n_iter=4,
    verbose=True,
)
print("\nResults:")
print(results_df3[["Model", "Accuracy"]].to_string(index=False))
print()


# ── 7. Optimization + experiment tracking ─────────────────────────────────────

print("=== Optimization + experiment tracking ===")

results_df4, pp = run_benchmark(
    str(csv_path),
    "target",
    task="classification",
    models_dict={
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    },
    output_dir=str(OUTPUT_DIR),
    optimize=True,
    optimization_method="random",
    n_iter=5,
    experiment_name="optimization_demo",
    verbose=False,
)

if "experiment" in pp:
    run_dir = Path(pp["experiment"]["run_dir"])
    config = json.loads((run_dir / "config.json").read_text())
    print(f"Run directory:  {run_dir}")
    print(f"Optimization:   {json.dumps(config.get('optimization', {}), indent=2)}")

print("\nAll outputs written to:", OUTPUT_DIR)
