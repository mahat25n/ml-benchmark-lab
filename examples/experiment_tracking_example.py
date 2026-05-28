"""
Experiment tracking example.

Demonstrates how to activate experiment tracking via experiment_name,
read back the saved artifacts, and use the standalone reproducibility
utilities (set_global_seed, capture_environment, ExperimentTracker).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import json
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.benchmark import run_benchmark
from src.experiment import (
    ExperimentTracker,
    capture_environment,
    generate_run_id,
    set_global_seed,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Reproducibility utilities ──────────────────────────────────────────────

print("=== Reproducibility utilities ===")

run_id = generate_run_id()
print(f"Generated run ID : {run_id}")

set_global_seed(42)
print("Global seed set  : 42")

env = capture_environment()
print(f"Python           : {env['python_version'].split()[0]}")
print(f"Platform         : {env['platform']}")
print(f"NumPy            : {env['packages'].get('numpy', 'unknown')}")
print(f"scikit-learn     : {env['packages'].get('scikit-learn', 'unknown')}")

print()


# ── 2. Standalone ExperimentTracker usage ─────────────────────────────────────

print("=== Standalone ExperimentTracker ===")

tracker = ExperimentTracker("manual_example", base_dir=OUTPUT_DIR, run_id="demo_run")
tracker.log_config(task="classification", random_state=42, note="standalone usage demo")
tracker.log_dataset(n_samples=500, n_features=10, train_size=400, test_size=100,
                    target_col="label")
tracker.log_models(["Decision Tree", "Random Forest"])

results_demo = pd.DataFrame({
    "Model":    ["Random Forest", "Decision Tree"],
    "Accuracy": [0.91, 0.83],
    "ROC AUC":  [0.96, 0.88],
})
tracker.log_metrics(results_demo)
tracker.save()

print(f"Run directory    : {tracker.run_dir}")
print(f"Elapsed          : {tracker.elapsed_seconds():.3f}s")

config = json.loads((tracker.run_dir / "config.json").read_text())
print(f"Config task      : {config['task']}")
print(f"Models saved     : {config['models']}")
print()


# ── 3. Integrated with run_benchmark ──────────────────────────────────────────

print("=== run_benchmark with experiment tracking ===")

set_global_seed(42)
X_arr, y_arr = make_classification(
    n_samples=400, n_features=12, n_informative=6,
    n_redundant=2, random_state=42,
)
df = pd.DataFrame(X_arr, columns=[f"feature_{i}" for i in range(X_arr.shape[1])])
df["target"] = y_arr

csv_path = OUTPUT_DIR / "exp_tracking_data.csv"
df.to_csv(csv_path, index=False)

results_df, preprocessor = run_benchmark(
    str(csv_path),
    "target",
    task="classification",
    models_dict={
        "Decision Tree":  DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":  RandomForestClassifier(n_estimators=50, random_state=42),
    },
    output_dir=str(OUTPUT_DIR),
    experiment_name="classification_demo",
    verbose=True,
)

print()
print("Results table:")
print(results_df[["Model", "Accuracy", "ROC AUC", "F1 Score"]].to_string(index=False))

if "experiment" in preprocessor:
    exp_info = preprocessor["experiment"]
    print(f"\nExperiment tracking:")
    print(f"  run_id  : {exp_info['run_id']}")
    print(f"  run_dir : {exp_info['run_dir']}")

    run_dir = Path(exp_info["run_dir"])
    summary = json.loads((run_dir / "experiment_summary.json").read_text())
    print(f"\nExperiment summary:")
    print(f"  elapsed : {summary['elapsed_seconds']:.2f}s")
    print(f"  models  : {summary['models']}")
    print(f"  dataset : {summary['dataset']}")

print("\nAll outputs written to:", OUTPUT_DIR)
