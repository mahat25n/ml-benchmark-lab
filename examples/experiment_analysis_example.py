"""
Experiment aggregation and benchmark analytics example.

Demonstrates:
  1. Running several benchmark experiments to populate the tracking store
  2. load_experiments()            — scan persisted run directories
  3. aggregate_experiments()       — per-model stats across runs
  4. compare_experiments()         — experiment-level comparison table
  5. summarize_experiment_history() — full summary with plots and exports
  6. export_analysis_word()        — Word document of the analysis
  7. CLI equivalent:  ml-benchmark analyze --base-dir ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.benchmark import run_benchmark
from src.experiment_analysis import (
    aggregate_experiments,
    compare_experiments,
    load_experiments,
    summarize_experiment_history,
)
from src.reporting import export_analysis_word

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "outputs" / "experiment_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 0. Generate two synthetic datasets ───────────────────────────────────────

def make_classification_csv(n=300, n_features=6, noise=0.0, path=None):
    X = np.random.randn(n, n_features)
    w = np.array([1.5, -1.0, 0.8, -0.5, 0.3, 0.1])
    p = 1 / (1 + np.exp(-(X @ w + noise * np.random.randn(n))))
    y = (p > 0.5).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    df.to_csv(path, index=False)
    return path


csv_easy = make_classification_csv(300, noise=0.0, path=OUTPUT_DIR / "data_easy.csv")
csv_hard = make_classification_csv(300, noise=2.0, path=OUTPUT_DIR / "data_hard.csv")


# ── 1. Run 3 experiments on easy dataset ─────────────────────────────────────

print("=== Running experiments on easy dataset ===")

EASY_MODELS = {
    "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
}

for i in range(3):
    run_benchmark(
        str(csv_easy), "target",
        task="classification",
        models_dict=EASY_MODELS,
        output_dir=str(OUTPUT_DIR),
        experiment_name="easy_dataset",
        verbose=False,
    )
    print(f"  Run {i+1}/3 complete")


# ── 2. Run 2 experiments on hard dataset ─────────────────────────────────────

print("\n=== Running experiments on hard dataset ===")

for i in range(2):
    run_benchmark(
        str(csv_hard), "target",
        task="classification",
        models_dict=EASY_MODELS,
        output_dir=str(OUTPUT_DIR),
        experiment_name="hard_dataset",
        verbose=False,
    )
    print(f"  Run {i+1}/2 complete")


# ── 3. load_experiments ───────────────────────────────────────────────────────

print("\n=== 3. load_experiments ===")

runs_all  = load_experiments(OUTPUT_DIR)
runs_easy = load_experiments(OUTPUT_DIR, experiment_name="easy_dataset")
runs_hard = load_experiments(OUTPUT_DIR, experiment_name="hard_dataset")

print(f"Total runs loaded : {len(runs_all)}")
print(f"  easy_dataset    : {len(runs_easy)}")
print(f"  hard_dataset    : {len(runs_hard)}")

print("\nFirst run sample:")
r0 = runs_easy[0]
print(f"  experiment_name : {r0['experiment_name']}")
print(f"  run_id          : {r0['run_id']}")
print(f"  task            : {r0['task']}")
print(f"  elapsed_seconds : {r0['elapsed_seconds']}")
print(f"  metrics:\n{r0['metrics'][['Model','Accuracy']].to_string(index=False)}")


# ── 4. aggregate_experiments ─────────────────────────────────────────────────

print("\n=== 4. aggregate_experiments (all runs) ===")

agg_all  = aggregate_experiments(runs_all, metric="Accuracy")
agg_easy = aggregate_experiments(runs_easy, metric="Accuracy")

print("\nAll runs aggregated:")
print(agg_all.to_string(index=False))

print("\nEasy dataset only:")
print(agg_easy.to_string(index=False))


# ── 5. compare_experiments ───────────────────────────────────────────────────

print("\n=== 5. compare_experiments (easy vs hard) ===")

comparison = compare_experiments(runs_all, metric="Accuracy")
print(comparison.to_string())


# ── 6. summarize_experiment_history ──────────────────────────────────────────

print("\n=== 6. summarize_experiment_history ===")

summary = summarize_experiment_history(
    OUTPUT_DIR,
    metric="Accuracy",
    export_dir=OUTPUT_DIR / "analysis",
    plots_dir=OUTPUT_DIR / "analysis" / "plots",
)

print(f"n_runs         : {summary['n_runs']}")
print(f"n_experiments  : {summary['n_experiments']}")
print(f"experiments    : {summary['experiment_names']}")
print(f"best_model     : {summary['best_model']}")
print(f"metric         : {summary['metric']}")

print("\nAggregate (all runs):")
print(summary["aggregate"].to_string(index=False))

print("\nTimeline (first 5 rows):")
print(summary["timeline"].head()[["run_id", "experiment_name", "best_model", "best_score"]].to_string(index=False))

print(f"\nExports written to: {OUTPUT_DIR / 'analysis'}")


# ── 7. export_analysis_word ───────────────────────────────────────────────────

print("\n=== 7. export_analysis_word ===")

word_path = OUTPUT_DIR / "analysis" / "analysis_report.docx"
export_analysis_word(summary, word_path, title="Benchmark History Report")
print(f"Word report: {word_path}")


print(f"\nAll outputs written to: {OUTPUT_DIR}")
