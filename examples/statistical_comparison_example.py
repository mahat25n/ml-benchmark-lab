"""
Advanced statistical comparison and ranking example.

Demonstrates:
  1. Paired t-test and corrected k-fold t-test
  2. Confidence intervals (t-distribution and bootstrap)
  3. Average rank computation and leaderboard
  4. Pairwise comparisons matrix and significance summary
  5. Critical difference (Nemenyi) preparation
  6. run_benchmark with compute_stats=True
  7. Rank bar plot and confidence interval plot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from src.stats import (
    run_paired_ttest,
    run_corrected_kfold_ttest,
    compute_confidence_interval,
    compute_bootstrap_ci,
    compute_average_ranks,
    compute_metric_leaderboard,
    compute_pairwise_comparisons,
    compute_significance_summary,
    compute_cd_nemenyi,
    prepare_cd_diagram_data,
)
from src.plots import plot_ranking_bar, plot_confidence_intervals
from src.benchmark import run_benchmark

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Paired t-test ──────────────────────────────────────────────────────────

print("=== 1. Paired t-test ===")

# Simulate 10-fold CV accuracy scores for two classifiers
np.random.seed(42)
scores_a = np.array([0.88, 0.90, 0.87, 0.91, 0.89, 0.92, 0.86, 0.90, 0.88, 0.91])
scores_b = np.array([0.83, 0.85, 0.82, 0.86, 0.84, 0.87, 0.81, 0.85, 0.83, 0.86])

res = run_paired_ttest(scores_a, scores_b)
print(f"t={res['statistic']:.4f}  p={res['p_value']:.4f}  df={res['df']}")
print(f"mean diff={res['mean_diff']:.4f}  95%CI=[{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
print(res["interpretation"])
print()


# ── 2. Corrected k-fold t-test ────────────────────────────────────────────────

print("=== 2. Corrected k-fold t-test (Nadeau-Bengio) ===")

res_corr = run_corrected_kfold_ttest(
    scores_a, scores_b,
    k=10,
    n_train=720, n_test=80,
)
print(f"t={res_corr['statistic']:.4f}  p={res_corr['p_value']:.4f}")
print(f"corrected variance={res_corr['corrected_variance']:.6f}")
print(res_corr["interpretation"])
print()


# ── 3. Confidence intervals ───────────────────────────────────────────────────

print("=== 3. Confidence intervals ===")

ci_t = compute_confidence_interval(scores_a, confidence=0.95, method="t")
print(f"t-CI:       mean={ci_t['mean']:.4f}  [{ci_t['lower']:.4f}, {ci_t['upper']:.4f}]")

ci_norm = compute_confidence_interval(scores_a, confidence=0.95, method="normal")
print(f"Normal-CI:  mean={ci_norm['mean']:.4f}  [{ci_norm['lower']:.4f}, {ci_norm['upper']:.4f}]")

ci_boot = compute_bootstrap_ci(scores_a, confidence=0.95, n_bootstrap=2000, random_state=42)
print(f"Bootstrap:  mean={ci_boot['mean']:.4f}  [{ci_boot['lower']:.4f}, {ci_boot['upper']:.4f}]")
print()


# ── 4. Average ranks and leaderboard ─────────────────────────────────────────

print("=== 4. Average ranks ===")

# Simulate 10 datasets × 4 classifiers
np.random.seed(0)
n_ds = 10
scores_matrix = pd.DataFrame({
    "Random Forest": np.random.uniform(0.85, 0.95, n_ds),
    "SVM":           np.random.uniform(0.80, 0.92, n_ds),
    "KNN":           np.random.uniform(0.75, 0.90, n_ds),
    "Decision Tree": np.random.uniform(0.70, 0.88, n_ds),
})

avg_ranks = compute_average_ranks(scores_matrix)
print("Average ranks (lower = better):")
for clf, rank in avg_ranks.items():
    print(f"  {clf:<20} {rank:.3f}")
print()

# Leaderboard from a results DataFrame
results_df = pd.DataFrame({
    "Model":    ["Random Forest", "SVM", "KNN", "Decision Tree"],
    "Accuracy": [0.91,            0.87,  0.84,  0.80],
    "ROC AUC":  [0.95,            0.91,  0.88,  0.83],
})

lb = compute_metric_leaderboard(results_df, "Accuracy")
print("Accuracy Leaderboard:")
print(lb.to_string(index=False))
print()


# ── 5. Pairwise comparisons and significance summary ─────────────────────────

print("=== 5. Pairwise comparisons ===")

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pw = compute_pairwise_comparisons(scores_matrix, test="wilcoxon", alpha=0.05)

print("p-values matrix:")
print(pw["p_values"].round(3).to_string())
print()

summary = compute_significance_summary(pw)
if len(summary) > 0:
    print("Significant pairs (alpha=0.05):")
    print(summary.to_string(index=False))
else:
    print("No significant pairs at alpha=0.05.")
print()


# ── 6. Critical difference (Nemenyi) ─────────────────────────────────────────

print("=== 6. Nemenyi critical difference ===")

cd = compute_cd_nemenyi(n_classifiers=4, n_datasets=n_ds, alpha=0.05)
print(f"CD (alpha=0.05, k=4, N={n_ds}) = {cd:.4f}")

cd_data = prepare_cd_diagram_data(scores_matrix, alpha=0.05)
print(f"Significant pairs: {cd_data['significant_pairs']}")
print()


# ── 7. Plot: ranking bar and confidence intervals ─────────────────────────────

print("=== 7. Plots ===")

rank_plot = OUTPUT_DIR / "ranking_bar.png"
plot_ranking_bar(avg_ranks, title="Model Average Ranks", save_path=rank_plot)
print(f"Ranking bar chart: {rank_plot}")

ci_rows = []
for clf in results_df["Model"]:
    fold_scores = scores_matrix[clf].values if clf in scores_matrix.columns else scores_a
    ci = compute_bootstrap_ci(fold_scores, random_state=42)
    ci_rows.append({"Model": clf, "mean": ci["mean"], "lower": ci["lower"], "upper": ci["upper"]})
ci_df = pd.DataFrame(ci_rows)

ci_plot = OUTPUT_DIR / "confidence_intervals.png"
plot_confidence_intervals(ci_df, metric="Accuracy", title="Bootstrap 95% CIs", save_path=ci_plot)
print(f"CI plot: {ci_plot}")
print()


# ── 8. run_benchmark with compute_stats=True ──────────────────────────────────

print("=== 8. run_benchmark with compute_stats=True ===")

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
df["target"] = y
csv_path = OUTPUT_DIR / "stats_data.csv"
df.to_csv(csv_path, index=False)

results_df2, preprocessor = run_benchmark(
    str(csv_path),
    "target",
    task="classification",
    models_dict={
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=20, random_state=42),
        "KNN":           KNeighborsClassifier(n_neighbors=5),
    },
    output_dir=str(OUTPUT_DIR / "stats_run"),
    compute_stats=True,
    verbose=False,
)

print("Results:")
print(results_df2[["Model", "Accuracy", "ROC AUC"]].to_string(index=False))

if "stats_summary" in preprocessor:
    ss = preprocessor["stats_summary"]
    print("\nBootstrap CIs:")
    for model_name, ci in ss["bootstrap_ci"].items():
        print(f"  {model_name:<20} {ci['metric']}  [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    if "mcnemar_pairs" in ss:
        print("\nMcNemar pairwise tests:")
        for pair_name, pair_res in ss["mcnemar_pairs"].items():
            print(f"  {pair_name:<40} p={pair_res['p_value']:.4f}")

print("\nAll outputs written to:", OUTPUT_DIR)
