"""
Feature Selection Example
=========================
Demonstrates all five selection methods plus benchmark integration.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import make_classification

from src.feature_selection import (
    correlation_selection,
    lasso_selection,
    mutual_information_selection,
    rfecv_selection,
    run_feature_selection,
    variance_threshold_selection,
)
from src.plots import plot_feature_importance_ranking, plot_selected_features_summary

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "feature_selection")
os.makedirs(OUTPUT, exist_ok=True)

# ── 1. Build dataset ──────────────────────────────────────────────────────────

print("=" * 60)
print("1. Dataset")
print("=" * 60)

X, y = make_classification(
    n_samples=500,
    n_features=15,
    n_informative=6,
    n_redundant=4,
    n_repeated=0,
    random_state=42,
)

# Add a constant column (zero variance) and a near-duplicate
X[:, 10] = 0.0                                            # constant
X[:, 11] = X[:, 0] + np.random.default_rng(7).standard_normal(500) * 0.001  # near-dup

feature_names = [f"feat_{i:02d}" for i in range(X.shape[1])]
print(f"  Shape          : {X.shape}")
print(f"  n_informative  : 6  (feat_00 … feat_05)")
print(f"  constant col   : feat_10")
print(f"  near-duplicate : feat_11 ≈ feat_00")
print()


def _show(label, result):
    print(f"  {label:<28}  n_after={result['n_after']:>2}  "
          f"removed={result['n_removed']:>2}  "
          f"kept: {result['selected_features'][:4]}{'...' if result['n_after'] > 4 else ''}")


# ── 2. Variance threshold ─────────────────────────────────────────────────────

print("=" * 60)
print("2. Variance Threshold  (removes constant columns)")
print("=" * 60)
vt = variance_threshold_selection(X, feature_names, threshold=0.01)
_show("variance (t=0.01)", vt)
plot_feature_importance_ranking(
    vt["scores"], "Variance Scores", os.path.join(OUTPUT, "variance_scores.png")
)
plot_selected_features_summary(
    vt, "Variance Selection", os.path.join(OUTPUT, "variance_summary.png")
)
print()

# ── 3. Correlation selection ──────────────────────────────────────────────────

print("=" * 60)
print("3. Correlation Selection  (removes near-duplicates)")
print("=" * 60)
cs = correlation_selection(X, feature_names, threshold=0.95)
_show("correlation (t=0.95)", cs)
plot_feature_importance_ranking(
    cs["scores"], "Mean Abs Correlation", os.path.join(OUTPUT, "correlation_scores.png")
)
print()

# ── 4. Mutual information ─────────────────────────────────────────────────────

print("=" * 60)
print("4. Mutual Information  (top-8 features)")
print("=" * 60)
mi = mutual_information_selection(X, y, feature_names, n_features=8,
                                  task="classification")
_show("mutual_info (k=8)", mi)
plot_feature_importance_ranking(
    mi["scores"], "Mutual Information Scores",
    os.path.join(OUTPUT, "mi_scores.png"),
)
print()

# ── 5. RFECV ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("5. RFECV  (auto-selects optimal n_features by CV)")
print("=" * 60)
rfe = rfecv_selection(X, y, feature_names, task="classification", cv=5)
_show("rfecv", rfe)
print()

# ── 6. Lasso (L1) ────────────────────────────────────────────────────────────

print("=" * 60)
print("6. Lasso / L1  (sparse coefficient selection)")
print("=" * 60)
ls = lasso_selection(X, y, feature_names, alpha=0.1, task="classification")
_show("lasso (alpha=0.1)", ls)
plot_feature_importance_ranking(
    ls["scores"], "Lasso |Coef|", os.path.join(OUTPUT, "lasso_scores.png"),
)
print()

# ── 7. Unified dispatcher ─────────────────────────────────────────────────────

print("=" * 60)
print("7. run_feature_selection  (dispatcher)")
print("=" * 60)
for method in ("variance", "correlation", "mutual_information", "lasso"):
    kw = {"n_features": 8} if method == "mutual_information" else {}
    r = run_feature_selection(X, y, feature_names, method=method,
                              task="classification", **kw)
    _show(method, r)
print()

# ── 8. Benchmark integration ──────────────────────────────────────────────────

print("=" * 60)
print("8. run_benchmark with feature_selection='mutual_information'")
print("=" * 60)

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
csv_path = os.path.join(OUTPUT, "data.csv")
df.to_csv(csv_path, index=False)

from src.benchmark import run_benchmark

results_df, preprocessor = run_benchmark(
    csv_path, "target",
    output_dir=os.path.join(OUTPUT, "benchmark_run"),
    task="classification",
    feature_selection="mutual_information",
    n_features=6,
    verbose=False,
)

fs_info = preprocessor.get("feature_selection", {})
print(f"  Features before : {fs_info.get('n_before')}")
print(f"  Features after  : {fs_info.get('n_after')}")
print(f"  Removed         : {fs_info.get('removed_features')}")
print()
print(results_df[["Model", "Accuracy", "F1 Score"]].to_string(index=False))
print()
print("Done.")
