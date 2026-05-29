"""
Dataset Profiling Example
=========================
Demonstrates profile_dataset(), summarize_columns(), summarize_target(),
summarize_missingness(), and the three profile plots.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_profile import (
    profile_dataset,
    summarize_columns,
    summarize_target,
    summarize_missingness,
)

# ── 1. Build a synthetic dataset ──────────────────────────────────────────────

rng = np.random.default_rng(42)
n = 500

df = pd.DataFrame({
    "age":         rng.integers(18, 80, n).astype(float),
    "income":      rng.exponential(50_000, n),
    "score":       rng.normal(100, 15, n),
    "category":    rng.choice(["A", "B", "C", "D"], n),
    "region":      rng.choice(["North", "South", "East", "West"], n),
    "flag":        rng.choice([True, False], n),
    "label":       rng.choice([0, 1], n, p=[0.7, 0.3]),
})

# Introduce some missing values
df.loc[rng.choice(n, 40, replace=False), "age"]      = np.nan
df.loc[rng.choice(n, 80, replace=False), "income"]   = np.nan
df.loc[rng.choice(n, 5,  replace=False), "category"] = np.nan

print("Dataset shape:", df.shape)
print()

# ── 2. Summarize columns ──────────────────────────────────────────────────────

print("=" * 60)
print("Column Summary")
print("=" * 60)
col_summary = summarize_columns(df)
print(col_summary[["column", "kind", "n_missing", "pct_missing", "mean", "skewness"]].to_string(index=False))
print()

# ── 3. Summarize target ───────────────────────────────────────────────────────

print("=" * 60)
print("Target Summary  (label)")
print("=" * 60)
target_info = summarize_target(df, "label")
print(f"  kind             : {target_info['kind']}")
print(f"  n_unique         : {target_info['n_unique']}")
print(f"  imbalance_ratio  : {target_info['imbalance_ratio']:.3f}")
print(f"  class_distribution:")
for cls, pct in target_info["class_distribution"].items():
    print(f"    {cls}: {pct:.1f}%")
print()

# ── 4. Summarize missingness ──────────────────────────────────────────────────

print("=" * 60)
print("Missingness Summary")
print("=" * 60)
miss = summarize_missingness(df)
print(miss[miss["n_missing"] > 0][["column", "n_missing", "pct_missing", "pattern"]].to_string(index=False))
print()

# ── 5. Full profile (no exports) ──────────────────────────────────────────────

print("=" * 60)
print("Full Profile (top-level stats)")
print("=" * 60)
profile = profile_dataset(df, target_col="label")

print(f"  Rows              : {profile['n_rows']}")
print(f"  Columns           : {profile['n_cols']}")
print(f"  Numeric cols      : {profile['n_numeric']}")
print(f"  Categorical cols  : {profile['n_categorical']}")
print(f"  Boolean cols      : {profile['n_boolean']}")
print(f"  Datetime cols     : {profile['n_datetime']}")
print(f"  Duplicate rows    : {profile['n_duplicate_rows']} ({profile['pct_duplicate_rows']:.1f}%)")
print(f"  Completeness      : {profile['dataset_completeness_pct']:.1f}%")
print(f"  Memory usage      : {profile['memory_usage_mb']:.3f} MB")
print(f"  Target kind       : {profile['target']['kind']}")
print(f"  Imbalance ratio   : {profile['target']['imbalance_ratio']:.3f}")
print()

# ── 6. Profile with export + plots ────────────────────────────────────────────

output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "profile_example")
plots_dir  = os.path.join(output_dir, "plots")

print("=" * 60)
print(f"Exporting profile to: {output_dir}")
print("=" * 60)

profile2 = profile_dataset(
    df,
    target_col="label",
    output_dir=output_dir,
    plots_dir=plots_dir,
)

exported = []
for fname in ["profile_summary.json", "profile_report.csv"]:
    path = os.path.join(output_dir, fname)
    if os.path.exists(path):
        exported.append(fname)
        print(f"  Created: {fname}")

for fname in ["missing_heatmap.png", "class_distribution.png", "numeric_distributions.png"]:
    path = os.path.join(plots_dir, fname)
    if os.path.exists(path):
        print(f"  Created: plots/{fname}")

print()
print("Done.")
