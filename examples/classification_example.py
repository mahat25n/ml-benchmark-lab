"""
Classification benchmark example.

Generates a synthetic binary dataset, runs the benchmark pipeline across
three lightweight models, and exports results + plots to examples/outputs/.
"""

import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.benchmark import run_benchmark

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "outputs" / "data"
OUT_DIR  = Path(__file__).parent / "outputs" / "classification"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic dataset (300 samples, 6 features, binary target) ────
X, y = make_classification(
    n_samples=300, n_features=6, n_informative=4,
    n_redundant=1, random_state=42,
)
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
df["target"] = y

csv_path = DATA_DIR / "classification.csv"
df.to_csv(csv_path, index=False)
print(f"[data] Saved {len(df)} rows -> {csv_path}")

# ── Lightweight models for the example ───────────────────────────
models = {
    "Decision Tree":   DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":   RandomForestClassifier(n_estimators=50, max_depth=6,
                                              random_state=42, n_jobs=-1),
    "Gradient Boost":  GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                                  learning_rate=0.1, random_state=42),
}

# ── Benchmark ─────────────────────────────────────────────────────
results_df, preprocessor = run_benchmark(
    csv_path, "target",
    task="classification",
    output_dir=str(OUT_DIR),
    models_dict=models,
    compute_importance=True,    # store feature importance in preprocessor
    export_formats=["csv", "excel", "word"],
    verbose=True,
)

# ── Feature importance for the best model ────────────────────────
best_model = results_df.iloc[0]["Model"]
if "importance" in preprocessor and best_model in preprocessor["importance"]:
    imp_df = preprocessor["importance"][best_model]
    print(f"\n[importance] Top features for {best_model}:")
    print(imp_df.head(6).to_string(index=False))

print(f"\n[done] Outputs saved -> {OUT_DIR.resolve()}")
