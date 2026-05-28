"""
Regression benchmark example.

Generates a synthetic continuous-target dataset, runs the benchmark
pipeline across three regressors, and exports results + plots to
examples/outputs/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor

from src.benchmark import run_benchmark

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "outputs" / "data"
OUT_DIR  = Path(__file__).parent / "outputs" / "regression"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic dataset (300 samples, 6 features, continuous target) ─
X, y = make_regression(
    n_samples=300, n_features=6, n_informative=4,
    noise=10.0, random_state=42,
)
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])
df["target"] = y

csv_path = DATA_DIR / "regression.csv"
df.to_csv(csv_path, index=False)
print(f"[data] Saved {len(df)} rows -> {csv_path}")

# ── Lightweight models ────────────────────────────────────────────
models = {
    "Ridge":         Ridge(alpha=1.0),
    "Lasso":         Lasso(alpha=0.1, max_iter=2000),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
}

# ── Benchmark ─────────────────────────────────────────────────────
# stratify must be False for regression (continuous target)
results_df, preprocessor = run_benchmark(
    csv_path, "target",
    task="regression",
    output_dir=str(OUT_DIR),
    models_dict=models,
    stratify=False,
    export_formats=["csv", "excel"],
    verbose=True,
)

# ── Summary ───────────────────────────────────────────────────────
print("\n[metrics] MAE / RMSE / R2 summary:")
print(results_df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))

print(f"\n[done] Outputs saved -> {OUT_DIR.resolve()}")
