"""
Diebold-Mariano forecast comparison example.

Demonstrates:
  1. compute_forecast_error_series — per-step loss arrays
  2. run_diebold_mariano_test      — one-step and multi-step
  3. compare_forecast_models       — all pairwise and baseline mode
  4. run_benchmark with compute_stats=True (time_series task)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.stats import (
    compute_forecast_error_series,
    run_diebold_mariano_test,
    compare_forecast_models,
)
from src.benchmark import run_benchmark

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ── Simulate forecast errors ──────────────────────────────────────────────────

T = 120
y_true = np.sin(np.linspace(0, 8 * np.pi, T)) + np.random.normal(0, 0.2, T)

# Three "models" with varying accuracy
errors_rf  = y_true - (y_true + np.random.normal(0, 0.15, T))   # best
errors_lr  = y_true - (y_true + np.random.normal(0, 0.50, T))   # medium
errors_naive = y_true - np.zeros(T)                              # naive (predict 0)


# ── 1. compute_forecast_error_series ─────────────────────────────────────────

print("=== 1. Per-step forecast errors ===")

y_pred_rf  = y_true + np.random.normal(0, 0.15, T)
mae_series = compute_forecast_error_series(y_true, y_pred_rf, loss="mae")
mse_series = compute_forecast_error_series(y_true, y_pred_rf, loss="mse")

print(f"MAE series: mean={mae_series.mean():.4f}  max={mae_series.max():.4f}")
print(f"MSE series: mean={mse_series.mean():.4f}  max={mse_series.max():.4f}")
print(f"RMSE (from MSE): {np.sqrt(mse_series.mean()):.4f}")
print()


# ── 2. run_diebold_mariano_test — one-step ────────────────────────────────────

print("=== 2. Diebold-Mariano test (h=1, MAE) ===")

res = run_diebold_mariano_test(errors_rf, errors_lr, h=1, loss="mae")
print(f"RF vs LR:    t={res['statistic']:.4f}  p={res['p_value']:.4f}  "
      f"mean_diff={res['mean_diff']:.4f}  (positive means RF has higher loss)")
print(res["interpretation"])
print()

res2 = run_diebold_mariano_test(errors_rf, errors_naive, h=1, loss="mae")
print(f"RF vs Naive: t={res2['statistic']:.4f}  p={res2['p_value']:.4f}  "
      f"mean_diff={res2['mean_diff']:.4f}")
print(res2["interpretation"])
print()


# ── 3. run_diebold_mariano_test — multi-step ──────────────────────────────────

print("=== 3. Diebold-Mariano test (h=5, MSE) ===")

for h in (1, 3, 5):
    res_h = run_diebold_mariano_test(errors_rf, errors_lr, h=h, loss="mse")
    print(f"h={h}: t={res_h['statistic']:.4f}  p={res_h['p_value']:.4f}")
print()


# ── 4. compare_forecast_models — all pairs ───────────────────────────────────

print("=== 4. Pairwise DM comparison (all models, MAE) ===")

error_dict = {
    "Random Forest": errors_rf,
    "Linear Regr.":  errors_lr,
    "Naive (zero)":  errors_naive,
}

comparison_df = compare_forecast_models(error_dict, h=1, loss="mae")
print(comparison_df.to_string(index=False))
print()


# ── 5. compare_forecast_models — baseline mode ───────────────────────────────

print("=== 5. Baseline comparison (Random Forest as baseline, MSE) ===")

comparison_base = compare_forecast_models(
    error_dict, baseline="Random Forest", h=1, loss="mse"
)
print(comparison_base.to_string(index=False))
print()


# ── 6. run_benchmark with compute_stats=True (time_series) ───────────────────

print("=== 6. run_benchmark — time_series + compute_stats=True ===")

t_idx = np.arange(200)
signal = np.sin(t_idx * 0.15) + np.random.normal(0, 0.3, 200)
ts_df  = pd.DataFrame({"t": t_idx, "value": signal})
ts_csv = OUTPUT_DIR / "dm_ts_data.csv"
ts_df.to_csv(ts_csv, index=False)

results_df, preprocessor = run_benchmark(
    str(ts_csv),
    "value",
    task="time_series",
    models_dict={
        "Ridge":    Ridge(alpha=1.0),
        "Lasso":    Lasso(alpha=0.1, max_iter=5000),
        "Dec Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
    },
    output_dir=str(OUTPUT_DIR / "dm_benchmark"),
    lags=[1, 2, 3],
    ts_n_splits=4,
    compute_stats=True,
    verbose=False,
)

print("Results:")
print(results_df[["Model", "MAE", "RMSE"]].to_string(index=False))

ss = preprocessor.get("stats_summary", {})
if ss:
    for loss_key in ("dm_mae", "dm_mse"):
        records = ss.get(loss_key)
        if records:
            label = loss_key.replace("dm_", "").upper()
            print(f"\nDiebold-Mariano ({label}):")
            dm_df = pd.DataFrame(records)
            print(dm_df[["Model_A", "Model_B", "DM_Statistic",
                          "p_value", "Significant", "Favored"]].to_string(index=False))

print("\nAll outputs written to:", OUTPUT_DIR)
