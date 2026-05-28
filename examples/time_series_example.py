import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.benchmark import run_benchmark
from src.time_series import create_lag_features, create_rolling_features

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Generate synthetic time series --
np.random.seed(42)
n = 300
t = np.arange(n)
# Sine wave with trend and noise
y = np.sin(2 * np.pi * t / 50) + 0.02 * t + 0.3 * np.random.randn(n)

df = pd.DataFrame({"step": t.astype(float), "value": y})
csv_path = OUTPUT_DIR / "ts_data.csv"
df.to_csv(csv_path, index=False)

print("Time-series benchmark")
print(f"  n_samples : {len(df)}")
print(f"  target    : value")
print(f"  lags      : [1, 2, 3, 5]")
print(f"  ts_splits : 4,  horizon : 20")

# -- Run benchmark with lag features --
results_df, preprocessor = run_benchmark(
    str(csv_path),
    "value",
    task="time_series",
    lags=[1, 2, 3, 5],
    ts_n_splits=4,
    ts_horizon=20,
    output_dir=str(OUTPUT_DIR),
    export_formats=["csv"],
    verbose=True,
)

print("\nFinal results:")
print(results_df[["Model", "MAE", "RMSE", "MAPE", "SMAPE", "n_folds"]].to_string(index=False))

# -- Demonstrate feature utilities standalone --
print("\nLag features demo:")
lag_df = create_lag_features(df.copy(), [1, 2, 3], "value")
print(f"  Original rows : {len(df)}")
print(f"  After lag=3   : {len(lag_df)} rows, {lag_df.shape[1]} columns")
print(f"  New columns   : {[c for c in lag_df.columns if 'lag' in c]}")

print("\nRolling features demo:")
roll_df = create_rolling_features(df.copy(), [5, 10], "value")
print(f"  After roll=10 : {len(roll_df)} rows, {roll_df.shape[1]} columns")
print(f"  New columns   : {[c for c in roll_df.columns if 'rolling' in c]}")

print(f"\nOutputs written to: {OUTPUT_DIR.resolve()}")
