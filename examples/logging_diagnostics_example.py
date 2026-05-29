"""
Logging and diagnostics example.

Demonstrates:
  1. get_logger       — file + console logger with selectable verbosity
  2. Timer            — context manager and standalone usage
  3. capture_warnings — redirect warnings.warn() to a logger
  4. run_diagnostics  — detect missing values, constant columns, duplicate
                        columns, class imbalance, leakage, and
                        train/test distribution mismatch
  5. run_benchmark integration — timing and diagnostics embedded in the
                        preprocessor dict returned by run_benchmark
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.logging_utils import Timer, capture_warnings, get_logger, run_diagnostics
from src.benchmark import run_benchmark

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)


# ── 1. get_logger ─────────────────────────────────────────────────────────────

print("=== 1. get_logger ===")

log_path = OUTPUT_DIR / "demo.log"
logger = get_logger(
    "demo",
    level="debug",
    log_file=log_path,
    console=True,
)

logger.debug("Debug message — only visible at debug level")
logger.info("Info message")
logger.warning("Warning message")
print(f"Log written to: {log_path}\n")


# ── 2. Timer — context manager ───────────────────────────────────────────────

print("=== 2. Timer (context manager) ===")

with Timer() as t:
    _ = [i ** 2 for i in range(100_000)]

print(f"List comprehension: {t.elapsed:.4f}s")

# Standalone usage
t2 = Timer()
t2.start()
_ = np.linalg.svd(rng.standard_normal((200, 200)))
t2.stop()
print(f"SVD (200×200):      {t2.elapsed:.4f}s\n")


# ── 3. capture_warnings ──────────────────────────────────────────────────────

print("=== 3. capture_warnings ===")

warn_logger = get_logger("warn_demo", level="warning", console=True)

with capture_warnings(warn_logger):
    warnings.warn("This UserWarning goes to warn_logger", UserWarning)
    warnings.warn("This DeprecationWarning goes to warn_logger", DeprecationWarning)

print("  Both warnings above were routed through warn_logger.\n")


# ── 4. run_diagnostics — clean data ──────────────────────────────────────────

print("=== 4. run_diagnostics — clean data ===")

n_tr, n_te = 200, 50
X_tr = rng.standard_normal((n_tr, 5))
X_te = rng.standard_normal((n_te, 5))
y_tr = rng.integers(0, 2, n_tr).astype(float)
y_te = rng.integers(0, 2, n_te).astype(float)

names = ["age", "income", "score", "days", "flag"]
diag = run_diagnostics(X_tr, X_te, y_tr, y_te, feature_names=names, task="classification")

print(f"has_issues:          {diag['has_issues']}")
print(f"issue_summary:       {diag['issue_summary']}")
print(f"missing (X_train):   {diag['missing']['X_train']}")
print(f"class imbalance:     {diag['class_imbalance']}")
print()


# ── 5. run_diagnostics — data with issues ────────────────────────────────────

print("=== 5. run_diagnostics — data with issues ===")

X_bad = rng.standard_normal((200, 5))
X_bad[:, 2] = 0.0                    # constant column  ("score")
X_bad[:, 4] = X_bad[:, 0]            # duplicate of "age"  ("flag")
X_bad[10:20, 1] = np.nan             # missing in "income"

# Perfect leakage: first column = target (correlation = 1.0)
y_bad = rng.standard_normal(200)
X_bad[:, 0] = y_bad + rng.standard_normal(200) * 0.001

# Train/test mean shift in feature index 3 ("days")
X_te_bad = rng.standard_normal((50, 5))
X_te_bad[:, 3] += 15.0               # large distribution shift

diag_bad = run_diagnostics(
    X_bad, X_te_bad,
    y_bad, rng.standard_normal(50),
    feature_names=names,
    task="regression",
)

print(f"has_issues:          {diag_bad['has_issues']}")
for issue in diag_bad["issue_summary"]:
    print(f"  * {issue}")
print(f"constant_cols:       {diag_bad['constant_cols']}")
print(f"duplicate_cols:      {diag_bad['duplicate_cols']}")
print(f"leakage_suspects:    {diag_bad['leakage_suspects']}")
print(f"train_test_mismatch: {diag_bad['train_test_mismatch']}")
print()


# ── 6. run_benchmark — timing and diagnostics ────────────────────────────────

print("=== 6. run_benchmark — timing and diagnostics ===")

n = 200
df = pd.DataFrame({
    "f1":     rng.standard_normal(n),
    "f2":     rng.standard_normal(n),
    "target": rng.integers(0, 2, n),
})
csv_path = OUTPUT_DIR / "log_demo_data.csv"
df.to_csv(csv_path, index=False)

results_df, prep = run_benchmark(
    str(csv_path),
    "target",
    models_dict={"LR": LogisticRegression(max_iter=500)},
    output_dir=str(OUTPUT_DIR / "log_demo_out"),
    log_level="info",
    verbose=True,
)

print("\nTiming:")
t = prep["timing"]
print(f"  total           : {t['total_seconds']:.3f}s")
print(f"  data loading    : {t['data_loading_seconds']:.3f}s")
print(f"  optimization    : {t['optimization_seconds']:.3f}s")
for m_name, secs in t["models"].items():
    print(f"  model '{m_name}'  : {secs:.3f}s")

print("\nDiagnostics:")
d = prep.get("diagnostics", {})
print(f"  has_issues      : {d.get('has_issues')}")
print(f"  issue_summary   : {d.get('issue_summary')}")

print(f"\nAll outputs written to: {OUTPUT_DIR}")
