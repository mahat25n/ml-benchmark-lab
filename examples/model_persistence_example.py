"""
Model Persistence Example
=========================
Demonstrates save_model(), load_model(), save_pipeline(), load_pipeline(),
predict_from_csv(), batch_predict(), and save_model_for_run().
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data import preprocess_data
from src.model_io import (
    batch_predict,
    load_model,
    load_pipeline,
    predict_from_csv,
    save_model,
    save_model_for_run,
    save_pipeline,
)

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "model_persistence")
os.makedirs(OUTPUT, exist_ok=True)

# ── 1. Generate and preprocess a synthetic dataset ───────────────────────────

print("=" * 60)
print("1. Generate dataset and train models")
print("=" * 60)

X_np, y = make_classification(
    n_samples=500, n_features=6, n_informative=4, random_state=42
)
feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

# Build a DataFrame so we can demonstrate preprocess_data() round-trip
df = pd.DataFrame(X_np, columns=feature_names)
df["target"] = y

# Save a CSV for batch inference demo
csv_path = os.path.join(OUTPUT, "inference_data.csv")
df.to_csv(csv_path, index=False)

# Preprocess (StandardScaler, no categoricals)
X, y_enc, preprocessor = preprocess_data(df, "target")
print(f"  Dataset shape  : {X.shape}")
print(f"  Feature names  : {preprocessor['feature_names']}")
print(f"  n_classes      : {preprocessor['n_classes']}")
print()

# Train two models
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y_enc)
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X, y_enc)
print(f"  RF accuracy  : {rf.score(X, y_enc):.3f}")
print(f"  LR accuracy  : {lr.score(X, y_enc):.3f}")
print()

# ── 2. save_model / load_model ────────────────────────────────────────────────

print("=" * 60)
print("2. save_model / load_model")
print("=" * 60)

model_dir = os.path.join(OUTPUT, "rf_model")
result = save_model(rf, model_dir, feature_names=feature_names)
print(f"  Model saved     : {result['model_path']}")
print(f"  Metadata saved  : {result['metadata_path']}")

loaded_rf = load_model(result["model_path"])
preds_orig   = rf.predict(X)
preds_loaded = loaded_rf.predict(X)
match = np.all(preds_orig == preds_loaded)
print(f"  Predictions match after reload : {match}")
print()

# ── 3. save_pipeline / load_pipeline ─────────────────────────────────────────

print("=" * 60)
print("3. save_pipeline / load_pipeline")
print("=" * 60)

pipeline_dir = os.path.join(OUTPUT, "lr_pipeline")
result = save_pipeline(lr, preprocessor, pipeline_dir)
print(f"  Pipeline saved  : {result['pipeline_path']}")
print(f"  Model saved     : {result['model_path']}")
print(f"  Metadata saved  : {result['metadata_path']}")

loaded_lr, loaded_prep = load_pipeline(pipeline_dir)
assert loaded_prep["feature_names"] == preprocessor["feature_names"]
assert loaded_prep["target_col"] == preprocessor["target_col"]
preds_pipe = loaded_lr.predict(X)
print(f"  Preprocessor feature_names preserved : True")
print(f"  Predictions match after reload       : {np.all(lr.predict(X) == preds_pipe)}")
print()

# ── 4. batch_predict ─────────────────────────────────────────────────────────

print("=" * 60)
print("4. batch_predict (chunk_size=100)")
print("=" * 60)

preds_direct  = rf.predict(X)
preds_batched = batch_predict(X, rf, chunk_size=100)
print(f"  Input rows       : {X.shape[0]}")
print(f"  Prediction shape : {preds_batched.shape}")
print(f"  All match direct : {np.all(preds_direct == preds_batched)}")

# Also from saved path
preds_from_path = batch_predict(X, result["model_path"])
print(f"  Predict from path works : {preds_from_path.shape == (X.shape[0],)}")
print()

# ── 5. predict_from_csv ───────────────────────────────────────────────────────

print("=" * 60)
print("5. predict_from_csv (pipeline directory)")
print("=" * 60)

out_csv = os.path.join(OUTPUT, "predictions.csv")
predictions = predict_from_csv(
    csv_path,
    pipeline_dir,
    target_col="target",
    output_path=out_csv,
)
print(f"  Input CSV        : {csv_path}")
print(f"  Predictions      : {len(predictions)} rows")
unique, counts = np.unique(predictions, return_counts=True)
for val, cnt in zip(unique, counts):
    print(f"    class {int(val)}: {cnt}  ({100 * cnt / len(predictions):.1f}%)")
print(f"  Output written   : {out_csv}")
print()

# ── 6. save_model_for_run ─────────────────────────────────────────────────────

print("=" * 60)
print("6. save_model_for_run (experiment tracker integration)")
print("=" * 60)

from src.experiment import ExperimentTracker

tracker = ExperimentTracker("model_persistence_demo", base_dir=OUTPUT)
tracker.log_config(task="classification", random_state=42)
tracker.log_dataset(n_samples=500, n_features=6, train_size=400, test_size=100)
tracker.log_models(["Random Forest", "Logistic Regression"])

# Save each model into the run directory
res_rf = save_model_for_run(
    rf, tracker.run_dir,
    name="random_forest",
    preprocessor=preprocessor,
)
res_lr = save_model_for_run(
    lr, tracker.run_dir,
    name="logistic_regression",
    feature_names=feature_names,
)

tracker.save()

print(f"  Run directory      : {tracker.run_dir}")
print(f"  RF pipeline saved  : {res_rf['pipeline_path']}")
print(f"  LR model saved     : {res_lr['model_path']}")
print()
print("Done.")
