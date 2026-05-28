"""
Explainability example — SHAP + permutation importance.

Trains a RandomForest on synthetic classification data, then demonstrates:
  - Global SHAP importance (summarise_shap_importance)
  - Local SHAP explanation for one prediction (explain_prediction)
  - SHAP summary, bar, and dependence plots
  - Permutation importance as a model-agnostic alternative
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.explainability import (
    compute_shap_values,
    compute_permutation_importance,
    explain_prediction,
    summarise_shap_importance,
)
from src.plots import (
    plot_shap_bar,
    plot_shap_dependence,
    plot_shap_summary,
)

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "outputs" / "data"
OUT_DIR  = Path(__file__).parent / "outputs" / "explainability"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic dataset ─────────────────────────────────────────────
X_raw, y = make_classification(
    n_samples=300, n_features=8, n_informative=5,
    n_redundant=2, random_state=42,
)
feature_names = [f"feat_{i}" for i in range(8)]

df = pd.DataFrame(X_raw, columns=feature_names)
df["target"] = y
csv_path = DATA_DIR / "explainability.csv"
df.to_csv(csv_path, index=False)
print(f"[data] Saved {len(df)} rows -> {csv_path}")

# ── Train / test split + model ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, random_state=42, stratify=y,
)
model = RandomForestClassifier(n_estimators=100, max_depth=6,
                               random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(f"[model] RandomForest trained - test accuracy: "
      f"{(model.predict(X_test) == y_test).mean():.3f}")

# ── SHAP values (TreeExplainer auto-selected) ─────────────────────
print("\n[shap] Computing SHAP values ...")
shap_values, explainer = compute_shap_values(
    model, X_test, feature_names=feature_names,
)
print(f"  shap_values shape: {shap_values.shape}")

# ── Global importance table ───────────────────────────────────────
importance_df = summarise_shap_importance(shap_values, feature_names)
print("\n[shap] Global feature importance (top 8):")
print(importance_df[["rank", "feature", "mean_abs_shap", "mean_shap"]]
      .to_string(index=False))

# ── Local explanation — single prediction ─────────────────────────
instance = X_test[0]
local_df = explain_prediction(
    model, instance, feature_names=feature_names,
    X_background=X_train[:50],
)
pred_class = model.predict([instance])[0]
pred_prob  = model.predict_proba([instance])[0, 1]
print(f"\n[local] Instance 0 - predicted class {pred_class} "
      f"(P(class=1) = {pred_prob:.3f})")
print(local_df[["rank", "feature", "feature_value", "shap_value"]]
      .to_string(index=False))

# ── SHAP plots ────────────────────────────────────────────────────
print("\n[plots] Saving SHAP plots ...")

# Beeswarm summary — shows direction, magnitude, feature-value correlation
plot_shap_summary(
    shap_values, X_test, feature_names,
    "Random Forest", OUT_DIR / "shap_summary.png",
)
print("  shap_summary.png")

# Bar chart — ranked global importance
plot_shap_bar(
    importance_df, "Random Forest",
    OUT_DIR / "shap_bar.png",
)
print("  shap_bar.png")

# Dependence plot for the top feature, coloured by the second feature
top_feature = importance_df.iloc[0]["feature"]
second_feature = importance_df.iloc[1]["feature"]
plot_shap_dependence(
    shap_values, X_test, feature_names,
    top_feature, "Random Forest",
    OUT_DIR / "shap_dependence.png",
    interaction_feature=second_feature,
)
print(f"  shap_dependence.png ({top_feature} vs {second_feature})")

# ── Permutation importance (model-agnostic cross-check) ───────────
print("\n[permutation] Computing permutation importance ...")
perm_df = compute_permutation_importance(
    model, X_test, y_test, feature_names,
    n_repeats=10, random_state=42,
)
print(perm_df[["rank", "feature", "importance_mean", "importance_sem"]]
      .head(8).to_string(index=False))

print(f"\n[done] All outputs saved -> {OUT_DIR.resolve()}")
