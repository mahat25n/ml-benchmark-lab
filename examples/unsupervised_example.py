"""
Unsupervised benchmark example.

Generates a synthetic blob dataset, runs the benchmark pipeline across
KMeans and DBSCAN, and exports clustering metrics + scatter plots to
examples/outputs/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from src.benchmark import run_benchmark

# ── Paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "outputs" / "data"
OUT_DIR  = Path(__file__).parent / "outputs" / "unsupervised"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Synthetic dataset (300 samples, 4 features, 3 natural clusters) ─
X, y = make_blobs(n_samples=300, n_features=4, centers=3,
                  cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
# "label" is stored in the CSV but not used by the unsupervised pipeline
df["label"] = y

csv_path = DATA_DIR / "unsupervised.csv"
df.to_csv(csv_path, index=False)
print(f"[data] Saved {len(df)} rows -> {csv_path}")

# ── Models: two clustering algorithms + PCA decomposition ─────────
models = {
    "KMeans":  KMeans(n_clusters=3, random_state=42, n_init=10),
    "DBSCAN":  DBSCAN(eps=1.5, min_samples=5),
    "PCA":     PCA(n_components=2, random_state=42),
}

# ── Benchmark ─────────────────────────────────────────────────────
results_df, preprocessor = run_benchmark(
    csv_path, "label",          # target_col is required but unused internally
    task="unsupervised",
    output_dir=str(OUT_DIR),
    models_dict=models,
    stratify=False,             # no stratification for unsupervised
    export_formats=["csv"],
    verbose=True,
)

# ── Clustering metric summary ─────────────────────────────────────
cluster_cols = [c for c in results_df.columns
                if c in ("Model", "Silhouette", "Davies-Bouldin",
                         "Calinski-Harabasz", "n_clusters", "n_noise",
                         "n_components", "cum_explained_variance_pct")]
print("\n[metrics] Clustering / PCA summary:")
print(results_df[cluster_cols].to_string(index=False))

print(f"\n[done] Outputs saved -> {OUT_DIR.resolve()}")
