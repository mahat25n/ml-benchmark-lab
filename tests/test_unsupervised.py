"""Smoke tests for the unsupervised learning pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ================================================================
# Blobs CSV fixture (module-scoped for speed)
# ================================================================

@pytest.fixture(scope="module")
def blobs_csv(tmp_path_factory):
    """150-row blobs CSV: 4 numeric features, 3 blob centres."""
    X, y = make_blobs(n_samples=150, n_features=4, centers=3, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["cluster"] = y  # present as target_col but ignored by unsupervised pipeline
    p = tmp_path_factory.mktemp("blobs") / "blobs.csv"
    df.to_csv(p, index=False)
    return str(p)


# ================================================================
# TestClusteringMetrics
# ================================================================

class TestClusteringMetrics:

    def test_returns_required_keys(self):
        from src.evaluation import compute_clustering_metrics
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 4))
        labels = np.repeat([0, 1, 2], 20)
        result = compute_clustering_metrics(X, labels)
        for key in ("Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "n_clusters", "n_noise"):
            assert key in result

    def test_n_clusters_correct(self):
        from src.evaluation import compute_clustering_metrics
        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4))
        labels = np.repeat([0, 1, 2], 20)
        result = compute_clustering_metrics(X, labels)
        assert result["n_clusters"] == 3

    def test_n_noise_zero_when_no_noise(self):
        from src.evaluation import compute_clustering_metrics
        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 4))
        labels = np.repeat([0, 1], 20)
        result = compute_clustering_metrics(X, labels)
        assert result["n_noise"] == 0

    def test_noise_points_excluded(self):
        from src.evaluation import compute_clustering_metrics
        rng = np.random.default_rng(3)
        X = rng.standard_normal((62, 4))
        labels = np.array([-1, -1] + [0] * 30 + [1] * 30)
        result = compute_clustering_metrics(X, labels)
        assert result["n_noise"] == 2
        assert result["n_clusters"] == 2
        assert not np.isnan(result["Silhouette"])

    def test_single_cluster_returns_nan_metrics(self):
        from src.evaluation import compute_clustering_metrics
        rng = np.random.default_rng(4)
        X = rng.standard_normal((30, 4))
        labels = np.zeros(30, dtype=int)
        result = compute_clustering_metrics(X, labels)
        assert np.isnan(result["Silhouette"])
        assert np.isnan(result["Davies-Bouldin"])
        assert np.isnan(result["Calinski-Harabasz"])

    def test_silhouette_in_valid_range(self):
        from src.evaluation import compute_clustering_metrics
        X, labels = make_blobs(n_samples=90, n_features=4, centers=3, random_state=42)
        result = compute_clustering_metrics(X, labels)
        assert -1.0 <= result["Silhouette"] <= 1.0

    def test_values_are_floats(self):
        from src.evaluation import compute_clustering_metrics
        X, labels = make_blobs(n_samples=60, n_features=3, centers=2, random_state=0)
        result = compute_clustering_metrics(X, labels)
        for k in ("Silhouette", "Davies-Bouldin", "Calinski-Harabasz"):
            assert isinstance(result[k], float)


# ================================================================
# TestPCAMetrics
# ================================================================

class TestPCAMetrics:

    def test_returns_required_keys(self):
        from src.evaluation import compute_pca_metrics
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 6))
        pca = PCA(n_components=3, random_state=42).fit(X)
        result = compute_pca_metrics(pca)
        for key in ("n_components", "cum_explained_variance_pct"):
            assert key in result

    def test_n_components_correct(self):
        from src.evaluation import compute_pca_metrics
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 6))
        pca = PCA(n_components=3, random_state=42).fit(X)
        result = compute_pca_metrics(pca)
        assert result["n_components"] == 3

    def test_cum_variance_in_valid_range(self):
        from src.evaluation import compute_pca_metrics
        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 6))
        pca = PCA(n_components=4, random_state=42).fit(X)
        result = compute_pca_metrics(pca)
        assert 0.0 < result["cum_explained_variance_pct"] <= 100.0

    def test_full_components_cumvar_near_100(self):
        from src.evaluation import compute_pca_metrics
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 4))
        pca = PCA(n_components=4, random_state=42).fit(X)
        result = compute_pca_metrics(pca)
        assert result["cum_explained_variance_pct"] == pytest.approx(100.0, abs=1e-3)


# ================================================================
# TestUnsupervisedModels
# ================================================================

class TestUnsupervisedModels:

    def test_unsupervised_models_all_registered(self):
        from src.models import UNSUPERVISED_MODELS
        expected = {"KMeans", "Agglomerative", "DBSCAN", "Gaussian Mixture", "PCA"}
        assert expected.issubset(set(UNSUPERVISED_MODELS.keys()))

    def test_get_models_unsupervised_returns_five(self):
        from src.models import get_models
        models = get_models("unsupervised")
        assert len(models) == 5

    def test_clustering_models_have_fit_predict(self):
        from src.models import UNSUPERVISED_MODELS
        for name in ("KMeans", "Agglomerative", "DBSCAN", "Gaussian Mixture"):
            assert hasattr(UNSUPERVISED_MODELS[name], "fit_predict"), \
                f"{name} should have fit_predict"

    def test_pca_does_not_have_fit_predict(self):
        from src.models import UNSUPERVISED_MODELS
        assert not hasattr(UNSUPERVISED_MODELS["PCA"], "fit_predict")

    def test_get_models_classification_unchanged(self):
        from src.models import get_models
        assert len(get_models("classification")) == 8

    def test_get_models_regression_unchanged(self):
        from src.models import get_models
        assert len(get_models("regression")) == 9


# ================================================================
# TestUnsupervisedPlots
# ================================================================

class TestUnsupervisedPlots:

    @pytest.fixture
    def blob_arrays(self):
        X, labels = make_blobs(n_samples=90, n_features=4, centers=3, random_state=42)
        return X, labels

    def test_plot_cluster_scatter_creates_png(self, tmp_path, blob_arrays):
        from src.plots import plot_cluster_scatter
        X, labels = blob_arrays
        p = tmp_path / "cs.png"
        plot_cluster_scatter(X, labels, "KMeans", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_cluster_scatter_two_feature_input(self, tmp_path):
        from src.plots import plot_cluster_scatter
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 2))
        labels = np.repeat([0, 1], 20)
        p = tmp_path / "cs2.png"
        plot_cluster_scatter(X, labels, "Test", p, feature_names=["A", "B"])
        assert p.exists() and p.stat().st_size > 0

    def test_plot_cluster_scatter_with_noise(self, tmp_path):
        from src.plots import plot_cluster_scatter
        rng = np.random.default_rng(1)
        X = rng.standard_normal((62, 4))
        labels = np.array([-1, -1] + [0] * 30 + [1] * 30)
        p = tmp_path / "cs_noise.png"
        plot_cluster_scatter(X, labels, "DBSCAN", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_pca_variance_creates_png(self, tmp_path):
        from src.plots import plot_pca_variance
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 6))
        pca = PCA(n_components=4, random_state=42).fit(X)
        p = tmp_path / "pv.png"
        plot_pca_variance(pca, "PCA", p)
        assert p.exists() and p.stat().st_size > 0


# ================================================================
# TestUnsupervisedBenchmark  (integration)
# ================================================================

@pytest.mark.integration
class TestUnsupervisedBenchmark:

    def test_returns_tuple_of_two(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        result = run_benchmark(
            blobs_csv, "cluster",
            task="unsupervised",
            output_dir=str(tmp_path),
            models_dict={"KMeans": KMeans(n_clusters=3, random_state=42, n_init=10)},
            stratify=False,
            verbose=False,
        )
        assert len(result) == 2

    def test_results_has_clustering_columns(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            blobs_csv, "cluster",
            task="unsupervised",
            output_dir=str(tmp_path),
            models_dict={"KMeans": KMeans(n_clusters=3, random_state=42, n_init=10)},
            stratify=False,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)
        for col in ("Model", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"):
            assert col in results_df.columns

    def test_cluster_scatter_png_created(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            blobs_csv, "cluster",
            task="unsupervised",
            output_dir=str(tmp_path),
            models_dict={"KMeans": KMeans(n_clusters=3, random_state=42, n_init=10)},
            stratify=False,
            verbose=False,
        )
        assert len(list(tmp_path.glob("cluster_*.png"))) >= 1

    def test_pca_run_creates_variance_png(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.decomposition import PCA as _PCA
        run_benchmark(
            blobs_csv, "cluster",
            task="unsupervised",
            output_dir=str(tmp_path),
            models_dict={"PCA": _PCA(n_components=2, random_state=42)},
            stratify=False,
            verbose=False,
        )
        assert len(list(tmp_path.glob("pca_*.png"))) >= 1

    def test_stratify_true_auto_corrected(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        with pytest.warns(UserWarning, match="stratify"):
            run_benchmark(
                blobs_csv, "cluster",
                task="unsupervised",
                output_dir=str(tmp_path),
                models_dict={"KMeans": KMeans(n_clusters=3, random_state=42, n_init=10)},
                stratify=True,
                verbose=False,
            )

    def test_export_csv_contains_clustering_metrics(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            blobs_csv, "cluster",
            task="unsupervised",
            output_dir=str(tmp_path),
            models_dict={"KMeans": KMeans(n_clusters=3, random_state=42, n_init=10)},
            stratify=False,
            export_formats=["csv"],
            verbose=False,
        )
        df = pd.read_csv(tmp_path / "results.csv")
        assert "Silhouette" in df.columns

    def test_preprocessor_keys_present(self, blobs_csv, tmp_path):
        from src.benchmark import run_benchmark
        _, pp = run_benchmark(
            blobs_csv, "cluster",
            task="unsupervised",
            output_dir=str(tmp_path),
            models_dict={"KMeans": KMeans(n_clusters=3, random_state=42, n_init=10)},
            stratify=False,
            verbose=False,
        )
        for key in ("scaler", "feature_names", "feature_encoders"):
            assert key in pp
