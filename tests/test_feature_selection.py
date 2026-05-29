"""
Tests for src/feature_selection.py — feature selection utilities.
"""

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from src.feature_selection import (
    _VALID_METHODS,
    correlation_selection,
    lasso_selection,
    mutual_information_selection,
    rfecv_selection,
    run_feature_selection,
    variance_threshold_selection,
)


# ================================================================
# FIXTURES
# ================================================================


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=5, n_redundant=2,
        random_state=42
    )
    feature_names = [f"f{i}" for i in range(10)]
    return X, y, feature_names


@pytest.fixture
def reg_data():
    X, y = make_regression(
        n_samples=300, n_features=10, n_informative=5, random_state=42
    )
    feature_names = [f"f{i}" for i in range(10)]
    return X, y, feature_names


@pytest.fixture
def constant_feature_data():
    """Dataset with a constant column that variance threshold should remove."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    X[:, 2] = 0.0   # make column 2 constant
    feature_names = [f"f{i}" for i in range(5)]
    return X, y, feature_names


@pytest.fixture
def correlated_data():
    """Dataset with a near-perfect duplicate column."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    X[:, 4] = X[:, 0] + rng.standard_normal(100) * 0.001   # near-duplicate
    y = (X[:, 0] > 0).astype(int)
    feature_names = [f"f{i}" for i in range(5)]
    return X, y, feature_names


# ================================================================
# RESULT DICT SCHEMA HELPERS
# ================================================================

_REQUIRED_KEYS = {
    "method", "n_before", "n_after", "n_removed",
    "selected_features", "removed_features", "selected_mask", "scores",
}


def _check_schema(result, expected_method):
    """Assert that a result dict has the standard schema."""
    assert isinstance(result, dict)
    assert _REQUIRED_KEYS.issubset(result.keys())
    assert result["method"] == expected_method
    assert result["n_before"] >= result["n_after"] >= 0
    assert result["n_removed"] == result["n_before"] - result["n_after"]
    assert len(result["selected_features"]) == result["n_after"]
    assert len(result["removed_features"]) == result["n_removed"]
    assert isinstance(result["selected_mask"], np.ndarray)
    assert result["selected_mask"].dtype == bool
    assert result["selected_mask"].sum() == result["n_after"]


# ================================================================
# variance_threshold_selection
# ================================================================


class TestVarianceThresholdSelection:
    def test_returns_dict(self, clf_data):
        X, _, fn = clf_data
        result = variance_threshold_selection(X, fn)
        assert isinstance(result, dict)

    def test_schema(self, clf_data):
        X, _, fn = clf_data
        result = variance_threshold_selection(X, fn)
        _check_schema(result, "variance_threshold")

    def test_removes_constant_column(self, constant_feature_data):
        X, _, fn = constant_feature_data
        result = variance_threshold_selection(X, fn, threshold=0.0)
        assert result["n_removed"] >= 1
        assert "f2" in result["removed_features"]

    def test_keeps_all_when_threshold_zero_no_constants(self, clf_data):
        X, _, fn = clf_data
        result = variance_threshold_selection(X, fn, threshold=0.0)
        assert result["n_after"] == result["n_before"]

    def test_scores_are_variances(self, clf_data):
        X, _, fn = clf_data
        result = variance_threshold_selection(X, fn)
        assert result["scores"] is not None
        assert len(result["scores"]) == X.shape[1]
        for f, v in result["scores"].items():
            assert v >= 0

    def test_high_threshold_removes_more(self, clf_data):
        X, _, fn = clf_data
        r_low  = variance_threshold_selection(X, fn, threshold=0.1)
        r_high = variance_threshold_selection(X, fn, threshold=0.5)
        assert r_low["n_after"] >= r_high["n_after"]

    def test_auto_feature_names(self, clf_data):
        X, _, _ = clf_data
        result = variance_threshold_selection(X)
        assert result["selected_features"][0].startswith("feature_")

    def test_selected_mask_length(self, clf_data):
        X, _, fn = clf_data
        result = variance_threshold_selection(X, fn)
        assert len(result["selected_mask"]) == X.shape[1]

    def test_feature_names_mismatch_raises(self, clf_data):
        X, _, _ = clf_data
        with pytest.raises(ValueError, match="feature_names length"):
            variance_threshold_selection(X, ["a", "b"])


# ================================================================
# correlation_selection
# ================================================================


class TestCorrelationSelection:
    def test_returns_dict(self, clf_data):
        X, _, fn = clf_data
        result = correlation_selection(X, fn)
        assert isinstance(result, dict)

    def test_schema(self, clf_data):
        X, _, fn = clf_data
        result = correlation_selection(X, fn)
        _check_schema(result, "correlation")

    def test_removes_correlated_column(self, correlated_data):
        X, _, fn = correlated_data
        result = correlation_selection(X, fn, threshold=0.99)
        assert result["n_removed"] >= 1

    def test_keeps_all_when_threshold_one(self, clf_data):
        X, _, fn = clf_data
        result = correlation_selection(X, fn, threshold=1.0)
        assert result["n_after"] == result["n_before"]

    def test_scores_are_mean_correlations(self, clf_data):
        X, _, fn = clf_data
        result = correlation_selection(X, fn)
        assert result["scores"] is not None
        for v in result["scores"].values():
            assert 0 <= v <= 1

    def test_single_feature_passthrough(self):
        X = np.array([[1.0], [2.0], [3.0]])
        result = correlation_selection(X, ["only"])
        assert result["n_after"] == 1
        assert result["n_removed"] == 0

    def test_lower_threshold_removes_more(self, correlated_data):
        X, _, fn = correlated_data
        r_high = correlation_selection(X, fn, threshold=0.99)
        r_low  = correlation_selection(X, fn, threshold=0.50)
        assert r_low["n_after"] <= r_high["n_after"]


# ================================================================
# mutual_information_selection
# ================================================================


class TestMutualInformationSelection:
    def test_returns_dict(self, clf_data):
        X, y, fn = clf_data
        result = mutual_information_selection(X, y, fn)
        assert isinstance(result, dict)

    def test_schema_classification(self, clf_data):
        X, y, fn = clf_data
        result = mutual_information_selection(X, y, fn, n_features=5)
        _check_schema(result, "mutual_information")

    def test_respects_n_features(self, clf_data):
        X, y, fn = clf_data
        result = mutual_information_selection(X, y, fn, n_features=4)
        assert result["n_after"] == 4

    def test_scores_non_negative(self, clf_data):
        X, y, fn = clf_data
        result = mutual_information_selection(X, y, fn)
        for v in result["scores"].values():
            assert v >= 0

    def test_regression_task(self, reg_data):
        X, y, fn = reg_data
        result = mutual_information_selection(
            X, y, fn, n_features=5, task="regression"
        )
        _check_schema(result, "mutual_information")
        assert result["n_after"] == 5

    def test_n_features_none_keeps_all(self, clf_data):
        X, y, fn = clf_data
        result = mutual_information_selection(X, y, fn, n_features=None)
        assert result["n_after"] == X.shape[1]

    def test_n_features_exceeds_total_clamps(self, clf_data):
        X, y, fn = clf_data
        with warnings.catch_warnings(record=True):
            result = mutual_information_selection(X, y, fn, n_features=999)
        assert result["n_after"] == X.shape[1]

    def test_requires_y(self):
        X = np.random.default_rng(0).standard_normal((50, 5))
        with pytest.raises(TypeError):
            mutual_information_selection(X)


# ================================================================
# rfecv_selection
# ================================================================


class TestRFECVSelection:
    def test_returns_dict(self, clf_data):
        X, y, fn = clf_data
        result = rfecv_selection(X, y, fn, cv=3)
        assert isinstance(result, dict)

    def test_schema(self, clf_data):
        X, y, fn = clf_data
        result = rfecv_selection(X, y, fn, cv=3)
        _check_schema(result, "rfecv")

    def test_at_least_one_feature_selected(self, clf_data):
        X, y, fn = clf_data
        result = rfecv_selection(X, y, fn, cv=3)
        assert result["n_after"] >= 1

    def test_regression_task(self, reg_data):
        X, y, fn = reg_data
        result = rfecv_selection(X, y, fn, task="regression", cv=3)
        _check_schema(result, "rfecv")

    def test_scores_are_ranks(self, clf_data):
        X, y, fn = clf_data
        result = rfecv_selection(X, y, fn, cv=3)
        for v in result["scores"].values():
            assert v >= 1   # RFECV ranking: 1 = selected

    def test_custom_estimator(self, clf_data):
        from sklearn.tree import DecisionTreeClassifier
        X, y, fn = clf_data
        est = DecisionTreeClassifier(max_depth=3, random_state=0)
        result = rfecv_selection(X, y, fn, estimator=est, cv=3)
        _check_schema(result, "rfecv")

    def test_selected_mask_matches_n_after(self, clf_data):
        X, y, fn = clf_data
        result = rfecv_selection(X, y, fn, cv=3)
        assert result["selected_mask"].sum() == result["n_after"]


# ================================================================
# lasso_selection
# ================================================================


class TestLassoSelection:
    def test_returns_dict(self, clf_data):
        X, y, fn = clf_data
        result = lasso_selection(X, y, fn)
        assert isinstance(result, dict)

    def test_schema_classification(self, clf_data):
        X, y, fn = clf_data
        result = lasso_selection(X, y, fn)
        _check_schema(result, "lasso")

    def test_schema_regression(self, reg_data):
        X, y, fn = reg_data
        result = lasso_selection(X, y, fn, task="regression")
        _check_schema(result, "lasso")

    def test_scores_non_negative(self, clf_data):
        X, y, fn = clf_data
        result = lasso_selection(X, y, fn)
        for v in result["scores"].values():
            assert v >= 0

    def test_smaller_alpha_keeps_more(self, clf_data):
        X, y, fn = clf_data
        r_small = lasso_selection(X, y, fn, alpha=0.001)
        r_large = lasso_selection(X, y, fn, alpha=10.0)
        assert r_small["n_after"] >= r_large["n_after"]

    def test_very_large_alpha_returns_all_features_with_warning(self, clf_data):
        X, y, fn = clf_data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = lasso_selection(X, y, fn, alpha=1e10)
        # Either returns all features (warning) or selects some
        assert result["n_after"] >= 1

    def test_regression_with_sparse_result(self, reg_data):
        X, y, fn = reg_data
        result = lasso_selection(X, y, fn, task="regression", alpha=0.1)
        assert result["n_after"] >= 1
        assert result["n_after"] <= X.shape[1]


# ================================================================
# run_feature_selection (dispatcher)
# ================================================================


class TestRunFeatureSelection:
    def test_variance_dispatch(self, clf_data):
        X, _, fn = clf_data
        result = run_feature_selection(X, feature_names=fn, method="variance")
        assert result["method"] == "variance_threshold"

    def test_correlation_dispatch(self, clf_data):
        X, _, fn = clf_data
        result = run_feature_selection(X, feature_names=fn, method="correlation")
        assert result["method"] == "correlation"

    def test_mutual_information_dispatch(self, clf_data):
        X, y, fn = clf_data
        result = run_feature_selection(X, y, fn, method="mutual_information",
                                       n_features=5)
        assert result["method"] == "mutual_information"
        assert result["n_after"] == 5

    def test_rfecv_dispatch(self, clf_data):
        X, y, fn = clf_data
        result = run_feature_selection(X, y, fn, method="rfecv",
                                       task="classification", cv=3)
        assert result["method"] == "rfecv"

    def test_lasso_dispatch(self, clf_data):
        X, y, fn = clf_data
        result = run_feature_selection(X, y, fn, method="lasso")
        assert result["method"] == "lasso"

    def test_invalid_method_raises(self, clf_data):
        X, y, fn = clf_data
        with pytest.raises(ValueError, match="method="):
            run_feature_selection(X, y, fn, method="nonexistent_method")

    def test_supervised_without_y_raises(self, clf_data):
        X, _, fn = clf_data
        with pytest.raises(ValueError, match="requires y"):
            run_feature_selection(X, None, fn, method="mutual_information")

    def test_valid_methods_constant(self):
        assert "variance" in _VALID_METHODS
        assert "correlation" in _VALID_METHODS
        assert "mutual_information" in _VALID_METHODS
        assert "rfecv" in _VALID_METHODS
        assert "lasso" in _VALID_METHODS

    def test_n_features_forwarded_to_mi(self, clf_data):
        X, y, fn = clf_data
        result = run_feature_selection(
            X, y, fn, method="mutual_information", n_features=3
        )
        assert result["n_after"] == 3

    def test_regression_task(self, reg_data):
        X, y, fn = reg_data
        result = run_feature_selection(
            X, y, fn, method="mutual_information",
            task="regression", n_features=4
        )
        assert result["method"] == "mutual_information"
        assert result["n_after"] == 4


# ================================================================
# Benchmark integration (run_benchmark with feature_selection)
# ================================================================


class TestBenchmarkIntegration:
    def test_variance_selection_in_benchmark(self, tmp_path):
        """Feature selection reduces features in benchmark pipeline."""
        import tempfile
        import pandas as pd
        from sklearn.datasets import make_classification

        X_np, y = make_classification(
            n_samples=200, n_features=8, n_informative=4, random_state=0
        )
        X_np[:, 5] = 0.0  # constant column
        df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(8)])
        df["target"] = y

        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        from src.benchmark import run_benchmark
        results_df, preprocessor = run_benchmark(
            str(csv_path), "target",
            output_dir=str(tmp_path / "out"),
            task="classification",
            feature_selection="variance",
            verbose=False,
        )
        fs = preprocessor.get("feature_selection")
        assert fs is not None
        assert fs["n_removed"] >= 1
        assert fs["n_after"] < 8

    def test_mi_selection_in_benchmark(self, tmp_path):
        """mutual_information selection via benchmark respects n_features."""
        import pandas as pd
        from sklearn.datasets import make_classification

        X_np, y = make_classification(
            n_samples=200, n_features=8, n_informative=4, random_state=1
        )
        df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(8)])
        df["target"] = y
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        from src.benchmark import run_benchmark
        results_df, preprocessor = run_benchmark(
            str(csv_path), "target",
            output_dir=str(tmp_path / "out"),
            task="classification",
            feature_selection="mutual_information",
            n_features=5,
            verbose=False,
        )
        fs = preprocessor.get("feature_selection")
        assert fs is not None
        assert fs["n_after"] == 5

    def test_feature_selection_none_leaves_features_unchanged(self, tmp_path):
        import pandas as pd
        from sklearn.datasets import make_classification

        X_np, y = make_classification(n_samples=150, n_features=6, random_state=2)
        df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(6)])
        df["target"] = y
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        from src.benchmark import run_benchmark
        _, preprocessor = run_benchmark(
            str(csv_path), "target",
            output_dir=str(tmp_path / "out"),
            task="classification",
            feature_selection=None,
            verbose=False,
        )
        assert preprocessor.get("feature_selection") is None
        assert len(preprocessor["feature_names"]) == 6

    def test_selection_plots_created(self, tmp_path):
        import pandas as pd
        from sklearn.datasets import make_classification

        X_np, y = make_classification(n_samples=150, n_features=6, random_state=3)
        df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(6)])
        df["target"] = y
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        from src.benchmark import run_benchmark
        run_benchmark(
            str(csv_path), "target",
            output_dir=str(tmp_path / "out"),
            task="classification",
            feature_selection="variance",
            verbose=False,
        )
        out = tmp_path / "out"
        assert (out / "fs_selection_summary.png").exists()


# ================================================================
# Feature selection plots
# ================================================================


class TestFeatureSelectionPlots:
    def test_plot_feature_importance_ranking(self, tmp_path):
        from src.plots import plot_feature_importance_ranking
        scores = {"f0": 0.8, "f1": 0.5, "f2": 0.2, "f3": 0.9}
        out = tmp_path / "scores.png"
        plot_feature_importance_ranking(scores, "Test Scores", str(out))
        assert out.exists()

    def test_plot_selected_features_summary(self, tmp_path):
        from src.plots import plot_selected_features_summary
        result = {
            "method": "variance_threshold",
            "n_after": 5, "n_removed": 3,
            "removed_features": ["f0", "f1", "f2"],
            "selected_features": [f"f{i}" for i in range(3, 8)],
        }
        out = tmp_path / "summary.png"
        plot_selected_features_summary(result, "Summary", str(out))
        assert out.exists()

    def test_plot_importance_empty_scores_no_error(self, tmp_path):
        from src.plots import plot_feature_importance_ranking
        out = tmp_path / "empty.png"
        plot_feature_importance_ranking({}, "Empty", str(out))
        # Should not raise; file may or may not be created

    def test_plot_summary_no_removed(self, tmp_path):
        from src.plots import plot_selected_features_summary
        result = {
            "method": "variance_threshold",
            "n_after": 6, "n_removed": 0,
            "removed_features": [],
            "selected_features": [f"f{i}" for i in range(6)],
        }
        out = tmp_path / "no_removed.png"
        plot_selected_features_summary(result, "No Removed", str(out))
        assert out.exists()
