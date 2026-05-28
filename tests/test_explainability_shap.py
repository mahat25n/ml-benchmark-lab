"""
Tests for SHAP-based explainability in src/explainability.py and src/plots.py.

All tests are skipped automatically when the shap package is not installed.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Skip entire module if shap is not available
pytest.importorskip("shap")


# ================================================================
# Module-scoped fixtures (built once, reused across all tests)
# ================================================================

@pytest.fixture(scope="module")
def clf_data():
    """Small binary classification dataset + fitted DecisionTree."""
    X, y = make_classification(
        n_samples=80, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_tr, y_tr)
    features = [f"feat_{i}" for i in range(5)]
    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "model": model, "features": features,
    }


@pytest.fixture(scope="module")
def reg_data():
    """Small regression dataset + fitted Ridge."""
    X, y = make_regression(n_samples=80, n_features=5, noise=0.1, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
    model = Ridge(alpha=1.0).fit(X_tr, y_tr)
    features = [f"feat_{i}" for i in range(5)]
    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "model": model, "features": features,
    }


# ================================================================
# TestComputeSHAPValues
# ================================================================

class TestComputeSHAPValues:

    def test_returns_tuple_of_two(self, clf_data):
        from src.explainability import compute_shap_values
        result = compute_shap_values(
            clf_data["model"], clf_data["X_test"][:10],
            feature_names=clf_data["features"],
        )
        assert len(result) == 2

    def test_shap_array_shape_classification(self, clf_data):
        from src.explainability import compute_shap_values
        X = clf_data["X_test"][:10]
        shap_arr, _ = compute_shap_values(
            clf_data["model"], X,
            feature_names=clf_data["features"],
        )
        assert shap_arr.shape == (10, 5)

    def test_shap_array_is_float(self, clf_data):
        from src.explainability import compute_shap_values
        shap_arr, _ = compute_shap_values(
            clf_data["model"], clf_data["X_test"][:10],
            feature_names=clf_data["features"],
        )
        assert shap_arr.dtype.kind == "f"

    def test_tree_explainer_classification(self, clf_data):
        from src.explainability import compute_shap_values
        X = clf_data["X_test"][:8]
        shap_arr, exp = compute_shap_values(
            clf_data["model"], X,
            feature_names=clf_data["features"],
            explainer="tree",
        )
        assert shap_arr.shape == (8, 5)

    def test_linear_explainer_regression(self, reg_data):
        from src.explainability import compute_shap_values
        X = reg_data["X_test"][:8]
        shap_arr, _ = compute_shap_values(
            reg_data["model"], X,
            feature_names=reg_data["features"],
            explainer="linear",
            X_background=reg_data["X_train"][:20],
            task="regression",
        )
        assert shap_arr.shape == (8, 5)

    def test_kernel_explainer_classification(self, clf_data):
        from src.explainability import compute_shap_values
        X_bg  = clf_data["X_train"][:10]
        X_exp = clf_data["X_test"][:5]
        shap_arr, _ = compute_shap_values(
            clf_data["model"], X_exp,
            feature_names=clf_data["features"],
            explainer="kernel",
            X_background=X_bg,
            task="classification",
        )
        assert shap_arr.shape == (5, 5)

    def test_kernel_explainer_regression(self, reg_data):
        from src.explainability import compute_shap_values
        X_bg  = reg_data["X_train"][:10]
        X_exp = reg_data["X_test"][:5]
        shap_arr, _ = compute_shap_values(
            reg_data["model"], X_exp,
            feature_names=reg_data["features"],
            explainer="kernel",
            X_background=X_bg,
            task="regression",
        )
        assert shap_arr.shape == (5, 5)

    def test_auto_selects_tree_for_tree_model(self, clf_data):
        import shap as _shap
        from src.explainability import _get_shap_explainer
        bg = clf_data["X_train"][:10]
        _, kind = _get_shap_explainer(
            _shap, clf_data["model"], bg, explainer="auto",
        )
        assert kind == "tree"

    def test_auto_selects_linear_for_ridge(self, reg_data):
        import shap as _shap
        from src.explainability import _get_shap_explainer
        bg = reg_data["X_train"][:10]
        _, kind = _get_shap_explainer(
            _shap, reg_data["model"], bg, explainer="auto",
        )
        assert kind == "linear"

    def test_custom_background_used(self, clf_data):
        from src.explainability import compute_shap_values
        X_bg = clf_data["X_train"][:5]
        shap_arr, _ = compute_shap_values(
            clf_data["model"], clf_data["X_test"][:8],
            feature_names=clf_data["features"],
            X_background=X_bg,
        )
        assert shap_arr.shape == (8, 5)

    def test_no_feature_names_generates_generic(self, clf_data):
        from src.explainability import compute_shap_values
        shap_arr, _ = compute_shap_values(
            clf_data["model"], clf_data["X_test"][:5],
        )
        assert shap_arr.shape == (5, 5)

    def test_invalid_explainer_raises_value_error(self, clf_data):
        from src.explainability import compute_shap_values
        with pytest.raises(ValueError, match="Unknown explainer"):
            compute_shap_values(
                clf_data["model"], clf_data["X_test"][:5],
                explainer="shap_forest",
            )

    def test_feature_names_length_mismatch_raises(self, clf_data):
        from src.explainability import compute_shap_values
        with pytest.raises(ValueError):
            compute_shap_values(
                clf_data["model"], clf_data["X_test"][:5],
                feature_names=["a", "b"],   # wrong length
            )


# ================================================================
# TestSummariseSHAPImportance
# ================================================================

class TestSummariseSHAPImportance:

    @pytest.fixture(scope="class")
    def shap_arr_and_names(self, clf_data):
        from src.explainability import compute_shap_values
        shap_arr, _ = compute_shap_values(
            clf_data["model"], clf_data["X_test"],
            feature_names=clf_data["features"],
        )
        return shap_arr, clf_data["features"]

    def test_returns_dataframe(self, shap_arr_and_names):
        from src.explainability import summarise_shap_importance
        shap_arr, names = shap_arr_and_names
        result = summarise_shap_importance(shap_arr, names)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, shap_arr_and_names):
        from src.explainability import summarise_shap_importance
        shap_arr, names = shap_arr_and_names
        result = summarise_shap_importance(shap_arr, names)
        for col in ("rank", "feature", "mean_abs_shap", "mean_shap",
                    "positive_mean", "negative_mean"):
            assert col in result.columns

    def test_row_count_equals_n_features(self, shap_arr_and_names):
        from src.explainability import summarise_shap_importance
        shap_arr, names = shap_arr_and_names
        result = summarise_shap_importance(shap_arr, names)
        assert len(result) == len(names)

    def test_rank_starts_at_one(self, shap_arr_and_names):
        from src.explainability import summarise_shap_importance
        shap_arr, names = shap_arr_and_names
        result = summarise_shap_importance(shap_arr, names)
        assert result["rank"].min() == 1

    def test_sorted_by_mean_abs_shap_descending(self, shap_arr_and_names):
        from src.explainability import summarise_shap_importance
        shap_arr, names = shap_arr_and_names
        result = summarise_shap_importance(shap_arr, names)
        vals = result["mean_abs_shap"].values
        assert list(vals) == sorted(vals, reverse=True)

    def test_mean_abs_shap_non_negative(self, shap_arr_and_names):
        from src.explainability import summarise_shap_importance
        shap_arr, names = shap_arr_and_names
        result = summarise_shap_importance(shap_arr, names)
        assert (result["mean_abs_shap"] >= 0).all()


# ================================================================
# TestExplainPrediction
# ================================================================

class TestExplainPrediction:

    def test_returns_dataframe(self, clf_data):
        from src.explainability import explain_prediction
        instance = clf_data["X_test"][0]
        result = explain_prediction(
            clf_data["model"], instance,
            feature_names=clf_data["features"],
            X_background=clf_data["X_train"][:20],
        )
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, clf_data):
        from src.explainability import explain_prediction
        instance = clf_data["X_test"][0]
        result = explain_prediction(
            clf_data["model"], instance,
            feature_names=clf_data["features"],
            X_background=clf_data["X_train"][:20],
        )
        for col in ("rank", "feature", "feature_value", "shap_value", "abs_shap"):
            assert col in result.columns

    def test_one_row_per_feature(self, clf_data):
        from src.explainability import explain_prediction
        result = explain_prediction(
            clf_data["model"], clf_data["X_test"][0],
            feature_names=clf_data["features"],
            X_background=clf_data["X_train"][:20],
        )
        assert len(result) == len(clf_data["features"])

    def test_rank_starts_at_one(self, clf_data):
        from src.explainability import explain_prediction
        result = explain_prediction(
            clf_data["model"], clf_data["X_test"][0],
            feature_names=clf_data["features"],
            X_background=clf_data["X_train"][:20],
        )
        assert result["rank"].min() == 1

    def test_feature_values_match_input(self, clf_data):
        from src.explainability import explain_prediction
        instance = clf_data["X_test"][0]
        result = explain_prediction(
            clf_data["model"], instance,
            feature_names=clf_data["features"],
            X_background=clf_data["X_train"][:20],
        )
        result_sorted = result.sort_values("feature").reset_index(drop=True)
        expected = sorted(zip(clf_data["features"], instance.tolist()))
        for i, (fname, fval) in enumerate(expected):
            assert result_sorted.loc[i, "feature"] == fname
            assert result_sorted.loc[i, "feature_value"] == pytest.approx(fval)

    def test_accepts_2d_input(self, clf_data):
        from src.explainability import explain_prediction
        instance = clf_data["X_test"][[0]]   # shape (1, 5)
        result = explain_prediction(
            clf_data["model"], instance,
            feature_names=clf_data["features"],
            X_background=clf_data["X_train"][:20],
        )
        assert len(result) == 5

    def test_regression_model(self, reg_data):
        from src.explainability import explain_prediction
        instance = reg_data["X_test"][0]
        result = explain_prediction(
            reg_data["model"], instance,
            feature_names=reg_data["features"],
            X_background=reg_data["X_train"][:20],
            task="regression",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5


# ================================================================
# TestSHAPPlots
# ================================================================

class TestSHAPPlots:

    @pytest.fixture(scope="class")
    def shap_bundle(self, clf_data):
        from src.explainability import compute_shap_values, summarise_shap_importance
        X = clf_data["X_test"]
        shap_arr, _ = compute_shap_values(
            clf_data["model"], X, feature_names=clf_data["features"],
        )
        importance_df = summarise_shap_importance(shap_arr, clf_data["features"])
        return shap_arr, X, clf_data["features"], importance_df

    def test_plot_shap_summary_creates_png(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_summary
        shap_arr, X, features, _ = shap_bundle
        p = tmp_path / "shap_summary.png"
        plot_shap_summary(shap_arr, X, features, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_shap_summary_max_display(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_summary
        shap_arr, X, features, _ = shap_bundle
        p = tmp_path / "shap_summary_3.png"
        plot_shap_summary(shap_arr, X, features, "TestModel", p, max_display=3)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_shap_bar_creates_png(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_bar
        _, _, _, importance_df = shap_bundle
        p = tmp_path / "shap_bar.png"
        plot_shap_bar(importance_df, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_shap_bar_top_n(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_bar
        _, _, _, importance_df = shap_bundle
        p = tmp_path / "shap_bar_top3.png"
        plot_shap_bar(importance_df, "TestModel", p, top_n=3)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_shap_dependence_creates_png(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_dependence
        shap_arr, X, features, _ = shap_bundle
        p = tmp_path / "shap_dep.png"
        plot_shap_dependence(shap_arr, X, features, features[0], "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_shap_dependence_by_index(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_dependence
        shap_arr, X, features, _ = shap_bundle
        p = tmp_path / "shap_dep_idx.png"
        plot_shap_dependence(shap_arr, X, features, 0, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_shap_dependence_with_interaction(self, tmp_path, shap_bundle):
        from src.plots import plot_shap_dependence
        shap_arr, X, features, _ = shap_bundle
        p = tmp_path / "shap_dep_int.png"
        plot_shap_dependence(
            shap_arr, X, features, features[0], "TestModel", p,
            interaction_feature=features[1],
        )
        assert p.exists() and p.stat().st_size > 0


# ================================================================
# TestMissingShap (graceful import error)
# ================================================================

class TestMissingShap:

    def test_import_error_message_contains_install_hint(self, monkeypatch):
        """_require_shap raises ImportError with pip install hint when shap absent."""
        import sys
        from src import explainability as expl

        real_require = expl._require_shap

        def _fake_require():
            raise ImportError("SHAP is required for this function. Install it with: pip install shap")

        monkeypatch.setattr(expl, "_require_shap", _fake_require)
        with pytest.raises(ImportError, match="pip install shap"):
            expl.compute_shap_values(None, np.zeros((5, 3)))
        monkeypatch.setattr(expl, "_require_shap", real_require)
