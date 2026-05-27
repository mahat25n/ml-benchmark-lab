"""Tests for src/explainability.py — feature importance."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


# ================================================================
# TestGetFeatureImportance
# ================================================================

class TestGetFeatureImportance:

    def test_returns_dataframe(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf)
        for col in ("rank", "feature", "importance", "source"):
            assert col in result.columns

    def test_rank_starts_at_one(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf)
        assert result["rank"].min() == 1

    def test_rank_sorted_ascending(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf)
        assert list(result["rank"]) == sorted(result["rank"])

    def test_row_count_matches_n_features(self, fitted_rf, split_binary):
        from src.explainability import get_feature_importance
        X_tr, _, _, _ = split_binary
        result = get_feature_importance(fitted_rf)
        assert len(result) == X_tr.shape[1]

    def test_custom_feature_names_used(self, fitted_rf, split_binary):
        from src.explainability import get_feature_importance
        X_tr, _, _, _ = split_binary
        names = [f"f{i}" for i in range(X_tr.shape[1])]
        result = get_feature_importance(fitted_rf, feature_names=names)
        assert set(result["feature"]) == set(names)

    def test_generic_names_when_none(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf)
        assert all(f.startswith("feature_") for f in result["feature"])

    def test_normalize_adds_importance_norm_column(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf, normalize=True)
        assert "importance_norm" in result.columns

    def test_normalize_max_is_one(self, fitted_rf):
        from src.explainability import get_feature_importance
        result = get_feature_importance(fitted_rf, normalize=True)
        assert abs(result["importance_norm"].max() - 1.0) < 1e-9

    def test_wrong_feature_names_length_raises(self, fitted_rf):
        from src.explainability import get_feature_importance
        with pytest.raises(ValueError, match="feature_names"):
            get_feature_importance(fitted_rf, feature_names=["a", "b"])

    def test_model_without_importance_raises(self):
        from src.explainability import get_feature_importance
        class _Dummy:
            pass
        with pytest.raises(ValueError):
            get_feature_importance(_Dummy())

    def test_linear_model_uses_coef(self, split_binary):
        from src.explainability import get_feature_importance
        X_tr, _, y_tr, _ = split_binary
        lr = LogisticRegression(max_iter=500, random_state=42)
        lr.fit(X_tr, y_tr)
        result = get_feature_importance(lr)
        assert isinstance(result, pd.DataFrame)
        assert "coef_" in result["source"].iloc[0]


# ================================================================
# TestComputePermutationImportance
# ================================================================

class TestComputePermutationImportance:

    def test_returns_dataframe(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        result = compute_permutation_importance(fitted_rf, X_te, y_te, n_repeats=5)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        result = compute_permutation_importance(fitted_rf, X_te, y_te, n_repeats=5)
        for col in ("rank", "feature", "importance_mean",
                    "importance_std", "importance_sem"):
            assert col in result.columns

    def test_rank_starts_at_one(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        result = compute_permutation_importance(fitted_rf, X_te, y_te, n_repeats=5)
        assert result["rank"].min() == 1

    def test_row_count_matches_n_features(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        result = compute_permutation_importance(fitted_rf, X_te, y_te, n_repeats=5)
        assert len(result) == X_te.shape[1]

    def test_sem_equals_std_over_sqrt_n_repeats(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        n = 5
        result = compute_permutation_importance(
            fitted_rf, X_te, y_te, n_repeats=n, random_state=42
        )
        expected = result["importance_std"] / np.sqrt(n)
        np.testing.assert_allclose(result["importance_sem"], expected)

    def test_reproducibility(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        a = compute_permutation_importance(
            fitted_rf, X_te, y_te, n_repeats=5, random_state=42
        )
        b = compute_permutation_importance(
            fitted_rf, X_te, y_te, n_repeats=5, random_state=42
        )
        pd.testing.assert_frame_equal(a, b)

    def test_wrong_feature_names_length_raises(self, fitted_rf, split_binary):
        from src.explainability import compute_permutation_importance
        _, X_te, _, y_te = split_binary
        with pytest.raises(ValueError, match="feature_names"):
            compute_permutation_importance(
                fitted_rf, X_te, y_te, feature_names=["only_one"]
            )
