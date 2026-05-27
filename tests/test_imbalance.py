"""Tests for src/imbalance.py — resampling strategies."""

import numpy as np
import pytest


# ================================================================
# TestGetSampler
# ================================================================

class TestGetSampler:

    @pytest.mark.parametrize("strategy", [
        "random_over", "random_under", "smote", "adasyn",
    ])
    def test_valid_strategy_returns_sampler(self, strategy):
        from src.imbalance import get_sampler
        sampler = get_sampler(strategy)
        assert hasattr(sampler, "fit_resample")

    def test_invalid_strategy_raises(self):
        from src.imbalance import get_sampler
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_sampler("nonexistent")

    def test_smotenc_requires_categorical_features(self):
        from src.imbalance import get_sampler
        with pytest.raises(ValueError, match="categorical_features"):
            get_sampler("smotenc")

    def test_smotenc_with_categorical_features_ok(self):
        from src.imbalance import get_sampler
        sampler = get_sampler("smotenc", categorical_features=[0])
        assert hasattr(sampler, "fit_resample")

    def test_kwargs_forwarded_to_sampler(self):
        from src.imbalance import get_sampler
        sampler = get_sampler("smote", k_neighbors=3)
        assert sampler.k_neighbors == 3

    def test_valid_strategies_set_not_empty(self):
        from src.imbalance import VALID_STRATEGIES
        assert len(VALID_STRATEGIES) >= 5


# ================================================================
# TestApplySampling
# ================================================================

class TestApplySampling:

    def test_returns_required_keys(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_over")
        for key in ("X_res", "y_res", "sampler", "class_dist_before",
                    "class_dist_after", "n_before", "n_after", "summary"):
            assert key in result

    def test_oversampling_increases_samples(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_over")
        assert result["n_after"] >= result["n_before"]

    def test_undersampling_decreases_samples(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_under")
        assert result["n_after"] <= result["n_before"]

    def test_resampled_shapes_consistent(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_over")
        assert result["X_res"].shape[0] == result["y_res"].shape[0]
        assert result["X_res"].shape[1] == X.shape[1]

    def test_n_before_matches_input_size(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_over")
        assert result["n_before"] == len(y)

    def test_class_dist_after_more_balanced(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_over")
        ratio_before = (result["class_dist_before"].max()
                        / result["class_dist_before"].min())
        ratio_after  = (result["class_dist_after"].max()
                        / result["class_dist_after"].min())
        assert ratio_after <= ratio_before

    def test_summary_is_string_with_arrow(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "random_over")
        assert isinstance(result["summary"], str)
        assert "->" in result["summary"]

    def test_smote_works_on_numeric(self, imbalanced_X_y):
        from src.imbalance import apply_sampling
        X, y = imbalanced_X_y
        result = apply_sampling(X, y, "smote")
        assert result["n_after"] >= result["n_before"]

    def test_balanced_data_triggers_warning(self, small_X_y):
        from src.imbalance import apply_sampling
        X, y = small_X_y
        with pytest.warns(UserWarning, match="imbalance ratio"):
            apply_sampling(X, y, "random_over")
