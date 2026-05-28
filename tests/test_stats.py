"""Tests for src/stats.py — McNemar, Wilcoxon, and Friedman tests."""

import warnings

import numpy as np
import pandas as pd
import pytest


# ================================================================
# Shared test data
# ================================================================

_Y_TRUE   = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
_PRED_A   = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0])  # 1 error
_PRED_B   = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1])  # 2 errors

_SCORES_A = np.array([0.90, 0.87, 0.88, 0.85, 0.91, 0.89, 0.86, 0.88])
_SCORES_B = np.array([0.80, 0.76, 0.78, 0.74, 0.82, 0.79, 0.77, 0.81])

_FRIEDMAN_SCORES = np.array([
    [0.90, 0.85, 0.78],
    [0.88, 0.83, 0.80],
    [0.91, 0.86, 0.79],
    [0.87, 0.84, 0.77],
    [0.92, 0.88, 0.82],
])


# ================================================================
# TestMcNemarTest
# ================================================================

class TestMcNemarTest:

    def test_returns_required_keys(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        for key in ("statistic", "p_value", "contingency_table", "n_discordant", "interpretation"):
            assert key in result

    def test_p_value_is_float(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        assert isinstance(result["p_value"], float)

    def test_p_value_in_valid_range(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_statistic_is_non_negative(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        assert result["statistic"] >= 0.0

    def test_contingency_table_shape(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        assert result["contingency_table"].shape == (2, 2)

    def test_n_discordant_is_int(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        assert isinstance(result["n_discordant"], int)

    def test_interpretation_is_string(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_B)
        assert isinstance(result["interpretation"], str)

    def test_identical_predictions_zero_discordant(self):
        from src.stats import run_mcnemar_test
        result = run_mcnemar_test(_Y_TRUE, _PRED_A, _PRED_A)
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0
        assert result["statistic"] == 0.0

    def test_no_correction_differs_from_correction(self):
        from src.stats import run_mcnemar_test
        rng = np.random.default_rng(0)
        y_true  = rng.integers(0, 2, 100)
        pred_a  = rng.integers(0, 2, 100)
        pred_b  = rng.integers(0, 2, 100)
        with_corr    = run_mcnemar_test(y_true, pred_a, pred_b, correction=True)
        without_corr = run_mcnemar_test(y_true, pred_a, pred_b, correction=False)
        # Yates correction reduces the statistic (or keeps it equal when diff==0)
        assert with_corr["statistic"] <= without_corr["statistic"] + 1e-9

    def test_small_n_discordant_warns(self):
        from src.stats import run_mcnemar_test
        y_true = np.array([0, 1, 0, 1, 0, 1])
        pred_a = np.array([0, 1, 0, 1, 0, 1])  # identical
        pred_b = np.array([1, 1, 0, 1, 0, 1])  # 1 discordant
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_mcnemar_test(y_true, pred_a, pred_b)
        assert any("discordant" in str(w.message).lower() for w in caught)

    def test_mismatched_shapes_raise_value_error(self):
        from src.stats import run_mcnemar_test
        with pytest.raises(ValueError, match="shape"):
            run_mcnemar_test(
                np.array([0, 1, 0]),
                np.array([0, 1]),
                np.array([0, 1, 0]),
            )

    def test_custom_alpha_in_interpretation(self):
        from src.stats import run_mcnemar_test
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 100)
        pred_a = rng.integers(0, 2, 100)
        pred_b = rng.integers(0, 2, 100)
        result = run_mcnemar_test(y_true, pred_a, pred_b, alpha=0.01)
        assert "alpha=0.01" in result["interpretation"]


# ================================================================
# TestWilcoxonTest
# ================================================================

class TestWilcoxonTest:

    def test_returns_required_keys(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B)
        for key in ("statistic", "p_value", "n_pairs", "effect_size_r", "interpretation"):
            assert key in result

    def test_p_value_is_float(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B)
        assert isinstance(result["p_value"], float)

    def test_p_value_in_valid_range(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_n_pairs_correct(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B)
        assert result["n_pairs"] == len(_SCORES_A)

    def test_effect_size_in_range(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B)
        assert -1.0 <= result["effect_size_r"] <= 1.0

    def test_interpretation_is_string(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B)
        assert isinstance(result["interpretation"], str)

    def test_clearly_different_scores_significant(self):
        from src.stats import run_wilcoxon_test
        # A is consistently 0.1 better than B across 10 folds
        a = np.array([0.90, 0.88, 0.91, 0.87, 0.93, 0.89, 0.92, 0.88, 0.90, 0.91])
        b = a - 0.10
        result = run_wilcoxon_test(a, b)
        assert result["p_value"] < 0.05

    def test_identical_scores_raises_or_warns(self):
        """Wilcoxon raises ValueError on all-zero differences (no ranks to compute)."""
        from src.stats import run_wilcoxon_test
        a = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        with pytest.raises(Exception):
            run_wilcoxon_test(a, a)

    def test_alternative_greater_accepted(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B, alternative="greater")
        assert "A > B" in result["interpretation"]

    def test_alternative_less_accepted(self):
        from src.stats import run_wilcoxon_test
        result = run_wilcoxon_test(_SCORES_A, _SCORES_B, alternative="less")
        assert "A < B" in result["interpretation"]

    def test_mismatched_lengths_raise_value_error(self):
        from src.stats import run_wilcoxon_test
        with pytest.raises(ValueError, match="length"):
            run_wilcoxon_test(
                np.array([0.8, 0.9, 0.7]),
                np.array([0.8, 0.9]),
            )

    def test_few_pairs_warns(self):
        from src.stats import run_wilcoxon_test
        a = np.array([0.90, 0.88, 0.87])
        b = np.array([0.80, 0.78, 0.77])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                run_wilcoxon_test(a, b)
            except Exception:
                pass  # Wilcoxon may raise on very few pairs — we only care about warning
        assert any("pair" in str(w.message).lower() for w in caught)


# ================================================================
# TestFriedmanTest
# ================================================================

class TestFriedmanTest:

    def test_returns_required_keys(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        for key in ("statistic", "p_value", "df", "avg_ranks", "interpretation"):
            assert key in result

    def test_p_value_is_float(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert isinstance(result["p_value"], float)

    def test_p_value_in_valid_range(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_df_equals_n_classifiers_minus_one(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert result["df"] == _FRIEDMAN_SCORES.shape[1] - 1

    def test_avg_ranks_is_series(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert isinstance(result["avg_ranks"], pd.Series)

    def test_avg_ranks_length_equals_n_classifiers(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert len(result["avg_ranks"]) == _FRIEDMAN_SCORES.shape[1]

    def test_avg_ranks_sorted_ascending(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        ranks = result["avg_ranks"].values
        assert list(ranks) == sorted(ranks)

    def test_interpretation_is_string(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert isinstance(result["interpretation"], str)

    def test_dataframe_input_uses_column_names(self):
        from src.stats import run_friedman_test
        df = pd.DataFrame(
            _FRIEDMAN_SCORES,
            columns=["ModelA", "ModelB", "ModelC"],
        )
        result = run_friedman_test(df)
        assert "ModelA" in result["avg_ranks"].index

    def test_clearly_different_classifiers_significant(self):
        from src.stats import run_friedman_test
        # First classifier is always best, third is always worst
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((10, 3)) * 0.01
        scores = np.column_stack([
            np.full(10, 0.95) + noise[:, 0],
            np.full(10, 0.80) + noise[:, 1],
            np.full(10, 0.65) + noise[:, 2],
        ])
        result = run_friedman_test(scores)
        assert result["p_value"] < 0.05

    def test_fewer_than_three_classifiers_raises(self):
        from src.stats import run_friedman_test
        with pytest.raises(ValueError, match="3 classifiers"):
            run_friedman_test(np.array([[0.9, 0.8], [0.85, 0.75]]))

    def test_fewer_than_two_datasets_raises(self):
        from src.stats import run_friedman_test
        with pytest.raises(ValueError, match="2 datasets"):
            run_friedman_test(np.array([[0.9, 0.8, 0.7]]))

    def test_non_2d_input_raises(self):
        from src.stats import run_friedman_test
        with pytest.raises(ValueError, match="2D"):
            run_friedman_test(np.array([0.9, 0.8, 0.7]))

    def test_few_datasets_warns(self):
        from src.stats import run_friedman_test
        scores = np.array([[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_friedman_test(scores)
        assert any("fold" in str(w.message).lower() or "dataset" in str(w.message).lower()
                   for w in caught)

    def test_statistic_non_negative(self):
        from src.stats import run_friedman_test
        result = run_friedman_test(_FRIEDMAN_SCORES)
        assert result["statistic"] >= 0.0
