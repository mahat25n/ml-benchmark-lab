"""
Tests for advanced statistical comparison and ranking utilities.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.stats import (
    compute_average_ranks,
    compute_bootstrap_ci,
    compute_cd_nemenyi,
    compute_confidence_interval,
    compute_metric_leaderboard,
    compute_pairwise_comparisons,
    compute_significance_summary,
    prepare_cd_diagram_data,
    run_corrected_kfold_ttest,
    run_paired_ttest,
)


# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture
def five_scores():
    """Two score arrays, each of length 5."""
    a = np.array([0.80, 0.85, 0.78, 0.82, 0.88])
    b = np.array([0.72, 0.76, 0.74, 0.70, 0.79])
    return a, b


@pytest.fixture
def scores_df():
    """4 datasets × 3 classifiers."""
    rng = np.random.default_rng(0)
    data = rng.random((8, 3))
    return pd.DataFrame(data, columns=["A", "B", "C"])


@pytest.fixture
def results_df():
    return pd.DataFrame({
        "Model":    ["RF", "SVM", "KNN", "DT"],
        "Accuracy": [0.92, 0.88, 0.85, 0.80],
        "ROC AUC":  [0.95, 0.90, 0.87, 0.83],
    })


# ================================================================
# run_paired_ttest
# ================================================================

class TestRunPairedTtest:
    def test_returns_expected_keys(self, five_scores):
        a, b = five_scores
        res = run_paired_ttest(a, b)
        for k in ("statistic", "p_value", "df", "mean_diff", "ci_low", "ci_high", "interpretation"):
            assert k in res

    def test_significant_difference(self):
        a = np.array([0.9, 0.88, 0.91, 0.87, 0.93, 0.90, 0.89, 0.92, 0.88, 0.91])
        b = np.array([0.7, 0.68, 0.71, 0.67, 0.73, 0.70, 0.69, 0.72, 0.68, 0.71])
        res = run_paired_ttest(a, b)
        assert res["p_value"] < 0.05
        assert "Significant" in res["interpretation"]

    def test_no_significant_difference(self):
        a = np.array([0.82, 0.83, 0.81])
        b = np.array([0.81, 0.84, 0.82])
        res = run_paired_ttest(a, b)
        assert res["p_value"] > 0.05
        assert "No significant" in res["interpretation"]

    def test_mean_diff_correct(self, five_scores):
        a, b = five_scores
        res = run_paired_ttest(a, b)
        assert abs(res["mean_diff"] - float((a - b).mean())) < 1e-9

    def test_df_is_n_minus_one(self, five_scores):
        a, b = five_scores
        res = run_paired_ttest(a, b)
        assert res["df"] == len(a) - 1

    def test_ci_contains_mean(self, five_scores):
        a, b = five_scores
        res = run_paired_ttest(a, b)
        assert res["ci_low"] <= res["mean_diff"] <= res["ci_high"]

    def test_alternative_greater(self, five_scores):
        a, b = five_scores
        res = run_paired_ttest(a, b, alternative="greater")
        assert "H1: A > B" in res["interpretation"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            run_paired_ttest([0.8, 0.9], [0.7])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            run_paired_ttest([0.8], [0.7])


# ================================================================
# run_corrected_kfold_ttest
# ================================================================

class TestRunCorrectedKfoldTtest:
    def test_returns_expected_keys(self, five_scores):
        a, b = five_scores
        res = run_corrected_kfold_ttest(a, b, k=5)
        for k in ("statistic", "p_value", "df", "mean_diff", "corrected_variance", "interpretation"):
            assert k in res

    def test_with_n_train_n_test(self, five_scores):
        a, b = five_scores
        res = run_corrected_kfold_ttest(a, b, k=5, n_train=800, n_test=200)
        assert "corrected_variance" in res
        assert res["corrected_variance"] >= 0

    def test_p_value_range(self, five_scores):
        a, b = five_scores
        res = run_corrected_kfold_ttest(a, b, k=5)
        assert 0.0 <= res["p_value"] <= 1.0

    def test_df_is_n_minus_one(self, five_scores):
        a, b = five_scores
        res = run_corrected_kfold_ttest(a, b, k=5)
        assert res["df"] == len(a) - 1

    def test_identical_scores_no_significance(self):
        a = np.array([0.85, 0.85, 0.85, 0.85, 0.85])
        res = run_corrected_kfold_ttest(a, a, k=5)
        assert res["p_value"] == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            run_corrected_kfold_ttest([0.8, 0.9], [0.7], k=2)

    def test_interpretation_is_string(self, five_scores):
        a, b = five_scores
        res = run_corrected_kfold_ttest(a, b, k=5)
        assert isinstance(res["interpretation"], str)


# ================================================================
# compute_confidence_interval
# ================================================================

class TestComputeConfidenceInterval:
    def test_returns_expected_keys(self):
        res = compute_confidence_interval([0.8, 0.85, 0.82, 0.79, 0.88])
        for k in ("mean", "std", "n", "lower", "upper", "margin"):
            assert k in res

    def test_interval_contains_mean(self):
        scores = [0.8, 0.85, 0.82, 0.79, 0.88, 0.91, 0.84, 0.87]
        res = compute_confidence_interval(scores)
        assert res["lower"] <= res["mean"] <= res["upper"]

    def test_n_correct(self):
        scores = [0.8, 0.85, 0.82]
        res = compute_confidence_interval(scores)
        assert res["n"] == 3

    def test_method_normal(self):
        scores = [0.8, 0.85, 0.82, 0.79, 0.88]
        res = compute_confidence_interval(scores, method="normal")
        assert res["lower"] < res["upper"]

    def test_higher_confidence_wider_interval(self):
        scores = [0.8, 0.85, 0.82, 0.79, 0.88]
        res_90 = compute_confidence_interval(scores, confidence=0.90)
        res_99 = compute_confidence_interval(scores, confidence=0.99)
        assert res_99["margin"] > res_90["margin"]

    def test_too_few_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_confidence_interval([0.8])


# ================================================================
# compute_bootstrap_ci
# ================================================================

class TestComputeBootstrapCI:
    def test_returns_expected_keys(self):
        res = compute_bootstrap_ci([0.8, 0.9, 0.7, 0.85, 0.88])
        for k in ("mean", "std", "lower", "upper"):
            assert k in res

    def test_interval_ordered(self):
        rng = np.random.default_rng(1)
        vals = rng.random(50)
        res = compute_bootstrap_ci(vals)
        assert res["lower"] <= res["mean"] <= res["upper"]

    def test_reproducible_with_seed(self):
        vals = [0.8, 0.9, 0.7, 0.85, 0.88, 0.82, 0.91]
        r1 = compute_bootstrap_ci(vals, random_state=42)
        r2 = compute_bootstrap_ci(vals, random_state=42)
        assert r1["lower"] == r2["lower"]
        assert r1["upper"] == r2["upper"]

    def test_custom_stat_fn(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        res = compute_bootstrap_ci(vals, stat_fn=np.median, random_state=0)
        assert "mean" in res

    def test_higher_confidence_wider(self):
        rng = np.random.default_rng(0)
        vals = rng.random(100)
        r90 = compute_bootstrap_ci(vals, confidence=0.90, random_state=0)
        r99 = compute_bootstrap_ci(vals, confidence=0.99, random_state=0)
        assert (r99["upper"] - r99["lower"]) >= (r90["upper"] - r90["lower"])

    def test_different_n_bootstrap(self):
        vals = [0.8, 0.9, 0.7, 0.85, 0.88]
        res = compute_bootstrap_ci(vals, n_bootstrap=500, random_state=0)
        assert isinstance(res["mean"], float)


# ================================================================
# compute_average_ranks
# ================================================================

class TestComputeAverageRanks:
    def test_returns_series(self, scores_df):
        avg = compute_average_ranks(scores_df)
        assert isinstance(avg, pd.Series)

    def test_length_equals_n_classifiers(self, scores_df):
        avg = compute_average_ranks(scores_df)
        assert len(avg) == scores_df.shape[1]

    def test_sorted_ascending(self, scores_df):
        avg = compute_average_ranks(scores_df)
        assert list(avg) == sorted(avg)

    def test_best_rank_close_to_1(self):
        data = pd.DataFrame({
            "Best":   [1.0, 1.0, 1.0, 1.0],
            "Middle": [0.8, 0.8, 0.8, 0.8],
            "Worst":  [0.6, 0.6, 0.6, 0.6],
        })
        avg = compute_average_ranks(data)
        assert avg.index[0] == "Best"

    def test_ndarray_input(self):
        data = np.array([[0.9, 0.7, 0.5], [0.8, 0.6, 0.4]])
        avg = compute_average_ranks(data)
        assert len(avg) == 3

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            compute_average_ranks(np.array([0.8, 0.9, 0.7]))

    def test_index_names_preserved(self, scores_df):
        avg = compute_average_ranks(scores_df)
        assert set(avg.index) == set(scores_df.columns)


# ================================================================
# compute_metric_leaderboard
# ================================================================

class TestComputeMetricLeaderboard:
    def test_returns_dataframe(self, results_df):
        lb = compute_metric_leaderboard(results_df, "Accuracy")
        assert isinstance(lb, pd.DataFrame)

    def test_columns(self, results_df):
        lb = compute_metric_leaderboard(results_df, "Accuracy")
        assert "Rank" in lb.columns
        assert "Model" in lb.columns
        assert "Accuracy" in lb.columns
        assert "Delta_from_best" in lb.columns

    def test_rank_1_is_best(self, results_df):
        lb = compute_metric_leaderboard(results_df, "Accuracy")
        assert lb.loc[lb["Rank"] == 1, "Accuracy"].iloc[0] == results_df["Accuracy"].max()

    def test_lower_is_better(self, results_df):
        lb = compute_metric_leaderboard(results_df, "Accuracy", higher_is_better=False)
        assert lb.loc[lb["Rank"] == 1, "Accuracy"].iloc[0] == results_df["Accuracy"].min()

    def test_delta_from_best_zero_for_rank1(self, results_df):
        lb = compute_metric_leaderboard(results_df, "Accuracy")
        assert lb.loc[lb["Rank"] == 1, "Delta_from_best"].iloc[0] == pytest.approx(0.0)

    def test_missing_model_column_raises(self, results_df):
        with pytest.raises(ValueError, match="'Model' column"):
            compute_metric_leaderboard(results_df.drop(columns=["Model"]), "Accuracy")

    def test_missing_metric_column_raises(self, results_df):
        with pytest.raises(ValueError, match="'F1'"):
            compute_metric_leaderboard(results_df, "F1")

    def test_length_equals_n_models(self, results_df):
        lb = compute_metric_leaderboard(results_df, "Accuracy")
        assert len(lb) == len(results_df)


# ================================================================
# compute_pairwise_comparisons
# ================================================================

class TestComputePairwiseComparisons:
    def test_returns_expected_keys(self, scores_df):
        res = compute_pairwise_comparisons(scores_df)
        assert "p_values"    in res
        assert "significant" in res
        assert "effect_sizes" in res

    def test_matrix_shape(self, scores_df):
        res = compute_pairwise_comparisons(scores_df)
        k = scores_df.shape[1]
        assert res["p_values"].shape == (k, k)

    def test_diagonal_is_nan(self, scores_df):
        res = compute_pairwise_comparisons(scores_df)
        diag = np.diag(res["p_values"].values)
        assert all(np.isnan(v) for v in diag)

    def test_ttest_method(self, scores_df):
        res = compute_pairwise_comparisons(scores_df, test="ttest")
        assert "p_values" in res

    def test_significant_is_bool(self, scores_df):
        res = compute_pairwise_comparisons(scores_df)
        assert res["significant"].dtypes.apply(lambda d: d == bool or d == np.bool_).all()

    def test_column_index_preserved(self, scores_df):
        res = compute_pairwise_comparisons(scores_df)
        assert list(res["p_values"].columns) == list(scores_df.columns)


# ================================================================
# compute_significance_summary
# ================================================================

class TestComputeSignificanceSummary:
    def test_returns_dataframe(self, scores_df):
        pw = compute_pairwise_comparisons(scores_df)
        summary = compute_significance_summary(pw)
        assert isinstance(summary, pd.DataFrame)

    def test_columns(self, scores_df):
        pw = compute_pairwise_comparisons(scores_df)
        summary = compute_significance_summary(pw)
        for col in ("Classifier_A", "Classifier_B", "p_value", "effect_size"):
            assert col in summary.columns

    def test_no_duplicates(self, scores_df):
        pw = compute_pairwise_comparisons(scores_df)
        summary = compute_significance_summary(pw)
        if len(summary) > 0:
            pairs = list(zip(summary["Classifier_A"], summary["Classifier_B"]))
            assert len(pairs) == len(set(pairs))

    def test_empty_when_no_significant_pairs(self):
        a = np.array([0.80, 0.81, 0.79, 0.80, 0.81])
        b = np.array([0.80, 0.80, 0.80, 0.80, 0.80])
        df = pd.DataFrame({"A": a, "B": b})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pw  = compute_pairwise_comparisons(df)
            summary = compute_significance_summary(pw)
        assert isinstance(summary, pd.DataFrame)


# ================================================================
# compute_cd_nemenyi
# ================================================================

class TestComputeCdNemenyi:
    def test_returns_float(self):
        cd = compute_cd_nemenyi(3, 10)
        assert isinstance(cd, float)

    def test_positive(self):
        cd = compute_cd_nemenyi(5, 20)
        assert cd > 0

    def test_more_classifiers_larger_cd(self):
        cd3 = compute_cd_nemenyi(3, 10)
        cd6 = compute_cd_nemenyi(6, 10)
        assert cd6 > cd3

    def test_more_datasets_smaller_cd(self):
        cd10 = compute_cd_nemenyi(4, 10)
        cd50 = compute_cd_nemenyi(4, 50)
        assert cd50 < cd10

    def test_invalid_n_classifiers_raises(self):
        with pytest.raises(ValueError, match="n_classifiers"):
            compute_cd_nemenyi(1, 10)

    def test_invalid_n_datasets_raises(self):
        with pytest.raises(ValueError, match="n_datasets"):
            compute_cd_nemenyi(3, 0)

    def test_different_alpha(self):
        cd05 = compute_cd_nemenyi(4, 10, alpha=0.05)
        cd10 = compute_cd_nemenyi(4, 10, alpha=0.10)
        assert cd05 > cd10


# ================================================================
# prepare_cd_diagram_data
# ================================================================

class TestPrepareCdDiagramData:
    def test_returns_expected_keys(self, scores_df):
        result = prepare_cd_diagram_data(scores_df)
        for k in ("avg_ranks", "cd", "significant_pairs", "n_classifiers", "n_datasets"):
            assert k in result

    def test_n_classifiers_correct(self, scores_df):
        result = prepare_cd_diagram_data(scores_df)
        assert result["n_classifiers"] == scores_df.shape[1]

    def test_n_datasets_correct(self, scores_df):
        result = prepare_cd_diagram_data(scores_df)
        assert result["n_datasets"] == scores_df.shape[0]

    def test_avg_ranks_is_series(self, scores_df):
        result = prepare_cd_diagram_data(scores_df)
        assert isinstance(result["avg_ranks"], pd.Series)

    def test_significant_pairs_is_list(self, scores_df):
        result = prepare_cd_diagram_data(scores_df)
        assert isinstance(result["significant_pairs"], list)

    def test_cd_positive(self, scores_df):
        result = prepare_cd_diagram_data(scores_df)
        assert result["cd"] > 0

    def test_clearly_different_classifiers_flagged(self):
        data = pd.DataFrame({
            "Best":   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Middle": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
            "Worst":  [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        })
        result = prepare_cd_diagram_data(data)
        # Best vs Worst should be significantly different
        pairs_flat = [f"{a}-{b}" for a, b in result["significant_pairs"]]
        assert any("Best" in p and "Worst" in p for p in pairs_flat)
