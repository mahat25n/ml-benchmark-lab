"""
Tests for src/data_profile.py — dataset profiling and automated data reports.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_profile import (
    profile_dataset,
    summarize_columns,
    summarize_missingness,
    summarize_target,
)


# ================================================================
# FIXTURES
# ================================================================


@pytest.fixture
def sample_df():
    """
    200-row DataFrame with:
    - Three numeric columns (age, income, score)
    - One categorical column (category)
    - One binary int column (target)
    - Missing values in age and income
    - One duplicate row
    """
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age":      np.random.randint(18, 80, n).astype(float),
        "income":   np.random.exponential(50_000, n),   # right-skewed
        "score":    np.random.normal(0.5, 0.1, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "target":   np.random.choice([0, 1], n, p=[0.3, 0.7]),
    })
    df.loc[10:20, "age"]    = np.nan   # 11 missing
    df.loc[50:55, "income"] = np.nan   # 6 missing
    # Inject one duplicate row
    df.loc[199] = df.loc[0]
    return df


@pytest.fixture
def sample_csv(sample_df, tmp_path):
    p = tmp_path / "data.csv"
    sample_df.to_csv(p, index=False)
    return p


@pytest.fixture
def multiclass_df():
    np.random.seed(0)
    return pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "label": np.random.choice(["cat", "dog", "bird", "fish"], 100),
    })


@pytest.fixture
def continuous_df():
    np.random.seed(0)
    return pd.DataFrame({
        "x1": np.random.randn(100),
        "y":  np.random.randn(100) * 10 + 5,   # 100 unique values → continuous
    })


@pytest.fixture
def complete_df():
    """No missing values."""
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "c": ["x", "y", "z", "x", "y"],
    })


# ================================================================
# summarize_columns
# ================================================================


class TestSummarizeColumns:
    def test_returns_dataframe(self, sample_df):
        result = summarize_columns(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_column(self, sample_df):
        result = summarize_columns(sample_df)
        assert len(result) == len(sample_df.columns)

    def test_expected_output_columns(self, sample_df):
        result = summarize_columns(sample_df)
        for col in ("column", "dtype", "kind", "n_missing", "pct_missing",
                    "n_unique", "pct_unique", "mean", "std", "min", "max",
                    "skewness", "kurtosis", "top_value", "top_freq"):
            assert col in result.columns, f"missing column: {col}"

    def test_column_names_match(self, sample_df):
        result = summarize_columns(sample_df)
        assert list(result["column"]) == list(sample_df.columns)

    def test_numeric_kind(self, sample_df):
        result = summarize_columns(sample_df)
        assert result.loc[result["column"] == "age", "kind"].iloc[0] == "numeric"

    def test_categorical_kind(self, sample_df):
        result = summarize_columns(sample_df)
        assert result.loc[result["column"] == "category", "kind"].iloc[0] == "categorical"

    def test_n_missing_age(self, sample_df):
        result = summarize_columns(sample_df)
        n = result.loc[result["column"] == "age", "n_missing"].iloc[0]
        assert n == 11

    def test_pct_missing_positive(self, sample_df):
        result = summarize_columns(sample_df)
        pct = result.loc[result["column"] == "age", "pct_missing"].iloc[0]
        assert pct > 0

    def test_mean_numeric(self, sample_df):
        result = summarize_columns(sample_df)
        mean = result.loc[result["column"] == "score", "mean"].iloc[0]
        expected = float(sample_df["score"].mean())
        assert abs(mean - expected) < 0.01

    def test_mean_categorical_is_nan(self, sample_df):
        result = summarize_columns(sample_df)
        mean_val = result.loc[result["column"] == "category", "mean"].iloc[0]
        assert np.isnan(mean_val)

    def test_skewness_income_positive(self, sample_df):
        # Exponential distribution is right-skewed → positive skewness
        result = summarize_columns(sample_df)
        skew = result.loc[result["column"] == "income", "skewness"].iloc[0]
        assert skew > 0

    def test_kurtosis_numeric(self, sample_df):
        result = summarize_columns(sample_df)
        kurt = result.loc[result["column"] == "income", "kurtosis"].iloc[0]
        assert np.isfinite(kurt)

    def test_n_unique_category(self, sample_df):
        result = summarize_columns(sample_df)
        n_uniq = result.loc[result["column"] == "category", "n_unique"].iloc[0]
        assert n_uniq == 3

    def test_top_value_present(self, sample_df):
        result = summarize_columns(sample_df)
        top = result.loc[result["column"] == "category", "top_value"].iloc[0]
        assert top in ("A", "B", "C")

    def test_top_freq_positive(self, sample_df):
        result = summarize_columns(sample_df)
        freq = result.loc[result["column"] == "category", "top_freq"].iloc[0]
        assert freq > 0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = summarize_columns(df)
        assert len(result) == 1

    def test_boolean_kind(self):
        df = pd.DataFrame({"flag": pd.array([True, False, True], dtype=bool)})
        result = summarize_columns(df)
        assert result.loc[0, "kind"] == "boolean"

    def test_percentiles_present(self, sample_df):
        result = summarize_columns(sample_df)
        for col_name in ("p25", "p50", "p75"):
            assert col_name in result.columns


# ================================================================
# summarize_target
# ================================================================


class TestSummarizeTarget:
    def test_returns_dict(self, sample_df):
        result = summarize_target(sample_df, "target")
        assert isinstance(result, dict)

    def test_expected_keys(self, sample_df):
        result = summarize_target(sample_df, "target")
        for key in ("column", "dtype", "kind", "n_unique", "n_missing",
                    "value_counts", "class_distribution", "imbalance_ratio"):
            assert key in result, f"missing key: {key}"

    def test_binary_kind(self, sample_df):
        result = summarize_target(sample_df, "target")
        assert result["kind"] == "binary"

    def test_multiclass_kind(self, multiclass_df):
        result = summarize_target(multiclass_df, "label")
        assert result["kind"] == "multiclass"

    def test_continuous_kind(self, continuous_df):
        result = summarize_target(continuous_df, "y")
        assert result["kind"] == "continuous"

    def test_value_counts_sum_n_rows(self, sample_df):
        result = summarize_target(sample_df, "target")
        total = sum(result["value_counts"].values())
        assert total == len(sample_df)

    def test_class_distribution_sums_100(self, sample_df):
        result = summarize_target(sample_df, "target")
        total = sum(result["class_distribution"].values())
        assert abs(total - 100.0) < 0.1

    def test_imbalance_ratio_between_0_and_1(self, sample_df):
        result = summarize_target(sample_df, "target")
        ratio = result["imbalance_ratio"]
        assert 0 < ratio <= 1.0

    def test_n_unique_binary(self, sample_df):
        result = summarize_target(sample_df, "target")
        assert result["n_unique"] == 2

    def test_missing_target_col_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            summarize_target(sample_df, "no_such_column")

    def test_column_name_stored(self, sample_df):
        result = summarize_target(sample_df, "target")
        assert result["column"] == "target"

    def test_continuous_imbalance_ratio_none(self, continuous_df):
        result = summarize_target(continuous_df, "y")
        assert result["imbalance_ratio"] is None


# ================================================================
# summarize_missingness
# ================================================================


class TestSummarizeMissingness:
    def test_returns_dataframe(self, sample_df):
        result = summarize_missingness(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_column(self, sample_df):
        result = summarize_missingness(sample_df)
        assert len(result) == len(sample_df.columns)

    def test_expected_columns(self, sample_df):
        result = summarize_missingness(sample_df)
        for col in ("column", "n_missing", "pct_missing", "pattern"):
            assert col in result.columns

    def test_complete_pattern(self, sample_df):
        result = summarize_missingness(sample_df)
        pat = result.loc[result["column"] == "score", "pattern"].iloc[0]
        assert pat == "complete"

    def test_partial_pattern(self, sample_df):
        result = summarize_missingness(sample_df)
        pat = result.loc[result["column"] == "age", "pattern"].iloc[0]
        assert pat == "partial"

    def test_mostly_missing_pattern(self):
        df = pd.DataFrame({"sparse": [np.nan] * 90 + [1.0] * 10})
        result = summarize_missingness(df)
        assert result.loc[0, "pattern"] == "mostly_missing"

    def test_n_missing_correct(self, sample_df):
        result = summarize_missingness(sample_df)
        n = result.loc[result["column"] == "age", "n_missing"].iloc[0]
        assert n == 11

    def test_pct_missing_in_range(self, sample_df):
        result = summarize_missingness(sample_df)
        pcts = result["pct_missing"]
        assert (pcts >= 0).all() and (pcts <= 100).all()

    def test_complete_df(self, complete_df):
        result = summarize_missingness(complete_df)
        assert (result["n_missing"] == 0).all()
        assert (result["pattern"] == "complete").all()


# ================================================================
# profile_dataset — scalar stats
# ================================================================


class TestProfileDatasetStats:
    def test_returns_dict(self, sample_df):
        p = profile_dataset(sample_df)
        assert isinstance(p, dict)

    def test_n_rows(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["n_rows"] == len(sample_df)

    def test_n_cols(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["n_cols"] == len(sample_df.columns)

    def test_n_numeric(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["n_numeric"] == 4  # age, income, score, target (int)

    def test_n_categorical(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["n_categorical"] == 1  # category

    def test_n_duplicate_rows(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["n_duplicate_rows"] >= 1

    def test_pct_duplicate_rows(self, sample_df):
        p = profile_dataset(sample_df)
        assert 0 <= p["pct_duplicate_rows"] <= 100

    def test_dataset_completeness_lt_100(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["dataset_completeness_pct"] < 100

    def test_dataset_completeness_100_for_complete(self, complete_df):
        p = profile_dataset(complete_df)
        assert p["dataset_completeness_pct"] == 100.0

    def test_memory_usage_positive(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["memory_usage_mb"] > 0

    def test_columns_is_dataframe(self, sample_df):
        p = profile_dataset(sample_df)
        assert isinstance(p["columns"], pd.DataFrame)

    def test_missingness_is_dataframe(self, sample_df):
        p = profile_dataset(sample_df)
        assert isinstance(p["missingness"], pd.DataFrame)

    def test_target_none_when_not_given(self, sample_df):
        p = profile_dataset(sample_df)
        assert p["target"] is None

    def test_target_dict_when_given(self, sample_df):
        p = profile_dataset(sample_df, target_col="target")
        assert isinstance(p["target"], dict)

    def test_target_kind_binary(self, sample_df):
        p = profile_dataset(sample_df, target_col="target")
        assert p["target"]["kind"] == "binary"

    def test_required_keys(self, sample_df):
        p = profile_dataset(sample_df)
        for key in ("n_rows", "n_cols", "n_numeric", "n_categorical",
                    "n_datetime", "n_boolean", "n_duplicate_rows",
                    "pct_duplicate_rows", "dataset_completeness_pct",
                    "memory_usage_mb", "columns", "target", "missingness"):
            assert key in p, f"missing key: {key}"


# ================================================================
# profile_dataset — CSV input
# ================================================================


class TestProfileDatasetCsvInput:
    def test_accepts_csv_path(self, sample_csv):
        p = profile_dataset(str(sample_csv))
        assert p["n_rows"] > 0

    def test_accepts_path_object(self, sample_csv):
        p = profile_dataset(sample_csv)
        assert p["n_rows"] > 0

    def test_csv_and_df_same_shape(self, sample_df, sample_csv):
        p_csv = profile_dataset(sample_csv)
        p_df  = profile_dataset(sample_df)
        assert p_csv["n_rows"] == p_df["n_rows"]
        assert p_csv["n_cols"] == p_df["n_cols"]

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            profile_dataset(tmp_path / "nonexistent.csv")


# ================================================================
# profile_dataset — export outputs
# ================================================================


class TestProfileDatasetExports:
    def test_json_created(self, sample_df, tmp_path):
        profile_dataset(sample_df, output_dir=tmp_path)
        assert (tmp_path / "profile_summary.json").exists()

    def test_csv_created(self, sample_df, tmp_path):
        profile_dataset(sample_df, output_dir=tmp_path)
        assert (tmp_path / "profile_report.csv").exists()

    def test_json_valid(self, sample_df, tmp_path):
        profile_dataset(sample_df, output_dir=tmp_path)
        raw = json.loads((tmp_path / "profile_summary.json").read_text())
        assert "n_rows" in raw
        assert "columns" in raw

    def test_csv_loadable(self, sample_df, tmp_path):
        profile_dataset(sample_df, output_dir=tmp_path)
        df = pd.read_csv(tmp_path / "profile_report.csv")
        assert "column" in df.columns
        assert len(df) == len(sample_df.columns)

    def test_output_dir_created(self, sample_df, tmp_path):
        out = tmp_path / "deep" / "nested" / "dir"
        profile_dataset(sample_df, output_dir=out)
        assert out.exists()

    def test_no_export_when_output_dir_none(self, sample_df, tmp_path):
        profile_dataset(sample_df, output_dir=None)
        assert not (tmp_path / "profile_summary.json").exists()


# ================================================================
# profile_dataset — plots
# ================================================================


class TestProfileDatasetPlots:
    def test_missing_heatmap_created(self, sample_df, tmp_path):
        profile_dataset(sample_df, plots_dir=tmp_path)
        assert (tmp_path / "plot_missing_heatmap.png").exists()

    def test_class_distribution_created(self, sample_df, tmp_path):
        profile_dataset(sample_df, target_col="target", plots_dir=tmp_path)
        assert (tmp_path / "plot_class_distribution.png").exists()

    def test_class_distribution_not_created_without_target(self, sample_df, tmp_path):
        profile_dataset(sample_df, target_col=None, plots_dir=tmp_path)
        assert not (tmp_path / "plot_class_distribution.png").exists()

    def test_numeric_distributions_created(self, sample_df, tmp_path):
        profile_dataset(sample_df, plots_dir=tmp_path)
        assert (tmp_path / "plot_numeric_distributions.png").exists()

    def test_plots_dir_created(self, sample_df, tmp_path):
        plots = tmp_path / "nested_plots"
        profile_dataset(sample_df, plots_dir=plots)
        assert plots.exists()


# ================================================================
# Standalone plot smoke tests
# ================================================================


class TestProfilePlots:
    def test_plot_missing_heatmap_with_missing(self, sample_df, tmp_path):
        from src.plots import plot_missing_heatmap
        p = tmp_path / "heatmap.png"
        plot_missing_heatmap(sample_df, "Test Heatmap", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_missing_heatmap_no_missing(self, complete_df, tmp_path):
        from src.plots import plot_missing_heatmap
        p = tmp_path / "heatmap_complete.png"
        plot_missing_heatmap(complete_df, "Complete", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_class_distribution_dict(self, tmp_path):
        from src.plots import plot_class_distribution
        vc = {"Class A": 80, "Class B": 20}
        p = tmp_path / "dist.png"
        plot_class_distribution(vc, "Class Dist", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_class_distribution_series(self, tmp_path):
        from src.plots import plot_class_distribution
        vc = pd.Series({"A": 50, "B": 30, "C": 20})
        p  = tmp_path / "dist2.png"
        plot_class_distribution(vc, "Multi-class", p)
        assert p.exists()

    def test_plot_numeric_distributions(self, sample_df, tmp_path):
        from src.plots import plot_numeric_distributions
        p = tmp_path / "num_dist.png"
        plot_numeric_distributions(sample_df, "Numerics", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_numeric_distributions_no_numeric_cols(self, tmp_path):
        from src.plots import plot_numeric_distributions
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        p  = tmp_path / "no_num.png"
        plot_numeric_distributions(df, "No Numerics", p)
        assert p.exists()

    def test_plot_numeric_distributions_max_cols(self, tmp_path):
        from src.plots import plot_numeric_distributions
        df = pd.DataFrame(np.random.randn(50, 20),
                          columns=[f"f{i}" for i in range(20)])
        p  = tmp_path / "many_cols.png"
        plot_numeric_distributions(df, "Many Cols", p, max_cols=4)
        assert p.exists()


# ================================================================
# ExperimentTracker.log_profile
# ================================================================


class TestExperimentTrackerLogProfile:
    def test_log_profile_stores_data(self, tmp_path, sample_df):
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker("test_exp", base_dir=str(tmp_path), run_id="r1")
        profile = profile_dataset(sample_df)
        tracker.log_profile(profile)
        assert tracker._profile["n_rows"] == len(sample_df)

    def test_log_profile_included_in_config(self, tmp_path, sample_df):
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker("test_exp", base_dir=str(tmp_path), run_id="r2")
        profile = profile_dataset(sample_df)
        tracker.log_profile(profile)
        tracker.save()
        cfg = json.loads((tracker.run_dir / "config.json").read_text())
        assert "profile" in cfg
        assert cfg["profile"]["n_rows"] == len(sample_df)

    def test_has_profile_in_summary(self, tmp_path, sample_df):
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker("test_exp", base_dir=str(tmp_path), run_id="r3")
        profile = profile_dataset(sample_df)
        tracker.log_profile(profile)
        tracker.save()
        summ = json.loads((tracker.run_dir / "experiment_summary.json").read_text())
        assert summ.get("has_profile") is True

    def test_profile_json_written(self, tmp_path, sample_df):
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker("test_exp", base_dir=str(tmp_path), run_id="r4")
        profile = profile_dataset(sample_df)
        tracker.log_profile(profile)
        tracker.save()
        assert (tracker.run_dir / "profile_summary.json").exists()

    def test_no_profile_has_profile_false(self, tmp_path):
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker("test_exp", base_dir=str(tmp_path), run_id="r5")
        tracker.save()
        summ = json.loads((tracker.run_dir / "experiment_summary.json").read_text())
        assert summ.get("has_profile") is False

    def test_log_profile_ignores_non_dict(self, tmp_path):
        from src.experiment import ExperimentTracker
        tracker = ExperimentTracker("test_exp", base_dir=str(tmp_path), run_id="r6")
        tracker.log_profile("not a dict")   # should not raise
        assert tracker._profile == {}
