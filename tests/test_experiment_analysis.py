"""
Tests for src/experiment_analysis.py — experiment aggregation and analytics.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment_analysis import (
    aggregate_experiments,
    compare_experiments,
    export_aggregate_csv,
    export_aggregate_json,
    load_experiments,
    summarize_experiment_history,
)


# ================================================================
# FIXTURES
# ================================================================


def _make_run(base, exp_name, run_id, models_metrics, task="classification"):
    """
    Create a minimal fake run directory with metrics.csv, config.json,
    and experiment_summary.json.

    models_metrics : list of (model_name, metric_value) tuples.
    Returns the run directory Path.
    """
    run_dir = Path(base) / "experiments" / exp_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Pick primary metric from task
    metric_col = {
        "classification": "Accuracy",
        "regression":     "R2",
        "time_series":    "MAE",
        "unsupervised":   "Silhouette",
    }.get(task, "Accuracy")

    models = [m for m, _ in models_metrics]

    df = pd.DataFrame({
        "Model":    [m for m, _ in models_metrics],
        metric_col: [v for _, v in models_metrics],
    })
    df.to_csv(run_dir / "metrics.csv", index=False)

    config = {"task": task, "dataset": {"n_samples": 100}}
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    summary = {
        "experiment_name":   exp_name,
        "run_id":            run_id,
        "run_dir":           str(run_dir),
        "elapsed_seconds":   5.0,
        "dataset":           {"n_samples": 100},
        "models":            models,
        "n_models_optimized": 0,
        "has_stats_summary": False,
        "has_diagnostic_issues": False,
    }
    (run_dir / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


@pytest.fixture
def exp_tree(tmp_path):
    """
    Two experiments (exp_A and exp_B), each with two runs.
    exp_A: models RF=0.92 vs LR=0.85  (run1) and RF=0.90 vs LR=0.88 (run2)
    exp_B: models RF=0.88 vs LR=0.91  (run1) and RF=0.87 vs LR=0.89 (run2)
    """
    _make_run(tmp_path, "exp_A", "20240101_100000_aaa000",
              [("Random Forest", 0.92), ("Logistic Regression", 0.85)])
    _make_run(tmp_path, "exp_A", "20240101_110000_aaa001",
              [("Random Forest", 0.90), ("Logistic Regression", 0.88)])
    _make_run(tmp_path, "exp_B", "20240101_120000_bbb000",
              [("Random Forest", 0.88), ("Logistic Regression", 0.91)])
    _make_run(tmp_path, "exp_B", "20240101_130000_bbb001",
              [("Random Forest", 0.87), ("Logistic Regression", 0.89)])
    return tmp_path


@pytest.fixture
def single_run(tmp_path):
    _make_run(tmp_path, "solo", "20240101_080000_xyz000",
              [("RF", 0.95), ("DT", 0.80), ("KNN", 0.85)])
    return tmp_path


# ================================================================
# load_experiments
# ================================================================


class TestLoadExperiments:
    def test_returns_list(self, exp_tree):
        runs = load_experiments(exp_tree)
        assert isinstance(runs, list)

    def test_count_all_experiments(self, exp_tree):
        runs = load_experiments(exp_tree)
        assert len(runs) == 4

    def test_filter_by_experiment_name(self, exp_tree):
        runs = load_experiments(exp_tree, experiment_name="exp_A")
        assert len(runs) == 2
        assert all(r["experiment_name"] == "exp_A" for r in runs)

    def test_run_keys(self, exp_tree):
        runs = load_experiments(exp_tree)
        for key in ("experiment_name", "run_id", "run_dir", "task",
                    "metrics", "config", "elapsed_seconds"):
            assert key in runs[0], f"missing key: {key}"

    def test_metrics_dataframe(self, exp_tree):
        runs = load_experiments(exp_tree)
        for run in runs:
            assert run["metrics"] is not None
            assert isinstance(run["metrics"], pd.DataFrame)
            assert "Model" in run["metrics"].columns

    def test_task_loaded(self, exp_tree):
        runs = load_experiments(exp_tree)
        assert all(r["task"] == "classification" for r in runs)

    def test_nonexistent_base_returns_empty(self, tmp_path):
        runs = load_experiments(tmp_path / "no_such_dir")
        assert runs == []

    def test_missing_experiment_name_returns_empty(self, exp_tree):
        runs = load_experiments(exp_tree, experiment_name="does_not_exist")
        assert runs == []

    def test_corrupt_summary_skipped(self, tmp_path):
        run_dir = tmp_path / "experiments" / "test" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "experiment_summary.json").write_text("NOT JSON", encoding="utf-8")
        runs = load_experiments(tmp_path)
        assert runs == []

    def test_sorted_by_run_id(self, exp_tree):
        runs = load_experiments(exp_tree, experiment_name="exp_A")
        ids = [r["run_id"] for r in runs]
        assert ids == sorted(ids)


# ================================================================
# aggregate_experiments
# ================================================================


class TestAggregateExperiments:
    def test_returns_dataframe(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        assert isinstance(agg, pd.DataFrame)

    def test_has_expected_columns(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        for col in ("Model", "mean_Accuracy", "std_Accuracy",
                    "win_count", "win_pct", "avg_rank", "n_runs"):
            assert col in agg.columns, f"missing column: {col}"

    def test_one_row_per_model(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        assert len(agg) == 2  # Random Forest + Logistic Regression

    def test_n_runs_correct(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        assert agg["n_runs"].sum() == 8  # 2 models × 4 runs each

    def test_win_count_sums_to_n_valid_runs(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        assert agg["win_count"].sum() == 4  # exactly one winner per run

    def test_avg_rank_between_1_and_n_models(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        assert (agg["avg_rank"] >= 1).all()
        assert (agg["avg_rank"] <= 2).all()

    def test_explicit_metric_parameter(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs, metric="Accuracy")
        assert "mean_Accuracy" in agg.columns

    def test_empty_runs_returns_empty(self):
        agg = aggregate_experiments([])
        assert isinstance(agg, pd.DataFrame)
        assert agg.empty

    def test_sorted_by_avg_rank(self, exp_tree):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        ranks = list(agg["avg_rank"])
        assert ranks == sorted(ranks)

    def test_mean_values_are_float(self, single_run):
        runs = load_experiments(single_run)
        agg  = aggregate_experiments(runs)
        assert agg["mean_Accuracy"].dtype in (float, np.float64)

    def test_std_zero_for_single_run(self, single_run):
        runs = load_experiments(single_run)
        agg  = aggregate_experiments(runs)
        # Only one run → ddof=1 → std = 0.0
        assert (agg["std_Accuracy"] == 0.0).all()

    def test_lower_is_better_metric(self, tmp_path):
        _make_run(tmp_path, "ts_exp", "20240101_000000_t000",
                  [("Ridge", 0.05), ("Lasso", 0.12)], task="time_series")
        runs = load_experiments(tmp_path)
        agg  = aggregate_experiments(runs, metric="MAE")
        # Ridge has lower MAE → should rank #1 (avg_rank=1)
        best = agg.iloc[0]["Model"]
        assert best == "Ridge"


# ================================================================
# compare_experiments
# ================================================================


class TestCompareExperiments:
    def test_returns_dataframe(self, exp_tree):
        runs = load_experiments(exp_tree)
        cmp  = compare_experiments(runs)
        assert isinstance(cmp, pd.DataFrame)

    def test_index_is_experiment_name(self, exp_tree):
        runs = load_experiments(exp_tree)
        cmp  = compare_experiments(runs)
        assert cmp.index.name == "experiment_name"

    def test_two_experiment_rows(self, exp_tree):
        runs = load_experiments(exp_tree)
        cmp  = compare_experiments(runs)
        assert len(cmp) == 2

    def test_two_model_columns(self, exp_tree):
        runs = load_experiments(exp_tree)
        cmp  = compare_experiments(runs)
        assert len(cmp.columns) == 2

    def test_values_are_means(self, exp_tree):
        runs = load_experiments(exp_tree, experiment_name="exp_A")
        cmp  = compare_experiments(runs, metric="Accuracy")
        rf_mean = cmp.loc["exp_A", "Random Forest"]
        expected = round((0.92 + 0.90) / 2, 4)
        assert abs(rf_mean - expected) < 1e-4

    def test_empty_runs_returns_empty(self):
        cmp = compare_experiments([])
        assert isinstance(cmp, pd.DataFrame)
        assert cmp.empty

    def test_group_by_run_id(self, exp_tree):
        runs = load_experiments(exp_tree, experiment_name="exp_A")
        cmp  = compare_experiments(runs, metric="Accuracy", group_by="run_id")
        assert cmp.index.name == "run_id"
        assert len(cmp) == 2


# ================================================================
# summarize_experiment_history
# ================================================================


class TestSummarizeExperimentHistory:
    def test_returns_dict(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        assert isinstance(s, dict)

    def test_required_keys(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        for key in ("runs", "aggregate", "comparison", "n_runs",
                    "n_experiments", "experiment_names", "metric",
                    "best_model", "timeline"):
            assert key in s, f"missing key: {key}"

    def test_n_runs(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        assert s["n_runs"] == 4

    def test_n_experiments(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        assert s["n_experiments"] == 2

    def test_best_model_is_string(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        assert isinstance(s["best_model"], str)

    def test_timeline_is_dataframe(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        assert isinstance(s["timeline"], pd.DataFrame)

    def test_timeline_columns(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        tl = s["timeline"]
        assert "run_id" in tl.columns
        assert "best_model" in tl.columns
        assert "best_score" in tl.columns

    def test_timeline_sorted_by_run_id(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        tl = s["timeline"]
        ids = list(tl["run_id"])
        assert ids == sorted(ids)

    def test_metric_inferred(self, exp_tree):
        s = summarize_experiment_history(exp_tree)
        assert s["metric"] == "Accuracy"

    def test_metric_explicit(self, exp_tree):
        s = summarize_experiment_history(exp_tree, metric="Accuracy")
        assert s["metric"] == "Accuracy"

    def test_filter_experiment_name(self, exp_tree):
        s = summarize_experiment_history(exp_tree, experiment_name="exp_A")
        assert s["n_runs"] == 2

    def test_empty_base_dir(self, tmp_path):
        s = summarize_experiment_history(tmp_path)
        assert s["n_runs"] == 0
        assert s["best_model"] is None


# ================================================================
# export_aggregate_csv
# ================================================================


class TestExportAggregateCsv:
    def test_file_created(self, exp_tree, tmp_path):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        out  = tmp_path / "out.csv"
        export_aggregate_csv(agg, out)
        assert out.exists()

    def test_readable(self, exp_tree, tmp_path):
        runs = load_experiments(exp_tree)
        agg  = aggregate_experiments(runs)
        out  = tmp_path / "out.csv"
        export_aggregate_csv(agg, out)
        reloaded = pd.read_csv(out)
        assert "Model" in reloaded.columns
        assert len(reloaded) == len(agg)


# ================================================================
# export_aggregate_json
# ================================================================


class TestExportAggregateJson:
    def test_file_created(self, exp_tree, tmp_path):
        s   = summarize_experiment_history(exp_tree)
        out = tmp_path / "summary.json"
        export_aggregate_json(s, out)
        assert out.exists()

    def test_valid_json(self, exp_tree, tmp_path):
        s   = summarize_experiment_history(exp_tree)
        out = tmp_path / "summary.json"
        export_aggregate_json(s, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "n_runs" in payload
        assert "aggregate" in payload
        assert "timeline" in payload

    def test_n_runs_matches(self, exp_tree, tmp_path):
        s   = summarize_experiment_history(exp_tree)
        out = tmp_path / "summary.json"
        export_aggregate_json(s, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["n_runs"] == 4


# ================================================================
# summarize_experiment_history — export_dir / plots_dir options
# ================================================================


class TestSummarizeWithExport:
    def test_export_dir_creates_csv(self, exp_tree, tmp_path):
        out = tmp_path / "analysis"
        summarize_experiment_history(exp_tree, export_dir=out)
        assert (out / "aggregated_results.csv").exists()

    def test_export_dir_creates_json(self, exp_tree, tmp_path):
        out = tmp_path / "analysis"
        summarize_experiment_history(exp_tree, export_dir=out)
        assert (out / "aggregated_summary.json").exists()

    def test_plots_dir_creates_pngs(self, exp_tree, tmp_path):
        plots = tmp_path / "plots"
        summarize_experiment_history(exp_tree, plots_dir=plots)
        png_files = list(plots.glob("*.png"))
        assert len(png_files) >= 2  # at least win_frequency and avg_rank


# ================================================================
# Plots — smoke tests (no assertions on pixel content)
# ================================================================


class TestAnalysisPlots:
    def test_plot_model_win_frequency(self, exp_tree, tmp_path):
        from src.plots import plot_model_win_frequency
        win = {"RF": 3, "LR": 1}
        p = tmp_path / "win.png"
        plot_model_win_frequency(win, "Win Freq", p)
        assert p.exists()

    def test_plot_average_rank(self, exp_tree, tmp_path):
        from src.plots import plot_average_rank
        rank = pd.Series({"RF": 1.2, "LR": 1.8})
        p = tmp_path / "rank.png"
        plot_average_rank(rank, "Avg Rank", p)
        assert p.exists()

    def test_plot_metric_distribution(self, tmp_path):
        from src.plots import plot_metric_distribution
        dist = pd.DataFrame({"RF": [0.9, 0.88, 0.91], "LR": [0.82, 0.85, 0.84]})
        p = tmp_path / "dist.png"
        plot_metric_distribution(dist, "Accuracy", "Distribution", p)
        assert p.exists()

    def test_plot_experiment_timeline(self, exp_tree, tmp_path):
        from src.plots import plot_experiment_timeline
        tl = pd.DataFrame({
            "run_id": ["r1", "r2", "r3"],
            "experiment_name": ["exp_A", "exp_A", "exp_B"],
            "best_model": ["RF", "RF", "LR"],
            "best_score": [0.92, 0.90, 0.91],
            "run_index": [0, 1, 2],
        })
        p = tmp_path / "timeline.png"
        plot_experiment_timeline(tl, "Accuracy", "Timeline", p)
        assert p.exists()


# ================================================================
# export_analysis_word
# ================================================================


class TestExportAnalysisWord:
    def test_creates_docx(self, exp_tree, tmp_path):
        from src.reporting import export_analysis_word
        s = summarize_experiment_history(exp_tree)
        p = tmp_path / "analysis.docx"
        export_analysis_word(s, p)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_empty_summary_creates_docx(self, tmp_path):
        from src.reporting import export_analysis_word
        s = summarize_experiment_history(tmp_path)
        p = tmp_path / "empty.docx"
        export_analysis_word(s, p)
        assert p.exists()
