"""
Tests for the ml-benchmark CLI (src/cli.py).

Strategy
--------
- Parser tests: exercise _build_parser() directly — no I/O, no ML.
- Command output tests: call handler functions with a mock Namespace and
  capture stdout/stderr via capsys.
- Integration smoke test: run `python -m src.cli <args>` via subprocess on
  a tiny CSV to verify the end-to-end path works without crashing.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.cli import _build_parser, _cmd_examples, _cmd_info, _cmd_version, main


# ================================================================
# HELPERS
# ================================================================

def _parse(argv):
    """Parse a list of CLI tokens and return the Namespace."""
    return _build_parser().parse_args(argv)


# ================================================================
# _build_parser — structural checks
# ================================================================

class TestBuildParser:
    def test_run_command_exists(self):
        args = _parse(["run", "data.csv", "target"])
        assert args.command == "run"

    def test_info_command_exists(self):
        args = _parse(["info"])
        assert args.command == "info"

    def test_examples_command_exists(self):
        args = _parse(["examples"])
        assert args.command == "examples"

    def test_version_command_exists(self):
        args = _parse(["version"])
        assert args.command == "version"

    def test_missing_command_exits(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_unknown_command_exits(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["frobnicate"])


# ================================================================
# run subcommand — argument parsing
# ================================================================

class TestRunArgParsing:
    def test_positional_csv_and_target(self):
        args = _parse(["run", "my_data.csv", "label"])
        assert args.csv_path == "my_data.csv"
        assert args.target_col == "label"

    def test_default_task_is_classification(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.task == "classification"

    def test_task_classification(self):
        args = _parse(["run", "d.csv", "y", "--task", "classification"])
        assert args.task == "classification"

    def test_task_regression(self):
        args = _parse(["run", "d.csv", "y", "--task", "regression"])
        assert args.task == "regression"

    def test_task_unsupervised(self):
        args = _parse(["run", "d.csv", "y", "--task", "unsupervised"])
        assert args.task == "unsupervised"

    def test_task_time_series(self):
        args = _parse(["run", "d.csv", "y", "--task", "time_series"])
        assert args.task == "time_series"

    def test_invalid_task_exits(self):
        with pytest.raises(SystemExit):
            _parse(["run", "d.csv", "y", "--task", "deep_learning"])

    def test_default_output_dir(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.output_dir == "outputs"

    def test_custom_output_dir(self):
        args = _parse(["run", "d.csv", "y", "--output-dir", "/tmp/out"])
        assert args.output_dir == "/tmp/out"

    def test_default_test_size(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.test_size == pytest.approx(0.2)

    def test_custom_test_size(self):
        args = _parse(["run", "d.csv", "y", "--test-size", "0.3"])
        assert args.test_size == pytest.approx(0.3)

    def test_default_random_state(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.random_state == 42

    def test_custom_random_state(self):
        args = _parse(["run", "d.csv", "y", "--random-state", "0"])
        assert args.random_state == 0

    def test_export_flag(self):
        args = _parse(["run", "d.csv", "y", "--export", "csv,word"])
        assert args.export_formats == "csv,word"

    def test_imbalance_flag(self):
        args = _parse(["run", "d.csv", "y", "--imbalance", "smote"])
        assert args.imbalance == "smote"

    def test_cv_strategy_flag(self):
        args = _parse(["run", "d.csv", "y", "--cv-strategy", "stratified_kfold"])
        assert args.cv_strategy == "stratified_kfold"

    def test_experiment_name_flag(self):
        args = _parse(["run", "d.csv", "y", "--experiment-name", "exp01"])
        assert args.experiment_name == "exp01"

    def test_optimize_flag_default_false(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.optimize is False

    def test_optimize_flag_true(self):
        args = _parse(["run", "d.csv", "y", "--optimize"])
        assert args.optimize is True

    def test_optimization_method_default(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.optimization_method == "random"

    def test_optimization_method_grid(self):
        args = _parse(["run", "d.csv", "y", "--optimization-method", "grid"])
        assert args.optimization_method == "grid"

    def test_invalid_optimization_method_exits(self):
        with pytest.raises(SystemExit):
            _parse(["run", "d.csv", "y", "--optimization-method", "bayesian"])

    def test_n_iter_default(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.n_iter == 20

    def test_n_iter_custom(self):
        args = _parse(["run", "d.csv", "y", "--n-iter", "5"])
        assert args.n_iter == 5

    def test_compute_stats_default_false(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.compute_stats is False

    def test_compute_stats_flag_true(self):
        args = _parse(["run", "d.csv", "y", "--compute-stats"])
        assert args.compute_stats is True

    def test_lags_flag(self):
        args = _parse(["run", "d.csv", "y", "--lags", "1,2,3"])
        assert args.lags == "1,2,3"

    def test_ts_n_splits_default(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.ts_n_splits == 5

    def test_ts_n_splits_custom(self):
        args = _parse(["run", "d.csv", "y", "--ts-n-splits", "4"])
        assert args.ts_n_splits == 4

    def test_ts_horizon_default_none(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.ts_horizon is None

    def test_ts_horizon_custom(self):
        args = _parse(["run", "d.csv", "y", "--ts-horizon", "10"])
        assert args.ts_horizon == 10

    def test_quiet_flag_default_false(self):
        args = _parse(["run", "d.csv", "y"])
        assert args.quiet is False

    def test_quiet_flag_true(self):
        args = _parse(["run", "d.csv", "y", "--quiet"])
        assert args.quiet is True

    def test_report_title_default(self):
        args = _parse(["run", "d.csv", "y"])
        assert "Benchmark" in args.report_title

    def test_report_title_custom(self):
        args = _parse(["run", "d.csv", "y", "--report-title", "My Report"])
        assert args.report_title == "My Report"

    def test_func_is_callable(self):
        args = _parse(["run", "d.csv", "y"])
        assert callable(args.func)


# ================================================================
# _cmd_version
# ================================================================

class TestCmdVersion:
    def test_prints_version(self, capsys):
        from src import __version__
        _cmd_version(None)
        out = capsys.readouterr().out
        assert __version__ in out

    def test_prints_package_name(self, capsys):
        _cmd_version(None)
        out = capsys.readouterr().out
        assert "ml-benchmark-lab" in out


# ================================================================
# _cmd_info
# ================================================================

class TestCmdInfo:
    def test_prints_version(self, capsys):
        from src import __version__
        _cmd_info(None)
        out = capsys.readouterr().out
        assert __version__ in out

    def test_lists_tasks(self, capsys):
        _cmd_info(None)
        out = capsys.readouterr().out
        for task in ("classification", "regression", "unsupervised", "time_series"):
            assert task in out

    def test_lists_export_formats(self, capsys):
        _cmd_info(None)
        out = capsys.readouterr().out
        assert "csv" in out
        assert "excel" in out
        assert "word" in out

    def test_lists_optimization_methods(self, capsys):
        _cmd_info(None)
        out = capsys.readouterr().out
        assert "random" in out
        assert "grid" in out

    def test_lists_model_counts(self, capsys):
        _cmd_info(None)
        out = capsys.readouterr().out
        # Each task row should show a count in parentheses
        assert "(8)" in out or "(9)" in out or "(5)" in out or "(3)" in out


# ================================================================
# _cmd_examples
# ================================================================

class TestCmdExamples:
    def test_lists_example_names(self, capsys):
        _cmd_examples(None)
        out = capsys.readouterr().out
        assert "classification_example.py" in out

    def test_lists_multiple_examples(self, capsys):
        _cmd_examples(None)
        out = capsys.readouterr().out
        for name in (
            "classification_example.py",
            "regression_example.py",
            "time_series_example.py",
        ):
            assert name in out

    def test_shows_how_to_run(self, capsys):
        _cmd_examples(None)
        out = capsys.readouterr().out
        assert "python examples/" in out

    def test_mentions_outputs_dir(self, capsys):
        _cmd_examples(None)
        out = capsys.readouterr().out
        assert "outputs" in out.lower()


# ================================================================
# _cmd_run — behaviour with mocked run_benchmark
# ================================================================

class TestCmdRun:
    def _make_args(self, **overrides):
        defaults = dict(
            csv_path="d.csv",
            target_col="y",
            task="classification",
            output_dir="outputs",
            test_size=0.2,
            random_state=42,
            imbalance=None,
            cv_strategy=None,
            export_formats=None,
            report_title="Benchmark Report",
            experiment_name=None,
            optimize=False,
            optimization_method="random",
            n_iter=20,
            compute_stats=False,
            lags=None,
            ts_n_splits=5,
            ts_horizon=None,
            quiet=False,
        )
        defaults.update(overrides)
        ns = MagicMock()
        for k, v in defaults.items():
            setattr(ns, k, v)
        return ns

    def test_calls_run_benchmark(self):
        from src.cli import _cmd_run
        args = self._make_args()
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        mock_rb.assert_called_once()

    def test_passes_task(self):
        from src.cli import _cmd_run
        args = self._make_args(task="regression")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["task"] == "regression"

    def test_passes_output_dir(self):
        from src.cli import _cmd_run
        args = self._make_args(output_dir="/tmp/my_outputs")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["output_dir"] == "/tmp/my_outputs"

    def test_passes_experiment_name(self):
        from src.cli import _cmd_run
        args = self._make_args(experiment_name="exp01")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["experiment_name"] == "exp01"

    def test_experiment_name_none_when_empty_string(self):
        from src.cli import _cmd_run
        args = self._make_args(experiment_name="")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["experiment_name"] is None

    def test_passes_optimize_true(self):
        from src.cli import _cmd_run
        args = self._make_args(optimize=True)
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["optimize"] is True

    def test_passes_compute_stats(self):
        from src.cli import _cmd_run
        args = self._make_args(compute_stats=True)
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["compute_stats"] is True

    def test_export_formats_parsed_from_string(self):
        from src.cli import _cmd_run
        args = self._make_args(export_formats="csv,word")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["export_formats"] == ["csv", "word"]

    def test_export_formats_none_when_not_set(self):
        from src.cli import _cmd_run
        args = self._make_args(export_formats=None)
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["export_formats"] is None

    def test_lags_parsed_for_time_series(self):
        from src.cli import _cmd_run
        args = self._make_args(task="time_series", lags="1,2,3")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs.get("lags") == [1, 2, 3]

    def test_lags_not_passed_for_non_ts_task(self):
        from src.cli import _cmd_run
        args = self._make_args(task="classification", lags="1,2,3")
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert "lags" not in kwargs

    def test_quiet_sets_verbose_false(self):
        from src.cli import _cmd_run
        args = self._make_args(quiet=True)
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["verbose"] is False

    def test_not_quiet_sets_verbose_true(self):
        from src.cli import _cmd_run
        args = self._make_args(quiet=False)
        with patch("src.benchmark.run_benchmark") as mock_rb:
            mock_rb.return_value = (pd.DataFrame(), {})
            _cmd_run(args)
        _, kwargs = mock_rb.call_args
        assert kwargs["verbose"] is True

    def test_invalid_lags_exits(self, capsys):
        from src.cli import _cmd_run
        args = self._make_args(lags="a,b,c")
        with pytest.raises(SystemExit) as exc_info:
            _cmd_run(args)
        assert exc_info.value.code == 1

    def test_run_benchmark_exception_exits_with_1(self):
        from src.cli import _cmd_run
        args = self._make_args()
        with patch("src.benchmark.run_benchmark", side_effect=RuntimeError("boom")):
            with pytest.raises(SystemExit) as exc_info:
                _cmd_run(args)
        assert exc_info.value.code == 1

    def test_file_not_found_exits_with_1(self):
        from src.cli import _cmd_run
        args = self._make_args()
        with patch("src.benchmark.run_benchmark",
                   side_effect=FileNotFoundError("no such file")):
            with pytest.raises(SystemExit) as exc_info:
                _cmd_run(args)
        assert exc_info.value.code == 1

    def test_not_implemented_error_exits_with_1(self):
        from src.cli import _cmd_run
        args = self._make_args(task="classification")
        with patch("src.benchmark.run_benchmark",
                   side_effect=NotImplementedError("not supported")):
            with pytest.raises(SystemExit) as exc_info:
                _cmd_run(args)
        assert exc_info.value.code == 1


# ================================================================
# main() — dispatch via sys.argv
# ================================================================

class TestMainDispatch:
    def test_version_dispatch(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["ml-benchmark", "version"])
        main()
        out = capsys.readouterr().out
        assert "ml-benchmark-lab" in out

    def test_info_dispatch(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["ml-benchmark", "info"])
        main()
        out = capsys.readouterr().out
        assert "classification" in out

    def test_examples_dispatch(self, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["ml-benchmark", "examples"])
        main()
        out = capsys.readouterr().out
        assert "classification_example.py" in out


# ================================================================
# Smoke integration test — subprocess with tiny real CSV
# ================================================================

class TestCLIIntegration:
    """Run the CLI as a subprocess against a real tiny CSV."""

    @pytest.fixture
    def tiny_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 60
        df = pd.DataFrame(
            rng.standard_normal((n, 4)),
            columns=["f0", "f1", "f2", "f3"],
        )
        df["target"] = (df["f0"] + df["f1"] > 0).astype(int)
        p = tmp_path / "tiny.csv"
        df.to_csv(p, index=False)
        return str(p)

    def _run_cli(self, *args, cwd=None):
        return subprocess.run(
            [sys.executable, "-m", "src.cli", *args],
            capture_output=True, text=True,
            cwd=cwd or str(Path(__file__).parent.parent),
        )

    def test_version_subprocess(self):
        result = self._run_cli("version")
        assert result.returncode == 0
        assert "ml-benchmark-lab" in result.stdout

    def test_info_subprocess(self):
        result = self._run_cli("info")
        assert result.returncode == 0
        assert "classification" in result.stdout

    def test_examples_subprocess(self):
        result = self._run_cli("examples")
        assert result.returncode == 0
        assert "classification_example.py" in result.stdout

    def test_run_classification(self, tiny_csv, tmp_path):
        result = self._run_cli(
            "run", tiny_csv, "target",
            "--task", "classification",
            "--output-dir", str(tmp_path / "out"),
            "--quiet",
        )
        assert result.returncode == 0

    def test_run_regression(self, tiny_csv, tmp_path):
        result = self._run_cli(
            "run", tiny_csv, "f3",
            "--task", "regression",
            "--output-dir", str(tmp_path / "out2"),
            "--quiet",
        )
        assert result.returncode == 0

    def test_run_with_export_csv(self, tiny_csv, tmp_path):
        outdir = tmp_path / "out3"
        result = self._run_cli(
            "run", tiny_csv, "target",
            "--task", "classification",
            "--output-dir", str(outdir),
            "--export", "csv",
            "--quiet",
        )
        assert result.returncode == 0
        assert (outdir / "results.csv").exists()

    def test_run_with_compute_stats(self, tiny_csv, tmp_path):
        result = self._run_cli(
            "run", tiny_csv, "target",
            "--task", "classification",
            "--output-dir", str(tmp_path / "out4"),
            "--compute-stats",
            "--quiet",
        )
        assert result.returncode == 0

    def test_run_missing_file_nonzero_exit(self, tmp_path):
        result = self._run_cli(
            "run", "/nonexistent/path/data.csv", "target",
            "--output-dir", str(tmp_path),
        )
        assert result.returncode != 0

    def test_no_args_nonzero_exit(self):
        result = self._run_cli()
        assert result.returncode != 0
