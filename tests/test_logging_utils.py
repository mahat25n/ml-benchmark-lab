"""
Tests for logging utilities and diagnostics (src/logging_utils.py).
"""

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.logging_utils import (
    Timer,
    capture_warnings,
    get_logger,
    run_diagnostics,
)


# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture
def clean_arrays():
    """Small, clean train/test arrays with no issues."""
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((80, 4))
    X_te = rng.standard_normal((20, 4))
    y_tr = rng.integers(0, 2, 80).astype(float)
    y_te = rng.integers(0, 2, 20).astype(float)
    return X_tr, X_te, y_tr, y_te


@pytest.fixture
def feature_names():
    return ["age", "income", "score", "days"]


# ================================================================
# get_logger
# ================================================================

class TestGetLogger:
    def test_returns_logger(self):
        lg = get_logger("test_rl_basic")
        assert isinstance(lg, logging.Logger)

    def test_name_is_set(self):
        lg = get_logger("test_rl_name")
        assert lg.name == "test_rl_name"

    def test_default_level_info(self):
        lg = get_logger("test_rl_lvl_default")
        assert lg.level == logging.INFO

    def test_level_debug(self):
        lg = get_logger("test_rl_lvl_debug", level="debug")
        assert lg.level == logging.DEBUG

    def test_level_warning(self):
        lg = get_logger("test_rl_lvl_warn", level="warning")
        assert lg.level == logging.WARNING

    def test_console_true_adds_stream_handler(self):
        lg = get_logger("test_rl_con_true", console=True)
        assert any(isinstance(h, logging.StreamHandler) for h in lg.handlers)

    def test_console_false_no_stream_handler(self):
        lg = get_logger("test_rl_con_false", console=False)
        assert not any(isinstance(h, logging.StreamHandler) for h in lg.handlers)

    def test_log_file_adds_file_handler(self, tmp_path):
        log_file = tmp_path / "test.log"
        lg = get_logger("test_rl_file", log_file=log_file, console=False)
        assert any(isinstance(h, logging.FileHandler) for h in lg.handlers)

    def test_log_file_creates_parent_dirs(self, tmp_path):
        log_file = tmp_path / "subdir" / "nested" / "test.log"
        get_logger("test_rl_dirs", log_file=log_file, console=False)
        assert log_file.parent.exists()

    def test_repeated_calls_clear_handlers(self):
        name = "test_rl_clear"
        get_logger(name, console=True)
        get_logger(name, console=True)
        lg = get_logger(name, console=True)
        stream_handlers = [h for h in lg.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) == 1

    def test_propagate_false(self):
        lg = get_logger("test_rl_prop")
        assert lg.propagate is False

    def test_writes_to_file(self, tmp_path):
        log_file = tmp_path / "out.log"
        lg = get_logger("test_rl_write", log_file=log_file, console=False)
        lg.info("hello from test")
        content = log_file.read_text(encoding="utf-8")
        assert "hello from test" in content

    def test_custom_fmt(self, tmp_path):
        log_file = tmp_path / "fmt.log"
        lg = get_logger("test_rl_fmt", log_file=log_file, console=False,
                        fmt="CUSTOM %(message)s")
        lg.info("my_message")
        content = log_file.read_text(encoding="utf-8")
        assert "CUSTOM my_message" in content


# ================================================================
# Timer
# ================================================================

class TestTimer:
    def test_context_manager_elapsed_positive(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed > 0

    def test_context_manager_elapsed_reasonable(self):
        with Timer() as t:
            time.sleep(0.02)
        assert 0.01 < t.elapsed < 2.0

    def test_standalone_start_stop(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        t.stop()
        assert t.elapsed > 0

    def test_elapsed_after_stop_is_fixed(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        t.stop()
        e1 = t.elapsed
        time.sleep(0.02)
        e2 = t.elapsed
        assert e1 == e2

    def test_elapsed_while_running_increases(self):
        t = Timer()
        t.start()
        e1 = t.elapsed
        time.sleep(0.02)
        e2 = t.elapsed
        assert e2 > e1

    def test_stop_before_start_raises(self):
        t = Timer()
        with pytest.raises(RuntimeError):
            t.stop()

    def test_elapsed_before_start_raises(self):
        t = Timer()
        with pytest.raises(RuntimeError):
            _ = t.elapsed

    def test_repr_not_started(self):
        t = Timer()
        assert "not started" in repr(t)

    def test_repr_after_start(self):
        with Timer() as t:
            pass
        assert "elapsed" in repr(t)

    def test_restart(self):
        t = Timer()
        t.start()
        time.sleep(0.01)
        t.stop()
        e1 = t.elapsed
        t.start()
        time.sleep(0.01)
        t.stop()
        e2 = t.elapsed
        assert abs(e1 - e2) < 0.5


# ================================================================
# capture_warnings
# ================================================================

class TestCaptureWarnings:
    """
    capture_warnings replaces warnings.showwarning; pytest may intercept
    warnings.warn() before showwarning is reached. Tests call showwarning
    directly to verify the redirect contract without fighting pytest's
    own warning capture machinery.
    """

    def _make_capturing_logger(self, name):
        """Return (logger, records_list) with a list-based handler."""
        captured = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())

        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.handlers.clear()
        lg.propagate = False
        lg.addHandler(_ListHandler())
        return lg, captured

    def test_showwarning_redirected_to_logger(self):
        lg, captured = self._make_capturing_logger("test_cw_basic")
        with capture_warnings(lg):
            warnings.showwarning("test warning message", UserWarning, "f.py", 1)
        assert any("test warning message" in msg for msg in captured)

    def test_original_showwarning_restored(self):
        lg = get_logger("test_cw_restore", console=False)
        original = warnings.showwarning
        with capture_warnings(lg):
            pass
        assert warnings.showwarning is original

    def test_original_restored_on_exception(self):
        lg = get_logger("test_cw_exc", console=False)
        original = warnings.showwarning
        try:
            with capture_warnings(lg):
                raise ValueError("boom")
        except ValueError:
            pass
        assert warnings.showwarning is original

    def test_multiple_warnings_captured(self):
        lg, captured = self._make_capturing_logger("test_cw_multi")
        with capture_warnings(lg):
            warnings.showwarning("first",  UserWarning, "f.py", 1)
            warnings.showwarning("second", DeprecationWarning, "f.py", 2)
        assert any("first" in msg  for msg in captured)
        assert any("second" in msg for msg in captured)

    def test_category_name_in_output(self):
        lg, captured = self._make_capturing_logger("test_cw_cat")
        with capture_warnings(lg):
            warnings.showwarning("something", RuntimeWarning, "f.py", 1)
        assert any("RuntimeWarning" in msg for msg in captured)


# ================================================================
# run_diagnostics
# ================================================================

class TestRunDiagnosticsReturnShape:
    def test_returns_dict(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert isinstance(result, dict)

    def test_has_all_keys(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        expected = {
            "has_issues", "issue_summary", "missing",
            "constant_cols", "duplicate_cols", "class_imbalance",
            "leakage_suspects", "train_test_mismatch",
        }
        assert expected == set(result.keys())

    def test_clean_data_no_issues(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert result["has_issues"] is False
        assert result["issue_summary"] == []

    def test_issue_summary_is_list(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert isinstance(result["issue_summary"], list)


class TestRunDiagnosticsMissing:
    def test_missing_in_X_train_detected(self):
        rng = np.random.default_rng(1)
        X_tr = rng.standard_normal((60, 3))
        X_tr[0, 0] = np.nan
        X_te = rng.standard_normal((20, 3))
        y_tr = rng.integers(0, 2, 60).astype(float)
        y_te = rng.integers(0, 2, 20).astype(float)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert result["has_issues"] is True
        assert result["missing"]["X_train"]["count"] == 1

    def test_no_missing_flag_when_fraction_zero(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, missing_threshold=0.0)
        assert result["missing"]["X_train"]["count"] == 0

    def test_missing_counts_correct(self):
        rng = np.random.default_rng(2)
        X_tr = rng.standard_normal((50, 2))
        X_tr[:5, 0] = np.nan   # 5 missing in col 0
        X_te = rng.standard_normal((10, 2))
        y_tr = rng.standard_normal(50)
        y_te = rng.standard_normal(10)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, task="regression")
        assert result["missing"]["X_train"]["count"] == 5


class TestRunDiagnosticsConstantCols:
    def test_constant_column_detected(self, feature_names):
        rng = np.random.default_rng(3)
        X_tr = rng.standard_normal((60, 4))
        X_tr[:, 2] = 5.0   # constant column at index 2
        X_te = rng.standard_normal((20, 4))
        y_tr = rng.integers(0, 2, 60).astype(float)
        y_te = rng.integers(0, 2, 20).astype(float)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, feature_names=feature_names)
        assert "score" in result["constant_cols"]
        assert result["has_issues"] is True

    def test_no_constant_cols_for_clean_data(self, clean_arrays, feature_names):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, feature_names=feature_names)
        assert result["constant_cols"] == []


class TestRunDiagnosticsDuplicateCols:
    def test_duplicate_column_detected(self, feature_names):
        rng = np.random.default_rng(4)
        X_tr = rng.standard_normal((60, 4))
        X_tr[:, 3] = X_tr[:, 1]   # col 3 is duplicate of col 1
        X_te = rng.standard_normal((20, 4))
        y_tr = rng.integers(0, 2, 60).astype(float)
        y_te = rng.integers(0, 2, 20).astype(float)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, feature_names=feature_names)
        assert len(result["duplicate_cols"]) > 0
        assert ("income", "days") in result["duplicate_cols"]

    def test_no_duplicates_for_clean_data(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert result["duplicate_cols"] == []


class TestRunDiagnosticsClassImbalance:
    def test_imbalance_computed_for_classification(self):
        rng = np.random.default_rng(5)
        X_tr = rng.standard_normal((100, 3))
        X_te = rng.standard_normal((20, 3))
        y_tr = np.array([0] * 95 + [1] * 5)   # 5:95 = 0.052 ratio
        y_te = rng.integers(0, 2, 20).astype(float)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, task="classification")
        assert result["class_imbalance"] is not None
        assert result["class_imbalance"]["minority_count"] == 5
        assert result["class_imbalance"]["majority_count"] == 95
        assert result["has_issues"] is True  # ratio 0.052 < 0.1

    def test_balanced_data_no_imbalance_issue(self):
        rng = np.random.default_rng(6)
        X_tr = rng.standard_normal((100, 3))
        X_te = rng.standard_normal((20, 3))
        y_tr = np.array([0] * 50 + [1] * 50)
        y_te = rng.integers(0, 2, 20).astype(float)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, task="classification")
        assert result["class_imbalance"]["ratio"] == 1.0
        assert not any("imbalance" in s.lower() for s in result["issue_summary"])

    def test_imbalance_none_for_regression(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, task="regression")
        assert result["class_imbalance"] is None


class TestRunDiagnosticsLeakage:
    def test_perfect_correlation_flagged(self):
        rng = np.random.default_rng(7)
        n = 80
        y_tr = rng.standard_normal(n)
        # First column perfectly predicts target
        X_tr = np.column_stack([y_tr, rng.standard_normal((n, 2))])
        X_te = rng.standard_normal((20, 3))
        y_te = rng.standard_normal(20)
        result = run_diagnostics(
            X_tr, X_te, y_tr, y_te,
            feature_names=["leak", "f1", "f2"],
            task="regression",
        )
        assert "leak" in result["leakage_suspects"]
        assert result["has_issues"] is True

    def test_no_leakage_for_unrelated_features(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert result["leakage_suspects"] == []

    def test_custom_leakage_threshold(self):
        rng = np.random.default_rng(8)
        n = 80
        y_tr = rng.standard_normal(n)
        noise = rng.standard_normal(n) * 0.1
        X_tr = np.column_stack([y_tr + noise, rng.standard_normal((n, 2))])
        X_te = rng.standard_normal((20, 3))
        y_te = rng.standard_normal(20)
        # High correlation but below default 0.95; should flag at 0.8
        result_default = run_diagnostics(
            X_tr, X_te, y_tr, y_te, task="regression", leakage_threshold=0.95
        )
        result_low = run_diagnostics(
            X_tr, X_te, y_tr, y_te, task="regression", leakage_threshold=0.5
        )
        # Lower threshold flags more
        assert len(result_low["leakage_suspects"]) >= len(result_default["leakage_suspects"])


class TestRunDiagnosticsMismatch:
    def test_large_mean_shift_flagged(self):
        rng = np.random.default_rng(9)
        X_tr = rng.standard_normal((80, 3))
        # Shift first column of test by 10 standard deviations
        X_te = rng.standard_normal((20, 3))
        X_te[:, 0] += 10.0
        y_tr = rng.integers(0, 2, 80).astype(float)
        y_te = rng.integers(0, 2, 20).astype(float)
        result = run_diagnostics(
            X_tr, X_te, y_tr, y_te,
            feature_names=["shifted", "f1", "f2"],
        )
        assert "shifted" in result["train_test_mismatch"]
        assert result["has_issues"] is True

    def test_no_mismatch_for_same_distribution(self, clean_arrays):
        X_tr, X_te, y_tr, y_te = clean_arrays
        result = run_diagnostics(X_tr, X_te, y_tr, y_te)
        assert result["train_test_mismatch"] == []


class TestRunDiagnosticsFeatureNames:
    def test_feature_names_used_in_constant_cols(self):
        rng = np.random.default_rng(10)
        X_tr = rng.standard_normal((40, 3))
        X_tr[:, 1] = 0.0
        X_te = rng.standard_normal((10, 3))
        y_tr = rng.integers(0, 2, 40).astype(float)
        y_te = rng.integers(0, 2, 10).astype(float)
        result = run_diagnostics(
            X_tr, X_te, y_tr, y_te, feature_names=["alpha", "beta", "gamma"]
        )
        assert "beta" in result["constant_cols"]

    def test_default_names_when_none(self):
        rng = np.random.default_rng(11)
        X_tr = rng.standard_normal((40, 3))
        X_tr[:, 0] = 7.0
        X_te = rng.standard_normal((10, 3))
        y_tr = rng.integers(0, 2, 40).astype(float)
        y_te = rng.integers(0, 2, 10).astype(float)
        result = run_diagnostics(X_tr, X_te, y_tr, y_te, feature_names=None)
        assert "feature_0" in result["constant_cols"]


# ================================================================
# Benchmark integration: timing and diagnostics
# ================================================================

class TestBenchmarkLoggingIntegration:
    """Integration tests: run_benchmark with logging/timing/diagnostics."""

    @pytest.fixture
    def small_csv(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 120
        df = pd.DataFrame({
            "f1":     rng.standard_normal(n),
            "f2":     rng.standard_normal(n),
            "target": rng.integers(0, 2, n),
        })
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        return str(p)

    @pytest.fixture
    def regression_csv(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "f1":     rng.standard_normal(n),
            "f2":     rng.standard_normal(n),
            "target": rng.standard_normal(n),
        })
        p = tmp_path / "reg.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_timing_key_in_preprocessor(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        _, prep = run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        assert "timing" in prep

    def test_timing_has_expected_keys(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        _, prep = run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        t = prep["timing"]
        assert "total_seconds" in t
        assert "data_loading_seconds" in t
        assert "optimization_seconds" in t
        assert "models" in t

    def test_timing_models_dict_has_model_name(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        _, prep = run_benchmark(
            small_csv, "target",
            models_dict={"MyLR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        assert "MyLR" in prep["timing"]["models"]

    def test_total_seconds_positive(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        _, prep = run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        assert prep["timing"]["total_seconds"] > 0

    def test_diagnostics_key_in_preprocessor(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        _, prep = run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        assert "diagnostics" in prep

    def test_diagnostics_has_expected_keys(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        _, prep = run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        d = prep["diagnostics"]
        assert "has_issues" in d
        assert "issue_summary" in d

    def test_benchmark_log_file_created(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        out_dir = tmp_path / "out"
        run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(out_dir),
            verbose=False,
        )
        assert (out_dir / "benchmark.log").exists()

    def test_diagnostics_summary_json_created(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        out_dir = tmp_path / "out"
        run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(out_dir),
            verbose=False,
        )
        assert (out_dir / "diagnostics_summary.json").exists()

    def test_diagnostics_summary_json_valid(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        out_dir = tmp_path / "out"
        run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(out_dir),
            verbose=False,
        )
        content = (out_dir / "diagnostics_summary.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert "has_issues" in data

    def test_log_level_parameter_accepted(self, small_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import LogisticRegression
        run_benchmark(
            small_csv, "target",
            models_dict={"LR": LogisticRegression(max_iter=200)},
            output_dir=str(tmp_path / "out"),
            verbose=False,
            log_level="warning",
        )

    def test_regression_has_timing_and_diagnostics(self, regression_csv, tmp_path):
        from src.benchmark import run_benchmark
        from sklearn.linear_model import Ridge
        _, prep = run_benchmark(
            regression_csv, "target",
            task="regression",
            models_dict={"Ridge": Ridge()},
            output_dir=str(tmp_path / "out"),
            verbose=False,
        )
        assert "timing" in prep
        assert "diagnostics" in prep
