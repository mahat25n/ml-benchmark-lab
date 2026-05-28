"""Tests for src/experiment.py — experiment tracking and reproducibility."""

import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiment import (
    ExperimentTracker,
    capture_environment,
    generate_run_id,
    set_global_seed,
)


# ================================================================
# generate_run_id
# ================================================================


class TestGenerateRunId:
    def test_format(self):
        run_id = generate_run_id()
        assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", run_id), run_id

    def test_uniqueness(self):
        ids = {generate_run_id() for _ in range(20)}
        assert len(ids) == 20

    def test_returns_string(self):
        assert isinstance(generate_run_id(), str)


# ================================================================
# set_global_seed
# ================================================================


class TestSetGlobalSeed:
    def test_numpy_reproducibility(self):
        set_global_seed(0)
        a = np.random.rand(5)
        set_global_seed(0)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_global_seed(0)
        a = np.random.rand(5)
        set_global_seed(1)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)

    def test_accepts_int(self):
        set_global_seed(42)  # must not raise


# ================================================================
# capture_environment
# ================================================================


class TestCaptureEnvironment:
    def test_required_keys(self):
        env = capture_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "packages" in env

    def test_packages_is_dict(self):
        packages = capture_environment()["packages"]
        assert isinstance(packages, dict)

    def test_known_packages_present(self):
        packages = capture_environment()["packages"]
        for pkg in ("numpy", "pandas", "scikit-learn"):
            assert pkg in packages

    def test_python_version_is_string(self):
        assert isinstance(capture_environment()["python_version"], str)


# ================================================================
# ExperimentTracker — init and path construction
# ================================================================


class TestExperimentTrackerInit:
    def test_run_dir_structure(self, tmp_path):
        tracker = ExperimentTracker("my_exp", base_dir=tmp_path)
        assert tracker.run_dir.parts[-3] == "experiments"
        assert tracker.run_dir.parts[-2] == "my_exp"
        assert re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", tracker.run_dir.name)

    def test_custom_run_id(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path, run_id="custom_id")
        assert tracker.run_id == "custom_id"
        assert tracker.run_dir.name == "custom_id"

    def test_elapsed_seconds_positive(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        assert tracker.elapsed_seconds() >= 0.0

    def test_experiment_name_stored(self, tmp_path):
        tracker = ExperimentTracker("test_exp", base_dir=tmp_path)
        assert tracker.experiment_name == "test_exp"


# ================================================================
# ExperimentTracker — logging helpers
# ================================================================


class TestExperimentTrackerLogging:
    def test_log_config_stores_values(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        tracker.log_config(task="classification", random_state=42)
        assert tracker._config["task"] == "classification"
        assert tracker._config["random_state"] == 42

    def test_log_config_updates(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        tracker.log_config(a=1)
        tracker.log_config(a=2, b=3)
        assert tracker._config["a"] == 2
        assert tracker._config["b"] == 3

    def test_log_dataset(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        tracker.log_dataset(100, 10, 80, 20, target_col="y")
        assert tracker._dataset["n_samples"] == 100
        assert tracker._dataset["target_col"] == "y"

    def test_log_models(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        tracker.log_models(["RF", "XGB"])
        assert tracker._models == ["RF", "XGB"]

    def test_log_metrics_dataframe(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        df = pd.DataFrame({"Model": ["RF"], "Accuracy": [0.9]})
        tracker.log_metrics(df)
        assert tracker._metrics is not None
        assert "Accuracy" in tracker._metrics.columns

    def test_log_metrics_ignores_non_dataframe(self, tmp_path):
        tracker = ExperimentTracker("exp", base_dir=tmp_path)
        tracker.log_metrics({"not": "a dataframe"})
        assert tracker._metrics is None


# ================================================================
# ExperimentTracker — save() file creation
# ================================================================


class TestExperimentTrackerSave:
    @pytest.fixture
    def saved_tracker(self, tmp_path):
        tracker = ExperimentTracker("save_test", base_dir=tmp_path, run_id="test_run")
        tracker.log_config(task="regression", random_state=7)
        tracker.log_dataset(200, 5, 160, 40, target_col="price")
        tracker.log_models(["Linear Regression", "Ridge"])
        tracker.log_metrics(
            pd.DataFrame({"Model": ["Linear Regression", "Ridge"], "MAE": [1.2, 1.1]})
        )
        tracker.save()
        return tracker

    def test_run_dir_created(self, saved_tracker):
        assert saved_tracker.run_dir.is_dir()

    def test_config_json_exists(self, saved_tracker):
        assert (saved_tracker.run_dir / "config.json").is_file()

    def test_metrics_csv_exists(self, saved_tracker):
        assert (saved_tracker.run_dir / "metrics.csv").is_file()

    def test_environment_txt_exists(self, saved_tracker):
        assert (saved_tracker.run_dir / "environment.txt").is_file()

    def test_experiment_summary_json_exists(self, saved_tracker):
        assert (saved_tracker.run_dir / "experiment_summary.json").is_file()

    def test_config_json_content(self, saved_tracker):
        config = json.loads((saved_tracker.run_dir / "config.json").read_text())
        assert config["experiment_name"] == "save_test"
        assert config["task"] == "regression"
        assert config["dataset"]["n_samples"] == 200
        assert config["models"] == ["Linear Regression", "Ridge"]

    def test_metrics_csv_content(self, saved_tracker):
        df = pd.read_csv(saved_tracker.run_dir / "metrics.csv")
        assert "Model" in df.columns
        assert "MAE" in df.columns
        assert len(df) == 2

    def test_environment_txt_content(self, saved_tracker):
        text = (saved_tracker.run_dir / "environment.txt").read_text()
        assert "python_version" in text
        assert "packages:" in text
        assert "numpy" in text

    def test_summary_json_content(self, saved_tracker):
        summary = json.loads(
            (saved_tracker.run_dir / "experiment_summary.json").read_text()
        )
        assert summary["run_id"] == "test_run"
        assert summary["experiment_name"] == "save_test"
        assert "elapsed_seconds" in summary
        assert "framework" in summary

    def test_metrics_csv_empty_when_no_metrics(self, tmp_path):
        tracker = ExperimentTracker("empty", base_dir=tmp_path, run_id="r0")
        tracker.save()
        path = tracker.run_dir / "metrics.csv"
        assert path.is_file()
        assert path.stat().st_size == 0


# ================================================================
# Integration: run_benchmark with experiment_name
# ================================================================


class TestRunBenchmarkExperimentTracking:
    @pytest.fixture
    def classification_csv(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["target"] = y
        p = tmp_path / "clf.csv"
        df.to_csv(p, index=False)
        return p

    def test_experiment_key_in_preprocessor(self, classification_csv, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        from src.benchmark import run_benchmark

        _, preprocessor = run_benchmark(
            str(classification_csv), "target",
            output_dir=str(tmp_path / "out"),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=0)},
            experiment_name="clf_test",
            verbose=False,
        )
        assert "experiment" in preprocessor
        assert "run_id" in preprocessor["experiment"]
        assert "run_dir" in preprocessor["experiment"]

    def test_artifact_files_written(self, classification_csv, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        from src.benchmark import run_benchmark

        out_dir = tmp_path / "out2"
        _, preprocessor = run_benchmark(
            str(classification_csv), "target",
            output_dir=str(out_dir),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=0)},
            experiment_name="clf_artifacts",
            verbose=False,
        )
        run_dir = Path(preprocessor["experiment"]["run_dir"])
        for fname in ("config.json", "metrics.csv", "environment.txt",
                      "experiment_summary.json"):
            assert (run_dir / fname).is_file(), f"Missing: {fname}"

    def test_no_tracking_when_experiment_name_none(self, classification_csv, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        from src.benchmark import run_benchmark

        _, preprocessor = run_benchmark(
            str(classification_csv), "target",
            output_dir=str(tmp_path / "out3"),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=0)},
            experiment_name=None,
            verbose=False,
        )
        assert "experiment" not in preprocessor

    def test_config_json_has_task(self, classification_csv, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        from src.benchmark import run_benchmark

        out_dir = tmp_path / "out4"
        _, preprocessor = run_benchmark(
            str(classification_csv), "target",
            output_dir=str(out_dir),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=0)},
            experiment_name="task_check",
            verbose=False,
        )
        run_dir = Path(preprocessor["experiment"]["run_dir"])
        config = json.loads((run_dir / "config.json").read_text())
        assert config["task"] == "classification"
        assert config["dataset"]["target_col"] == "target"
