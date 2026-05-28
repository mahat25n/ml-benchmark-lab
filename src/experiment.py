"""
Lightweight experiment tracking and reproducibility utilities.

No external dependencies beyond the standard library and numpy/pandas.
All data is stored as plain files (JSON, CSV, text) in a local directory tree:

    <base_dir>/experiments/<experiment_name>/<run_id>/
        config.json
        metrics.csv
        environment.txt
        experiment_summary.json
"""

import json
import platform
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def generate_run_id():
    """
    Generate a unique, sortable run identifier.

    Format: ``YYYYMMDD_HHMMSS_<6char_hex>``

    Returns
    -------
    str
    """
    ts  = time.strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{ts}_{uid}"


def set_global_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, and optionally PyTorch.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def capture_environment():
    """
    Capture the current runtime environment.

    Returns
    -------
    dict with keys:
        python_version, platform, packages (dict of {name: version})
    """
    import importlib.metadata

    packages_of_interest = [
        "numpy", "pandas", "scikit-learn", "xgboost",
        "imbalanced-learn", "scipy", "matplotlib", "seaborn",
        "openpyxl", "python-docx", "shap",
    ]

    versions = {}
    for pkg in packages_of_interest:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    try:
        versions["ml-benchmark-lab"] = importlib.metadata.version("ml-benchmark-lab")
    except importlib.metadata.PackageNotFoundError:
        try:
            import src as _src
            versions["ml-benchmark-lab"] = getattr(_src, "__version__", "unknown")
        except Exception:
            versions["ml-benchmark-lab"] = "unknown"

    return {
        "python_version": sys.version,
        "platform":       platform.platform(),
        "packages":       versions,
    }


class ExperimentTracker:
    """
    Lightweight tracker for a single benchmark run.

    Collects config, dataset info, model names, and metrics, then writes
    four artifact files when save() is called.

    Usage
    -----
    tracker = ExperimentTracker("my_experiment", base_dir="outputs")
    tracker.log_config(task="classification", random_state=42)
    tracker.log_dataset(n_samples=800, n_features=10, train_size=640, test_size=160)
    tracker.log_models(["Random Forest", "XGBoost"])
    tracker.log_metrics(results_df)
    tracker.save()
    print(tracker.run_dir)
    """

    def __init__(self, experiment_name, base_dir="outputs", run_id=None):
        """
        Parameters
        ----------
        experiment_name : str
            Logical name for this experiment (used as a sub-directory name).
        base_dir        : str or Path
            Root output directory (same as output_dir in run_benchmark).
        run_id          : str or None
            Custom run ID. When None, generated via generate_run_id().
        """
        self.experiment_name = experiment_name
        self.run_id  = run_id if run_id is not None else generate_run_id()
        self.run_dir = Path(base_dir) / "experiments" / experiment_name / self.run_id
        self._start_time = time.time()

        self._config       = {}
        self._dataset      = {}
        self._models       = []
        self._metrics      = None   # pd.DataFrame or None
        self._optimization = {}     # {model_name: {best_params, best_score, …}}

    # ----------------------------------------------------------------
    # Logging helpers
    # ----------------------------------------------------------------

    def log_config(self, **kwargs):
        """
        Store arbitrary key-value config entries (task, random_state, …).

        Can be called multiple times; later calls update earlier values.
        """
        self._config.update(kwargs)

    def log_dataset(self, n_samples, n_features, train_size, test_size, target_col=None):
        """Record dataset shape and split sizes."""
        self._dataset.update({
            "n_samples":  n_samples,
            "n_features": n_features,
            "train_size": train_size,
            "test_size":  test_size,
            "target_col": target_col,
        })

    def log_models(self, model_names):
        """Record which models were benchmarked."""
        self._models = list(model_names)

    def log_metrics(self, results_df):
        """
        Store per-model metrics from the benchmark results table.

        Parameters
        ----------
        results_df : pd.DataFrame  Output of build_results_table().
        """
        if isinstance(results_df, pd.DataFrame):
            self._metrics = results_df.copy()

    def log_optimization(self, optimization_results):
        """
        Store per-model hyperparameter optimization results.

        Parameters
        ----------
        optimization_results : dict
            {model_name: result_dict} where each result_dict is the output
            of optimization.optimize_model() and contains at least:
            best_params, best_score, n_evaluations, search_duration, method.
        """
        self._optimization = {
            name: {
                "best_params":     res.get("best_params"),
                "best_score":      res.get("best_score"),
                "n_evaluations":   res.get("n_evaluations"),
                "search_duration": res.get("search_duration"),
                "method":          res.get("method"),
            }
            for name, res in optimization_results.items()
        }

    def elapsed_seconds(self):
        """Return wall-clock seconds since tracker was instantiated."""
        return time.time() - self._start_time

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self):
        """
        Write all collected data to the run directory.

        Files written
        -------------
        config.json              Task config + dataset info + model list.
        metrics.csv              Per-model metrics table.
        environment.txt          Human-readable runtime environment snapshot.
        experiment_summary.json  Top-level summary for programmatic access.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        env = capture_environment()

        self._save_config()
        self._save_metrics()
        self._save_environment(env)
        self._save_summary(env)

    def _save_config(self):
        payload = {
            "experiment_name": self.experiment_name,
            "run_id":          self.run_id,
            **self._config,
            "dataset":         self._dataset,
            "models":          self._models,
            "optimization":    self._optimization,
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def _save_metrics(self):
        path = self.run_dir / "metrics.csv"
        if self._metrics is not None:
            self._metrics.to_csv(path, index=False)
        else:
            path.write_text("", encoding="utf-8")

    def _save_environment(self, env):
        lines = [
            f"python_version : {env['python_version']}",
            f"platform       : {env['platform']}",
            "",
            "packages:",
        ]
        for pkg, ver in env["packages"].items():
            lines.append(f"  {pkg:<25} {ver}")
        (self.run_dir / "environment.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    def _save_summary(self, env):
        payload = {
            "experiment_name":      self.experiment_name,
            "run_id":               self.run_id,
            "run_dir":              str(self.run_dir),
            "elapsed_seconds":      round(self.elapsed_seconds(), 3),
            "dataset":              self._dataset,
            "models":               self._models,
            "n_models_optimized":   len(self._optimization),
            "framework": {
                "ml_benchmark_lab": env["packages"].get("ml-benchmark-lab", "unknown"),
                "python":           env["python_version"].split()[0],
            },
        }
        (self.run_dir / "experiment_summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
