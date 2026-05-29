"""
Experiment aggregation and benchmark analytics.

Reads persisted run artifacts written by ExperimentTracker
(metrics.csv, config.json, experiment_summary.json) and produces
cross-run statistics, comparison tables, export files, and plots.

No database, no MLflow, no cloud storage — plain files only.

Public API
----------
load_experiments(base_dir, experiment_name=None)
aggregate_experiments(runs, metric=None)
compare_experiments(runs, metric=None, group_by="experiment_name")
summarize_experiment_history(base_dir, ...)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# INTERNAL HELPERS
# ================================================================

# Metrics where lower values are better
_LOWER_IS_BETTER = {"mae", "mse", "rmse", "mape", "smape", "loss",
                    "davies-bouldin", "davies_bouldin"}

_TASK_PRIMARY_METRIC = {
    "classification": "Accuracy",
    "regression":     "R2",
    "time_series":    "MAE",
    "unsupervised":   "Silhouette",
}


def _lower_is_better(metric):
    return metric.lower().replace(" ", "_") in _LOWER_IS_BETTER


def _infer_primary_metric(task):
    """Return a sensible default metric column name for each task type."""
    return _TASK_PRIMARY_METRIC.get(task or "", "Accuracy")


def _parse_run_dir(run_dir):
    """
    Load one run directory into a dict.

    Returns None if required files are missing or malformed.
    """
    run_dir = Path(run_dir)
    summary_path = run_dir / "experiment_summary.json"
    metrics_path = run_dir / "metrics.csv"
    config_path  = run_dir / "config.json"

    if not summary_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    metrics_df = None
    if metrics_path.exists():
        try:
            raw = pd.read_csv(metrics_path)
            if not raw.empty and "Model" in raw.columns:
                metrics_df = raw
        except Exception:
            pass

    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "experiment_name":   summary.get("experiment_name", run_dir.parent.name),
        "run_id":            summary.get("run_id", run_dir.name),
        "run_dir":           str(run_dir),
        "elapsed_seconds":   summary.get("elapsed_seconds"),
        "task":              config.get("task"),
        "dataset":           summary.get("dataset", {}),
        "models":            summary.get("models", []),
        "n_models_optimized": summary.get("n_models_optimized", 0),
        "has_stats_summary": summary.get("has_stats_summary", False),
        "metrics":           metrics_df,
        "config":            config,
    }


# ================================================================
# PUBLIC API
# ================================================================


def load_experiments(base_dir, experiment_name=None):
    """
    Load all benchmark runs from an experiment directory tree.

    Reads each run directory under
    ``<base_dir>/experiments/<experiment_name>/<run_id>/``.
    Each directory must contain an ``experiment_summary.json`` file.

    Parameters
    ----------
    base_dir        : str or Path
        Root output directory (same value passed as output_dir to run_benchmark).
    experiment_name : str or None
        When provided, only load runs for that experiment.
        When None, load ALL experiments under base_dir/experiments/.

    Returns
    -------
    list[dict]
        Each dict has keys:
        experiment_name, run_id, run_dir, task, dataset, models,
        elapsed_seconds, metrics (pd.DataFrame or None), config (dict).
        Sorted by run_id (lexicographic == temporal for the default format).
    """
    base_dir = Path(base_dir)
    experiments_root = base_dir / "experiments"

    if not experiments_root.exists():
        return []

    if experiment_name is not None:
        exp_dirs = [experiments_root / experiment_name]
    else:
        exp_dirs = sorted(
            d for d in experiments_root.iterdir() if d.is_dir()
        )

    runs = []
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = _parse_run_dir(run_dir)
            if run is not None:
                runs.append(run)

    return runs


def aggregate_experiments(runs, metric=None):
    """
    Compute per-model statistics aggregated across multiple benchmark runs.

    For each model name that appears in the runs the function computes:

    - ``mean_{metric}``  — mean metric value across runs
    - ``std_{metric}``   — standard deviation (ddof=1; 0 for single run)
    - ``win_count``      — runs where this model ranked #1
    - ``win_pct``        — win_count / model_n_runs * 100
    - ``avg_rank``       — mean rank (1 = best) across runs
    - ``n_runs``         — number of runs where this model appeared

    Parameters
    ----------
    runs   : list[dict]  Output of load_experiments().
    metric : str or None
        Metric column to aggregate. Inferred from the first run's task
        when None (Accuracy, R2, MAE, or Silhouette).

    Returns
    -------
    pd.DataFrame  sorted by avg_rank ascending.
    """
    if not runs:
        return pd.DataFrame(columns=["Model"])

    if metric is None:
        task   = next((r.get("task") for r in runs if r.get("task")), None)
        metric = _infer_primary_metric(task)

    lower = _lower_is_better(metric)

    model_scores = {}  # {name: [score, ...]}
    win_counts   = {}  # {name: int}
    rank_sums    = {}  # {name: float}
    rank_counts  = {}  # {name: int}

    for run in runs:
        df = run.get("metrics")
        if df is None or "Model" not in df.columns or metric not in df.columns:
            continue
        df_v = df.dropna(subset=[metric])
        if df_v.empty:
            continue

        # Rank models in this run (1 = best)
        scores = df_v[metric].values.astype(float)
        if lower:
            order = scores.argsort()
        else:
            order = (-scores).argsort()

        ranked_names = df_v["Model"].iloc[order].tolist()
        winner = ranked_names[0]

        for rank_pos, name in enumerate(ranked_names, start=1):
            row_score = float(df_v.loc[df_v["Model"] == name, metric].iloc[0])
            model_scores.setdefault(name, []).append(row_score)
            rank_sums[name]   = rank_sums.get(name, 0.0) + rank_pos
            rank_counts[name] = rank_counts.get(name, 0) + 1

        win_counts[winner] = win_counts.get(winner, 0) + 1

    if not model_scores:
        return pd.DataFrame(columns=[
            "Model", f"mean_{metric}", f"std_{metric}",
            "win_count", "win_pct", "avg_rank", "n_runs",
        ])

    rows = []
    for name, scores in model_scores.items():
        n = len(scores)
        rows.append({
            "Model":          name,
            f"mean_{metric}": round(float(np.mean(scores)), 4),
            f"std_{metric}":  round(float(np.std(scores, ddof=1)) if n > 1 else 0.0, 4),
            "win_count":      win_counts.get(name, 0),
            "win_pct":        round(win_counts.get(name, 0) / n * 100, 1),
            "avg_rank":       round(rank_sums[name] / rank_counts[name], 2),
            "n_runs":         n,
        })

    df_agg = pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
    return df_agg


def compare_experiments(runs, metric=None, group_by="experiment_name"):
    """
    Compare mean metric values across experiment groups.

    Produces a wide DataFrame where:
    - rows    = groups (default: experiment name)
    - columns = model names
    - values  = mean metric value for that (group, model) pair

    Parameters
    ----------
    runs     : list[dict]  Output of load_experiments().
    metric   : str or None  Metric column. Inferred when None.
    group_by : str          Key in each run dict to group by.

    Returns
    -------
    pd.DataFrame  shape (n_groups, n_models)
    """
    if not runs:
        return pd.DataFrame()

    if metric is None:
        task   = next((r.get("task") for r in runs if r.get("task")), None)
        metric = _infer_primary_metric(task)

    records = []
    for run in runs:
        df = run.get("metrics")
        if df is None or "Model" not in df.columns or metric not in df.columns:
            continue
        group_val = run.get(group_by, "unknown")
        for _, row in df.iterrows():
            val = row.get(metric)
            if pd.notna(val):
                records.append({"group": group_val, "model": row["Model"], "value": val})

    if not records:
        return pd.DataFrame()

    long_df = pd.DataFrame(records)
    wide_df = (
        long_df
        .groupby(["group", "model"])["value"]
        .mean()
        .unstack("model")
        .round(4)
    )
    wide_df.index.name = group_by
    return wide_df


def summarize_experiment_history(
    base_dir,
    experiment_name=None,
    metric=None,
    export_dir=None,
    plots_dir=None,
):
    """
    Load, aggregate, and summarize the full history of benchmark runs.

    Parameters
    ----------
    base_dir        : str or Path
        Root output directory.
    experiment_name : str or None
        Filter to a single experiment. None scans all experiments.
    metric          : str or None
        Metric to aggregate. Inferred from task when None.
    export_dir      : str or Path or None
        When set, writes ``aggregated_results.csv`` and
        ``aggregated_summary.json`` to this directory.
    plots_dir       : str or Path or None
        When set, saves the four analysis plots to this directory.

    Returns
    -------
    dict with keys:
        runs             : list[dict]
        aggregate        : pd.DataFrame
        comparison       : pd.DataFrame
        n_runs           : int
        n_experiments    : int
        experiment_names : list[str]
        metric           : str
        best_model       : str or None
        timeline         : pd.DataFrame
    """
    runs = load_experiments(base_dir, experiment_name=experiment_name)

    if metric is None:
        task   = next((r.get("task") for r in runs if r.get("task")), None)
        metric = _infer_primary_metric(task)

    aggregate  = aggregate_experiments(runs, metric=metric)
    comparison = compare_experiments(runs, metric=metric)
    timeline   = _build_timeline(runs, metric)

    best_model = (
        aggregate.iloc[0]["Model"]
        if not aggregate.empty and "Model" in aggregate.columns
        else None
    )

    exp_names = sorted({r["experiment_name"] for r in runs if r.get("experiment_name")})

    summary = {
        "runs":             runs,
        "aggregate":        aggregate,
        "comparison":       comparison,
        "n_runs":           len(runs),
        "n_experiments":    len(exp_names),
        "experiment_names": exp_names,
        "metric":           metric,
        "best_model":       best_model,
        "timeline":         timeline,
    }

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_aggregate_csv(aggregate, export_dir / "aggregated_results.csv")
        export_aggregate_json(summary, export_dir / "aggregated_summary.json")

    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        _save_analysis_plots(summary, metric, plots_dir)

    return summary


# ================================================================
# EXPORT HELPERS
# ================================================================


def export_aggregate_csv(aggregate_df, save_path):
    """
    Write the aggregate results table to a CSV file.

    Parameters
    ----------
    aggregate_df : pd.DataFrame  Output of aggregate_experiments().
    save_path    : str or Path
    """
    aggregate_df.to_csv(save_path, index=False)


def export_aggregate_json(summary, save_path):
    """
    Write a JSON summary of the experiment history.

    Parameters
    ----------
    summary   : dict  Output of summarize_experiment_history().
    save_path : str or Path
    """
    agg = summary.get("aggregate", pd.DataFrame())
    tl  = summary.get("timeline",  pd.DataFrame())

    payload = {
        "n_runs":           summary.get("n_runs", 0),
        "n_experiments":    summary.get("n_experiments", 0),
        "experiment_names": summary.get("experiment_names", []),
        "metric":           summary.get("metric"),
        "best_model":       summary.get("best_model"),
        "aggregate": (
            agg.to_dict(orient="records") if not agg.empty else []
        ),
        "timeline": (
            tl[["run_id", "experiment_name", "best_model", "best_score"]]
            .to_dict(orient="records")
            if not tl.empty else []
        ),
    }
    Path(save_path).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


# ================================================================
# INTERNAL: timeline
# ================================================================


def _build_timeline(runs, metric):
    """
    One row per run: run_id, experiment_name, best_model, best_score.

    Sorted by run_id (lexicographic == temporal for YYYYMMDD_HHMMSS format).
    """
    lower = _lower_is_better(metric)
    rows  = []

    for run in runs:
        df = run.get("metrics")
        if df is None or "Model" not in df.columns or metric not in df.columns:
            continue
        df_v = df.dropna(subset=[metric])
        if df_v.empty:
            continue

        if lower:
            best_idx = df_v[metric].idxmin()
        else:
            best_idx = df_v[metric].idxmax()

        rows.append({
            "run_id":          run["run_id"],
            "experiment_name": run.get("experiment_name", ""),
            "best_model":      df_v.loc[best_idx, "Model"],
            "best_score":      round(float(df_v.loc[best_idx, metric]), 4),
            "elapsed_seconds": run.get("elapsed_seconds"),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "run_id", "experiment_name", "best_model",
            "best_score", "elapsed_seconds",
        ])

    tl = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    tl["run_index"] = range(len(tl))
    return tl


# ================================================================
# INTERNAL: plot orchestration
# ================================================================


def _save_analysis_plots(summary, metric, plots_dir):
    """Generate and save all four analysis plots to plots_dir."""
    try:
        from src.plots import (
            plot_average_rank,
            plot_experiment_timeline,
            plot_metric_distribution,
            plot_model_win_frequency,
        )
    except ImportError as exc:
        warnings.warn(f"experiment_analysis: could not import plot functions: {exc}")
        return

    agg  = summary.get("aggregate", pd.DataFrame())
    runs = summary.get("runs", [])
    tl   = summary.get("timeline", pd.DataFrame())

    if not agg.empty and "Model" in agg.columns:
        # Win frequency
        if "win_count" in agg.columns:
            plot_model_win_frequency(
                dict(zip(agg["Model"], agg["win_count"])),
                title="Model Win Frequency",
                save_path=plots_dir / "plot_win_frequency.png",
            )

        # Average rank
        if "avg_rank" in agg.columns:
            plot_average_rank(
                agg.set_index("Model")["avg_rank"],
                title="Average Model Rank",
                save_path=plots_dir / "plot_avg_rank.png",
            )

        # Metric distribution (box plot per model across all runs)
        model_scores = {}
        for run in runs:
            df = run.get("metrics")
            if df is None or "Model" not in df.columns or metric not in df.columns:
                continue
            for _, row in df.iterrows():
                v = row.get(metric)
                if pd.notna(v):
                    model_scores.setdefault(row["Model"], []).append(float(v))

        if model_scores:
            max_len = max(len(v) for v in model_scores.values())
            dist_df = pd.DataFrame({
                m: scores + [float("nan")] * (max_len - len(scores))
                for m, scores in model_scores.items()
            })
            plot_metric_distribution(
                dist_df, metric,
                title=f"Distribution of {metric} Across Runs",
                save_path=plots_dir / "plot_metric_distribution.png",
            )

    # Timeline
    if not tl.empty:
        plot_experiment_timeline(
            tl, metric,
            title=f"Best {metric} per Run",
            save_path=plots_dir / "plot_experiment_timeline.png",
        )
