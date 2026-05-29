"""
ml-benchmark-lab command-line interface.

Entry point registered as `ml-benchmark` by pyproject.toml.

Usage
-----
    ml-benchmark run data.csv target
    ml-benchmark run data.csv target --task regression --export csv,word
    ml-benchmark run data.csv target --optimize --experiment-name my_run
    ml-benchmark run ts.csv value --task time_series --lags 1,2,3 --compute-stats
    ml-benchmark info
    ml-benchmark examples
    ml-benchmark version
"""

import argparse
import sys


# ================================================================
# COMMAND HANDLERS
# ================================================================


def _cmd_run(args):
    """Run the full benchmark pipeline with CLI-supplied arguments."""
    from src.benchmark import run_benchmark

    # Parse comma-separated flags
    export_formats = (
        [f.strip() for f in args.export_formats.split(",")]
        if args.export_formats
        else None
    )

    lags = None
    if getattr(args, "lags", None):
        try:
            lags = [int(x.strip()) for x in args.lags.split(",")]
        except ValueError:
            print(
                f"error: --lags must be comma-separated integers (e.g. 1,2,3). "
                f"Got: {args.lags!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    kwargs = dict(
        output_dir=args.output_dir,
        task=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
        imbalance_strategy=args.imbalance or None,
        cv_strategy=args.cv_strategy or None,
        export_formats=export_formats,
        report_title=args.report_title,
        experiment_name=args.experiment_name or None,
        optimize=args.optimize,
        optimization_method=args.optimization_method,
        n_iter=args.n_iter,
        compute_stats=args.compute_stats,
        log_level=args.log_level,
        verbose=not args.quiet,
    )

    # Time-series extras (only passed for time_series task to avoid confusion)
    if args.task == "time_series":
        if lags:
            kwargs["lags"] = lags
        kwargs["ts_n_splits"] = args.ts_n_splits
        if args.ts_horizon is not None:
            kwargs["ts_horizon"] = args.ts_horizon

    try:
        run_benchmark(args.csv_path, args.target_col, **kwargs)
    except FileNotFoundError as exc:
        print(f"error: input file not found — {exc}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_info(_args):
    """Print framework version and registered capabilities."""
    from src import __version__
    from src.config import VALID_SECTIONS
    from src.imbalance import VALID_STRATEGIES as VALID_SAMPLERS
    from src.models import (
        CLASSIFICATION_MODELS,
        REGRESSION_MODELS,
        TIME_SERIES_MODELS,
        UNSUPERVISED_MODELS,
    )
    from src.validation import VALID_STRATEGIES

    print(f"ml-benchmark-lab  v{__version__}")
    print()
    print("Supported tasks and default models:")
    for label, registry in (
        ("classification", CLASSIFICATION_MODELS),
        ("regression",     REGRESSION_MODELS),
        ("unsupervised",   UNSUPERVISED_MODELS),
        ("time_series",    TIME_SERIES_MODELS),
    ):
        names = list(registry.keys())
        print(f"  {label:<16}  ({len(names)}) {', '.join(names)}")
    print()
    print(f"CV strategies        : {', '.join(sorted(VALID_STRATEGIES))}")
    print(f"Imbalance strategies : {', '.join(sorted(VALID_SAMPLERS))}")
    print(f"Export formats       : csv, excel, word")
    print(f"Optimization methods : random, grid")
    print(f"Config sections      : {', '.join(sorted(VALID_SECTIONS))}")


def _cmd_examples(_args):
    """List available example scripts with descriptions."""
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parent.parent / "examples"

    # Ordered list: (filename, description)
    catalog = [
        ("classification_example.py",
         "Binary/multiclass classification benchmark"),
        ("regression_example.py",
         "Regression benchmark with residual plots"),
        ("unsupervised_example.py",
         "Clustering (KMeans, DBSCAN) and PCA benchmark"),
        ("time_series_example.py",
         "Walk-forward forecasting benchmark"),
        ("explainability_example.py",
         "SHAP and permutation importance  [requires: pip install shap]"),
        ("optimization_example.py",
         "Grid and random hyperparameter search"),
        ("statistical_comparison_example.py",
         "Advanced statistical comparison and ranking"),
        ("diebold_mariano_example.py",
         "Diebold-Mariano forecasting comparison"),
        ("logging_diagnostics_example.py",
         "Logging, timing, and data-quality diagnostics"),
        ("experiment_analysis_example.py",
         "Experiment aggregation and benchmark analytics"),
        ("data_profile_example.py",
         "Dataset profiling and automated data reports"),
    ]

    print("Available examples  (run with: python examples/<name>)\n")
    for name, desc in catalog:
        exists = (examples_dir / name).exists()
        marker = "  " if exists else "  [missing]  "
        print(f"  {name:<44}{marker}{desc}")

    print(f"\nOutputs are written to: examples/outputs/")
    print(
        "\nRun any example:\n"
        "  python examples/classification_example.py\n"
        "  python examples/time_series_example.py"
    )


def _cmd_profile(args):
    """Profile a CSV dataset and print a statistical summary."""
    from src.data_profile import profile_dataset

    na_values = (
        [v.strip() for v in args.na_values.split(",")]
        if getattr(args, "na_values", None)
        else None
    )

    export_dir = args.output_dir if not args.no_export else None
    plots_dir  = args.output_dir if args.plots else None

    try:
        profile = profile_dataset(
            args.csv_path,
            target_col=args.target or None,
            na_values=na_values,
            output_dir=export_dir,
            plots_dir=plots_dir,
        )
    except FileNotFoundError as exc:
        print(f"error: input file not found — {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Console output ──────────────────────────────────────────────────────
    print(f"Dataset      : {args.csv_path}")
    print(f"Rows         : {profile['n_rows']:,}")
    print(f"Columns      : {profile['n_cols']}  "
          f"(numeric={profile['n_numeric']}, "
          f"categorical={profile['n_categorical']}"
          + (f", datetime={profile['n_datetime']}" if profile["n_datetime"] else "")
          + ")")
    print(f"Duplicate rows: {profile['n_duplicate_rows']} "
          f"({profile['pct_duplicate_rows']:.1f}%)")
    print(f"Completeness : {profile['dataset_completeness_pct']:.1f}%")
    print(f"Memory       : {profile['memory_usage_mb']:.3f} MB")

    if profile.get("target"):
        t = profile["target"]
        print(f"\nTarget       : {t['column']}  "
              f"kind={t['kind']}  n_unique={t['n_unique']}")
        if t.get("imbalance_ratio") is not None:
            print(f"  Imbalance ratio: {t['imbalance_ratio']:.4f}")
        top_classes = list(t.get("value_counts", {}).items())[:5]
        for cls, cnt in top_classes:
            pct = t["class_distribution"].get(cls, 0)
            print(f"  {str(cls):<20} {cnt:>6}  ({pct:.1f}%)")

    print()
    cols_df = profile["columns"]
    display_cols = ["column", "dtype", "n_missing", "n_unique", "mean", "skewness"]
    print(cols_df[display_cols].to_string(index=False))

    if export_dir:
        print(f"\nExports written to : {export_dir}")
    if plots_dir:
        print(f"Plots saved to     : {plots_dir}")


def _cmd_analyze(args):
    """Aggregate and summarize experiment history from persisted run artifacts."""
    from src.experiment_analysis import summarize_experiment_history
    from src.reporting import export_analysis_word

    export_formats = (
        [f.strip() for f in args.export_formats.split(",")]
        if args.export_formats
        else []
    )

    out = args.output_dir

    try:
        summary = summarize_experiment_history(
            args.base_dir,
            experiment_name=args.experiment_name or None,
            metric=args.metric or None,
            export_dir=out if export_formats or args.plots else None,
            plots_dir=out if args.plots else None,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Console output
    n = summary["n_runs"]
    ne = summary["n_experiments"]
    if n == 0:
        print("No experiment runs found under the specified base directory.")
        return

    print(f"Loaded {n} run(s) from {ne} experiment(s)")
    print(f"Metric : {summary['metric']}")
    if summary["best_model"]:
        print(f"Best   : {summary['best_model']}")
    print()

    agg = summary["aggregate"]
    if not agg.empty:
        print(agg.to_string(index=False))

    # Optional CSV / JSON export (already done inside summarize when export_dir is set)
    if "csv" in export_formats or "json" in export_formats:
        print(f"\nExports written to: {out}")

    # Optional Word export
    if "word" in export_formats:
        from pathlib import Path
        p = Path(out) / "analysis_report.docx"
        try:
            export_analysis_word(summary, p, title="Experiment Analysis Report")
            print(f"Word report : {p}")
        except Exception as exc:
            print(f"warning: Word export failed — {exc}", file=sys.stderr)


def _cmd_version(_args):
    """Print the package version string."""
    from src import __version__
    print(f"ml-benchmark-lab {__version__}")


# ================================================================
# ARGUMENT PARSER
# ================================================================


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ml-benchmark",
        description=(
            "ML Benchmark Lab — research-grade ML benchmarking framework.\n"
            "Supports classification, regression, unsupervised, and time-series tasks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ml-benchmark run churn.csv Churn\n"
            "  ml-benchmark run churn.csv Churn --task classification --export csv,word\n"
            "  ml-benchmark run prices.csv price --task regression\n"
            "  ml-benchmark run ts.csv value --task time_series --lags 1,2,3\n"
            "  ml-benchmark run churn.csv Churn --optimize --experiment-name exp01\n"
            "  ml-benchmark run churn.csv Churn --compute-stats --export word\n"
            "  ml-benchmark profile data.csv --target label --plots\n"
            "  ml-benchmark analyze --base-dir outputs --experiment-name exp01\n"
            "  ml-benchmark info\n"
            "  ml-benchmark examples\n"
            "  ml-benchmark version\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Run a full benchmark pipeline on a CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Load a CSV, preprocess, train all default models for the selected task,\n"
            "evaluate, plot, and optionally export results."
        ),
    )
    run_p.add_argument("csv_path",   help="Path to the input CSV file.")
    run_p.add_argument("target_col", help="Name of the target column.")

    # Core
    run_p.add_argument(
        "--task", default="classification",
        choices=["classification", "regression", "unsupervised", "time_series"],
        help="Learning task (default: classification).",
    )
    run_p.add_argument(
        "--output-dir", default="outputs", dest="output_dir", metavar="DIR",
        help="Directory for all output files (default: outputs).",
    )
    run_p.add_argument(
        "--test-size", default=0.2, type=float, dest="test_size", metavar="FLOAT",
        help="Held-out test fraction (default: 0.2).",
    )
    run_p.add_argument(
        "--random-state", default=42, type=int, dest="random_state", metavar="INT",
        help="Global random seed (default: 42).",
    )

    # Optional features
    run_p.add_argument(
        "--imbalance", default=None, metavar="STRATEGY",
        help="Resampling strategy: random_over, random_under, smote, smotenc, adasyn.",
    )
    run_p.add_argument(
        "--cv-strategy", default=None, dest="cv_strategy", metavar="STRATEGY",
        help=(
            "Inner CV strategy for hyperparameter search. "
            "Choices: stratified_kfold, kfold, repeated_stratified_kfold, "
            "group_kfold, time_series_split, holdout."
        ),
    )
    run_p.add_argument(
        "--export", default=None, dest="export_formats", metavar="FMT",
        help="Comma-separated export formats: csv, excel, word  (e.g. csv,word).",
    )
    run_p.add_argument(
        "--report-title", default="Benchmark Report", dest="report_title",
        metavar="TITLE",
        help="Title heading for the Word report (default: 'Benchmark Report').",
    )

    # Experiment tracking
    run_p.add_argument(
        "--experiment-name", default=None, dest="experiment_name", metavar="NAME",
        help=(
            "Activate experiment tracking. Writes config.json, metrics.csv, "
            "environment.txt, and experiment_summary.json under "
            "<output-dir>/experiments/<name>/<run-id>/."
        ),
    )

    # Optimization
    run_p.add_argument(
        "--optimize", action="store_true", default=False,
        help="Optimize every model with built-in default search spaces before evaluation.",
    )
    run_p.add_argument(
        "--optimization-method", default="random",
        choices=["random", "grid"], dest="optimization_method",
        help="Search method when --optimize is set (default: random).",
    )
    run_p.add_argument(
        "--n-iter", default=20, type=int, dest="n_iter", metavar="INT",
        help="Number of parameter combinations for random search (default: 20).",
    )

    # Statistical comparison
    run_p.add_argument(
        "--compute-stats", action="store_true", default=False, dest="compute_stats",
        help=(
            "Compute bootstrap confidence intervals and pairwise statistical tests "
            "(McNemar for classification, Diebold-Mariano for time_series)."
        ),
    )

    # Time-series extras
    run_p.add_argument(
        "--lags", default=None, metavar="1,2,3",
        help="Comma-separated lag values for time-series feature engineering (e.g. 1,2,3).",
    )
    run_p.add_argument(
        "--ts-n-splits", default=5, type=int, dest="ts_n_splits", metavar="INT",
        help="Number of walk-forward folds for time_series task (default: 5).",
    )
    run_p.add_argument(
        "--ts-horizon", default=None, type=int, dest="ts_horizon", metavar="INT",
        help="Test-window size per fold for time_series task (default: auto).",
    )

    # Logging
    run_p.add_argument(
        "--log-level", default="info", dest="log_level",
        choices=["debug", "info", "warning", "error"],
        help="Logging verbosity for benchmark.log and console (default: info).",
    )

    # Output control
    run_p.add_argument(
        "--quiet", action="store_true",
        help="Suppress all progress output.",
    )
    run_p.set_defaults(func=_cmd_run)

    # ── info ─────────────────────────────────────────────────────────────────
    info_p = sub.add_parser(
        "info",
        help="Print framework version, supported models, and registered capabilities.",
    )
    info_p.set_defaults(func=_cmd_info)

    # ── examples ─────────────────────────────────────────────────────────────
    examples_p = sub.add_parser(
        "examples",
        help="List available example scripts with descriptions.",
    )
    examples_p.set_defaults(func=_cmd_examples)

    # ── profile ──────────────────────────────────────────────────────────────
    profile_p = sub.add_parser(
        "profile",
        help="Profile a CSV dataset and produce statistical reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compute per-column statistics (mean, std, skewness, kurtosis,\n"
            "cardinality, missingness), analyse the target distribution,\n"
            "and optionally export profile_summary.json, profile_report.csv,\n"
            "and three diagnostic plots."
        ),
    )
    profile_p.add_argument("csv_path", help="Path to the input CSV file.")
    profile_p.add_argument(
        "--target", default=None, metavar="COL",
        help="Target column for class-distribution analysis (optional).",
    )
    profile_p.add_argument(
        "--output-dir", default="outputs", dest="output_dir", metavar="DIR",
        help="Directory for exports and plots (default: outputs).",
    )
    profile_p.add_argument(
        "--na-values", default=None, dest="na_values", metavar="V,V",
        help="Comma-separated extra NA markers (e.g. '?,N/A').",
    )
    profile_p.add_argument(
        "--plots", action="store_true", default=False,
        help="Save missing heatmap, class distribution, and numeric histograms.",
    )
    profile_p.add_argument(
        "--no-export", action="store_true", default=False, dest="no_export",
        help="Skip writing profile_summary.json and profile_report.csv.",
    )
    profile_p.set_defaults(func=_cmd_profile)

    # ── analyze ──────────────────────────────────────────────────────────────
    analyze_p = sub.add_parser(
        "analyze",
        help="Aggregate and compare results across experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Load all persisted benchmark runs, compute per-model statistics\n"
            "across runs (mean, std, win rate, avg rank), and optionally export\n"
            "results and plots."
        ),
    )
    analyze_p.add_argument(
        "--base-dir", default="outputs", dest="base_dir", metavar="DIR",
        help="Root output directory to scan for experiment runs (default: outputs).",
    )
    analyze_p.add_argument(
        "--experiment-name", default=None, dest="experiment_name", metavar="NAME",
        help="Filter to a single experiment name. Scans all experiments when omitted.",
    )
    analyze_p.add_argument(
        "--metric", default=None, metavar="METRIC",
        help=(
            "Metric column to aggregate (e.g. Accuracy, R2, MAE). "
            "Inferred from task when omitted."
        ),
    )
    analyze_p.add_argument(
        "--output-dir", default="outputs", dest="output_dir", metavar="DIR",
        help="Directory for exported files and plots (default: outputs).",
    )
    analyze_p.add_argument(
        "--export", default=None, dest="export_formats", metavar="FMT",
        help="Comma-separated export formats: csv, json, word  (e.g. csv,word).",
    )
    analyze_p.add_argument(
        "--plots", action="store_true", default=False,
        help="Save analysis plots (win frequency, avg rank, distribution, timeline).",
    )
    analyze_p.set_defaults(func=_cmd_analyze)

    # ── version ──────────────────────────────────────────────────────────────
    version_p = sub.add_parser(
        "version",
        help="Print the package version.",
    )
    version_p.set_defaults(func=_cmd_version)

    return parser


# ================================================================
# ENTRY POINT
# ================================================================


def main():
    parser = _build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
