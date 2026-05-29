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
