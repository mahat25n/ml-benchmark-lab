"""
ml-benchmark-lab command-line interface.

Entry point registered as `ml-benchmark` by pyproject.toml.

Usage
-----
    ml-benchmark run data.csv target --output-dir outputs
    ml-benchmark run data.csv target --imbalance smote --export csv,excel,word
    ml-benchmark info
"""

import argparse
import sys


# ================================================================
# COMMAND HANDLERS
# ================================================================


def _cmd_run(args):
    """Delegate to run_benchmark with CLI-supplied arguments."""
    from src.benchmark import run_benchmark

    export_formats = (
        [f.strip() for f in args.export_formats.split(",")]
        if args.export_formats
        else None
    )

    run_benchmark(
        args.csv_path,
        args.target_col,
        output_dir=args.output_dir,
        task=args.task,
        test_size=args.test_size,
        imbalance_strategy=args.imbalance or None,
        cv_strategy=args.cv_strategy or None,
        export_formats=export_formats,
        report_title=args.report_title,
        verbose=not args.quiet,
    )


def _cmd_info(_args):
    """Print framework version and registered capabilities."""
    from src import __version__
    from src.config import VALID_SECTIONS
    from src.imbalance import VALID_STRATEGIES as VALID_SAMPLERS
    from src.validation import VALID_STRATEGIES

    print(f"ml-benchmark-lab  v{__version__}")
    print(f"  Validation strategies : {sorted(VALID_STRATEGIES)}")
    print(f"  Imbalance strategies  : {sorted(VALID_SAMPLERS)}")
    print(f"  Config sections       : {sorted(VALID_SECTIONS)}")
    print(f"  Supported tasks       : classification")
    print(f"  Export formats        : csv, excel, word  (latex, pdf, html: planned)")


# ================================================================
# ARGUMENT PARSER
# ================================================================


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ml-benchmark",
        description="ML Benchmark Lab — research-grade ML benchmarking framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ml-benchmark run churn.csv Churn\n"
            "  ml-benchmark run churn.csv Churn --imbalance smote --export csv,word\n"
            "  ml-benchmark run churn.csv Churn --cv-strategy stratified_kfold\n"
            "  ml-benchmark info\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── run ──────────────────────────────────────────────────────────────
    run_p = sub.add_parser(
        "run",
        help="Run a full benchmark pipeline on a CSV dataset.",
    )
    run_p.add_argument("csv_path",   help="Path to input CSV file.")
    run_p.add_argument("target_col", help="Name of the target column.")

    run_p.add_argument(
        "--output-dir", default="outputs", dest="output_dir",
        metavar="DIR",
        help="Directory for all output files (default: outputs).",
    )
    run_p.add_argument(
        "--task", default="classification",
        choices=["classification"],
        help="Learning task (default: classification).",
    )
    run_p.add_argument(
        "--test-size", default=0.2, type=float, dest="test_size",
        metavar="FLOAT",
        help="Held-out test fraction (default: 0.2).",
    )
    run_p.add_argument(
        "--imbalance", default=None, metavar="STRATEGY",
        help=(
            "Resampling strategy applied to the training set. "
            "Choices: random_over, random_under, smote, smotenc, adasyn."
        ),
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
        "--export", default=None, dest="export_formats",
        metavar="csv,excel,word",
        help="Comma-separated export formats (e.g. csv,excel,word).",
    )
    run_p.add_argument(
        "--report-title", default="Benchmark Report", dest="report_title",
        metavar="TITLE",
        help="Title heading for the Word report (default: 'Benchmark Report').",
    )
    run_p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    run_p.set_defaults(func=_cmd_run)

    # ── info ─────────────────────────────────────────────────────────────
    info_p = sub.add_parser(
        "info",
        help="Print framework version and registered capabilities.",
    )
    info_p.set_defaults(func=_cmd_info)

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
