"""
Logging utilities and data-quality diagnostics for ml-benchmark-lab.

Provides:
  - get_logger()        : centralized stdlib-only logger factory
  - Timer               : perf_counter-based timing context manager
  - capture_warnings()  : redirect warnings.warn() to a logger
  - run_diagnostics()   : detect common data quality / leakage issues
"""

import logging
import time
import warnings
from contextlib import contextmanager

import numpy as np


# ================================================================
# LOGGER FACTORY
# ================================================================

_FILE_FMT    = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_CONSOLE_FMT = "%(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

_LEVEL_MAP = {
    "debug":   logging.DEBUG,
    "info":    logging.INFO,
    "warning": logging.WARNING,
    "error":   logging.ERROR,
}


def get_logger(
    name,
    *,
    level="info",
    log_file=None,
    console=True,
    fmt=None,
):
    """
    Create or retrieve a named logger with file and/or console handlers.

    Always clears existing handlers before attaching new ones, preventing
    duplicate log lines when run_benchmark is called multiple times.

    Parameters
    ----------
    name     : str    Logger name (e.g. "ml_benchmark").
    level    : str    One of "debug", "info", "warning", "error". Default "info".
    log_file : str or Path or None
               When set, an append-mode FileHandler is added.
               Parent directories are created automatically.
    console  : bool   Add a StreamHandler to stdout. Default True.
    fmt      : str or None
               Custom format for the file handler. Defaults to the module-level
               _FILE_FMT (includes timestamp, level, name). Console always uses
               plain %(message)s so output looks identical to the original print
               style.

    Returns
    -------
    logging.Logger
    """
    log_level   = _LEVEL_MAP.get(str(level).lower(), logging.INFO)
    file_fmt    = fmt or _FILE_FMT
    file_fmtr   = logging.Formatter(file_fmt, datefmt=_DATE_FMT)
    cons_fmtr   = logging.Formatter(_CONSOLE_FMT)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear handlers from previous calls to avoid duplicate output.
    logger.handlers.clear()
    logger.propagate = False

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(cons_fmtr)
        logger.addHandler(ch)

    if log_file is not None:
        from pathlib import Path
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(file_fmtr)
        logger.addHandler(fh)

    return logger


# ================================================================
# TIMER
# ================================================================


class Timer:
    """
    Lightweight wall-clock timer based on time.perf_counter().

    Context manager usage::

        with Timer() as t:
            do_work()
        print(f"{t.elapsed:.3f}s")

    Standalone usage::

        t = Timer()
        t.start()
        do_work()
        t.stop()
        print(f"{t.elapsed:.3f}s")
    """

    def __init__(self):
        self._start = None
        self._end   = None

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── standalone API ────────────────────────────────────────────────────────

    def start(self):
        """Record the start time (resets any previous measurement)."""
        self._start = time.perf_counter()
        self._end   = None

    def stop(self):
        """Record the stop time. Raises RuntimeError if start() was never called."""
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start()")
        self._end = time.perf_counter()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def elapsed(self):
        """
        Elapsed seconds as a float.

        Returns the final duration when stop() has been called, or the running
        duration when the timer is still active. Raises RuntimeError if start()
        was never called.
        """
        if self._start is None:
            raise RuntimeError("Timer has not been started.")
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    def __repr__(self):
        try:
            return f"Timer(elapsed={self.elapsed:.3f}s)"
        except RuntimeError:
            return "Timer(not started)"


# ================================================================
# WARNING CAPTURE
# ================================================================


@contextmanager
def capture_warnings(logger):
    """
    Context manager that redirects warnings.warn() to logger.warning().

    Restores the original showwarning function on exit even if an
    exception is raised.

    Parameters
    ----------
    logger : logging.Logger

    Usage
    -----
    with capture_warnings(logger):
        some_function_that_warns()
    """
    original = warnings.showwarning

    def _redirect(message, category, filename, lineno, file=None, line=None):
        logger.warning("%s:%d: %s: %s", filename, lineno, category.__name__, message)

    warnings.showwarning = _redirect
    try:
        yield
    finally:
        warnings.showwarning = original


# ================================================================
# DATA-QUALITY DIAGNOSTICS
# ================================================================


def run_diagnostics(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    feature_names=None,
    task="classification",
    missing_threshold=0.0,
    leakage_threshold=0.95,
    mismatch_threshold=2.0,
):
    """
    Run a suite of data-quality and leakage diagnostics on train/test splits.

    Parameters
    ----------
    X_train, X_test : array-like of shape (n, p)
    y_train, y_test : array-like of shape (n,)
    feature_names   : list[str] or None
        Column names; indices are used when None.
    task            : str
        Task type. Only "classification" checks class imbalance.
    missing_threshold : float
        Fraction above which a split is flagged for missing values.
        0.0 means any NaN triggers the flag (default).
    leakage_threshold : float
        |Pearson r| >= this triggers a leakage suspect flag. Default 0.95.
    mismatch_threshold : float
        |mean_diff| / pooled_std > this flags a train/test shift. Default 2.0.

    Returns
    -------
    dict with keys
        has_issues          : bool
        issue_summary       : list[str]
        missing             : dict  {split: {count, fraction}}
        constant_cols       : list[str]
        duplicate_cols      : list[tuple[str, str]]
        class_imbalance     : dict or None  (classification only)
        leakage_suspects    : list[str]
        train_test_mismatch : list[str]
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    n_features = X_train.shape[1] if X_train.ndim == 2 else 0
    if feature_names is None:
        names = [f"feature_{i}" for i in range(n_features)]
    else:
        names = list(feature_names)

    issues = []
    result = {
        "has_issues":           False,
        "issue_summary":        [],
        "missing":              {},
        "constant_cols":        [],
        "duplicate_cols":       [],
        "class_imbalance":      None,
        "leakage_suspects":     [],
        "train_test_mismatch":  [],
    }

    if n_features == 0:
        return result

    # ── 1. Missing values ────────────────────────────────────────────────────
    missing_info = {}
    for label, arr in (
        ("X_train", X_train),
        ("X_test",  X_test),
        ("y_train", y_train.reshape(-1, 1)),
        ("y_test",  y_test.reshape(-1, 1)),
    ):
        n_miss = int(np.isnan(arr).sum())
        frac   = n_miss / max(arr.size, 1)
        missing_info[label] = {"count": n_miss, "fraction": round(frac, 6)}
        if n_miss > 0 and frac > missing_threshold:
            issues.append(f"Missing values in {label}: {n_miss} ({frac:.2%})")

    result["missing"] = missing_info

    # ── 2. Constant columns (zero variance in training set) ──────────────────
    stds      = np.nanstd(X_train, axis=0)
    const_idx = [i for i in range(n_features) if stds[i] == 0.0]
    const_names = [names[i] for i in const_idx]
    result["constant_cols"] = const_names
    if const_names:
        issues.append(f"Constant (zero-variance) columns: {const_names}")

    # ── 3. Duplicate columns (exact pairwise equality) ───────────────────────
    dup_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.array_equal(X_train[:, i], X_train[:, j], equal_nan=True):
                dup_pairs.append((names[i], names[j]))
    result["duplicate_cols"] = dup_pairs
    if dup_pairs:
        issues.append(f"Duplicate columns (exact equality): {dup_pairs}")

    # ── 4. Class imbalance (classification only) ─────────────────────────────
    if task == "classification":
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) >= 2:
            minority = int(counts.min())
            majority = int(counts.max())
            ratio    = minority / majority
            imb = {
                "n_classes":      int(len(classes)),
                "minority_count": minority,
                "majority_count": majority,
                "ratio":          round(float(ratio), 4),
            }
            result["class_imbalance"] = imb
            if ratio < 0.1:
                issues.append(
                    f"Class imbalance: minority/majority={ratio:.4f} "
                    f"(minority={minority}, majority={majority})"
                )

    # ── 5. Leakage suspects (|Pearson r| >= leakage_threshold) ───────────────
    try:
        y_num = y_train.astype(float)
        leakage = []
        for i in range(n_features):
            if stds[i] == 0.0:
                continue
            col   = X_train[:, i]
            valid = ~(np.isnan(col) | np.isnan(y_num))
            if valid.sum() < 5:
                continue
            corr = float(np.corrcoef(col[valid], y_num[valid])[0, 1])
            if np.isfinite(corr) and abs(corr) >= leakage_threshold:
                leakage.append(names[i])
        result["leakage_suspects"] = leakage
        if leakage:
            issues.append(
                f"Potential leakage (|r|>={leakage_threshold}): {leakage}"
            )
    except Exception:
        pass

    # ── 6. Train/test distribution mismatch ──────────────────────────────────
    mismatch = []
    for i in range(n_features):
        col_tr, col_te = X_train[:, i], X_test[:, i]
        std_tr = float(np.nanstd(col_tr))
        std_te = float(np.nanstd(col_te))
        pooled = (std_tr + std_te) / 2.0
        if pooled == 0.0:
            continue
        diff = abs(float(np.nanmean(col_tr)) - float(np.nanmean(col_te)))
        if diff / pooled > mismatch_threshold:
            mismatch.append(names[i])
    result["train_test_mismatch"] = mismatch
    if mismatch:
        issues.append(
            f"Train/test mean mismatch (>{mismatch_threshold}*std) in: {mismatch}"
        )

    result["has_issues"]    = bool(issues)
    result["issue_summary"] = issues
    return result
