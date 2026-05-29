"""
Dataset profiling and automated data reports.

Produces per-column statistics, target analysis, missingness summaries,
and optional plot/export artefacts — with no dependency on pandas-profiling,
ydata-profiling, or any other heavy profiling library.

Public API
----------
profile_dataset(csv_or_df, target_col=None, ...)
summarize_columns(df)
summarize_target(df, target_col)
summarize_missingness(df)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# INTERNAL HELPERS
# ================================================================

_DEFAULT_NA_VALUES = ["?", "N/A", "n/a", "na", "none", "None", "unknown", "Unknown", "-"]

_NAN = float("nan")


def _safe_round(value, digits=4):
    """Round a scalar, returning NaN for non-finite values."""
    try:
        f = float(value)
        return round(f, digits) if np.isfinite(f) else _NAN
    except (TypeError, ValueError):
        return _NAN


def _col_kind(series):
    """Classify a column as 'numeric', 'categorical', 'boolean', or 'datetime'."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "numeric"
    return "categorical"


def _json_safe(obj):
    """Custom JSON serializer default — converts numpy / NaN scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _profile_to_dict(profile):
    """
    Convert a profile dict (which may contain DataFrames) to a
    fully JSON-serialisable dict.
    """
    def _fix(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return None if np.isnan(v) else float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    def _fix_record(record):
        return {k: _fix(v) for k, v in record.items()}

    target = None
    if profile.get("target"):
        t = profile["target"]
        target = {
            k: (
                {str(kk): _fix(vv) for kk, vv in v.items()}
                if isinstance(v, dict) else _fix(v)
            )
            for k, v in t.items()
        }

    return {
        "n_rows":                   profile["n_rows"],
        "n_cols":                   profile["n_cols"],
        "n_numeric":                profile["n_numeric"],
        "n_categorical":            profile["n_categorical"],
        "n_datetime":               profile["n_datetime"],
        "n_boolean":                profile["n_boolean"],
        "n_duplicate_rows":         profile["n_duplicate_rows"],
        "pct_duplicate_rows":       profile["pct_duplicate_rows"],
        "dataset_completeness_pct": profile["dataset_completeness_pct"],
        "memory_usage_mb":          profile["memory_usage_mb"],
        "target":                   target,
        "columns":  [_fix_record(r) for r in profile["columns"].to_dict(orient="records")],
        "missingness": [
            _fix_record(r) for r in profile["missingness"].to_dict(orient="records")
        ],
    }


# ================================================================
# COLUMN SUMMARY
# ================================================================


def summarize_columns(df):
    """
    Compute per-column descriptive statistics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  with one row per column and columns:
        column, dtype, kind, n_missing, pct_missing, n_unique, pct_unique,
        mean, std, min, p25, p50, p75, max, skewness, kurtosis,
        top_value, top_freq, top_pct
    """
    n = len(df)
    rows = []

    for col in df.columns:
        series = df[col]
        kind   = _col_kind(series)

        n_missing  = int(series.isnull().sum())
        pct_missing = round(n_missing / max(n, 1) * 100, 2)
        n_unique   = int(series.nunique(dropna=True))
        pct_unique = round(n_unique / max(n, 1) * 100, 2)

        # Most frequent value
        try:
            vc        = series.value_counts(dropna=True)
            top_value = str(vc.index[0]) if len(vc) > 0 else None
            top_freq  = int(vc.iloc[0])  if len(vc) > 0 else 0
            top_pct   = round(top_freq / max(n, 1) * 100, 2)
        except Exception:
            top_value, top_freq, top_pct = None, 0, 0.0

        # Numeric-only stats
        if kind == "numeric":
            s_v  = series.dropna()
            nv   = len(s_v)
            mean = _safe_round(s_v.mean())   if nv > 0 else _NAN
            std  = _safe_round(s_v.std())    if nv > 1 else _NAN
            mn   = _safe_round(s_v.min())    if nv > 0 else _NAN
            p25  = _safe_round(s_v.quantile(0.25)) if nv > 0 else _NAN
            p50  = _safe_round(s_v.quantile(0.50)) if nv > 0 else _NAN
            p75  = _safe_round(s_v.quantile(0.75)) if nv > 0 else _NAN
            mx   = _safe_round(s_v.max())    if nv > 0 else _NAN
            skew = _safe_round(s_v.skew())   if nv > 2 else _NAN
            kurt = _safe_round(s_v.kurt())   if nv > 3 else _NAN
        else:
            mean = std = mn = p25 = p50 = p75 = mx = skew = kurt = _NAN

        rows.append({
            "column":     col,
            "dtype":      str(series.dtype),
            "kind":       kind,
            "n_missing":  n_missing,
            "pct_missing": pct_missing,
            "n_unique":   n_unique,
            "pct_unique": pct_unique,
            "mean":       mean,
            "std":        std,
            "min":        mn,
            "p25":        p25,
            "p50":        p50,
            "p75":        p75,
            "max":        mx,
            "skewness":   skew,
            "kurtosis":   kurt,
            "top_value":  top_value,
            "top_freq":   top_freq,
            "top_pct":    top_pct,
        })

    return pd.DataFrame(rows)


# ================================================================
# TARGET SUMMARY
# ================================================================


def summarize_target(df, target_col):
    """
    Analyse the target column: distribution, kind, and imbalance.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str

    Returns
    -------
    dict with keys:
        column, dtype, kind, n_unique, n_missing,
        value_counts, class_distribution, imbalance_ratio
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} not found in DataFrame")

    series  = df[target_col]
    n       = len(series)
    n_unique = int(series.nunique(dropna=True))

    # Classify kind
    if pd.api.types.is_numeric_dtype(series.dtype):
        if n_unique == 2:
            kind = "binary"
        elif n_unique <= 20:
            kind = "multiclass"
        else:
            kind = "continuous"
    elif n_unique == 2:
        kind = "binary"
    else:
        kind = "multiclass"

    vc = series.value_counts(dropna=True)
    value_counts       = {str(k): int(v) for k, v in vc.items()}
    class_distribution = {str(k): round(v / max(n, 1) * 100, 2) for k, v in vc.items()}

    imbalance_ratio = None
    if kind in ("binary", "multiclass") and len(vc) >= 2:
        imbalance_ratio = round(float(vc.min()) / float(vc.max()), 4)

    return {
        "column":             target_col,
        "dtype":              str(series.dtype),
        "kind":               kind,
        "n_unique":           n_unique,
        "n_missing":          int(series.isnull().sum()),
        "value_counts":       value_counts,
        "class_distribution": class_distribution,
        "imbalance_ratio":    imbalance_ratio,
    }


# ================================================================
# MISSINGNESS SUMMARY
# ================================================================


def summarize_missingness(df):
    """
    Per-column missing-value statistics.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  with columns:
        column, n_missing, pct_missing, pattern

    pattern is one of:
        "complete"       — no missing values
        "partial"        — some missing values (pct_missing <= 50%)
        "mostly_missing" — majority missing (pct_missing > 50%)
    """
    n = len(df)
    rows = []
    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        pct    = round(n_miss / max(n, 1) * 100, 2)
        if n_miss == 0:
            pattern = "complete"
        elif pct > 50:
            pattern = "mostly_missing"
        else:
            pattern = "partial"
        rows.append({
            "column":    col,
            "n_missing": n_miss,
            "pct_missing": pct,
            "pattern":   pattern,
        })
    return pd.DataFrame(rows)


# ================================================================
# PROFILE DATASET
# ================================================================


def profile_dataset(
    csv_or_df,
    target_col=None,
    *,
    na_values=None,
    output_dir=None,
    plots_dir=None,
):
    """
    Compute a full statistical profile of a dataset.

    Parameters
    ----------
    csv_or_df   : str, Path, or pd.DataFrame
        CSV file path or already-loaded DataFrame.
    target_col  : str or None
        Target column for class-distribution and imbalance analysis.
    na_values   : list[str] or None
        Additional strings to treat as NA when loading from CSV.
    output_dir  : str or Path or None
        When set, writes ``profile_summary.json`` and ``profile_report.csv``
        to this directory.
    plots_dir   : str or Path or None
        When set, saves three plot PNGs to this directory:
        ``plot_missing_heatmap.png``, ``plot_class_distribution.png``
        (only when target_col is given), ``plot_numeric_distributions.png``.

    Returns
    -------
    dict with keys:
        n_rows, n_cols, n_numeric, n_categorical, n_datetime, n_boolean,
        n_duplicate_rows, pct_duplicate_rows, dataset_completeness_pct,
        memory_usage_mb, columns (pd.DataFrame), target (dict or None),
        missingness (pd.DataFrame)
    """
    # ── Load ──────────────────────────────────────────────────────────────
    if isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df.copy()
    else:
        extra = _DEFAULT_NA_VALUES if na_values is None else list(na_values)
        df    = pd.read_csv(csv_or_df, na_values=extra, keep_default_na=True)
        df    = df.reset_index(drop=True)

    n_rows = len(df)
    n_cols = len(df.columns)

    # ── Column type counts ────────────────────────────────────────────────
    kinds = {col: _col_kind(df[col]) for col in df.columns}
    n_numeric     = sum(1 for k in kinds.values() if k == "numeric")
    n_categorical = sum(1 for k in kinds.values() if k == "categorical")
    n_datetime    = sum(1 for k in kinds.values() if k == "datetime")
    n_boolean     = sum(1 for k in kinds.values() if k == "boolean")

    # ── Duplicate rows ────────────────────────────────────────────────────
    n_duplicate_rows  = int(df.duplicated().sum())
    pct_duplicate_rows = round(n_duplicate_rows / max(n_rows, 1) * 100, 2)

    # ── Completeness ──────────────────────────────────────────────────────
    total_cells = n_rows * n_cols
    total_missing = int(df.isnull().sum().sum())
    dataset_completeness_pct = round(
        (1 - total_missing / max(total_cells, 1)) * 100, 2
    )

    # ── Memory ────────────────────────────────────────────────────────────
    memory_usage_mb = round(
        df.memory_usage(deep=True).sum() / (1024 ** 2), 4
    )

    # ── Sub-summaries ─────────────────────────────────────────────────────
    columns_df    = summarize_columns(df)
    missingness_df = summarize_missingness(df)

    target_info = None
    if target_col is not None:
        try:
            target_info = summarize_target(df, target_col)
        except ValueError as exc:
            warnings.warn(f"profile_dataset: {exc}", stacklevel=2)

    profile = {
        "n_rows":                   n_rows,
        "n_cols":                   n_cols,
        "n_numeric":                n_numeric,
        "n_categorical":            n_categorical,
        "n_datetime":               n_datetime,
        "n_boolean":                n_boolean,
        "n_duplicate_rows":         n_duplicate_rows,
        "pct_duplicate_rows":       pct_duplicate_rows,
        "dataset_completeness_pct": dataset_completeness_pct,
        "memory_usage_mb":          memory_usage_mb,
        "columns":                  columns_df,
        "target":                   target_info,
        "missingness":              missingness_df,
    }

    # ── Optional exports ──────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _export_profile_json(profile, out / "profile_summary.json")
        _export_profile_csv(profile, out / "profile_report.csv")

    # ── Optional plots ────────────────────────────────────────────────────
    if plots_dir is not None:
        plots = Path(plots_dir)
        plots.mkdir(parents=True, exist_ok=True)
        _save_profile_plots(profile, df, plots)

    return profile


# ================================================================
# EXPORT HELPERS
# ================================================================


def _export_profile_json(profile, path):
    payload = _profile_to_dict(profile)
    Path(path).write_text(
        json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8"
    )


def _export_profile_csv(profile, path):
    profile["columns"].to_csv(path, index=False)


# ================================================================
# PLOT ORCHESTRATION
# ================================================================


def _save_profile_plots(profile, df, plots_dir):
    """Save all three profile plots to plots_dir."""
    try:
        from src.plots import (
            plot_class_distribution,
            plot_missing_heatmap,
            plot_numeric_distributions,
        )
    except ImportError as exc:
        warnings.warn(f"data_profile: could not import plot functions: {exc}")
        return

    plot_missing_heatmap(
        df,
        title="Missing Value Heatmap",
        save_path=plots_dir / "plot_missing_heatmap.png",
    )

    if profile.get("target"):
        vc = profile["target"].get("value_counts", {})
        if vc:
            plot_class_distribution(
                vc,
                title=f"Class Distribution — {profile['target']['column']}",
                save_path=plots_dir / "plot_class_distribution.png",
            )

    num_cols = profile["columns"].loc[
        profile["columns"]["kind"] == "numeric", "column"
    ].tolist()
    if num_cols:
        plot_numeric_distributions(
            df[num_cols],
            title="Numeric Column Distributions",
            save_path=plots_dir / "plot_numeric_distributions.png",
        )
