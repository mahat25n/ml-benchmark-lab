"""
Utility functions for time-series data preparation.

Transforms sequential data into supervised learning format by adding
lag and rolling features. All transformations preserve temporal order
so that only past observations are used to construct each row —
no future leakage.
"""

import warnings

import pandas as pd


def _validate_col(df, target_col, fn_name):
    if target_col not in df.columns:
        raise ValueError(
            f"{fn_name}: target_col '{target_col}' not found in DataFrame columns."
        )


def _drop_nan_with_warning(df, fn_name):
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        warnings.warn(
            f"{fn_name}: dropped {dropped} row(s) containing NaN.",
            stacklevel=3,
        )
    return df


def create_lag_features(df, lags, target_col):
    """
    Add lagged copies of the target column to the DataFrame.

    Each row receives the target values from ``lags`` prior steps.
    Rows at the start of the sequence where lag values are unavailable
    are dropped.

    Parameters
    ----------
    df         : pd.DataFrame  Input data containing ``target_col``.
    lags       : list[int]     Positive integers, e.g. ``[1, 2, 3]``.
    target_col : str           Column to lag.

    Returns
    -------
    pd.DataFrame with added columns ``{target_col}_lag_{k}`` for each k.
    Row count is reduced by ``max(lags)`` due to NaN-drop.

    Raises
    ------
    ValueError  If ``target_col`` is absent or any lag value <= 0.
    """
    _validate_col(df, target_col, "create_lag_features")

    lags = sorted(set(int(k) for k in lags))
    if any(k <= 0 for k in lags):
        raise ValueError("create_lag_features: all lag values must be positive integers.")

    result = df.copy()
    for k in lags:
        result[f"{target_col}_lag_{k}"] = result[target_col].shift(k)

    return _drop_nan_with_warning(result, "create_lag_features")


def create_rolling_features(df, windows, target_col):
    """
    Add rolling summary statistics of the target column to the DataFrame.

    For each window size ``w``, adds:

    * ``{target_col}_rolling_mean_{w}``
    * ``{target_col}_rolling_std_{w}``

    All windows use ``min_periods=w`` so the first ``(w - 1)`` rows
    produce NaN and are dropped.

    Parameters
    ----------
    df         : pd.DataFrame  Input data containing ``target_col``.
    windows    : list[int]     Window sizes in number of observations (>= 2).
    target_col : str           Column to compute rolling stats over.

    Returns
    -------
    pd.DataFrame with added rolling feature columns.
    Row count is reduced by ``max(windows) - 1`` due to NaN-drop.

    Raises
    ------
    ValueError  If ``target_col`` is absent or any window size < 2.
    """
    _validate_col(df, target_col, "create_rolling_features")

    windows = sorted(set(int(w) for w in windows))
    if any(w < 2 for w in windows):
        raise ValueError("create_rolling_features: all window sizes must be >= 2.")

    result = df.copy()
    for w in windows:
        rolled = result[target_col].rolling(window=w, min_periods=w)
        result[f"{target_col}_rolling_mean_{w}"] = rolled.mean()
        result[f"{target_col}_rolling_std_{w}"]  = rolled.std()

    return _drop_nan_with_warning(result, "create_rolling_features")
