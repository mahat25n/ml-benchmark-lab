import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Strings the notebook used as missing-value placeholders; extended for
# common real-world variants. Merged with pandas' own default NA set.
_DEFAULT_NA_VALUES = ["?", "N/A", "n/a", "na", "none", "None", "unknown", "Unknown", "-"]


# ================================================================
# LOAD
# ================================================================


def load_data(csv_path, *, na_values=None, drop_na=True):
    """
    Read a CSV and normalise missing-value markers.

    Parameters
    ----------
    csv_path   : str or Path
    na_values  : list[str] or None
        Additional strings to recognise as NA, merged with the built-in
        defaults above. Pass an empty list to disable the extended defaults.
    drop_na    : bool
        Drop every row that contains at least one NA after replacement.

    Returns
    -------
    pd.DataFrame  (reset index)
    """
    extra = _DEFAULT_NA_VALUES if na_values is None else list(na_values)
    df = pd.read_csv(csv_path, na_values=extra, keep_default_na=True)
    if drop_na:
        df = df.dropna()
    return df.reset_index(drop=True)


# ================================================================
# PREPROCESS
# ================================================================


def preprocess_data(df, target_col, *, scaler=None, drop_na=False):
    """
    Encode categoricals, optionally scale features, and separate X from y.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str   Name of the target column — never hardcoded.
    scaler     : sklearn-compatible transformer, None, or False
        None  → StandardScaler() is created and fitted (default).
        False → scaling is skipped; X is returned as a float array.
        Any sklearn scaler → that scaler is fitted and applied.
    drop_na    : bool
        Drop NA rows before processing. Set True when calling standalone
        without a prior load_data(..., drop_na=True).

    Returns
    -------
    X           : np.ndarray  shape (n_samples, n_features), float
    y           : np.ndarray  shape (n_samples,), int or float
    preprocessor: dict
        {
          "scaler"          : fitted scaler or None,
          "feature_encoders": {col: fitted LabelEncoder},
          "target_encoder"  : fitted LabelEncoder or None,
          "feature_names"   : [str, ...],
          "target_col"      : str,
          "n_classes"       : int,   # unique values in encoded y
        }
    """
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # --- Encode categorical feature columns ---
    X = X_raw.copy()
    feature_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        feature_encoders[col] = enc

    # --- Encode target if non-numeric ---
    # Handles string labels ("yes"/"no", class names) and pandas Categorical.
    target_encoder = None
    if not pd.api.types.is_numeric_dtype(y_raw):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.to_numpy()

    # --- Scale ---
    feature_names = X.columns.tolist()
    if scaler is False:
        X_out = X.to_numpy(dtype=float)
        fitted_scaler = None
    else:
        fitted_scaler = StandardScaler() if scaler is None else scaler
        X_out = fitted_scaler.fit_transform(X)

    preprocessor = {
        "scaler":           fitted_scaler,
        "feature_encoders": feature_encoders,
        "target_encoder":   target_encoder,
        "feature_names":    feature_names,
        "target_col":       target_col,
        "n_classes":        int(np.unique(y).size),
    }

    return X_out, y.astype(float) if y.dtype.kind == "f" else y, preprocessor


# ================================================================
# SPLIT
# ================================================================


def split_data(X, y, *, test_size=0.2, random_state=42, stratify=True):
    """
    Partition X and y into train and test sets.

    Parameters
    ----------
    test_size    : float   Fraction of data held out for testing.
    random_state : int
    stratify     : bool
        True  → stratified split preserving class proportions (classification).
        False → plain random split (regression or continuous targets).
        When True but y is continuous, falls back to unstratified with a warning.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    try:
        return train_test_split(
            X, y, test_size=test_size, stratify=strat, random_state=random_state
        )
    except ValueError as exc:
        warnings.warn(
            f"Stratified split failed ({exc}). "
            "Falling back to non-stratified split. "
            "Pass stratify=False to suppress this warning."
        )
        return train_test_split(
            X, y, test_size=test_size, stratify=None, random_state=random_state
        )


# ================================================================
# PIPELINE CONVENIENCE WRAPPER
# ================================================================


def load_and_preprocess(
    csv_path,
    target_col,
    *,
    na_values=None,
    drop_na=True,
    scaler=None,
    test_size=0.2,
    random_state=42,
    stratify=True,
):
    """
    Full pipeline: load → preprocess → split.

    Parameters mirror load_data(), preprocess_data(), and split_data().

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    preprocessor                      : dict  (see preprocess_data)
    """
    df = load_data(csv_path, na_values=na_values, drop_na=drop_na)

    # drop_na=False here — load_data already handled it
    X, y, preprocessor = preprocess_data(
        df, target_col, scaler=scaler, drop_na=False
    )

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train, X_test, y_train, y_test, preprocessor


# ================================================================
# FUTURE LOADERS  (not yet implemented)
# ================================================================
#
# Time-series / sequential data:
#   load_time_series(csv_path, target_col, datetime_col, freq=None)
#   split_time_series(X, y, test_size, gap=0)          # no shuffling
#
# Panel / longitudinal data:
#   load_panel(csv_path, target_col, entity_col, time_col)
#   split_panel(X, y, entity_col, test_entities=None)
#
# Image datasets:
#   load_image_folder(root_dir, target_size, grayscale=False)
#
# Text / NLP:
#   load_text_corpus(csv_path, text_col, target_col, max_length=None)
#
# Large / chunked CSV:
#   load_data_chunked(csv_path, chunksize=10_000, ...)  -> generator
