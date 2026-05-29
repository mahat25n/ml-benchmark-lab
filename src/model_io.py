"""
Model persistence and inference utilities.

Saves and loads sklearn estimators, preprocessing objects, and benchmark
pipelines using joblib. No external model-serving dependencies required.

Directory layout (when save_dir is supplied)
--------------------------------------------
    <save_dir>/
        model.pkl              Fitted sklearn estimator
        pipeline.pkl           Estimator + preprocessor dict together
        model_metadata.json    Type, timestamps, feature names, versions

Batch inference
---------------
    predict_from_csv(csv_path, model_path, preprocessor_path)
    batch_predict(X, model_or_path)
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np

_JOBLIB_MISSING = (
    "joblib is required for model persistence. "
    "Install it with:  pip install joblib\n"
    "(It ships with scikit-learn, so it is almost certainly already installed.)"
)


def _joblib():
    try:
        import joblib
        return joblib
    except ImportError as exc:
        raise ImportError(_JOBLIB_MISSING) from exc


# ================================================================
# METADATA HELPERS
# ================================================================


def _collect_metadata(model, *, feature_names=None, extra=None):
    """Build the model_metadata.json payload."""
    import sys
    import importlib.metadata

    model_type  = type(model).__name__
    model_module = type(model).__module__

    versions = {}
    for pkg in ("scikit-learn", "numpy", "joblib", "xgboost"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    try:
        import src as _src
        versions["ml-benchmark-lab"] = getattr(_src, "__version__", "unknown")
    except Exception:
        versions["ml-benchmark-lab"] = "unknown"

    payload = {
        "model_type":        model_type,
        "model_module":      model_module,
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_names":     list(feature_names) if feature_names is not None else None,
        "python_version":    sys.version.split()[0],
        "framework_versions": versions,
    }
    if extra:
        payload.update(extra)
    return payload


def _write_metadata(metadata, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )


# ================================================================
# SAVE / LOAD — MODEL
# ================================================================


def save_model(model, save_path, *, feature_names=None, extra_metadata=None):
    """
    Persist a fitted sklearn estimator to disk.

    Writes two files:
        <save_path>          The pickled model  (model.pkl if save_path is a dir)
        <save_path>.meta.json  OR  <dir>/model_metadata.json

    Parameters
    ----------
    model          : fitted sklearn estimator
    save_path      : str or Path
        Target file (e.g. ``outputs/model.pkl``) or directory.
        When a directory is given, the file is saved as ``<dir>/model.pkl``.
    feature_names  : list[str] or None
        Column names the model was trained on — stored in metadata.
    extra_metadata : dict or None
        Additional key-value pairs merged into model_metadata.json.

    Returns
    -------
    dict with keys ``model_path`` and ``metadata_path``.
    """
    jl = _joblib()
    save_path = Path(save_path)

    if save_path.is_dir() or not save_path.suffix:
        save_path.mkdir(parents=True, exist_ok=True)
        model_path    = save_path / "model.pkl"
        metadata_path = save_path / "model_metadata.json"
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_path    = save_path
        metadata_path = save_path.with_suffix(".meta.json")

    jl.dump(model, model_path)

    metadata = _collect_metadata(
        model,
        feature_names=feature_names,
        extra=extra_metadata,
    )
    _write_metadata(metadata, metadata_path)

    return {"model_path": str(model_path), "metadata_path": str(metadata_path)}


def load_model(model_path):
    """
    Load a joblib-serialised sklearn estimator from disk.

    Parameters
    ----------
    model_path : str or Path

    Returns
    -------
    Fitted sklearn estimator.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    """
    jl = _joblib()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return jl.load(model_path)


# ================================================================
# SAVE / LOAD — PIPELINE
# ================================================================


def save_pipeline(model, preprocessor, save_path, *, extra_metadata=None):
    """
    Persist a (model, preprocessor) pair as a single pipeline artifact.

    ``preprocessor`` is the dict returned by ``preprocess_data()``::

        {
          "scaler":           fitted StandardScaler or None,
          "feature_encoders": {col: fitted LabelEncoder},
          "target_encoder":   fitted LabelEncoder or None,
          "feature_names":    [str, ...],
          "target_col":       str,
          "n_classes":        int,
        }

    Writes three files:
        <dir>/pipeline.pkl            Combined artifact
        <dir>/model.pkl               Model-only convenience copy
        <dir>/model_metadata.json     Metadata

    Parameters
    ----------
    model        : fitted sklearn estimator
    preprocessor : dict   Output of preprocess_data().
    save_path    : str or Path   Target directory (created if needed).
    extra_metadata : dict or None

    Returns
    -------
    dict with keys ``pipeline_path``, ``model_path``, ``metadata_path``.
    """
    jl = _joblib()
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    pipeline_path = save_path / "pipeline.pkl"
    model_path    = save_path / "model.pkl"
    metadata_path = save_path / "model_metadata.json"

    jl.dump({"model": model, "preprocessor": preprocessor}, pipeline_path)
    jl.dump(model, model_path)

    feature_names = (
        preprocessor.get("feature_names") if isinstance(preprocessor, dict) else None
    )
    metadata = _collect_metadata(
        model,
        feature_names=feature_names,
        extra={
            "has_pipeline":   True,
            "target_col":     preprocessor.get("target_col") if isinstance(preprocessor, dict) else None,
            "n_classes":      preprocessor.get("n_classes")  if isinstance(preprocessor, dict) else None,
            **(extra_metadata or {}),
        },
    )
    _write_metadata(metadata, metadata_path)

    return {
        "pipeline_path": str(pipeline_path),
        "model_path":    str(model_path),
        "metadata_path": str(metadata_path),
    }


def load_pipeline(pipeline_path):
    """
    Load a (model, preprocessor) pair saved by ``save_pipeline()``.

    Parameters
    ----------
    pipeline_path : str or Path
        Path to ``pipeline.pkl``, or the directory that contains it.

    Returns
    -------
    tuple (model, preprocessor)
        model        : fitted sklearn estimator
        preprocessor : dict (same schema as preprocess_data() output)

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    """
    jl = _joblib()
    path = Path(pipeline_path)
    if path.is_dir():
        path = path / "pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    obj = jl.load(path)
    if isinstance(obj, dict) and "model" in obj and "preprocessor" in obj:
        return obj["model"], obj["preprocessor"]
    # Backwards-compat: plain estimator with no preprocessor
    return obj, {}


# ================================================================
# INFERENCE
# ================================================================


def _apply_preprocessor(df, preprocessor):
    """
    Transform a raw DataFrame using a saved preprocessor dict.

    Handles unseen categories by mapping them to the most-frequent known class
    (or 0) with a warning rather than raising.

    Returns
    -------
    np.ndarray  (n_samples, n_features)  float
    """
    import pandas as pd

    preprocessor = preprocessor or {}
    feature_names    = preprocessor.get("feature_names")
    feature_encoders = preprocessor.get("feature_encoders", {})
    scaler           = preprocessor.get("scaler")

    # Align columns to training order
    if feature_names:
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Input is missing columns that were present at training time: {missing}"
            )
        df = df[feature_names].copy()
    else:
        df = df.copy()

    # Encode categoricals
    for col, enc in feature_encoders.items():
        if col not in df.columns:
            continue
        known = set(enc.classes_)
        unseen = set(df[col].dropna().astype(str).unique()) - known
        if unseen:
            warnings.warn(
                f"Column '{col}': unseen categories {unseen} mapped to first known class.",
                stacklevel=3,
            )
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in known else enc.classes_[0]
            )
        df[col] = enc.transform(df[col].astype(str))

    X = df.to_numpy(dtype=float)

    if scaler is not None:
        X = scaler.transform(X)

    return X


def predict_from_csv(
    csv_path,
    model_path,
    *,
    preprocessor_path=None,
    target_col=None,
    na_values=None,
    output_path=None,
):
    """
    Load a CSV, apply preprocessing, run inference, and return predictions.

    Parameters
    ----------
    csv_path         : str or Path   Path to the input CSV.
    model_path       : str or Path
        Path to ``model.pkl`` **or** a directory containing ``pipeline.pkl``.
        When a directory with ``pipeline.pkl`` is supplied, the bundled
        preprocessor is used automatically (no need for preprocessor_path).
    preprocessor_path : str or Path or None
        Path to a separately-saved ``pipeline.pkl`` whose preprocessor is
        extracted.  Ignored when model_path resolves to a pipeline directory.
    target_col       : str or None
        Column to drop before inference (the training label, if present).
    na_values        : list[str] or None
        Extra NA markers passed to ``pd.read_csv``.
    output_path      : str or Path or None
        When set, predictions are written to this CSV path.

    Returns
    -------
    np.ndarray  Predicted labels / values (shape: (n_samples,)).
    """
    import pandas as pd
    from src.data import _DEFAULT_NA_VALUES

    # Load CSV
    extra_na = _DEFAULT_NA_VALUES if na_values is None else list(na_values)
    df = pd.read_csv(csv_path, na_values=extra_na, keep_default_na=True)
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Resolve model + preprocessor
    mp = Path(model_path)
    pipeline_candidate = (mp / "pipeline.pkl") if mp.is_dir() else None

    if pipeline_candidate and pipeline_candidate.exists():
        model, preprocessor = load_pipeline(mp)
    else:
        model = load_model(model_path)
        preprocessor = {}
        if preprocessor_path:
            _, preprocessor = load_pipeline(preprocessor_path)

    # Apply preprocessing
    if preprocessor:
        X = _apply_preprocessor(df, preprocessor)
    else:
        X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    predictions = model.predict(X)

    if output_path:
        result_df = df.copy()
        result_df["prediction"] = predictions
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)

    return predictions


def batch_predict(X, model_or_path, *, chunk_size=10_000):
    """
    Run inference on a numpy array (or large array in chunks).

    Parameters
    ----------
    X            : np.ndarray  shape (n_samples, n_features)
    model_or_path: fitted sklearn estimator **or** path to ``model.pkl``
    chunk_size   : int   Number of rows per inference chunk (default 10 000).
        Use a smaller value to bound peak memory on very large arrays.

    Returns
    -------
    np.ndarray  (n_samples,)
    """
    if isinstance(model_or_path, (str, Path)):
        model = load_model(model_or_path)
    else:
        model = model_or_path

    X = np.asarray(X)
    n = X.shape[0]

    if n <= chunk_size:
        return model.predict(X)

    parts = []
    for start in range(0, n, chunk_size):
        parts.append(model.predict(X[start: start + chunk_size]))
    return np.concatenate(parts)


# ================================================================
# EXPERIMENT TRACKER INTEGRATION
# ================================================================


def save_model_for_run(model, run_dir, *, name="model", feature_names=None,
                       preprocessor=None, extra_metadata=None):
    """
    Save a model (and optional pipeline) into an existing experiment run directory.

    Writes artifacts into ``<run_dir>/models/<name>/``.

    Parameters
    ----------
    model         : fitted sklearn estimator
    run_dir       : str or Path   Path to the run directory (tracker.run_dir).
    name          : str           Sub-directory name (default: "model").
    feature_names : list[str] or None
    preprocessor  : dict or None  Output of preprocess_data(); saves pipeline.pkl.
    extra_metadata: dict or None

    Returns
    -------
    dict   Same as save_pipeline() or save_model().
    """
    target_dir = Path(run_dir) / "models" / name
    if preprocessor is not None:
        return save_pipeline(
            model, preprocessor, target_dir, extra_metadata=extra_metadata
        )
    return save_model(
        model, target_dir, feature_names=feature_names, extra_metadata=extra_metadata
    )
