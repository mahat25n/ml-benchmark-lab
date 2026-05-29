"""
Tests for src/model_io.py — model persistence and inference utilities.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.model_io import (
    batch_predict,
    load_model,
    load_pipeline,
    predict_from_csv,
    save_model,
    save_model_for_run,
    save_pipeline,
)


# ================================================================
# FIXTURES
# ================================================================


@pytest.fixture
def clf_model():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    m = RandomForestClassifier(n_estimators=10, random_state=42)
    m.fit(X, y)
    return m, X, y


@pytest.fixture
def reg_model():
    X, y = make_regression(n_samples=200, n_features=4, random_state=42)
    m = LinearRegression()
    m.fit(X, y)
    return m, X, y


@pytest.fixture
def preprocessor():
    """Minimal preprocessor dict matching preprocess_data() output."""
    scaler = StandardScaler()
    X_dummy = np.random.default_rng(0).standard_normal((50, 3))
    scaler.fit(X_dummy)
    return {
        "scaler":           scaler,
        "feature_encoders": {},
        "target_encoder":   None,
        "feature_names":    ["a", "b", "c"],
        "target_col":       "label",
        "n_classes":        2,
    }


@pytest.fixture
def simple_csv(tmp_path):
    df = pd.DataFrame({
        "a": np.random.default_rng(1).standard_normal(30),
        "b": np.random.default_rng(2).standard_normal(30),
        "c": np.random.default_rng(3).standard_normal(30),
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p, df


# ================================================================
# save_model / load_model
# ================================================================


class TestSaveModel:
    def test_returns_dict(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path / "model.pkl")
        assert isinstance(result, dict)

    def test_model_path_key(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path / "model.pkl")
        assert "model_path" in result

    def test_metadata_path_key(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path / "model.pkl")
        assert "metadata_path" in result

    def test_file_created(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path / "model.pkl")
        assert Path(result["model_path"]).exists()

    def test_metadata_file_created(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path / "model.pkl")
        assert Path(result["metadata_path"]).exists()

    def test_dir_input_creates_model_pkl(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path)
        assert Path(result["model_path"]).name == "model.pkl"

    def test_dir_input_creates_metadata_json(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path)
        assert Path(result["metadata_path"]).name == "model_metadata.json"

    def test_feature_names_in_metadata(self, clf_model, tmp_path):
        m, X, _ = clf_model
        names = ["f1", "f2", "f3", "f4", "f5"]
        result = save_model(m, tmp_path, feature_names=names)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta["feature_names"] == names

    def test_model_type_in_metadata(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta["model_type"] == "RandomForestClassifier"

    def test_training_timestamp_in_metadata(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert "training_timestamp" in meta
        assert meta["training_timestamp"]  # non-empty

    def test_framework_versions_in_metadata(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert "framework_versions" in meta
        assert "scikit-learn" in meta["framework_versions"]

    def test_extra_metadata_merged(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path, extra_metadata={"experiment": "test_run"})
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta["experiment"] == "test_run"

    def test_creates_parent_dir(self, clf_model, tmp_path):
        m, X, _ = clf_model
        nested = tmp_path / "a" / "b" / "c"
        save_model(m, nested)
        assert nested.exists()


class TestLoadModel:
    def test_returns_estimator(self, clf_model, tmp_path):
        m, X, y = clf_model
        result = save_model(m, tmp_path)
        loaded = load_model(result["model_path"])
        assert hasattr(loaded, "predict")

    def test_predictions_match(self, clf_model, tmp_path):
        m, X, y = clf_model
        orig_preds = m.predict(X)
        result = save_model(m, tmp_path)
        loaded = load_model(result["model_path"])
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")

    def test_load_regression_model(self, reg_model, tmp_path):
        m, X, y = reg_model
        result = save_model(m, tmp_path)
        loaded = load_model(result["model_path"])
        np.testing.assert_allclose(m.predict(X), loaded.predict(X), rtol=1e-6)


# ================================================================
# save_pipeline / load_pipeline
# ================================================================


class TestSavePipeline:
    def test_returns_dict(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        assert isinstance(result, dict)

    def test_pipeline_path_key(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        assert "pipeline_path" in result

    def test_model_path_key(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        assert "model_path" in result

    def test_pipeline_pkl_created(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        assert Path(result["pipeline_path"]).exists()

    def test_model_pkl_created(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        assert Path(result["model_path"]).exists()

    def test_metadata_pkl_created(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        assert Path(result["metadata_path"]).exists()

    def test_has_pipeline_in_metadata(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta.get("has_pipeline") is True

    def test_target_col_in_metadata(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta.get("target_col") == "label"

    def test_feature_names_in_metadata(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        meta = json.loads(Path(result["metadata_path"]).read_text())
        assert meta["feature_names"] == ["a", "b", "c"]


class TestLoadPipeline:
    def test_returns_tuple(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        out = load_pipeline(result["pipeline_path"])
        assert isinstance(out, tuple) and len(out) == 2

    def test_model_has_predict(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        loaded_m, _ = load_pipeline(result["pipeline_path"])
        assert hasattr(loaded_m, "predict")

    def test_preprocessor_is_dict(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        _, loaded_p = load_pipeline(result["pipeline_path"])
        assert isinstance(loaded_p, dict)

    def test_feature_names_preserved(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        result = save_pipeline(m, preprocessor, tmp_path)
        _, loaded_p = load_pipeline(result["pipeline_path"])
        assert loaded_p["feature_names"] == ["a", "b", "c"]

    def test_directory_input_finds_pipeline_pkl(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        save_pipeline(m, preprocessor, tmp_path)
        loaded_m, loaded_p = load_pipeline(tmp_path)
        assert hasattr(loaded_m, "predict")
        assert isinstance(loaded_p, dict)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "no_pipeline.pkl")

    def test_predictions_match_original(self, clf_model, preprocessor, tmp_path):
        m, X, _ = clf_model
        orig_preds = m.predict(X)
        result = save_pipeline(m, preprocessor, tmp_path)
        loaded_m, _ = load_pipeline(result["pipeline_path"])
        loaded_preds = loaded_m.predict(X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)


# ================================================================
# batch_predict
# ================================================================


class TestBatchPredict:
    def test_returns_ndarray(self, clf_model):
        m, X, _ = clf_model
        preds = batch_predict(X, m)
        assert isinstance(preds, np.ndarray)

    def test_shape_matches_input(self, clf_model):
        m, X, _ = clf_model
        preds = batch_predict(X, m)
        assert preds.shape == (X.shape[0],)

    def test_matches_direct_predict(self, clf_model):
        m, X, _ = clf_model
        direct = m.predict(X)
        batched = batch_predict(X, m)
        np.testing.assert_array_equal(direct, batched)

    def test_chunked_matches_direct(self, clf_model):
        m, X, _ = clf_model
        direct = m.predict(X)
        chunked = batch_predict(X, m, chunk_size=30)
        np.testing.assert_array_equal(direct, chunked)

    def test_accepts_path(self, clf_model, tmp_path):
        m, X, _ = clf_model
        result = save_model(m, tmp_path)
        preds = batch_predict(X, result["model_path"])
        assert preds.shape == (X.shape[0],)

    def test_regression_model(self, reg_model):
        m, X, _ = reg_model
        preds = batch_predict(X, m)
        np.testing.assert_allclose(m.predict(X), preds, rtol=1e-6)

    def test_chunk_smaller_than_data(self, clf_model):
        m, X, _ = clf_model
        direct = m.predict(X)
        preds = batch_predict(X, m, chunk_size=1)
        np.testing.assert_array_equal(direct, preds)


# ================================================================
# predict_from_csv
# ================================================================


class TestPredictFromCsv:
    def test_returns_ndarray(self, simple_csv, preprocessor, tmp_path):
        csv_path, df = simple_csv
        X_train = preprocessor["scaler"].transform(df.values)
        y_train = np.tile([0, 1], len(df))[:len(df)]
        m = LogisticRegression(max_iter=200)
        m.fit(X_train, y_train)
        save_pipeline(m, preprocessor, tmp_path)
        preds = predict_from_csv(csv_path, tmp_path)
        assert isinstance(preds, np.ndarray)

    def test_shape_matches_rows(self, simple_csv, preprocessor, tmp_path):
        csv_path, df = simple_csv
        X_train = preprocessor["scaler"].transform(df.values)
        y_train = np.tile([0, 1], len(df))[:len(df)]
        m = LogisticRegression(max_iter=200)
        m.fit(X_train, y_train)
        save_pipeline(m, preprocessor, tmp_path)
        preds = predict_from_csv(csv_path, tmp_path)
        assert len(preds) == len(df)

    def test_target_col_dropped(self, tmp_path):
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "label": [0, 1, 0],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        X = df[["x1", "x2"]].values.astype(float)
        m = LogisticRegression(max_iter=200)
        m.fit(X, df["label"].values)
        save_model(m, tmp_path / "model.pkl")

        preds = predict_from_csv(csv_path, tmp_path / "model.pkl", target_col="label")
        assert len(preds) == 3

    def test_output_csv_written(self, simple_csv, preprocessor, tmp_path):
        csv_path, df = simple_csv
        X_train = preprocessor["scaler"].transform(df.values)
        y_train = np.tile([0, 1], len(df))[:len(df)]
        m = LogisticRegression(max_iter=200)
        m.fit(X_train, y_train)
        save_pipeline(m, preprocessor, tmp_path / "pipeline_dir")
        out_path = tmp_path / "predictions.csv"
        predict_from_csv(csv_path, tmp_path / "pipeline_dir", output_path=out_path)
        assert out_path.exists()

    def test_output_csv_has_prediction_column(self, simple_csv, preprocessor, tmp_path):
        csv_path, df = simple_csv
        X_train = preprocessor["scaler"].transform(df.values)
        y_train = np.tile([0, 1], len(df))[:len(df)]
        m = LogisticRegression(max_iter=200)
        m.fit(X_train, y_train)
        save_pipeline(m, preprocessor, tmp_path / "pipeline_dir")
        out_path = tmp_path / "predictions.csv"
        predict_from_csv(csv_path, tmp_path / "pipeline_dir", output_path=out_path)
        result_df = pd.read_csv(out_path)
        assert "prediction" in result_df.columns

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            predict_from_csv(tmp_path / "missing.csv", tmp_path / "model.pkl")

    def test_missing_column_raises(self, tmp_path, preprocessor):
        df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})  # missing "c"
        csv_path = tmp_path / "partial.csv"
        df.to_csv(csv_path, index=False)

        X_train = preprocessor["scaler"].transform(
            np.random.default_rng(0).standard_normal((10, 3))
        )
        m = LogisticRegression(max_iter=200)
        m.fit(X_train, [0, 1] * 5)
        save_pipeline(m, preprocessor, tmp_path / "pl")

        with pytest.raises(ValueError, match="missing columns"):
            predict_from_csv(csv_path, tmp_path / "pl")


# ================================================================
# save_model_for_run
# ================================================================


class TestSaveModelForRun:
    def test_creates_models_subdir(self, clf_model, tmp_path):
        m, _, _ = clf_model
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_model_for_run(m, run_dir, name="rf")
        assert (run_dir / "models" / "rf").exists()

    def test_model_pkl_in_subdir(self, clf_model, tmp_path):
        m, _, _ = clf_model
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_model_for_run(m, run_dir, name="rf")
        assert (run_dir / "models" / "rf" / "model.pkl").exists()

    def test_pipeline_pkl_when_preprocessor_given(self, clf_model, preprocessor, tmp_path):
        m, _, _ = clf_model
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_model_for_run(m, run_dir, name="rf_pipe", preprocessor=preprocessor)
        assert (run_dir / "models" / "rf_pipe" / "pipeline.pkl").exists()

    def test_returns_paths_dict(self, clf_model, tmp_path):
        m, _, _ = clf_model
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = save_model_for_run(m, run_dir, name="rf")
        assert isinstance(result, dict)
        assert "model_path" in result

    def test_feature_names_stored(self, clf_model, tmp_path):
        m, _, _ = clf_model
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        result = save_model_for_run(m, run_dir, name="rf", feature_names=["f1", "f2"])
        meta_path = Path(result["metadata_path"])
        meta = json.loads(meta_path.read_text())
        assert meta["feature_names"] == ["f1", "f2"]

    def test_default_name_is_model(self, clf_model, tmp_path):
        m, _, _ = clf_model
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        save_model_for_run(m, run_dir)
        assert (run_dir / "models" / "model").exists()
