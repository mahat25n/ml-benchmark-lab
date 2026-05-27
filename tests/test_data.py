"""Tests for src/data.py — loading, preprocessing, splitting."""

import numpy as np
import pandas as pd
import pytest


# ================================================================
# TestLoadData
# ================================================================

class TestLoadData:

    def test_reads_csv(self, tmp_path):
        from src.data import load_data
        df_orig = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p = tmp_path / "data.csv"
        df_orig.to_csv(p, index=False)
        result = load_data(str(p))
        assert result.shape == (3, 2)

    def test_drops_na_by_default(self, tmp_path):
        from src.data import load_data
        df_orig = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        p = tmp_path / "data.csv"
        df_orig.to_csv(p, index=False)
        result = load_data(str(p))
        assert result.shape == (2, 2)

    def test_default_na_strings_recognized(self, tmp_path):
        from src.data import load_data
        # "?" is in _DEFAULT_NA_VALUES and should be treated as NA
        df_orig = pd.DataFrame({"a": ["1", "?", "3"], "b": ["4", "5", "6"]})
        p = tmp_path / "data.csv"
        df_orig.to_csv(p, index=False)
        result = load_data(str(p))
        assert result.shape == (2, 2)

    def test_custom_na_values_accepted(self, tmp_path):
        from src.data import load_data
        df_orig = pd.DataFrame({"a": ["1", "MISSING", "3"], "b": [4, 5, 6]})
        p = tmp_path / "data.csv"
        df_orig.to_csv(p, index=False)
        # "MISSING" is not in the default list; pass it explicitly
        result = load_data(str(p), na_values=["MISSING"])
        assert result.shape == (2, 2)

    def test_drop_na_false_keeps_rows(self, tmp_path):
        from src.data import load_data
        df_orig = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        p = tmp_path / "data.csv"
        df_orig.to_csv(p, index=False)
        result = load_data(str(p), drop_na=False)
        assert result.shape == (3, 2)


# ================================================================
# TestPreprocessData
# ================================================================

class TestPreprocessData:

    def _make_df(self, n=8):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "num1": rng.standard_normal(n),
            "num2": rng.standard_normal(n),
            "target": [0, 1] * (n // 2),
        })

    def test_returns_arrays_and_preprocessor(self):
        from src.data import preprocess_data
        df = self._make_df()
        X, y, pp = preprocess_data(df, "target")
        assert X.shape == (8, 2)
        assert y.shape == (8,)
        assert isinstance(pp, dict)

    def test_preprocessor_has_required_keys(self):
        from src.data import preprocess_data
        _, _, pp = preprocess_data(self._make_df(), "target")
        for key in ("scaler", "feature_encoders", "target_encoder",
                    "feature_names", "target_col", "n_classes"):
            assert key in pp

    def test_categorical_columns_encoded(self):
        from src.data import preprocess_data
        df = pd.DataFrame({
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        })
        X, _, pp = preprocess_data(df, "target")
        assert X.dtype.kind == "f"           # float after encoding
        assert "cat" in pp["feature_encoders"]

    def test_string_target_label_encoded(self):
        from src.data import preprocess_data
        df = pd.DataFrame({
            "x": np.arange(8, dtype=float),
            "label": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        })
        _, y, pp = preprocess_data(df, "label")
        assert y.dtype.kind in ("i", "u")    # integer after encoding
        assert pp["target_encoder"] is not None

    def test_numeric_target_preserved(self):
        from src.data import preprocess_data
        df = self._make_df()
        _, y, pp = preprocess_data(df, "target")
        assert pp["target_encoder"] is None

    def test_scaler_false_skips_scaling(self):
        from src.data import preprocess_data
        _, _, pp = preprocess_data(self._make_df(), "target", scaler=False)
        assert pp["scaler"] is None

    def test_n_classes_binary(self):
        from src.data import preprocess_data
        _, _, pp = preprocess_data(self._make_df(), "target")
        assert pp["n_classes"] == 2

    def test_n_classes_multiclass(self):
        from src.data import preprocess_data
        df = pd.DataFrame({
            "x": np.arange(6, dtype=float),
            "target": [0, 1, 2, 0, 1, 2],
        })
        _, _, pp = preprocess_data(df, "target")
        assert pp["n_classes"] == 3

    def test_target_excluded_from_features(self):
        from src.data import preprocess_data
        df = self._make_df()
        X, _, pp = preprocess_data(df, "target")
        assert "target" not in pp["feature_names"]
        assert X.shape[1] == 2


# ================================================================
# TestSplitData
# ================================================================

class TestSplitData:

    @pytest.fixture
    def arrays(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        y = np.array([0] * 50 + [1] * 50)
        return X, y

    def test_total_samples_preserved(self, arrays):
        from src.data import split_data
        X, y = arrays
        X_tr, X_te, y_tr, y_te = split_data(X, y)
        assert X_tr.shape[0] + X_te.shape[0] == 100

    def test_test_size_respected(self, arrays):
        from src.data import split_data
        X, y = arrays
        _, X_te, _, _ = split_data(X, y, test_size=0.3)
        assert X_te.shape[0] == 30

    def test_feature_dim_preserved(self, arrays):
        from src.data import split_data
        X, y = arrays
        X_tr, X_te, _, _ = split_data(X, y)
        assert X_tr.shape[1] == X_te.shape[1] == 4

    def test_reproducibility(self, arrays):
        from src.data import split_data
        X, y = arrays
        a = split_data(X, y, random_state=42)
        b = split_data(X, y, random_state=42)
        np.testing.assert_array_equal(a[0], b[0])   # same X_train

    def test_different_seeds_differ(self, arrays):
        from src.data import split_data
        X, y = arrays
        a = split_data(X, y, random_state=0)
        b = split_data(X, y, random_state=1)
        assert not np.array_equal(a[0], b[0])


# ================================================================
# TestLoadAndPreprocess  (integration)
# ================================================================

class TestLoadAndPreprocess:

    def test_returns_five_values(self, tmp_csv):
        from src.data import load_and_preprocess
        result = load_and_preprocess(tmp_csv, "target")
        assert len(result) == 5

    def test_shapes_consistent(self, tmp_csv):
        from src.data import load_and_preprocess
        X_tr, X_te, y_tr, y_te, _ = load_and_preprocess(tmp_csv, "target")
        assert X_tr.shape[0] == y_tr.shape[0]
        assert X_te.shape[0] == y_te.shape[0]
        assert X_tr.shape[1] == X_te.shape[1]

    def test_total_samples(self, tmp_csv):
        from src.data import load_and_preprocess
        X_tr, X_te, _, _, _ = load_and_preprocess(tmp_csv, "target", test_size=0.25)
        # conftest tmp_csv has 200 rows
        assert X_tr.shape[0] + X_te.shape[0] == 200

    def test_preprocessor_keys_complete(self, tmp_csv):
        from src.data import load_and_preprocess
        *_, pp = load_and_preprocess(tmp_csv, "target")
        for key in ("scaler", "feature_encoders", "target_encoder",
                    "feature_names", "target_col", "n_classes"):
            assert key in pp
