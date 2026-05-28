"""Tests for time-series benchmarking support."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression


# ================================================================
# Module fixtures
# ================================================================


@pytest.fixture(scope="module")
def ts_df():
    """Simple time series DataFrame: 200 steps of sine + noise."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    y = np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(n)
    return pd.DataFrame({"step": t.astype(float), "value": y})


@pytest.fixture(scope="module")
def ts_csv(tmp_path_factory, ts_df):
    """Temporary CSV for the time-series benchmark integration tests."""
    p = tmp_path_factory.mktemp("ts_data") / "ts.csv"
    ts_df.to_csv(p, index=False)
    return str(p)


# ================================================================
# TestCreateLagFeatures
# ================================================================


class TestCreateLagFeatures:

    def test_returns_dataframe(self, ts_df):
        from src.time_series import create_lag_features
        out = create_lag_features(ts_df.copy(), [1, 2], "value")
        assert isinstance(out, pd.DataFrame)

    def test_adds_lag_columns(self, ts_df):
        from src.time_series import create_lag_features
        out = create_lag_features(ts_df.copy(), [1, 3], "value")
        assert "value_lag_1" in out.columns
        assert "value_lag_3" in out.columns

    def test_drops_nan_rows(self, ts_df):
        from src.time_series import create_lag_features
        out = create_lag_features(ts_df.copy(), [1, 2, 3], "value")
        assert out.isnull().sum().sum() == 0
        assert len(out) == len(ts_df) - 3

    def test_lag_values_are_correct(self, ts_df):
        from src.time_series import create_lag_features
        out = create_lag_features(ts_df.copy(), [1], "value")
        # After dropping row 0 (NaN lag), row index 0 of out corresponds
        # to original row 1: out["value"][0] == ts_df["value"][1]
        # and out["value_lag_1"][0] == ts_df["value"][0]
        assert np.isclose(out["value_lag_1"].iloc[0], ts_df["value"].iloc[0])
        assert np.isclose(out["value"].iloc[0], ts_df["value"].iloc[1])

    def test_missing_column_raises(self, ts_df):
        from src.time_series import create_lag_features
        with pytest.raises(ValueError, match="not found"):
            create_lag_features(ts_df.copy(), [1], "nonexistent")

    def test_nonpositive_lag_raises(self, ts_df):
        from src.time_series import create_lag_features
        with pytest.raises(ValueError, match="positive"):
            create_lag_features(ts_df.copy(), [0, 1], "value")

    def test_single_lag(self, ts_df):
        from src.time_series import create_lag_features
        out = create_lag_features(ts_df.copy(), [5], "value")
        assert "value_lag_5" in out.columns
        assert len(out) == len(ts_df) - 5

    def test_original_columns_preserved(self, ts_df):
        from src.time_series import create_lag_features
        out = create_lag_features(ts_df.copy(), [1], "value")
        assert "step" in out.columns
        assert "value" in out.columns


# ================================================================
# TestCreateRollingFeatures
# ================================================================


class TestCreateRollingFeatures:

    def test_returns_dataframe(self, ts_df):
        from src.time_series import create_rolling_features
        out = create_rolling_features(ts_df.copy(), [3], "value")
        assert isinstance(out, pd.DataFrame)

    def test_adds_mean_and_std_columns(self, ts_df):
        from src.time_series import create_rolling_features
        out = create_rolling_features(ts_df.copy(), [3, 5], "value")
        for w in [3, 5]:
            assert f"value_rolling_mean_{w}" in out.columns
            assert f"value_rolling_std_{w}" in out.columns

    def test_no_nan_after_drop(self, ts_df):
        from src.time_series import create_rolling_features
        out = create_rolling_features(ts_df.copy(), [3], "value")
        assert out.isnull().sum().sum() == 0

    def test_row_count_reduced_correctly(self, ts_df):
        from src.time_series import create_rolling_features
        out = create_rolling_features(ts_df.copy(), [4], "value")
        assert len(out) == len(ts_df) - 3  # window=4 removes first 3 rows

    def test_missing_column_raises(self, ts_df):
        from src.time_series import create_rolling_features
        with pytest.raises(ValueError, match="not found"):
            create_rolling_features(ts_df.copy(), [3], "nonexistent")

    def test_window_less_than_two_raises(self, ts_df):
        from src.time_series import create_rolling_features
        with pytest.raises(ValueError, match="2"):
            create_rolling_features(ts_df.copy(), [1], "value")


# ================================================================
# TestForecastMetrics
# ================================================================


class TestForecastMetrics:

    def test_required_keys(self):
        from src.evaluation import compute_forecast_metrics
        m = compute_forecast_metrics([1, 2, 3], [1, 2, 3])
        assert set(m.keys()) == {"MAE", "RMSE", "MAPE", "SMAPE"}

    def test_perfect_prediction_zeros(self):
        from src.evaluation import compute_forecast_metrics
        y = [1.0, 2.0, 3.0, 4.0]
        m = compute_forecast_metrics(y, y)
        assert m["MAE"]   == pytest.approx(0.0)
        assert m["RMSE"]  == pytest.approx(0.0)
        assert m["MAPE"]  == pytest.approx(0.0)
        assert m["SMAPE"] == pytest.approx(0.0)

    def test_mae_value(self):
        from src.evaluation import compute_forecast_metrics
        m = compute_forecast_metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert m["MAE"] == pytest.approx(1.0)

    def test_rmse_value(self):
        from src.evaluation import compute_forecast_metrics
        m = compute_forecast_metrics([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        assert m["RMSE"] == pytest.approx(1.0)

    def test_mape_all_zero_y_true_returns_nan(self):
        from src.evaluation import compute_forecast_metrics
        m = compute_forecast_metrics([0.0, 0.0], [1.0, 2.0])
        assert np.isnan(m["MAPE"])

    def test_smape_symmetric(self):
        from src.evaluation import compute_forecast_metrics
        # SMAPE should be the same regardless of which is y_true/y_pred
        m1 = compute_forecast_metrics([1.0, 2.0], [1.5, 2.5])
        m2 = compute_forecast_metrics([1.5, 2.5], [1.0, 2.0])
        assert m1["SMAPE"] == pytest.approx(m2["SMAPE"], rel=1e-6)

    def test_smape_range(self):
        from src.evaluation import compute_forecast_metrics
        m = compute_forecast_metrics([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        assert 0.0 <= m["SMAPE"] <= 200.0

    def test_all_values_are_float(self):
        from src.evaluation import compute_forecast_metrics
        m = compute_forecast_metrics([1.0, 2.0], [1.5, 2.5])
        for k, v in m.items():
            assert isinstance(v, float), f"{k} should be float"


# ================================================================
# TestWalkForwardSplit
# ================================================================


class TestWalkForwardSplit:

    def test_yields_correct_number_of_folds(self):
        from src.validation import walk_forward_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        splits = list(walk_forward_split(X, y, n_splits=3, horizon=10))
        assert len(splits) == 3

    def test_no_leakage(self):
        from src.validation import walk_forward_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        for train_idx, test_idx in walk_forward_split(X, y, n_splits=4, horizon=10):
            assert max(train_idx) < min(test_idx)

    def test_test_size_equals_horizon(self):
        from src.validation import walk_forward_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        horizon = 10
        for _, test_idx in walk_forward_split(X, y, n_splits=3, horizon=horizon):
            assert len(test_idx) == horizon

    def test_train_window_expands(self):
        from src.validation import walk_forward_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        train_sizes = [len(tr) for tr, _ in walk_forward_split(X, y, n_splits=4, horizon=10)]
        assert train_sizes == sorted(train_sizes)

    def test_raises_when_not_enough_samples(self):
        from src.validation import walk_forward_split
        X = np.arange(10).reshape(-1, 1)
        y = np.arange(10)
        with pytest.raises(ValueError):
            list(walk_forward_split(X, y, n_splits=5, horizon=5))

    def test_min_train_size_override(self):
        from src.validation import walk_forward_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        splits = list(walk_forward_split(X, y, n_splits=3, horizon=10, min_train_size=20))
        assert len(splits[0][0]) == 20


# ================================================================
# TestRollingWindowSplit
# ================================================================


class TestRollingWindowSplit:

    def test_yields_tuples(self):
        from src.validation import rolling_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        splits = list(rolling_window_split(X, y, train_size=50, test_size=10))
        assert len(splits) > 0

    def test_fixed_train_size(self):
        from src.validation import rolling_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        for train_idx, _ in rolling_window_split(X, y, train_size=50, test_size=10):
            assert len(train_idx) == 50

    def test_fixed_test_size(self):
        from src.validation import rolling_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        for _, test_idx in rolling_window_split(X, y, train_size=50, test_size=10):
            assert len(test_idx) == 10

    def test_no_leakage(self):
        from src.validation import rolling_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        for train_idx, test_idx in rolling_window_split(X, y, train_size=50, test_size=10):
            assert max(train_idx) < min(test_idx)

    def test_raises_when_too_small(self):
        from src.validation import rolling_window_split
        X = np.arange(10).reshape(-1, 1)
        y = np.arange(10)
        with pytest.raises(ValueError):
            list(rolling_window_split(X, y, train_size=8, test_size=5))

    def test_step_controls_overlap(self):
        from src.validation import rolling_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        splits_step1 = list(rolling_window_split(X, y, train_size=40, test_size=10, step=1))
        splits_step5 = list(rolling_window_split(X, y, train_size=40, test_size=10, step=5))
        assert len(splits_step1) > len(splits_step5)


# ================================================================
# TestExpandingWindowSplit
# ================================================================


class TestExpandingWindowSplit:

    def test_yields_tuples(self):
        from src.validation import expanding_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        splits = list(expanding_window_split(X, y, min_train_size=50, test_size=10))
        assert len(splits) > 0

    def test_train_size_grows(self):
        from src.validation import expanding_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        train_sizes = [
            len(tr)
            for tr, _ in expanding_window_split(X, y, min_train_size=50, test_size=5, step=5)
        ]
        assert train_sizes == sorted(train_sizes)

    def test_fixed_test_size(self):
        from src.validation import expanding_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        for _, test_idx in expanding_window_split(X, y, min_train_size=50, test_size=10, step=10):
            assert len(test_idx) == 10

    def test_no_leakage(self):
        from src.validation import expanding_window_split
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        for train_idx, test_idx in expanding_window_split(
            X, y, min_train_size=40, test_size=10, step=5
        ):
            assert max(train_idx) < min(test_idx)

    def test_raises_when_too_small(self):
        from src.validation import expanding_window_split
        X = np.arange(10).reshape(-1, 1)
        y = np.arange(10)
        with pytest.raises(ValueError):
            list(expanding_window_split(X, y, min_train_size=8, test_size=5))


# ================================================================
# TestForecastPlots
# ================================================================


class TestForecastPlots:

    def test_plot_forecast_creates_file(self, tmp_path):
        from src.plots import plot_forecast
        y = np.sin(np.arange(50, dtype=float))
        p = tmp_path / "forecast.png"
        plot_forecast(y, y + 0.1, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_rolling_forecast_creates_file(self, tmp_path):
        from src.plots import plot_rolling_forecast
        folds = [
            (np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1]))
            for _ in range(3)
        ]
        p = tmp_path / "rolling_forecast.png"
        plot_rolling_forecast(folds, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_residuals_over_time_creates_file(self, tmp_path):
        from src.plots import plot_residuals_over_time
        y = np.sin(np.arange(50, dtype=float))
        p = tmp_path / "res_time.png"
        plot_residuals_over_time(y, y + 0.1, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_forecast_single_point(self, tmp_path):
        from src.plots import plot_forecast
        p = tmp_path / "single.png"
        plot_forecast([1.0], [1.1], "M", p)
        assert p.exists()


# ================================================================
# TestTimeSeriesModels
# ================================================================


class TestTimeSeriesModels:

    def test_registry_has_time_series(self):
        from src.models import _REGISTRY
        assert "time_series" in _REGISTRY

    def test_get_models_time_series_returns_dict(self):
        from src.models import get_models
        m = get_models("time_series")
        assert isinstance(m, dict)
        assert len(m) >= 3

    def test_expected_model_names_present(self):
        from src.models import get_models
        m = get_models("time_series")
        assert "Linear Regression" in m
        assert "Random Forest" in m
        assert "XGBoost" in m

    def test_models_have_fit_and_predict(self):
        from src.models import get_models
        for name, model in get_models("time_series").items():
            assert hasattr(model, "fit"),    f"{name} missing fit"
            assert hasattr(model, "predict"), f"{name} missing predict"

    def test_existing_tasks_unchanged(self):
        from src.models import get_models
        assert len(get_models("classification")) >= 8
        assert len(get_models("regression"))     >= 9
        assert len(get_models("unsupervised"))   >= 5


# ================================================================
# TestBenchmarkTimeSeries  (integration — use -m integration to run)
# ================================================================


@pytest.mark.integration
class TestBenchmarkTimeSeries:

    def test_returns_dataframe_and_preprocessor(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, preprocessor = run_benchmark(
            ts_csv, "value",
            task="time_series",
            lags=[1, 2, 3],
            ts_n_splits=3,
            ts_horizon=10,
            output_dir=str(tmp_path),
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(preprocessor, dict)

    def test_all_metric_columns_present(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            ts_csv, "value",
            task="time_series",
            lags=[1, 2],
            ts_n_splits=2,
            ts_horizon=15,
            output_dir=str(tmp_path),
            verbose=False,
        )
        for col in ["Model", "MAE", "RMSE", "MAPE", "SMAPE", "n_folds"]:
            assert col in results_df.columns, f"Missing column: {col}"

    def test_n_rows_equals_n_models(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        models = {"LR": LinearRegression()}
        results_df, _ = run_benchmark(
            ts_csv, "value",
            task="time_series",
            lags=[1, 2],
            ts_n_splits=2,
            ts_horizon=15,
            models_dict=models,
            output_dir=str(tmp_path),
            verbose=False,
        )
        assert len(results_df) == 1

    def test_forecast_plots_created(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            ts_csv, "value",
            task="time_series",
            lags=[1, 2],
            ts_n_splits=2,
            ts_horizon=15,
            models_dict={"LR": LinearRegression()},
            output_dir=str(tmp_path),
            verbose=False,
        )
        forecast_pngs = list(tmp_path.glob("forecast_*.png"))
        assert len(forecast_pngs) >= 1

    def test_csv_export_created(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            ts_csv, "value",
            task="time_series",
            lags=[1],
            ts_n_splits=2,
            ts_horizon=15,
            models_dict={"LR": LinearRegression()},
            output_dir=str(tmp_path),
            export_formats=["csv"],
            verbose=False,
        )
        assert (tmp_path / "results.csv").exists()

    def test_rolling_features_pipeline(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            ts_csv, "value",
            task="time_series",
            rolling_windows=[3, 5],
            ts_n_splits=2,
            ts_horizon=10,
            models_dict={"LR": LinearRegression()},
            output_dir=str(tmp_path),
            verbose=False,
        )
        assert not results_df.empty

    def test_lags_and_rolling_combined(self, ts_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            ts_csv, "value",
            task="time_series",
            lags=[1, 2],
            rolling_windows=[3],
            ts_n_splits=2,
            ts_horizon=10,
            models_dict={"LR": LinearRegression()},
            output_dir=str(tmp_path),
            verbose=False,
        )
        assert not results_df.empty
