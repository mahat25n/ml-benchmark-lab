"""
Tests for forecasting-specific statistical comparison utilities:
  compute_forecast_error_series
  run_diebold_mariano_test
  compare_forecast_models
"""

import numpy as np
import pandas as pd
import pytest

from src.stats import (
    compare_forecast_models,
    compute_forecast_error_series,
    run_diebold_mariano_test,
)


# ================================================================
# SHARED FIXTURES
# ================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def true_pred_good_bad(rng):
    """y_true, y_pred_good (small error), y_pred_bad (large error)."""
    T = 50
    y_true = rng.standard_normal(T)
    y_pred_good = y_true + rng.normal(0, 0.1, T)
    y_pred_bad  = y_true + rng.normal(0, 1.0, T)
    return y_true, y_pred_good, y_pred_bad


@pytest.fixture
def errors_good_bad(true_pred_good_bad):
    y_true, y_pred_good, y_pred_bad = true_pred_good_bad
    return y_true - y_pred_good, y_true - y_pred_bad


# ================================================================
# compute_forecast_error_series
# ================================================================

class TestComputeForecastErrorSeries:
    def test_mae_shape(self, true_pred_good_bad):
        y_true, y_pred_good, _ = true_pred_good_bad
        result = compute_forecast_error_series(y_true, y_pred_good, loss="mae")
        assert result.shape == y_true.shape

    def test_mae_nonnegative(self, true_pred_good_bad):
        y_true, y_pred_good, _ = true_pred_good_bad
        result = compute_forecast_error_series(y_true, y_pred_good, loss="mae")
        assert np.all(result >= 0)

    def test_mse_nonnegative(self, true_pred_good_bad):
        y_true, y_pred_good, _ = true_pred_good_bad
        result = compute_forecast_error_series(y_true, y_pred_good, loss="mse")
        assert np.all(result >= 0)

    def test_rmse_equals_mse(self, true_pred_good_bad):
        y_true, y_pred_good, _ = true_pred_good_bad
        r_mse  = compute_forecast_error_series(y_true, y_pred_good, loss="mse")
        r_rmse = compute_forecast_error_series(y_true, y_pred_good, loss="rmse")
        np.testing.assert_array_equal(r_mse, r_rmse)

    def test_perfect_prediction_zero_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        result = compute_forecast_error_series(y_true, y_true, loss="mae")
        np.testing.assert_array_almost_equal(result, [0, 0, 0])

    def test_mae_formula(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 1.5, 3.5])
        result = compute_forecast_error_series(y_true, y_pred, loss="mae")
        np.testing.assert_array_almost_equal(result, [0.5, 0.5, 0.5])

    def test_mse_formula(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        result = compute_forecast_error_series(y_true, y_pred, loss="mse")
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 1.0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_forecast_error_series([1, 2, 3], [1, 2])

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            compute_forecast_error_series([1, 2, 3], [1, 2, 3], loss="huber")

    def test_returns_ndarray(self, true_pred_good_bad):
        y_true, y_pred_good, _ = true_pred_good_bad
        result = compute_forecast_error_series(y_true, y_pred_good)
        assert isinstance(result, np.ndarray)


# ================================================================
# run_diebold_mariano_test
# ================================================================

class TestRunDieboldMarianoTest:
    def test_returns_expected_keys(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb)
        for k in ("statistic", "p_value", "h", "loss", "mean_diff",
                  "n_obs", "interpretation"):
            assert k in res

    def test_significant_when_clearly_different(self, errors_good_bad):
        ea, eb = errors_good_bad   # ea ~ N(0, 0.1), eb ~ N(0, 1.0)
        res = run_diebold_mariano_test(ea, eb, loss="mae")
        # good forecaster has smaller MAE → d = L(A) - L(B) < 0 → A is better
        assert res["p_value"] < 0.05

    def test_not_significant_for_equal_forecasts(self, rng):
        y_true = rng.standard_normal(60)
        err = y_true + rng.normal(0, 0.5, 60)
        # Same errors for both models
        res = run_diebold_mariano_test(err, err, loss="mae")
        assert res["p_value"] == pytest.approx(1.0)
        assert res["statistic"] == pytest.approx(0.0)

    def test_mae_loss(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb, loss="mae")
        assert res["loss"] == "mae"

    def test_mse_loss(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb, loss="mse")
        assert res["loss"] == "mse"

    def test_rmse_loss_accepted(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb, loss="rmse")
        assert "loss" in res

    def test_h_stored_in_result(self, errors_good_bad):
        ea, eb = errors_good_bad
        for h in (1, 3, 5):
            res = run_diebold_mariano_test(ea, eb, h=h)
            assert res["h"] == h

    def test_multistep_does_not_crash(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb, h=5, loss="mse")
        assert 0.0 <= res["p_value"] <= 1.0

    def test_n_obs_correct(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb)
        assert res["n_obs"] == len(ea)

    def test_mean_diff_sign(self, errors_good_bad):
        ea, eb = errors_good_bad
        res_mae = run_diebold_mariano_test(ea, eb, loss="mae")
        la = np.abs(ea)
        lb = np.abs(eb)
        expected_diff = float((la - lb).mean())
        assert res_mae["mean_diff"] == pytest.approx(expected_diff, rel=1e-6)

    def test_p_value_range(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb)
        assert 0.0 <= res["p_value"] <= 1.0

    def test_interpretation_is_string(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb)
        assert isinstance(res["interpretation"], str)
        assert len(res["interpretation"]) > 0

    def test_significant_interpretation_contains_alpha(self, errors_good_bad):
        ea, eb = errors_good_bad
        res = run_diebold_mariano_test(ea, eb, alpha=0.05)
        assert "0.05" in res["interpretation"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            run_diebold_mariano_test([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            run_diebold_mariano_test([0.1], [0.2])

    def test_invalid_h_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            run_diebold_mariano_test([0.1, 0.2], [0.2, 0.1], h=0)

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            run_diebold_mariano_test([0.1, 0.2], [0.2, 0.1], loss="huber")

    def test_symmetry_of_p_value(self, errors_good_bad):
        """p-value is the same when A and B are swapped (two-sided test)."""
        ea, eb = errors_good_bad
        res_ab = run_diebold_mariano_test(ea, eb)
        res_ba = run_diebold_mariano_test(eb, ea)
        assert res_ab["p_value"] == pytest.approx(res_ba["p_value"], rel=1e-6)

    def test_statistic_negated_when_swapped(self, errors_good_bad):
        ea, eb = errors_good_bad
        res_ab = run_diebold_mariano_test(ea, eb)
        res_ba = run_diebold_mariano_test(eb, ea)
        assert res_ab["statistic"] == pytest.approx(-res_ba["statistic"], rel=1e-6)


# ================================================================
# compare_forecast_models
# ================================================================

class TestCompareForecastModels:
    def test_returns_dataframe(self, errors_good_bad, rng):
        ea, eb = errors_good_bad
        ec = rng.normal(0, 0.5, len(ea))
        df = compare_forecast_models({"A": ea, "B": eb, "C": ec})
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, errors_good_bad, rng):
        ea, eb = errors_good_bad
        ec = rng.normal(0, 0.5, len(ea))
        df = compare_forecast_models({"A": ea, "B": eb, "C": ec})
        for col in ("Model_A", "Model_B", "DM_Statistic", "p_value",
                    "Mean_Loss_Diff", "Significant", "Favored"):
            assert col in df.columns

    def test_correct_n_pairs_all(self, errors_good_bad, rng):
        ea, eb = errors_good_bad
        ec = rng.normal(0, 0.5, len(ea))
        df = compare_forecast_models({"A": ea, "B": eb, "C": ec})
        # C(3,2) = 3 pairs
        assert len(df) == 3

    def test_correct_n_pairs_two_models(self, errors_good_bad):
        ea, eb = errors_good_bad
        df = compare_forecast_models({"A": ea, "B": eb})
        assert len(df) == 1

    def test_baseline_mode(self, errors_good_bad, rng):
        ea, eb = errors_good_bad
        ec = rng.normal(0, 0.5, len(ea))
        df = compare_forecast_models({"A": ea, "B": eb, "C": ec}, baseline="A")
        # baseline=A → A vs B, A vs C → 2 pairs
        assert len(df) == 2
        assert all(df["Model_A"] == "A")

    def test_invalid_baseline_raises(self, errors_good_bad):
        ea, eb = errors_good_bad
        with pytest.raises(ValueError, match="not found"):
            compare_forecast_models({"A": ea, "B": eb}, baseline="X")

    def test_too_few_models_raises(self, errors_good_bad):
        ea, _ = errors_good_bad
        with pytest.raises(ValueError, match="at least 2"):
            compare_forecast_models({"A": ea})

    def test_significant_column_is_bool(self, errors_good_bad):
        ea, eb = errors_good_bad
        df = compare_forecast_models({"A": ea, "B": eb})
        assert df["Significant"].dtype == bool or df["Significant"].dtype == object

    def test_good_vs_bad_significant(self, errors_good_bad):
        ea, eb = errors_good_bad
        df = compare_forecast_models({"Good": ea, "Bad": eb}, loss="mae")
        assert df["Significant"].iloc[0]

    def test_mse_loss_option(self, errors_good_bad):
        ea, eb = errors_good_bad
        df = compare_forecast_models({"A": ea, "B": eb}, loss="mse")
        assert "DM_Statistic" in df.columns

    def test_multistep_horizon(self, errors_good_bad):
        ea, eb = errors_good_bad
        df = compare_forecast_models({"A": ea, "B": eb}, h=3)
        assert len(df) == 1

    def test_p_value_in_range(self, errors_good_bad):
        ea, eb = errors_good_bad
        df = compare_forecast_models({"A": ea, "B": eb})
        p = float(df["p_value"].iloc[0])
        assert 0.0 <= p <= 1.0


# ================================================================
# Integration: run_benchmark with compute_stats=True (time series)
# ================================================================

class TestRunBenchmarkForecastingStats:
    """Integration tests for compute_stats=True in time series pipelines."""

    @pytest.fixture
    def ts_csv(self, tmp_path, rng):
        t = np.arange(200)
        y = np.sin(t * 0.2) + rng.normal(0, 0.3, 200)
        df = pd.DataFrame({"t": t, "value": y})
        p = tmp_path / "ts_data.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_stats_summary_in_preprocessor(self, ts_csv, tmp_path):
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        from src.benchmark import run_benchmark

        _, pp = run_benchmark(
            ts_csv, "value",
            task="time_series",
            models_dict={
                "Ridge": Ridge(),
                "DT":    DecisionTreeRegressor(max_depth=3, random_state=42),
            },
            output_dir=str(tmp_path),
            lags=[1, 2],
            ts_n_splits=3,
            compute_stats=True,
            verbose=False,
        )
        assert "stats_summary" in pp

    def test_dm_mae_key_present(self, ts_csv, tmp_path):
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        from src.benchmark import run_benchmark

        _, pp = run_benchmark(
            ts_csv, "value",
            task="time_series",
            models_dict={
                "Ridge": Ridge(),
                "DT":    DecisionTreeRegressor(max_depth=3, random_state=42),
            },
            output_dir=str(tmp_path / "b"),
            lags=[1, 2],
            ts_n_splits=3,
            compute_stats=True,
            verbose=False,
        )
        ss = pp.get("stats_summary", {})
        assert "dm_mae" in ss or "dm_mse" in ss

    def test_no_stats_summary_when_flag_false(self, ts_csv, tmp_path):
        from sklearn.linear_model import Ridge
        from src.benchmark import run_benchmark

        _, pp = run_benchmark(
            ts_csv, "value",
            task="time_series",
            models_dict={"Ridge": Ridge()},
            output_dir=str(tmp_path / "c"),
            lags=[1],
            ts_n_splits=3,
            compute_stats=False,
            verbose=False,
        )
        assert "stats_summary" not in pp

    def test_dm_records_have_expected_fields(self, ts_csv, tmp_path):
        from sklearn.linear_model import Ridge
        from sklearn.tree import DecisionTreeRegressor
        from src.benchmark import run_benchmark

        _, pp = run_benchmark(
            ts_csv, "value",
            task="time_series",
            models_dict={
                "Ridge": Ridge(),
                "DT":    DecisionTreeRegressor(max_depth=3, random_state=42),
            },
            output_dir=str(tmp_path / "d"),
            lags=[1, 2],
            ts_n_splits=3,
            compute_stats=True,
            verbose=False,
        )
        ss = pp.get("stats_summary", {})
        records = ss.get("dm_mae", ss.get("dm_mse", []))
        assert len(records) >= 1
        first = records[0]
        assert "Model_A" in first
        assert "Model_B" in first
        assert "DM_Statistic" in first
        assert "p_value" in first
