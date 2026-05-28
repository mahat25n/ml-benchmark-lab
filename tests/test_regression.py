"""Smoke tests for regression pipeline integration."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge


# ================================================================
# Regression CSV fixture (module-scoped for speed)
# ================================================================

@pytest.fixture(scope="module")
def reg_csv(tmp_path_factory):
    """100-row regression CSV: 4 numeric features, continuous target."""
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    df["target"] = y
    p = tmp_path_factory.mktemp("reg_data") / "reg.csv"
    df.to_csv(p, index=False)
    return str(p)


# ================================================================
# TestRegressionMetrics
# ================================================================

class TestRegressionMetrics:

    def test_returns_required_keys(self):
        from src.evaluation import compute_regression_metrics
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        result = compute_regression_metrics(y_true, y_pred)
        for key in ("MAE", "MSE", "RMSE", "R2", "MAPE"):
            assert key in result

    def test_perfect_predictions_r2_is_one(self):
        from src.evaluation import compute_regression_metrics
        y = np.arange(1.0, 11.0)
        result = compute_regression_metrics(y, y)
        assert result["R2"] == pytest.approx(1.0)

    def test_rmse_equals_sqrt_mse(self):
        from src.evaluation import compute_regression_metrics
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(50)
        y_pred = y_true + rng.standard_normal(50) * 0.1
        result = compute_regression_metrics(y_true, y_pred)
        assert result["RMSE"] == pytest.approx(np.sqrt(result["MSE"]))

    def test_mae_non_negative(self):
        from src.evaluation import compute_regression_metrics
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        result = compute_regression_metrics(y_true, y_pred)
        assert result["MAE"] >= 0.0

    def test_mse_non_negative(self):
        from src.evaluation import compute_regression_metrics
        rng = np.random.default_rng(2)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        result = compute_regression_metrics(y_true, y_pred)
        assert result["MSE"] >= 0.0

    def test_all_zero_target_mape_is_nan(self):
        from src.evaluation import compute_regression_metrics
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="MAPE"):
            result = compute_regression_metrics(y_true, y_pred)
        assert np.isnan(result["MAPE"])

    def test_values_are_floats(self):
        from src.evaluation import compute_regression_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        result = compute_regression_metrics(y_true, y_pred)
        for v in result.values():
            assert isinstance(v, float)


# ================================================================
# TestRegressionModels
# ================================================================

class TestRegressionModels:

    def test_regression_models_all_registered(self):
        from src.models import REGRESSION_MODELS
        expected = {
            "Linear Regression", "Ridge", "Lasso", "ElasticNet",
            "Random Forest", "GradientBoost", "XGBoost", "SVR", "KNN",
        }
        assert expected.issubset(set(REGRESSION_MODELS.keys()))

    def test_get_models_regression_returns_nine(self):
        from src.models import get_models
        models = get_models("regression")
        assert len(models) == 9

    def test_regression_models_have_fit_predict(self):
        from src.models import get_models
        import copy
        rng = np.random.default_rng(0)
        X_tr = rng.standard_normal((30, 4))
        y_tr = rng.standard_normal(30)
        for name, model in list(get_models("regression").items())[:4]:
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_tr)
            assert preds.shape == (30,), f"{name}: wrong predict shape"

    def test_get_models_classification_unchanged(self):
        from src.models import get_models
        clf_models = get_models("classification")
        assert len(clf_models) == 8


# ================================================================
# TestRegressionPlots
# ================================================================

class TestRegressionPlots:

    @pytest.fixture
    def reg_arrays(self):
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(50)
        y_pred = y_true + rng.standard_normal(50) * 0.3
        return y_true, y_pred

    def test_plot_actual_vs_predicted_creates_png(self, tmp_path, reg_arrays):
        from src.plots import plot_actual_vs_predicted
        y_true, y_pred = reg_arrays
        p = tmp_path / "avp.png"
        plot_actual_vs_predicted(y_true, y_pred, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_residuals_creates_png(self, tmp_path, reg_arrays):
        from src.plots import plot_residuals
        y_true, y_pred = reg_arrays
        p = tmp_path / "res.png"
        plot_residuals(y_true, y_pred, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0

    def test_plot_error_distribution_creates_png(self, tmp_path, reg_arrays):
        from src.plots import plot_error_distribution
        y_true, y_pred = reg_arrays
        p = tmp_path / "err.png"
        plot_error_distribution(y_true, y_pred, "TestModel", p)
        assert p.exists() and p.stat().st_size > 0


# ================================================================
# TestRegressionBenchmark  (integration)
# ================================================================

@pytest.mark.integration
class TestRegressionBenchmark:

    def test_returns_tuple_of_two(self, reg_csv, tmp_path):
        from src.benchmark import run_benchmark
        result = run_benchmark(
            reg_csv, "target",
            task="regression",
            output_dir=str(tmp_path),
            models_dict={"Ridge": Ridge(alpha=1.0)},
            stratify=False,
            verbose=False,
        )
        assert len(result) == 2

    def test_results_has_regression_metric_columns(self, reg_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            reg_csv, "target",
            task="regression",
            output_dir=str(tmp_path),
            models_dict={"Ridge": Ridge(alpha=1.0)},
            stratify=False,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)
        for col in ("Model", "MAE", "MSE", "RMSE", "R2", "MAPE"):
            assert col in results_df.columns

    def test_avp_and_residual_plots_created(self, reg_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            reg_csv, "target",
            task="regression",
            output_dir=str(tmp_path),
            models_dict={"Ridge": Ridge(alpha=1.0)},
            stratify=False,
            verbose=False,
        )
        assert len(list(tmp_path.glob("avp_*.png"))) >= 1
        assert len(list(tmp_path.glob("res_*.png"))) >= 1

    def test_export_csv_contains_regression_metrics(self, reg_csv, tmp_path):
        from src.benchmark import run_benchmark
        run_benchmark(
            reg_csv, "target",
            task="regression",
            output_dir=str(tmp_path),
            models_dict={"Ridge": Ridge(alpha=1.0)},
            stratify=False,
            export_formats=["csv"],
            verbose=False,
        )
        df = pd.read_csv(tmp_path / "results.csv")
        assert "R2" in df.columns

    def test_stratify_true_auto_corrected(self, reg_csv, tmp_path):
        from src.benchmark import run_benchmark
        with pytest.warns(UserWarning, match="stratify"):
            run_benchmark(
                reg_csv, "target",
                task="regression",
                output_dir=str(tmp_path),
                models_dict={"Ridge": Ridge(alpha=1.0)},
                stratify=True,   # should auto-correct
                verbose=False,
            )

    def test_preprocessor_keys_present(self, reg_csv, tmp_path):
        from src.benchmark import run_benchmark
        _, pp = run_benchmark(
            reg_csv, "target",
            task="regression",
            output_dir=str(tmp_path),
            models_dict={"Ridge": Ridge(alpha=1.0)},
            stratify=False,
            verbose=False,
        )
        for key in ("scaler", "feature_names", "feature_encoders"):
            assert key in pp
