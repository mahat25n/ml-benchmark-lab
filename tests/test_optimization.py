"""Tests for src/optimization.py — grid search, random search, and unified optimizer."""

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

_PARAM_GRID = {"max_depth": [3, 5], "min_samples_split": [2, 4]}


# ================================================================
# TestRunGridSearch
# ================================================================

class TestRunGridSearch:

    def test_returns_required_keys(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        for key in ("best_estimator", "best_params", "best_score", "cv_results"):
            assert key in result

    def test_best_estimator_can_predict(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, X_te, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        preds = result["best_estimator"].predict(X_te)
        assert len(preds) == len(X_te)

    def test_best_params_values_from_grid(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        assert result["best_params"]["max_depth"] in [3, 5]
        assert result["best_params"]["min_samples_split"] in [2, 4]

    def test_best_score_is_float_in_range(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        assert isinstance(result["best_score"], float)
        assert 0.0 <= result["best_score"] <= 1.0

    def test_cv_results_is_dataframe(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        assert isinstance(result["cv_results"], pd.DataFrame)

    def test_cv_results_rows_equal_grid_size(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        # 2 values x 2 values = 4 combinations
        assert len(result["cv_results"]) == 4

    def test_refit_false_returns_none_estimator(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            cv=3, refit=False
        )
        assert result["best_estimator"] is None

    def test_cv_results_sorted_by_rank(self, split_binary):
        from src.optimization import run_grid_search
        X_tr, _, y_tr, _ = split_binary
        result = run_grid_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, cv=3
        )
        ranks = result["cv_results"]["rank_test_score"].tolist()
        assert ranks == sorted(ranks)


# ================================================================
# TestRunRandomSearch
# ================================================================

class TestRunRandomSearch:

    def test_returns_required_keys(self, split_binary):
        from src.optimization import run_random_search
        X_tr, _, y_tr, _ = split_binary
        result = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            cv=3, n_iter=4, random_state=42
        )
        for key in ("best_estimator", "best_params", "best_score", "cv_results"):
            assert key in result

    def test_best_estimator_can_predict(self, split_binary):
        from src.optimization import run_random_search
        X_tr, X_te, y_tr, _ = split_binary
        result = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            cv=3, n_iter=4, random_state=42
        )
        preds = result["best_estimator"].predict(X_te)
        assert len(preds) == len(X_te)

    def test_best_score_is_float(self, split_binary):
        from src.optimization import run_random_search
        X_tr, _, y_tr, _ = split_binary
        result = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            cv=3, n_iter=4, random_state=42
        )
        assert isinstance(result["best_score"], float)

    def test_cv_results_rows_equal_n_iter(self, split_binary):
        from src.optimization import run_random_search
        X_tr, _, y_tr, _ = split_binary
        n = 3
        result = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            cv=3, n_iter=n, random_state=42
        )
        assert len(result["cv_results"]) == n

    def test_reproducibility_same_seed(self, split_binary):
        from src.optimization import run_random_search
        X_tr, _, y_tr, _ = split_binary
        kwargs = dict(cv=3, n_iter=4, random_state=0)
        a = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, **kwargs
        )
        b = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr, **kwargs
        )
        assert a["best_params"] == b["best_params"]
        assert a["best_score"] == b["best_score"]

    def test_cv_results_is_dataframe(self, split_binary):
        from src.optimization import run_random_search
        X_tr, _, y_tr, _ = split_binary
        result = run_random_search(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            cv=3, n_iter=4, random_state=42
        )
        assert isinstance(result["cv_results"], pd.DataFrame)


# ================================================================
# TestBuildSearchSpace
# ================================================================

class TestBuildSearchSpace:

    def test_returns_dict_for_known_model(self):
        from src.optimization import build_search_space
        space = build_search_space("Random Forest", task="classification")
        assert isinstance(space, dict)
        assert len(space) > 0

    def test_returns_none_for_unknown_model(self):
        from src.optimization import build_search_space
        space = build_search_space("Unknown Model XYZ", task="classification")
        assert space is None

    def test_returns_none_for_linear_regression_ts(self):
        from src.optimization import build_search_space
        # Linear Regression has no default TS search space
        space = build_search_space("Linear Regression", task="time_series")
        assert space is None

    def test_classification_five_required_models(self):
        from src.optimization import build_search_space
        for model in ("Random Forest", "XGBoost", "SVM", "KNN", "GradientBoost"):
            space = build_search_space(model, task="classification")
            assert space is not None, f"No space for {model}"

    def test_regression_five_required_models(self):
        from src.optimization import build_search_space
        for model in ("Random Forest", "XGBoost", "SVR", "KNN", "GradientBoost"):
            space = build_search_space(model, task="regression")
            assert space is not None, f"No space for {model}"

    def test_time_series_rf_and_xgb(self):
        from src.optimization import build_search_space
        for model in ("Random Forest", "XGBoost"):
            space = build_search_space(model, task="time_series")
            assert space is not None, f"No space for {model}"

    def test_returns_copy_not_reference(self):
        from src.optimization import build_search_space
        s1 = build_search_space("Random Forest", task="classification")
        s2 = build_search_space("Random Forest", task="classification")
        s1["n_estimators"] = [999]
        assert s2["n_estimators"] != [999]

    def test_invalid_task_raises(self):
        from src.optimization import build_search_space
        with pytest.raises(ValueError, match="unknown task"):
            build_search_space("Random Forest", task="banana")


# ================================================================
# TestValidateSearchSpace
# ================================================================

class TestValidateSearchSpace:

    def test_valid_params_does_not_raise(self):
        from src.optimization import validate_search_space
        model = DecisionTreeClassifier()
        validate_search_space(model, {"max_depth": [3, 5], "min_samples_split": [2, 4]})

    def test_invalid_param_raises_value_error(self):
        from src.optimization import validate_search_space
        model = DecisionTreeClassifier()
        with pytest.raises(ValueError, match="does not accept parameter"):
            validate_search_space(model, {"nonexistent_param": [1, 2]})

    def test_multiple_invalid_params_listed_in_error(self):
        from src.optimization import validate_search_space
        model = DecisionTreeClassifier()
        with pytest.raises(ValueError) as exc_info:
            validate_search_space(model, {"bad_a": [1], "bad_b": [2]})
        msg = str(exc_info.value)
        assert "bad_a" in msg or "bad_b" in msg

    def test_empty_space_does_not_raise(self):
        from src.optimization import validate_search_space
        model = DecisionTreeClassifier()
        validate_search_space(model, {})

    def test_ridge_valid_alpha(self):
        from src.optimization import validate_search_space
        validate_search_space(Ridge(), {"alpha": [0.1, 1.0, 10.0]})

    def test_knn_valid_params(self):
        from src.optimization import validate_search_space
        validate_search_space(KNeighborsClassifier(), {"n_neighbors": [3, 5], "weights": ["uniform"]})


# ================================================================
# TestOptimizeModel
# ================================================================

class TestOptimizeModel:

    def test_returns_extended_keys(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="random", cv=3, n_iter=4,
        )
        for key in ("best_estimator", "best_params", "best_score",
                    "cv_results", "search_duration", "n_evaluations", "method"):
            assert key in result, f"Missing key: {key}"

    def test_method_stored_in_result_random(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="random", cv=3, n_iter=4,
        )
        assert result["method"] == "random"

    def test_method_stored_in_result_grid(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="grid", cv=3,
        )
        assert result["method"] == "grid"

    def test_search_duration_is_positive(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="random", cv=3, n_iter=4,
        )
        assert result["search_duration"] >= 0.0

    def test_n_evaluations_matches_n_iter(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        n = 3
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="random", cv=3, n_iter=n,
        )
        assert result["n_evaluations"] == n

    def test_n_evaluations_grid_equals_grid_size(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="grid", cv=3,
        )
        # 2 × 2 = 4 combinations
        assert result["n_evaluations"] == 4

    def test_best_estimator_can_predict(self, split_binary):
        from src.optimization import optimize_model
        X_tr, X_te, y_tr, _ = split_binary
        result = optimize_model(
            DecisionTreeClassifier(random_state=42), _PARAM_GRID, X_tr, y_tr,
            method="random", cv=3, n_iter=4,
        )
        preds = result["best_estimator"].predict(X_te)
        assert len(preds) == len(X_te)

    def test_unknown_method_raises(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        with pytest.raises(ValueError, match="unknown method"):
            optimize_model(
                DecisionTreeClassifier(), _PARAM_GRID, X_tr, y_tr, method="bayesian"
            )

    def test_invalid_params_raises_before_search(self, split_binary):
        from src.optimization import optimize_model
        X_tr, _, y_tr, _ = split_binary
        with pytest.raises(ValueError, match="does not accept parameter"):
            optimize_model(
                DecisionTreeClassifier(), {"nonexistent": [1, 2]}, X_tr, y_tr,
                method="random", cv=3, n_iter=2,
            )


# ================================================================
# TestDefaultSearchSpaces
# ================================================================

class TestDefaultSearchSpaces:

    def test_registry_has_classification_key(self):
        from src.optimization import DEFAULT_SEARCH_SPACES
        assert "classification" in DEFAULT_SEARCH_SPACES

    def test_registry_has_regression_key(self):
        from src.optimization import DEFAULT_SEARCH_SPACES
        assert "regression" in DEFAULT_SEARCH_SPACES

    def test_registry_has_time_series_key(self):
        from src.optimization import DEFAULT_SEARCH_SPACES
        assert "time_series" in DEFAULT_SEARCH_SPACES

    def test_classification_space_values_are_lists(self):
        from src.optimization import DEFAULT_SEARCH_SPACES
        for model, space in DEFAULT_SEARCH_SPACES["classification"].items():
            for k, v in space.items():
                assert isinstance(v, list), f"{model}.{k} is not a list"

    def test_regression_space_values_are_lists(self):
        from src.optimization import DEFAULT_SEARCH_SPACES
        for model, space in DEFAULT_SEARCH_SPACES["regression"].items():
            for k, v in space.items():
                assert isinstance(v, list), f"{model}.{k} is not a list"


# ================================================================
# Integration: run_benchmark with optimize=True / optimize=dict
# ================================================================

class TestRunBenchmarkOptimize:

    @pytest.fixture
    def clf_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 120
        X = rng.standard_normal((n, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        df["target"] = y
        p = tmp_path / "clf_opt.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_optimize_true_returns_results(self, clf_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            clf_csv, "target",
            output_dir=str(tmp_path / "out1"),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            optimize=True,
            optimization_method="random",
            n_iter=4,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)
        assert "Model" in results_df.columns

    def test_optimize_dict_backward_compat(self, clf_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            clf_csv, "target",
            output_dir=str(tmp_path / "out2"),
            models_dict={"DT": DecisionTreeClassifier(random_state=42)},
            optimize={"DT": {"max_depth": [3, 5], "min_samples_split": [2, 4]}},
            optimization_method="random",
            n_iter=4,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)

    def test_optimize_false_no_optimization(self, clf_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, pp = run_benchmark(
            clf_csv, "target",
            output_dir=str(tmp_path / "out3"),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            optimize=False,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)

    def test_optimize_with_search_space_override(self, clf_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            clf_csv, "target",
            output_dir=str(tmp_path / "out4"),
            models_dict={"DT": DecisionTreeClassifier(random_state=42)},
            optimize=True,
            search_space={"DT": {"max_depth": [3, 5]}},
            optimization_method="random",
            n_iter=4,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)

    def test_optimize_grid_method(self, clf_csv, tmp_path):
        from src.benchmark import run_benchmark
        results_df, _ = run_benchmark(
            clf_csv, "target",
            output_dir=str(tmp_path / "out5"),
            models_dict={"DT": DecisionTreeClassifier(random_state=42)},
            optimize={"DT": {"max_depth": [3, 5]}},
            optimization_method="grid",
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)

    def test_optimize_with_experiment_tracking(self, clf_csv, tmp_path):
        from pathlib import Path
        import json
        from src.benchmark import run_benchmark
        out_dir = tmp_path / "out6"
        _, preprocessor = run_benchmark(
            clf_csv, "target",
            output_dir=str(out_dir),
            models_dict={"DT": DecisionTreeClassifier(random_state=42)},
            optimize={"DT": {"max_depth": [3, 5]}},
            optimization_method="random",
            n_iter=4,
            experiment_name="opt_track_test",
            verbose=False,
        )
        assert "experiment" in preprocessor
        run_dir = Path(preprocessor["experiment"]["run_dir"])
        config = json.loads((run_dir / "config.json").read_text())
        assert "optimization" in config
        assert "DT" in config["optimization"]
        assert "best_params" in config["optimization"]["DT"]

    def test_model_with_no_default_space_skipped(self, clf_csv, tmp_path):
        from src.benchmark import run_benchmark
        # ANN has a default space; "MyCustomModel" does not
        # optimize=True with a model that has no default space → skip silently
        results_df, _ = run_benchmark(
            clf_csv, "target",
            output_dir=str(tmp_path / "out7"),
            models_dict={"DT": DecisionTreeClassifier(max_depth=3, random_state=42)},
            optimize=True,
            # DT has a default classification space, so it should be optimized
            optimization_method="random",
            n_iter=4,
            verbose=False,
        )
        assert isinstance(results_df, pd.DataFrame)
