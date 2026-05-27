"""Tests for src/optimization.py — grid search and random search."""

import pandas as pd
import pytest
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
