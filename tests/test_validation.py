"""Tests for src/validation.py — CV strategy creation and description."""

import numpy as np
import pytest
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)


# ================================================================
# TestGetCvStrategy
# ================================================================

class TestGetCvStrategy:

    def test_invalid_strategy_raises(self):
        from src.validation import get_cv_strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_cv_strategy("nonexistent")

    @pytest.mark.parametrize("strategy,expected_cls", [
        ("holdout",                   ShuffleSplit),
        ("kfold",                     KFold),
        ("stratified_kfold",          StratifiedKFold),
        ("repeated_stratified_kfold", RepeatedStratifiedKFold),
        ("group_kfold",               GroupKFold),
        ("time_series_split",         TimeSeriesSplit),
    ])
    def test_returns_correct_class(self, strategy, expected_cls):
        from src.validation import get_cv_strategy
        cv = get_cv_strategy(strategy)
        assert isinstance(cv, expected_cls)

    def test_custom_n_splits_overrides_default(self):
        from src.validation import get_cv_strategy
        cv = get_cv_strategy("stratified_kfold", n_splits=10)
        assert vars(cv)["n_splits"] == 10

    def test_kfold_shuffle_false_does_not_raise(self):
        from src.validation import get_cv_strategy
        cv = get_cv_strategy("kfold", shuffle=False)
        assert isinstance(cv, KFold)

    def test_holdout_produces_one_split(self, small_X_y):
        from src.validation import get_cv_strategy
        X, y = small_X_y
        cv = get_cv_strategy("holdout")
        splits = list(cv.split(X, y))
        assert len(splits) == 1

    def test_stratified_kfold_split_count(self, small_X_y):
        from src.validation import get_cv_strategy
        X, y = small_X_y
        cv = get_cv_strategy("stratified_kfold", n_splits=3)
        splits = list(cv.split(X, y))
        assert len(splits) == 3

    def test_split_indices_cover_all_samples(self, small_X_y):
        from src.validation import get_cv_strategy
        X, y = small_X_y
        cv = get_cv_strategy("kfold", n_splits=5)
        all_test = np.concatenate([te for _, te in cv.split(X, y)])
        assert sorted(all_test) == list(range(len(y)))

    def test_valid_strategies_not_empty(self):
        from src.validation import VALID_STRATEGIES
        assert len(VALID_STRATEGIES) >= 6


# ================================================================
# TestDescribeCvStrategy
# ================================================================

class TestDescribeCvStrategy:

    def test_returns_string(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        cv = get_cv_strategy("stratified_kfold")
        assert isinstance(describe_cv_strategy(cv), str)

    def test_describes_holdout(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        result = describe_cv_strategy(get_cv_strategy("holdout"))
        assert "split" in result.lower() or "holdout" in result.lower()

    def test_describes_kfold_with_n_splits(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        result = describe_cv_strategy(get_cv_strategy("kfold", n_splits=7))
        assert "7" in result

    def test_describes_stratified_kfold(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        result = describe_cv_strategy(get_cv_strategy("stratified_kfold"))
        assert "stratified" in result.lower()

    def test_describes_repeated_stratified_kfold(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        result = describe_cv_strategy(get_cv_strategy("repeated_stratified_kfold"))
        assert "repeat" in result.lower()

    def test_describes_group_kfold(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        result = describe_cv_strategy(get_cv_strategy("group_kfold"))
        assert "group" in result.lower()

    def test_describes_time_series_split(self):
        from src.validation import describe_cv_strategy, get_cv_strategy
        result = describe_cv_strategy(get_cv_strategy("time_series_split"))
        assert "time" in result.lower() or "series" in result.lower()

    def test_unknown_type_returns_class_name(self):
        from src.validation import describe_cv_strategy
        from sklearn.model_selection import LeaveOneOut
        result = describe_cv_strategy(LeaveOneOut())
        assert "LeaveOneOut" in result
