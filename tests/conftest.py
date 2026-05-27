"""
Shared pytest fixtures for ML Benchmark Lab test suite.

Session-scoped fixtures are used for expensive objects (datasets,
fitted models) so they are created once and reused across all tests.
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# ── Datasets ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def small_X_y():
    """200 samples, 8 numeric features, balanced binary target."""
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5,
        n_redundant=1, random_state=42,
    )
    return X, y


@pytest.fixture(scope="session")
def imbalanced_X_y():
    """200 samples, 8 numeric features, imbalanced binary target (80/20)."""
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=5,
        n_redundant=1, weights=[0.8, 0.2], random_state=42,
    )
    return X, y


@pytest.fixture(scope="session")
def split_binary(small_X_y):
    """Pre-split training / test arrays (75 / 25 split)."""
    from sklearn.model_selection import train_test_split
    X, y = small_X_y
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


@pytest.fixture(scope="session")
def fitted_rf(split_binary):
    """RandomForest fitted on split_binary training data."""
    X_train, _, y_train, _ = split_binary
    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    return rf


# ── CSV file ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def tmp_csv(tmp_path_factory, small_X_y):
    """Temporary CSV file with numeric features and binary target."""
    X, y = small_X_y
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["target"] = y
    p = tmp_path_factory.mktemp("data") / "data.csv"
    df.to_csv(p, index=False)
    return str(p)


# ── Small fast models dict for benchmark tests ─────────────────────


@pytest.fixture
def fast_models():
    """Two lightweight models for benchmark integration tests."""
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=1),
    }
