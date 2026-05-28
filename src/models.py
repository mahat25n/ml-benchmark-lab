from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# CLASSIFICATION_MODELS
# Currently populated with the 8 prototype models from the notebook.
# Future targets: Logistic Regression, Ridge, SGD, Passive Aggressive, LDA,
# QDA, Linear SVM, Poly SVM, Extra Tree, Extra Trees, HistGradientBoosting,
# Bagging, LightGBM, CatBoost, Gaussian NB, Bernoulli NB, Gaussian Process.
# ---------------------------------------------------------------------------
CLASSIFICATION_MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.5,
        random_state=42,
    ),
    "GradientBoost": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    ),
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced",
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
    ),
    "ANN": MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    ),
}

# ---------------------------------------------------------------------------
# REGRESSION_MODELS
# Future targets: Decision Tree Regressor, Extra Trees Regressor,
# HistGradientBoosting Regressor, LightGBM Regressor,
# CatBoost Regressor, MLP Regressor, Bayesian Ridge, HuberRegressor.
# ---------------------------------------------------------------------------
REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(
        alpha=1.0,
    ),
    "Lasso": Lasso(
        alpha=0.1,
        random_state=42,
        max_iter=2000,
    ),
    "ElasticNet": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=42,
        max_iter=2000,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoost": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    ),
    "SVR": SVR(
        kernel="rbf",
        C=1.0,
        epsilon=0.1,
    ),
    "KNN": KNeighborsRegressor(
        n_neighbors=5,
    ),
}

# ---------------------------------------------------------------------------
# UNSUPERVISED_MODELS
# Future targets: UMAP, t-SNE, Spectral Clustering, Birch, MiniBatchKMeans.
# ---------------------------------------------------------------------------
UNSUPERVISED_MODELS = {
    "KMeans": KMeans(n_clusters=3, random_state=42, n_init=10),
    "Agglomerative": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42),
    "PCA": PCA(n_components=2, random_state=42),
}

# ---------------------------------------------------------------------------
# TIME_SERIES_MODELS
# Lag-feature–based forecasters; use these with create_lag_features() to
# convert a time series into a supervised regression problem before fitting.
# Future targets: LightGBM, CatBoost, LSTM (via scikit-learn wrapper).
# ---------------------------------------------------------------------------
TIME_SERIES_MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    ),
}

_REGISTRY = {
    "classification": CLASSIFICATION_MODELS,
    "regression": REGRESSION_MODELS,
    "unsupervised": UNSUPERVISED_MODELS,
    "time_series": TIME_SERIES_MODELS,
}


def get_models(task="classification"):
    """Return the model registry for the given task type."""
    if task not in _REGISTRY:
        raise ValueError(
            f"Unknown task '{task}'. Valid options: {list(_REGISTRY)}"
        )
    return _REGISTRY[task]
