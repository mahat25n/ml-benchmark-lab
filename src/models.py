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
        use_label_encoder=False,
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
# Future targets: Linear Regression, Ridge, Lasso, ElasticNet, SVR,
# KNN Regressor, Decision Tree Regressor, Random Forest Regressor,
# Extra Trees Regressor, Gradient Boosting Regressor,
# HistGradientBoosting Regressor, XGBoost Regressor,
# LightGBM Regressor, CatBoost Regressor, MLP Regressor.
# ---------------------------------------------------------------------------
REGRESSION_MODELS = {}

# ---------------------------------------------------------------------------
# UNSUPERVISED_MODELS
# Future targets: KMeans, Hierarchical Clustering, DBSCAN,
# Gaussian Mixture Model, PCA, UMAP, t-SNE.
# ---------------------------------------------------------------------------
UNSUPERVISED_MODELS = {}

_REGISTRY = {
    "classification": CLASSIFICATION_MODELS,
    "regression": REGRESSION_MODELS,
    "unsupervised": UNSUPERVISED_MODELS,
}


def get_models(task="classification"):
    """Return the model registry for the given task type."""
    if task not in _REGISTRY:
        raise ValueError(
            f"Unknown task '{task}'. Valid options: {list(_REGISTRY)}"
        )
    return _REGISTRY[task]
