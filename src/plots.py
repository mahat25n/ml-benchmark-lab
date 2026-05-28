import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as _PCA

from src.config import DEFAULT_PLOTTING


# ================================================================
# CONFUSION MATRIX
# ================================================================


def plot_confusion_matrix(cm, name, save_path, class_labels=None):
    """Save a publication-quality heatmap of a precomputed confusion matrix."""
    if class_labels is None:
        n = cm.shape[0]
        class_labels = ["Negative", "Positive"] if n == 2 else [f"Class {i}" for i in range(n)]

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_cm"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[f"Predicted: {c}" for c in class_labels],
        yticklabels=[f"Actual: {c}" for c in class_labels],
        annot_kws={"size": DEFAULT_PLOTTING["annot_size"]},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}", fontsize=DEFAULT_PLOTTING["title_size"], pad=10)
    ax.set_xlabel("Predicted Label", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("True Label", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# ROC CURVE
#
# Usage pattern (mirrors the notebook's figure lifecycle):
#   fig, ax = init_roc_figure()
#   for name, fpr, tpr, auc in ...:
#       add_roc_curve(ax, fpr, tpr, auc, name)
#   finalize_roc_plot(fig, ax, "Combined_ROC.png")
# ================================================================


def init_roc_figure(figsize=DEFAULT_PLOTTING["figsize_roc"]):
    """Create and return (fig, ax) ready to receive ROC curves."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate  (1 – Specificity)", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title("Combined ROC Curves — All Models", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"], alpha=DEFAULT_PLOTTING["grid_alpha"])
    return fig, ax


def add_roc_curve(ax, fpr, tpr, auc, name):
    """Plot one model's ROC curve onto an existing axis."""
    ax.plot(fpr, tpr, lw=DEFAULT_PLOTTING["line_width"], label=f"{name}  (AUC = {auc:.3f})")


def finalize_roc_plot(fig, ax, save_path):
    """Add the random-guess diagonal, legend, then save and close."""
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Random Guess")
    ax.legend(loc="lower right", fontsize=DEFAULT_PLOTTING["legend_font_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# PRECISION-RECALL CURVE
#
# Usage pattern:
#   fig, ax = init_pr_figure()
#   for name, precision, recall, ap in ...:
#       add_pr_curve(ax, precision, recall, ap, name)
#   finalize_pr_plot(fig, ax, "Combined_PR.png", baseline=prevalence)
# ================================================================


def init_pr_figure(figsize=DEFAULT_PLOTTING["figsize_pr"]):
    """Create and return (fig, ax) ready to receive PR curves."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall  (Sensitivity)", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Precision", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title("Combined Precision-Recall Curves — All Models", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"], alpha=DEFAULT_PLOTTING["grid_alpha"])
    return fig, ax


def add_pr_curve(ax, precision, recall, ap, name):
    """Plot one model's PR curve onto an existing axis."""
    ax.plot(recall, precision, lw=DEFAULT_PLOTTING["line_width"], label=f"{name}  (AP = {ap:.3f})")


def finalize_pr_plot(fig, ax, save_path, baseline=None):
    """
    Add the no-skill baseline, legend, then save and close.

    baseline : float or None
        Positive-class prevalence (n_pos / n_total) drawn as a horizontal
        reference line. Pass None to omit it.
    """
    if baseline is not None:
        ax.axhline(
            y=baseline,
            color="k",
            linestyle="--",
            lw=1.0,
            label=f"No Skill  (baseline = {baseline:.3f})",
        )
    ax.legend(loc="upper right", fontsize=DEFAULT_PLOTTING["legend_font_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# REGRESSION PLOTS
# ================================================================


def plot_actual_vs_predicted(y_true, y_pred, name, save_path):
    """
    Scatter of actual vs predicted values with a perfect-fit diagonal.

    Parameters
    ----------
    y_true    : array-like   Ground-truth continuous values.
    y_pred    : array-like   Model predictions.
    name      : str          Model name used in the plot title.
    save_path : str or Path  Destination PNG path.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    ax.scatter(y_true, y_pred, alpha=0.5,
               s=DEFAULT_PLOTTING["marker_size"] * 4, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--",
            lw=DEFAULT_PLOTTING["line_width"], label="Perfect fit")
    ax.set_xlabel("Actual", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Predicted", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Actual vs Predicted — {name}",
                 fontsize=DEFAULT_PLOTTING["title_size"])
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_residuals(y_true, y_pred, name, save_path):
    """
    Residuals (predicted − actual) vs predicted values with a zero reference line.

    Parameters
    ----------
    y_true    : array-like
    y_pred    : array-like
    name      : str
    save_path : str or Path
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    ax.scatter(y_pred, residuals, alpha=0.5,
               s=DEFAULT_PLOTTING["marker_size"] * 4, edgecolors="none")
    ax.axhline(y=0, color="k", linestyle="--", lw=DEFAULT_PLOTTING["line_width"])
    ax.set_xlabel("Predicted", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Residual  (Predicted - Actual)",
                  fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Residual Plot — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_error_distribution(y_true, y_pred, name, save_path):
    """
    Histogram with KDE overlay of prediction errors (residuals).

    Parameters
    ----------
    y_true    : array-like
    y_pred    : array-like
    name      : str
    save_path : str or Path
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    sns.histplot(residuals, kde=True, ax=ax, bins=30)
    ax.axvline(x=0, color="k", linestyle="--",
               lw=DEFAULT_PLOTTING["line_width"], label="Zero error")
    ax.set_xlabel("Residual  (Predicted - Actual)",
                  fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Count", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Error Distribution — {name}",
                 fontsize=DEFAULT_PLOTTING["title_size"])
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# UNSUPERVISED PLOTS
# ================================================================


def plot_cluster_scatter(X, labels, name, save_path, feature_names=None):
    """
    2-D scatter of cluster assignments.

    Uses the first two features when X already has exactly two columns.
    Otherwise projects down to 2-D with PCA (fitted on X, no leakage
    concern for unsupervised exploration).

    Noise points (label == -1, DBSCAN) are rendered as gray "x" markers
    and excluded from the colour legend.

    Parameters
    ----------
    X             : array-like (n_samples, n_features)
    labels        : array-like (n_samples,)  Cluster labels.
    name          : str                       Model name for the title.
    save_path     : str or Path
    feature_names : list[str] or None         Used to label axes when X has 2 cols.
    """
    X_arr  = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X_arr.shape[1] == 2:
        X_2d   = X_arr
        xlabel = feature_names[0] if feature_names and len(feature_names) >= 2 else "Feature 0"
        ylabel = feature_names[1] if feature_names and len(feature_names) >= 2 else "Feature 1"
    else:
        pca    = _PCA(n_components=2, random_state=42)
        X_2d   = pca.fit_transform(X_arr)
        xlabel = "PC 1"
        ylabel = "PC 2"

    unique_labels = sorted(set(labels) - {-1})
    palette       = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            color=palette[idx % len(palette)],
            alpha=0.7,
            s=DEFAULT_PLOTTING["marker_size"] * 4,
            edgecolors="none",
            label=f"Cluster {label}",
        )

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            X_2d[noise_mask, 0], X_2d[noise_mask, 1],
            c="gray", marker="x",
            s=DEFAULT_PLOTTING["marker_size"] * 3,
            label="Noise",
        )

    ax.set_xlabel(xlabel, fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel(ylabel, fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Cluster Scatter — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"], alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_pca_variance(pca_model, name, save_path):
    """
    Bar chart of per-component explained variance % with a red cumulative line.

    Parameters
    ----------
    pca_model : fitted sklearn PCA instance
    name      : str          Model name for the title.
    save_path : str or Path
    """
    evr      = np.asarray(pca_model.explained_variance_ratio_) * 100
    cum_evr  = np.cumsum(evr)
    n        = len(evr)
    x        = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    ax.bar(x, evr, alpha=0.7, label="Per-component variance %")
    ax.plot(x, cum_evr, "r-o", lw=DEFAULT_PLOTTING["line_width"],
            markersize=5, label="Cumulative variance %")

    ax.set_xlabel("Principal Component", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Explained Variance (%)", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"PCA Explained Variance — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"], alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# FORECASTING PLOTS
# ================================================================


def plot_forecast(y_true, y_pred, name, save_path):
    """
    Line plot of actual vs forecasted values over the time index.

    Parameters
    ----------
    y_true    : array-like  Observed values.
    y_pred    : array-like  Model forecasts.
    name      : str         Model name for the title.
    save_path : str or Path
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    ax.plot(x, y_true, lw=DEFAULT_PLOTTING["line_width"], label="Actual")
    ax.plot(x, y_pred, lw=DEFAULT_PLOTTING["line_width"], linestyle="--", label="Forecast")
    ax.set_xlabel("Time Step", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Value", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Forecast vs Actual — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_rolling_forecast(fold_results, name, save_path):
    """
    Show all walk-forward fold predictions concatenated on one plot.

    Actual values and forecast values are drawn in different colours.
    Vertical grey dotted lines separate folds.

    Parameters
    ----------
    fold_results : list of (y_true_fold, y_pred_fold) tuples
        Per-fold arrays from a walk-forward evaluation loop.
    name         : str   Model name for the title.
    save_path    : str or Path
    """
    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])

    offset = 0
    for fold_idx, (y_true_fold, y_pred_fold) in enumerate(fold_results):
        y_true_fold = np.asarray(y_true_fold, dtype=float)
        y_pred_fold = np.asarray(y_pred_fold, dtype=float)
        x = np.arange(offset, offset + len(y_true_fold))
        label_true = "Actual"   if fold_idx == 0 else None
        label_pred = "Forecast" if fold_idx == 0 else None
        ax.plot(x, y_true_fold, color="steelblue", lw=DEFAULT_PLOTTING["line_width"],
                alpha=0.8, label=label_true)
        ax.plot(x, y_pred_fold, color="tomato", lw=DEFAULT_PLOTTING["line_width"],
                linestyle="--", alpha=0.9, label=label_pred)
        if fold_idx > 0:
            ax.axvline(x=offset, color="gray", linestyle=":", lw=0.8, alpha=0.5)
        offset += len(y_true_fold)

    ax.set_xlabel("Time Step", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Value", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Walk-Forward Forecast — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_residuals_over_time(y_true, y_pred, name, save_path):
    """
    Scatter of residuals (predicted - actual) over the time index.

    Reveals whether forecast errors have temporal structure such as
    trend drift, heteroscedasticity, or seasonal bias.

    Parameters
    ----------
    y_true    : array-like
    y_pred    : array-like
    name      : str
    save_path : str or Path
    """
    y_true    = np.asarray(y_true, dtype=float)
    y_pred    = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true
    x         = np.arange(len(residuals))

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    ax.scatter(x, residuals, alpha=0.6,
               s=DEFAULT_PLOTTING["marker_size"] * 4, edgecolors="none")
    ax.axhline(y=0, color="k", linestyle="--", lw=DEFAULT_PLOTTING["line_width"])
    ax.set_xlabel("Time Step", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Residual (Predicted - Actual)", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"Residuals Over Time — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# SHAP PLOTS
# Requires: pip install shap
# ================================================================


def plot_shap_summary(shap_values, X, feature_names, name, save_path, *, max_display=20):
    """
    SHAP beeswarm summary plot — one dot per sample, coloured by feature value.

    Shows direction, magnitude, and feature-value correlation simultaneously.
    Requires the shap package.

    Parameters
    ----------
    shap_values   : ndarray (n_samples, n_features)   Raw SHAP values.
    X             : array-like (n_samples, n_features) Corresponding input data.
    feature_names : list[str]
    name          : str   Model name for the title.
    save_path     : str or Path
    max_display   : int   Maximum number of features shown (most important first).
    """
    try:
        import shap as _shap
    except ImportError:
        raise ImportError("shap is required for plot_shap_summary. pip install shap")

    X_arr = np.asarray(X, dtype=float)
    _shap.summary_plot(
        shap_values,
        X_arr,
        feature_names=list(feature_names),
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"], bbox_inches="tight")
    plt.close("all")


def plot_shap_bar(shap_importance_df, name, save_path, *, top_n=20):
    """
    Horizontal bar chart of global SHAP feature importance (mean |SHAP|).

    Parameters
    ----------
    shap_importance_df : pd.DataFrame
        Output of explainability.summarise_shap_importance().
        Must have "feature" and "mean_abs_shap" columns.
    name      : str   Model name for the title.
    save_path : str or Path
    top_n     : int   Number of top features to display. Default 20.
    """
    df = shap_importance_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    ax.barh(df["feature"].iloc[::-1], df["mean_abs_shap"].iloc[::-1], alpha=0.8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("Feature", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"SHAP Feature Importance — {name}", fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, axis="x", linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_shap_dependence(shap_values, X, feature_names, feature, name, save_path,
                         *, interaction_feature=None):
    """
    SHAP dependence scatter plot for one feature.

    Scatter of feature value (x-axis) vs SHAP value (y-axis). Reveals
    non-linear effects and interaction structure.

    Parameters
    ----------
    shap_values         : ndarray (n_samples, n_features)
    X                   : array-like (n_samples, n_features)
    feature_names       : list[str]
    feature             : str or int   Feature to plot (name or column index).
    name                : str          Model name for the title.
    save_path           : str or Path
    interaction_feature : str, int, or None
        When provided, dots are coloured by this feature's value to reveal
        interaction effects. Default None (no colour encoding).
    """
    names = list(feature_names)
    X_arr = np.asarray(X, dtype=float)
    shap_arr = np.asarray(shap_values, dtype=float)

    feat_idx = names.index(feature) if isinstance(feature, str) else int(feature)
    feat_name = names[feat_idx]

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])

    if interaction_feature is not None:
        int_idx = (names.index(interaction_feature)
                   if isinstance(interaction_feature, str)
                   else int(interaction_feature))
        sc = ax.scatter(
            X_arr[:, feat_idx], shap_arr[:, feat_idx],
            c=X_arr[:, int_idx],
            cmap="viridis",
            alpha=0.6,
            s=DEFAULT_PLOTTING["marker_size"] * 4,
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label=names[int_idx])
    else:
        ax.scatter(
            X_arr[:, feat_idx], shap_arr[:, feat_idx],
            alpha=0.6,
            s=DEFAULT_PLOTTING["marker_size"] * 4,
            edgecolors="none",
        )

    ax.axhline(y=0, color="k", linestyle="--", lw=DEFAULT_PLOTTING["line_width"])
    ax.set_xlabel(feat_name, fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_ylabel("SHAP value", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(f"SHAP Dependence — {feat_name} ({name})",
                 fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# STATISTICAL COMPARISON PLOTS
# ================================================================


def plot_ranking_bar(avg_ranks, title, save_path):
    """
    Horizontal bar chart of average classifier ranks (lower = better).

    Parameters
    ----------
    avg_ranks : pd.Series   avg_rank per classifier, sorted ascending (best first).
    title     : str
    save_path : str or Path
    """
    ranks = avg_ranks.sort_values(ascending=False)  # worst at top, best at bottom

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    bars = ax.barh(range(len(ranks)), ranks.values, alpha=0.8)

    # Colour best bar differently
    if len(bars) > 0:
        bars[-1].set_color("steelblue")

    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels(list(ranks.index), fontsize=DEFAULT_PLOTTING["tick_size"])
    ax.set_xlabel("Average Rank (lower is better)", fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(title, fontsize=DEFAULT_PLOTTING["title_size"])
    ax.axvline(x=ranks.values.mean(), color="red", linestyle="--",
               lw=DEFAULT_PLOTTING["line_width"], alpha=0.6, label="Mean rank")
    ax.legend(fontsize=DEFAULT_PLOTTING["legend_font_size"])
    ax.grid(True, axis="x", linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(axis="x", labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


def plot_confidence_intervals(ci_df, metric, title, save_path):
    """
    Point-and-error-bar plot of per-model confidence intervals.

    Parameters
    ----------
    ci_df     : pd.DataFrame
        Must contain columns: Model, mean, lower, upper.
    metric    : str   Metric name for the x-axis label.
    title     : str
    save_path : str or Path
    """
    df = ci_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=DEFAULT_PLOTTING["figsize_default"])
    y_pos   = np.arange(len(df))

    ax.errorbar(
        df["mean"], y_pos,
        xerr=[df["mean"] - df["lower"], df["upper"] - df["mean"]],
        fmt="o",
        capsize=4,
        markersize=DEFAULT_PLOTTING["marker_size"],
        lw=DEFAULT_PLOTTING["line_width"],
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(df["Model"]), fontsize=DEFAULT_PLOTTING["tick_size"])
    ax.set_xlabel(metric, fontsize=DEFAULT_PLOTTING["font_size"])
    ax.set_title(title, fontsize=DEFAULT_PLOTTING["title_size"])
    ax.grid(True, axis="x", linestyle=DEFAULT_PLOTTING["grid_linestyle"],
            alpha=DEFAULT_PLOTTING["grid_alpha"])
    ax.tick_params(axis="x", labelsize=DEFAULT_PLOTTING["tick_size"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=DEFAULT_PLOTTING["dpi"])
    plt.close(fig)


# ================================================================
# FUTURE PLOTS  (not yet implemented)
# ================================================================
#
# SHAP:
#   plot_shap_summary(shap_values, X, feature_names, save_path, max_display=20)
#   plot_shap_waterfall(shap_values, sample_idx, save_path)
#   plot_shap_dependence(shap_values, feature_name, X, save_path)
#
# Calibration:
#   plot_calibration_curve(y_true, y_prob, name, save_path, n_bins=10)
#   plot_calibration_comparison(calibration_data, save_path, n_bins=10)
#
# Feature Importance:
#   plot_feature_importance(importances, feature_names, save_path, top_n=20)
#   plot_permutation_importance(result, feature_names, save_path, top_n=20)
