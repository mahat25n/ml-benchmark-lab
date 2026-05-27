import matplotlib.pyplot as plt
import seaborn as sns

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
