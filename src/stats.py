"""
Statistical comparison utilities for ML benchmark results.

Reference methodology:
  Demsar, J. (2006). Statistical comparisons of classifiers over
  multiple data sets. JMLR, 7, 1–30.

Test selection guide
--------------------
  McNemar       : two classifiers, one dataset, requires raw predictions
  Wilcoxon      : two classifiers, multiple datasets/folds, metric scores
  Friedman      : three or more classifiers, multiple datasets/folds
  (Nemenyi)     : post-hoc after a significant Friedman result
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata


# ================================================================
# INTERNAL HELPERS
# ================================================================


def _build_mcnemar_table(y_true, y_pred_a, y_pred_b):
    """
    Build the McNemar 2×2 contingency table from paired predictions.

    Returns
    -------
    table : ndarray, shape (2, 2)
        [[n_both_correct,    n_a_correct_b_wrong],
         [n_a_wrong_b_correct, n_both_wrong]]

    b : int  — A correct, B wrong (discordant, favours A)
    c : int  — A wrong, B correct (discordant, favours B)
    """
    y_true  = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    if not (y_true.shape == y_pred_a.shape == y_pred_b.shape):
        raise ValueError(
            "y_true, y_pred_a, and y_pred_b must have the same shape. "
            f"Got {y_true.shape}, {y_pred_a.shape}, {y_pred_b.shape}."
        )

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    a = int(np.sum( correct_a &  correct_b))
    b = int(np.sum( correct_a & ~correct_b))
    c = int(np.sum(~correct_a &  correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    return np.array([[a, b], [c, d]]), b, c


def _direction_label(b, c):
    """Human-readable direction from discordant cell counts."""
    if b > c:
        return "Classifier A outperforms Classifier B on discordant cases."
    if c > b:
        return "Classifier B outperforms Classifier A on discordant cases."
    return "Classifiers are tied on discordant cases."


# ================================================================
# PUBLIC API
# ================================================================


def run_mcnemar_test(
    y_true,
    y_pred_a,
    y_pred_b,
    *,
    correction=True,
    alpha=0.05,
):
    """
    McNemar's test for pairwise classifier comparison on a single dataset.

    Tests whether two classifiers make significantly different errors on the
    same test set. The test operates on discordant pairs only (cases where
    exactly one classifier is correct). It does not require class balance.

    When the number of discordant pairs (b + c) is fewer than 25, the
    asymptotic chi-squared p-value may be unreliable; interpret with caution
    or collect more data.

    Parameters
    ----------
    y_true   : array-like, shape (n_samples,)
        Ground-truth class labels.
    y_pred_a : array-like, shape (n_samples,)
        Predictions from classifier A.
    y_pred_b : array-like, shape (n_samples,)
        Predictions from classifier B.
    correction : bool
        Apply Yates' continuity correction (default True). Recommended for
        small discordant-pair counts; reduces inflated Type I error.
    alpha : float
        Significance level for the interpretation string. Default 0.05.

    Returns
    -------
    dict with keys:
        "statistic"         : float  chi-squared statistic
        "p_value"           : float
        "contingency_table" : ndarray (2,2)
                              [[both_correct,    A_correct_B_wrong],
                               [A_wrong_B_correct, both_wrong]]
        "n_discordant"      : int    b + c
        "interpretation"    : str    plain-language conclusion
    """
    table, b, c = _build_mcnemar_table(y_true, y_pred_a, y_pred_b)

    if b + c == 0:
        return {
            "statistic":         0.0,
            "p_value":           1.0,
            "contingency_table": table,
            "n_discordant":      0,
            "interpretation": (
                "McNemar test: no discordant pairs found. "
                "The two classifiers make identical predictions on every sample."
            ),
        }

    if b + c < 25:
        warnings.warn(
            f"McNemar test: only {b + c} discordant pair(s). "
            "The chi-squared approximation may be unreliable for n_discordant < 25. "
            "Consider collecting more test samples.",
            stacklevel=2,
        )

    denom = b + c
    if correction:
        stat = (max(abs(b - c) - 1.0, 0.0) ** 2) / denom
    else:
        stat = (b - c) ** 2 / denom

    p_value = float(stats.chi2.sf(stat, df=1))
    stat    = float(stat)

    correction_note = " (Yates' continuity correction applied)" if correction else ""
    direction       = _direction_label(b, c)

    if p_value < alpha:
        interp = (
            f"McNemar test{correction_note}: chi2={stat:.4f}, p={p_value:.4f}. "
            f"Significant difference in error rates at alpha={alpha} "
            f"(discordant pairs: b={b}, c={c}). {direction}"
        )
    else:
        interp = (
            f"McNemar test{correction_note}: chi2={stat:.4f}, p={p_value:.4f}. "
            f"No significant difference in error rates at alpha={alpha} "
            f"(discordant pairs: b={b}, c={c}). {direction}"
        )

    return {
        "statistic":         stat,
        "p_value":           p_value,
        "contingency_table": table,
        "n_discordant":      b + c,
        "interpretation":    interp,
    }


def run_wilcoxon_test(
    scores_a,
    scores_b,
    *,
    alternative="two-sided",
    zero_method="wilcox",
    alpha=0.05,
):
    """
    Wilcoxon signed-rank test for pairwise classifier comparison across
    multiple datasets or CV folds.

    Non-parametric alternative to the paired t-test. Does not assume
    normality of score differences. Recommended when comparing two
    classifiers across k ≥ 5 independent datasets or folds.

    Parameters
    ----------
    scores_a    : array-like, shape (k,)
        Metric scores for classifier A (one per dataset or fold).
    scores_b    : array-like, shape (k,)
        Metric scores for classifier B.
    alternative : {"two-sided", "greater", "less"}
        Hypothesis direction. "two-sided" tests whether distributions
        differ. "greater" tests whether A scores higher than B.
        Default "two-sided".
    zero_method : {"wilcox", "pratt", "zsplit"}
        Treatment of zero differences (tied pairs). "wilcox" discards them
        (default). "pratt" includes them in ranking. "zsplit" splits them.
    alpha       : float
        Significance level. Default 0.05.

    Returns
    -------
    dict with keys:
        "statistic"      : float   Wilcoxon W statistic (sum of signed ranks)
        "p_value"        : float
        "n_pairs"        : int     number of score pairs
        "effect_size_r"  : float   matched-pairs rank-biserial correlation
                           r = W / (k*(k+1)/2); range [−1, 1]
        "interpretation" : str     plain-language conclusion
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must have the same length. "
            f"Got {len(a)} and {len(b)}."
        )
    if len(a) < 5:
        warnings.warn(
            f"Wilcoxon test: only {len(a)} pair(s). Minimum recommended is 5; "
            "the test has very low power with fewer pairs.",
            stacklevel=2,
        )

    result  = stats.wilcoxon(a, b, zero_method=zero_method, alternative=alternative)
    stat    = float(result.statistic)
    p_value = float(result.pvalue)
    k       = len(a)

    # Rank-biserial correlation as effect size
    max_w       = k * (k + 1) / 2
    effect_size = stat / max_w if max_w > 0 else 0.0

    alt_labels = {
        "two-sided": "A != B",
        "greater":   "A > B",
        "less":      "A < B",
    }
    alt_str = alt_labels.get(alternative, alternative)

    if p_value < alpha:
        direction = (
            "A scores higher." if np.median(a - b) > 0
            else "B scores higher."
        )
        interp = (
            f"Wilcoxon signed-rank test (H1: {alt_str}): "
            f"W={stat:.4f}, p={p_value:.4f}. "
            f"Significant difference at alpha={alpha} (n={k} pairs). "
            f"{direction} Effect size r={effect_size:.3f}."
        )
    else:
        interp = (
            f"Wilcoxon signed-rank test (H1: {alt_str}): "
            f"W={stat:.4f}, p={p_value:.4f}. "
            f"No significant difference at alpha={alpha} (n={k} pairs). "
            f"Effect size r={effect_size:.3f}."
        )

    return {
        "statistic":      stat,
        "p_value":        p_value,
        "n_pairs":        k,
        "effect_size_r":  float(effect_size),
        "interpretation": interp,
    }


def run_friedman_test(scores, *, alpha=0.05):
    """
    Friedman test for comparing three or more classifiers across multiple
    datasets or CV folds.

    Non-parametric omnibus test (Demsar 2006, JMLR). Tests the null
    hypothesis that all classifiers perform equally. A significant result
    warrants a post-hoc Nemenyi test to identify which specific pairs differ.

    Parameters
    ----------
    scores : pd.DataFrame or 2D array-like, shape (n_datasets, n_classifiers)
        Each row is one independent dataset or fold.
        Each column is one classifier.
        When a DataFrame is passed, column names are used in the output.
        Higher values are assumed better (ranks are assigned accordingly).
    alpha : float
        Significance level. Default 0.05.

    Returns
    -------
    dict with keys:
        "statistic"      : float   Friedman chi-squared statistic
        "p_value"        : float
        "df"             : int     degrees of freedom (n_classifiers − 1)
        "avg_ranks"      : pd.Series
                           Mean rank per classifier, rank 1 = best.
                           Sorted ascending (best first).
        "interpretation" : str     plain-language conclusion with follow-up guidance
    """
    if isinstance(scores, pd.DataFrame):
        col_names = list(scores.columns)
        data = scores.to_numpy(dtype=float)
    else:
        data = np.asarray(scores, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"scores must be a 2D array (n_datasets × n_classifiers); "
                f"received shape {data.shape}."
            )
        col_names = [f"Classifier_{i}" for i in range(data.shape[1])]

    if data.ndim != 2:
        raise ValueError(
            f"scores must be a 2D array (n_datasets × n_classifiers); "
            f"received shape {data.shape}."
        )

    n_datasets, n_classifiers = data.shape

    if n_classifiers < 3:
        raise ValueError(
            f"Friedman test requires at least 3 classifiers; "
            f"received {n_classifiers}. Use run_wilcoxon_test() for 2 classifiers."
        )
    if n_datasets < 2:
        raise ValueError(
            f"Friedman test requires at least 2 datasets/folds; "
            f"received {n_datasets}."
        )
    if n_datasets < 5:
        warnings.warn(
            f"Friedman test: only {n_datasets} dataset(s)/fold(s). "
            "The test has low power with fewer than 5 observations.",
            stacklevel=2,
        )

    # Rank within each row: rank 1 = highest score (best classifier)
    ranks = np.apply_along_axis(
        lambda row: rankdata(-row, method="average"),
        axis=1,
        arr=data,
    )
    avg_ranks_arr = ranks.mean(axis=0)
    avg_ranks = (
        pd.Series(avg_ranks_arr, index=col_names, name="avg_rank")
        .sort_values()
    )

    stat, p_value = stats.friedmanchisquare(
        *[data[:, i] for i in range(n_classifiers)]
    )
    stat    = float(stat)
    p_value = float(p_value)
    df      = n_classifiers - 1

    if p_value < alpha:
        best = avg_ranks.index[0]
        interp = (
            f"Friedman test: chi2={stat:.4f}, df={df}, p={p_value:.4f}. "
            f"Significant differences exist among classifiers at alpha={alpha} "
            f"(n={n_datasets} datasets, k={n_classifiers} classifiers). "
            f"'{best}' ranks best (avg rank={avg_ranks.iloc[0]:.2f}). "
            f"Post-hoc Nemenyi test recommended to identify differing pairs."
        )
    else:
        interp = (
            f"Friedman test: chi2={stat:.4f}, df={df}, p={p_value:.4f}. "
            f"No significant differences among classifiers at alpha={alpha} "
            f"(n={n_datasets} datasets, k={n_classifiers} classifiers)."
        )

    return {
        "statistic":      stat,
        "p_value":        p_value,
        "df":             df,
        "avg_ranks":      avg_ranks,
        "interpretation": interp,
    }


# ================================================================
# FUTURE: ADVANCED STATISTICAL TESTS  (not yet implemented)
# ================================================================
#
# Nemenyi post-hoc test (follows a significant Friedman result):
#   run_nemenyi_test(scores, *, alpha=0.05)
#   scores : same (n_datasets × n_classifiers) input as run_friedman_test.
#   Computes the critical difference (CD) at the given alpha level using
#   the Studentized range distribution. Two classifiers differ significantly
#   when |avg_rank_i − avg_rank_j| > CD.
#   Returns: pairwise p-value matrix (DataFrame), CD value, and a
#   significance mask DataFrame (bool) for easy heatmap plotting.
#   Requires: pip install scikit-posthocs  OR  manual implementation
#             via scipy.stats.studentized_range.
#
# Diebold-Mariano test (pairwise forecast accuracy for time series):
#   run_diebold_mariano_test(errors_a, errors_b, *, h=1,
#                            power=2, alternative="two-sided", alpha=0.05)
#   errors_a, errors_b : 1-D arrays of forecast errors (one per time step).
#   h : forecast horizon (accounts for serial correlation in multi-step forecasts).
#   power : loss differential power (1 = absolute error, 2 = squared error).
#   Returns: DM statistic, p-value, interpretation.
#   Requires: no additional dependencies (scipy.stats.t for the p-value).
#
# Bayesian comparison tests:
#   run_bayesian_signed_rank_test(scores_a, scores_b, *,
#                                 rope=0.01, prior_strength=0.75)
#   Bayesian counterpart to the Wilcoxon signed-rank (Benavoli et al. 2017).
#   Returns posterior probabilities: P(A > B), P(A ≈ B), P(B > A).
#   rope : Region Of Practical Equivalence half-width; differences smaller
#          than rope are considered practically equivalent.
#   Requires: pip install baycomp
#
#   run_bayesian_correlated_t_test(scores_a, scores_b, *,
#                                  rope=0.01, runs=1, folds=10)
#   Correlated Bayesian t-test for k-fold cross-validation results
#   (Nadeau & Bengio 2003 correction applied).
#   Returns: P(A > B), P(A ≈ B), P(B > A).
#   Requires: pip install baycomp
