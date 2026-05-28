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
# ADVANCED STATISTICAL COMPARISON AND RANKING
# ================================================================


def run_paired_ttest(scores_a, scores_b, *, alternative="two-sided", alpha=0.05):
    """
    Paired t-test for two classifiers compared across multiple folds/datasets.

    Parameters
    ----------
    scores_a    : array-like, shape (k,)
    scores_b    : array-like, shape (k,)
    alternative : {"two-sided", "greater", "less"}
    alpha       : float

    Returns
    -------
    dict with keys: statistic, p_value, df, mean_diff, ci_low, ci_high,
                    interpretation
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must have the same length. "
            f"Got {len(a)} and {len(b)}."
        )
    if len(a) < 2:
        raise ValueError("run_paired_ttest requires at least 2 score pairs.")

    result = stats.ttest_rel(a, b, alternative=alternative)
    stat    = float(result.statistic)
    p_value = float(result.pvalue)
    df      = len(a) - 1

    diff    = a - b
    mean_diff = float(diff.mean())
    # 95% CI for mean difference using t distribution
    se      = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    t_crit  = float(stats.t.ppf(0.975, df=df))
    ci_low  = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    alt_labels = {"two-sided": "A != B", "greater": "A > B", "less": "A < B"}
    alt_str    = alt_labels.get(alternative, alternative)

    if p_value < alpha:
        direction = "A scores higher." if mean_diff > 0 else "B scores higher."
        interp = (
            f"Paired t-test (H1: {alt_str}): t={stat:.4f}, p={p_value:.4f}, df={df}. "
            f"Significant difference at alpha={alpha}. {direction} "
            f"mean diff={mean_diff:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]."
        )
    else:
        interp = (
            f"Paired t-test (H1: {alt_str}): t={stat:.4f}, p={p_value:.4f}, df={df}. "
            f"No significant difference at alpha={alpha}. "
            f"mean diff={mean_diff:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]."
        )

    return {
        "statistic":      stat,
        "p_value":        p_value,
        "df":             df,
        "mean_diff":      mean_diff,
        "ci_low":         ci_low,
        "ci_high":        ci_high,
        "interpretation": interp,
    }


def run_corrected_kfold_ttest(
    scores_a,
    scores_b,
    *,
    k=10,
    n_train=None,
    n_test=None,
    alpha=0.05,
):
    """
    Nadeau-Bengio corrected repeated k-fold t-test.

    Corrects the variance estimate for the positive correlation between
    training folds sharing the same data (Nadeau & Bengio, 2003).

    Parameters
    ----------
    scores_a : array-like, shape (k,)
    scores_b : array-like, shape (k,)
    k        : int   Number of folds.
    n_train  : int or None   Training set size. When None, correction=1/k only.
    n_test   : int or None   Test set size.
    alpha    : float

    Returns
    -------
    dict with keys: statistic, p_value, df, mean_diff, corrected_variance,
                    interpretation
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must have the same length. "
            f"Got {len(a)} and {len(b)}."
        )
    if len(a) < 2:
        raise ValueError("run_corrected_kfold_ttest requires at least 2 score pairs.")

    diff      = a - b
    mean_diff = float(diff.mean())
    n         = len(diff)

    # Nadeau-Bengio correction factor
    correction = 1.0 / k
    if n_train is not None and n_test is not None and n_train > 0:
        correction += n_test / n_train

    sample_var  = float(diff.var(ddof=1))
    corrected_var = correction * sample_var
    se = float(np.sqrt(corrected_var / n))

    if se == 0:
        stat    = 0.0
        p_value = 1.0
    else:
        stat    = float(mean_diff / se)
        p_value = float(2 * stats.t.sf(abs(stat), df=n - 1))

    df = n - 1

    if p_value < alpha:
        direction = "A scores higher." if mean_diff > 0 else "B scores higher."
        interp = (
            f"Corrected k-fold t-test (k={k}): t={stat:.4f}, p={p_value:.4f}, df={df}. "
            f"Significant difference at alpha={alpha}. {direction} "
            f"mean diff={mean_diff:.4f}, corrected variance={corrected_var:.6f}."
        )
    else:
        interp = (
            f"Corrected k-fold t-test (k={k}): t={stat:.4f}, p={p_value:.4f}, df={df}. "
            f"No significant difference at alpha={alpha}. "
            f"mean diff={mean_diff:.4f}, corrected variance={corrected_var:.6f}."
        )

    return {
        "statistic":          stat,
        "p_value":            p_value,
        "df":                 df,
        "mean_diff":          mean_diff,
        "corrected_variance": corrected_var,
        "interpretation":     interp,
    }


def compute_confidence_interval(scores, *, confidence=0.95, method="t"):
    """
    Compute a confidence interval for a 1-D array of scores.

    Parameters
    ----------
    scores     : array-like, shape (n,)
    confidence : float   Coverage level, e.g. 0.95. Default 0.95.
    method     : "t" | "normal"
        "t"      — uses t distribution (recommended for n < 30).
        "normal" — uses standard normal (z-score).

    Returns
    -------
    dict with keys: mean, std, n, lower, upper, margin
    """
    x = np.asarray(scores, dtype=float)
    if len(x) < 2:
        raise ValueError("compute_confidence_interval requires at least 2 values.")

    n    = len(x)
    mean = float(x.mean())
    std  = float(x.std(ddof=1))
    se   = std / np.sqrt(n)

    alpha = 1 - confidence
    if method == "t":
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        margin = t_crit * se
    else:
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        margin = z_crit * se

    return {
        "mean":   mean,
        "std":    std,
        "n":      n,
        "lower":  mean - margin,
        "upper":  mean + margin,
        "margin": margin,
    }


def compute_bootstrap_ci(values, *, confidence=0.95, n_bootstrap=1000,
                         stat_fn=None, random_state=42):
    """
    Bootstrap confidence interval for a scalar statistic of *values*.

    Parameters
    ----------
    values      : array-like, shape (n,)
    confidence  : float   Coverage level. Default 0.95.
    n_bootstrap : int     Number of bootstrap resamples. Default 1000.
    stat_fn     : callable or None
        Function applied to each resample to compute the statistic.
        Default is np.mean.
    random_state : int or None

    Returns
    -------
    dict with keys: mean, std, lower, upper
    """
    x = np.asarray(values, dtype=float)
    if stat_fn is None:
        stat_fn = np.mean

    rng       = np.random.default_rng(random_state)
    boot_stats = np.array([
        stat_fn(rng.choice(x, size=len(x), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "mean":  float(boot_stats.mean()),
        "std":   float(boot_stats.std()),
        "lower": lower,
        "upper": upper,
    }


def compute_average_ranks(scores):
    """
    Compute average classifier ranks from a scores matrix.

    Parameters
    ----------
    scores : pd.DataFrame or 2-D array-like, shape (n_datasets, n_classifiers)
        Each row is one dataset/fold; each column one classifier.
        Higher scores are assumed better (rank 1 = best).

    Returns
    -------
    pd.Series
        Average rank per classifier, indexed by classifier name.
        Sorted ascending (rank 1 = best classifier first).
    """
    if isinstance(scores, pd.DataFrame):
        col_names = list(scores.columns)
        data      = scores.to_numpy(dtype=float)
    else:
        data      = np.asarray(scores, dtype=float)
        if data.ndim != 2:
            raise ValueError(
                f"scores must be 2-D (n_datasets × n_classifiers); got {data.shape}."
            )
        col_names = [f"Classifier_{i}" for i in range(data.shape[1])]

    ranks = np.apply_along_axis(
        lambda row: rankdata(-row, method="average"),
        axis=1,
        arr=data,
    )
    avg_ranks = pd.Series(ranks.mean(axis=0), index=col_names, name="avg_rank")
    return avg_ranks.sort_values()


def compute_metric_leaderboard(results_df, metric, *, higher_is_better=True):
    """
    Build a ranked leaderboard table for one metric from a results DataFrame.

    Parameters
    ----------
    results_df       : pd.DataFrame   Must contain "Model" and *metric* columns.
    metric           : str            Column name to rank.
    higher_is_better : bool           Default True.

    Returns
    -------
    pd.DataFrame with columns: Rank, Model, <metric>, Delta_from_best
        Delta_from_best is always >= 0 (distance from the top-ranked model).
    """
    if "Model" not in results_df.columns:
        raise ValueError("results_df must contain a 'Model' column.")
    if metric not in results_df.columns:
        raise ValueError(f"results_df does not contain column '{metric}'.")

    df = results_df[["Model", metric]].copy().dropna(subset=[metric])
    df = df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    df["Rank"] = df.index + 1

    best_val       = float(df[metric].iloc[0])
    df["Delta_from_best"] = (
        (best_val - df[metric]).abs() if higher_is_better
        else (df[metric] - best_val).abs()
    )
    df["Delta_from_best"] = df["Delta_from_best"].round(6)

    return df[["Rank", "Model", metric, "Delta_from_best"]].reset_index(drop=True)


def compute_pairwise_comparisons(scores_df, *, test="wilcoxon", alpha=0.05):
    """
    Compute all pairwise statistical comparisons between classifiers.

    Parameters
    ----------
    scores_df : pd.DataFrame, shape (n_datasets, n_classifiers)
        Each row is one dataset/fold; each column one classifier.
        Higher values are assumed better.
    test      : "wilcoxon" | "ttest"
        Statistical test to use for each pair.
    alpha     : float   Significance level.

    Returns
    -------
    dict with keys:
        "p_values"    : pd.DataFrame (n_cls × n_cls) — NaN on diagonal
        "significant" : pd.DataFrame (n_cls × n_cls, bool)
        "effect_sizes": pd.DataFrame (n_cls × n_cls) — NaN on diagonal
    """
    classifiers = list(scores_df.columns)
    k = len(classifiers)

    p_mat    = pd.DataFrame(np.full((k, k), np.nan), index=classifiers, columns=classifiers)
    sig_mat  = pd.DataFrame(np.zeros((k, k), dtype=bool), index=classifiers, columns=classifiers)
    eff_mat  = pd.DataFrame(np.full((k, k), np.nan), index=classifiers, columns=classifiers)

    for i, ci in enumerate(classifiers):
        for j, cj in enumerate(classifiers):
            if i == j:
                continue
            a = scores_df[ci].to_numpy(dtype=float)
            b = scores_df[cj].to_numpy(dtype=float)

            try:
                if test == "wilcoxon":
                    res = run_wilcoxon_test(a, b, alpha=alpha)
                    p   = res["p_value"]
                    eff = res["effect_size_r"]
                else:
                    res = run_paired_ttest(a, b, alpha=alpha)
                    p   = res["p_value"]
                    eff = float(res["mean_diff"] / (np.std(a - b, ddof=1) + 1e-12))
                p_mat.loc[ci, cj]   = p
                sig_mat.loc[ci, cj] = p < alpha
                eff_mat.loc[ci, cj] = eff
            except Exception:
                pass

    return {
        "p_values":    p_mat,
        "significant": sig_mat,
        "effect_sizes": eff_mat,
    }


def compute_significance_summary(pairwise_result):
    """
    Extract significant pairs from a compute_pairwise_comparisons() result.

    Parameters
    ----------
    pairwise_result : dict   Output of compute_pairwise_comparisons().

    Returns
    -------
    pd.DataFrame with columns: Classifier_A, Classifier_B, p_value, effect_size
        Only pairs where A beats B significantly (p < alpha) are listed.
        Each pair appears once (A < B lexicographically).
    """
    p_mat   = pairwise_result["p_values"]
    sig_mat = pairwise_result["significant"]
    eff_mat = pairwise_result["effect_sizes"]

    classifiers = list(p_mat.index)
    rows = []
    for i, ci in enumerate(classifiers):
        for j, cj in enumerate(classifiers):
            if j <= i:
                continue
            if sig_mat.loc[ci, cj] or sig_mat.loc[cj, ci]:
                rows.append({
                    "Classifier_A":  ci,
                    "Classifier_B":  cj,
                    "p_value":       round(float(p_mat.loc[ci, cj]), 6),
                    "effect_size":   round(float(eff_mat.loc[ci, cj]), 4),
                })

    if not rows:
        return pd.DataFrame(columns=["Classifier_A", "Classifier_B", "p_value", "effect_size"])
    return pd.DataFrame(rows).reset_index(drop=True)


def compute_cd_nemenyi(n_classifiers, n_datasets, *, alpha=0.05):
    """
    Compute Nemenyi critical difference (CD) for post-hoc analysis after Friedman.

    CD = q_alpha * sqrt(k*(k+1) / (6*N))

    where q_alpha is from the Studentized range distribution divided by sqrt(2).

    Parameters
    ----------
    n_classifiers : int   k — number of classifiers.
    n_datasets    : int   N — number of datasets/folds.
    alpha         : float   Significance level. Default 0.05.

    Returns
    -------
    float   Critical difference value.
    """
    if n_classifiers < 2:
        raise ValueError("n_classifiers must be >= 2.")
    if n_datasets < 1:
        raise ValueError("n_datasets must be >= 1.")

    # q_alpha from Studentized range; divide by sqrt(2) for Nemenyi
    q_alpha = float(
        stats.studentized_range.ppf(1 - alpha, k=n_classifiers, df=np.inf) / np.sqrt(2)
    )
    cd = q_alpha * np.sqrt(n_classifiers * (n_classifiers + 1) / (6 * n_datasets))
    return float(cd)


def prepare_cd_diagram_data(scores_df, *, alpha=0.05):
    """
    Prepare data for a critical difference diagram (Demsar 2006).

    Parameters
    ----------
    scores_df : pd.DataFrame, shape (n_datasets, n_classifiers)
        Each row is one dataset/fold; each column one classifier.
        Higher values are assumed better.
    alpha     : float   Significance level. Default 0.05.

    Returns
    -------
    dict with keys:
        "avg_ranks"        : pd.Series   Average rank per classifier (ascending).
        "cd"               : float       Nemenyi critical difference.
        "significant_pairs": list[tuple] Pairs (A, B) where |rank_A - rank_B| > CD.
        "n_classifiers"    : int
        "n_datasets"       : int
    """
    n_datasets, n_classifiers = scores_df.shape

    avg_ranks = compute_average_ranks(scores_df)
    cd        = compute_cd_nemenyi(n_classifiers, n_datasets, alpha=alpha)

    classifiers = list(avg_ranks.index)
    sig_pairs   = []
    for i, ci in enumerate(classifiers):
        for j, cj in enumerate(classifiers):
            if j <= i:
                continue
            if abs(float(avg_ranks[ci]) - float(avg_ranks[cj])) > cd:
                sig_pairs.append((ci, cj))

    return {
        "avg_ranks":         avg_ranks,
        "cd":                cd,
        "significant_pairs": sig_pairs,
        "n_classifiers":     n_classifiers,
        "n_datasets":        n_datasets,
    }


# ================================================================
# FUTURE: BAYESIAN TESTS  (not yet implemented)
# ================================================================
#
#   run_bayesian_signed_rank_test(scores_a, scores_b, *,
#                                 rope=0.01, prior_strength=0.75)
#   Bayesian counterpart to the Wilcoxon signed-rank (Benavoli et al. 2017).
#   Returns posterior probabilities: P(A > B), P(A ≈ B), P(B > A).
#   Requires: pip install baycomp
#
#   run_bayesian_correlated_t_test(scores_a, scores_b, *,
#                                  rope=0.01, runs=1, folds=10)
#   Correlated Bayesian t-test for k-fold cross-validation results
#   (Nadeau & Bengio 2003 correction applied).
#   Returns: P(A > B), P(A ≈ B), P(B > A).
#   Requires: pip install baycomp
