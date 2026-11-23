"""Label-free hyperparameter selection for GCVAE.

Two signals, used in this order (thesis, Ch. 5, "ranking by relevance, then
selecting the most reproducible"):

1. relevance: eta^2, the share of the guide variance explained by the clusters;
2. reliability: agreement, the mean pairwise ARI between the assignments of
   independently seeded runs of the same configuration.

Configurations whose seeds diverge are discarded, the others are ranked by
eta^2 and a short list of the most relevant is kept; within it, the most
reproducible configuration is selected. The two signals are never combined
into one score, and labels are never used.
"""
import warnings
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score


def eta2(y, c):
    """Share of the guide variance explained by the clusters.

    eta^2 = sum_c n_c ||ybar_c - ybar||^2 / sum_i ||y_i - ybar||^2, in [0, 1].

    Args:
        y (array-like): The guide, first dimension = observations. Any other
            dimensions are flattened per row (a binary or scalar guide works).
        c (array-like): Cluster assignment of each observation.

    Returns:
        float: eta^2, or NaN if the guide is constant.
    """
    y = np.asarray(y, dtype=np.float64).reshape(len(y), -1)
    c = np.asarray(c)
    if len(c) != len(y):
        raise ValueError(f"y and c have different lengths: {len(y)} and {len(c)}")
    grand = y.mean(axis=0)
    ss_total = ((y - grand) ** 2).sum()
    if ss_total <= 0:
        return float("nan")
    ss_between = 0.0
    for k in np.unique(c):
        block = y[c == k]
        ss_between += len(block) * ((block.mean(axis=0) - grand) ** 2).sum()
    return float(ss_between / ss_total)


def agreement(assignments):
    """Mean pairwise adjusted Rand index between the assignments of several seeds.

    Args:
        assignments (list[array-like]): One assignment vector per seed, all on
            the same observations in the same order. `None` entries (failed
            runs) are ignored.

    Returns:
        float: The mean pairwise ARI, or NaN if fewer than 2 assignments remain.
    """
    usable = [np.asarray(a) for a in assignments if a is not None]
    if len(usable) < 2:
        return float("nan")
    if len({len(a) for a in usable}) > 1:
        raise ValueError("all assignments must cover the same observations")
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(usable, 2)]))


def select(table, top_k=5, tol=None, min_cluster_frac=None):
    """Select a configuration: relevance first, then reproducibility.

    1. Drop the configurations that cannot be used: `diverged` is True (if the
       column exists), eta^2 is not finite, or the smallest cluster is below
       `min_cluster_frac` (if given; needs a `min_cluster_frac` column).
    2. Keep the shortlist of the most relevant: the `top_k` largest `eta2`, or,
       if `tol` is given, every configuration with `eta2 >= max(eta2) - tol`.
    3. Within the shortlist, pick the highest `agreement`. If agreement is NaN
       for the whole shortlist (fewer than 2 seeds), fall back to the highest
       eta^2 with a warning.

    A warning is also raised when the selected configuration has eta^2 = 0: its
    clusters explain none of the guide variance (e.g. all observations in one
    cluster, which agree perfectly across seeds). That is always a bad result.

    Other columns (e.g. ARI against labels) are never read.

    Args:
        table (pandas.DataFrame): One row per configuration, with columns
            `eta2` and `agreement`, and optionally `diverged` and
            `min_cluster_frac`.
        top_k (int): Size of the shortlist when `tol` is None.
        tol (float or None): Keep the configurations within `tol` of the best eta^2.
        min_cluster_frac (float or None): Minimum share of observations in the
            smallest cluster.

    Returns:
        tuple[pandas.Series, pandas.DataFrame]: The selected row and the shortlist
            (sorted by decreasing eta^2).
    """
    usable = table[np.isfinite(table["eta2"].astype(float))]
    if "diverged" in usable:
        usable = usable[~usable["diverged"].astype(bool)]
    if min_cluster_frac is not None:
        if "min_cluster_frac" not in usable:
            raise ValueError("min_cluster_frac is set but the table has no min_cluster_frac column")
        usable = usable[usable["min_cluster_frac"] >= min_cluster_frac]
    if usable.empty:
        raise ValueError("no usable configuration: all diverged, collapsed or have no eta2")

    if tol is None:
        shortlist = usable.nlargest(top_k, "eta2")
    else:
        shortlist = usable[usable["eta2"] >= usable["eta2"].max() - tol]
        shortlist = shortlist.sort_values("eta2", ascending=False)

    if shortlist["agreement"].isna().all():
        warnings.warn("agreement is NaN for the whole shortlist (fewer than 2 seeds); "
                      "selecting the highest eta2 instead")
        row = shortlist.iloc[0]
    else:
        row = shortlist.loc[shortlist["agreement"].idxmax()]
    if row["eta2"] <= 1e-12:
        warnings.warn("the selected configuration has eta2 = 0: its clusters explain none "
                      "of the guide variance. This is a bad result; train more "
                      "configurations or check the training.")
    return row, shortlist
