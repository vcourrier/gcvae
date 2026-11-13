import numpy as np
import pandas as pd
import pytest

from gcvae.selection import agreement, eta2, select


def test_eta2_perfect_partition_is_one():
    c = np.repeat([0, 1, 2], 50)
    y = np.repeat([[0.0, 1.0], [5.0, -2.0], [10.0, 3.0]], 50, axis=0)
    assert eta2(y, c) == pytest.approx(1.0)


def test_eta2_random_partition_is_near_zero():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(5000, 3))
    c = rng.integers(0, 5, size=5000)
    assert eta2(y, c) < 0.01


def test_eta2_binary_guide_and_image_shape():
    c = np.repeat([0, 1], 100)
    y = np.repeat([0, 1], 100)  # binary guide, constant within clusters
    assert eta2(y, c) == pytest.approx(1.0)
    imgs = np.random.default_rng(1).random((200, 1, 4, 4))  # flattened per row
    assert 0.0 <= eta2(imgs, c) <= 1.0


def test_eta2_constant_guide_is_nan():
    assert np.isnan(eta2(np.ones(10), np.arange(10) % 2))


def test_agreement_identical_and_permuted():
    a = np.array([0, 0, 1, 1, 2, 2, 2])
    perm = np.array([2, 0, 1])[a]  # same partition, labels permuted
    assert agreement([a, a]) == pytest.approx(1.0)
    assert agreement([a, perm, a]) == pytest.approx(1.0)


def test_agreement_needs_two_runs():
    assert np.isnan(agreement([np.array([0, 1]), None]))


def test_select_is_relevance_first():
    table = pd.DataFrame({
        "config":    ["nuisance", "good", "ok", "diverged", "collapsed"],
        "eta2":      [0.05,       0.70,   0.68, 0.90,       0.80],
        "agreement": [0.99,       0.60,   0.90, 0.95,       0.95],
        "diverged":  [False,      False,  False, True,      False],
        "min_cluster_frac": [0.1, 0.1,    0.1,  0.1,        0.001],
    })
    # top-2 by eta2 among usable rows = good, ok -> most reproducible = ok.
    # The reproducible nuisance (agreement 0.99) is never considered.
    row, shortlist = select(table, top_k=2, min_cluster_frac=0.01)
    assert row["config"] == "ok"
    assert list(shortlist["config"]) == ["good", "ok"]
    # with a tolerance of 0.01 only "good" is within reach of the best eta2
    row, shortlist = select(table, tol=0.01, min_cluster_frac=0.01)
    assert row["config"] == "good"
    # without the size guard the collapsed config has the best eta2 and enters
    row, _ = select(table, top_k=2)
    assert row["config"] == "collapsed"


def test_select_falls_back_to_eta2_without_agreement():
    table = pd.DataFrame({"eta2": [0.2, 0.5], "agreement": [np.nan, np.nan]})
    with pytest.warns(UserWarning):
        row, _ = select(table)
    assert row["eta2"] == 0.5


def test_select_warns_when_eta2_is_zero():
    # a collapsed configuration: one cluster, perfect agreement, eta2 = 0
    table = pd.DataFrame({"eta2": [0.0, 0.001], "agreement": [1.0, 0.0]})
    with pytest.warns(UserWarning, match="eta2 = 0"):
        row, _ = select(table)
    assert row["eta2"] == 0.0   # reported, not hidden
