import multiprocessing as mp
import os

from gcvae.utils import ResilientPool, resolve_device


def _square(v):
    return v * v


def _crash(v):
    os._exit(1)  # hard crash of the worker process


def test_resolve_device():
    assert resolve_device() in ("cpu", "cuda")


def test_resilient_pool_maps_and_survives_crash():
    with ResilientPool(2, mp.get_context("spawn")) as pool:
        assert pool.map(_square, [1, 2, 3]) == [1, 4, 9]
        assert pool.map(_crash, [1, 2]) == [None, None]
        assert pool.map(_square, [4]) == [16]  # still usable after the crash


def test_hungarian_accuracy_is_permutation_invariant():
    from gcvae.utils import hungarian_accuracy
    labels = [0, 0, 1, 1, 2, 2]
    assert hungarian_accuracy(labels, [2, 2, 0, 0, 1, 1]) == 1.0
    assert hungarian_accuracy(labels, [0, 0, 0, 1, 1, 1]) == 4 / 6
