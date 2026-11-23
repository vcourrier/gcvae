import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def set_seed(seed_value=42):
    """
    Sets the random seed for reproducibility across multiple libraries.

    This function sets the seed for PyTorch (on CPU and CUDA), NumPy, and
    Python's built-in random module. It also configures CUDA operations to be
    deterministic for consistent results.

    Args:
        seed_value (int): The integer value to use as the seed. Defaults to 42.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=1)
def resolve_device():
    """Return 'cuda' if a CUDA device actually works, else 'cpu'.

    `torch.cuda.is_available()` can return True while every kernel launch fails
    (e.g. an unsupported GPU), so a small operation is run on the device first.
    The result is cached per process.

    The environment variable `GCVAE_DEVICE=cpu` forces the CPU.
    `GCVAE_DEVICE=cuda` still runs the check and warns if it falls back to CPU.

    Returns:
        str: 'cuda' or 'cpu'.
    """
    override = os.environ.get("GCVAE_DEVICE")
    if override not in (None, "", "cpu", "cuda"):
        raise ValueError(f"GCVAE_DEVICE must be 'cpu' or 'cuda', got {override!r}")
    if override == "cpu" or not torch.cuda.is_available():
        return "cpu"
    try:
        probe = torch.nn.Linear(4, 2).to("cuda")
        probe(torch.randn(2, 4, device="cuda")).sum().item()
        return "cuda"
    except Exception:
        if override == "cuda":
            warnings.warn("GCVAE_DEVICE=cuda was requested but CUDA does not work; using the CPU.")
        return "cpu"


def pool_worker_init():
    """Initializer of the `ResilientPool` workers: one torch thread per worker,
    so that n workers do not oversubscribe the CPU. It runs in the workers only,
    never in the parent process."""
    torch.set_num_threads(1)


class ResilientPool:
    """A process pool that survives a worker crash.

    A segfault or an out-of-memory kill of one worker breaks a
    `ProcessPoolExecutor` (`BrokenProcessPool`) and every later call fails. Here
    the pool is rebuilt and the call retried once. If it breaks again on the
    same inputs, `map` returns None for every input instead of raising, so the
    caller can treat those runs as failed and go on.

    The function passed to `map` must be defined at module level (picklable).
    Use the 'spawn' context with CUDA.

    Args:
        n_workers (int): Number of worker processes.
        ctx (multiprocessing context, optional): e.g. `multiprocessing.get_context("spawn")`.
        initializer (callable, optional): Run once in each worker. Defaults to
            `pool_worker_init`.
    """

    def __init__(self, n_workers, ctx=None, initializer=pool_worker_init):
        self.n_workers = n_workers
        self.ctx = ctx
        self.initializer = initializer
        self._pool = None
        self._make()

    def _make(self):
        self._pool = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=self.ctx,
                                         initializer=self.initializer)

    def map(self, fn, args_list):
        """Like `Executor.map`, as a list; None for every input after two crashes."""
        args_list = list(args_list)
        try:
            return list(self._pool.map(fn, args_list))
        except BrokenProcessPool:
            self.shutdown()
            self._make()
            try:
                return list(self._pool.map(fn, args_list))
            except BrokenProcessPool:
                self.shutdown()
                self._make()
                return [None] * len(args_list)

    def shutdown(self):
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

def hungarian_accuracy(labels, c):
    """Clustering accuracy against labels, after the best one-to-one matching
    of clusters to labels (Hungarian algorithm). Needs labels: use it as an
    oracle metric, never for selection.

    Args:
        labels (array-like): True labels (non-negative integers).
        c (array-like): Cluster assignments (non-negative integers).

    Returns:
        float: The accuracy, in [0, 1].
    """
    labels, c = np.asarray(labels).astype(int), np.asarray(c).astype(int)
    n = max(labels.max(), c.max()) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (c, labels), 1)
    rows, cols = linear_sum_assignment(-cm)
    return float(cm[rows, cols].sum() / len(labels))
