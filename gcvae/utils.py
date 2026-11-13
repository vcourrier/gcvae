import os
import random
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import lru_cache

import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score


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

def assign_clusters_using_gmm(z, pi_c, mu_c, log_sigma_c):
    """
    Assigns latent vectors to clusters using a vectorized GMM calculation.

    Calculates the log-probability of each latent vector `z` belonging to each
    cluster `c` and assigns the cluster with the highest probability. 

    Args:
        z (torch.Tensor): The latent space vectors.
        pi_c (torch.Tensor): The mixing coefficients (priors) of the GMM.
        mu_c (torch.Tensor): The means of the GMM components.
        log_sigma_c (torch.Tensor): The log of the standard deviations of the GMM components.

    Returns:
        np.ndarray: The cluster assignments for each data point.
    """
    batch_size, latent_dim = z.shape
    n_clusters = pi_c.shape[0]

    sigma_2_c = torch.exp(log_sigma_c)**2
    log_pi_c = torch.log(pi_c)

    log_prob = torch.zeros(batch_size, n_clusters).to(z.device)
    
    for c in range(n_clusters):
        log_prob[:, c] = log_pi_c[c] - 0.5 * torch.sum(torch.log(sigma_2_c[:, c])) - \
                         0.5 * torch.sum((z - mu_c[:, c].unsqueeze(0)) ** 2 / sigma_2_c[:, c].unsqueeze(0), dim=1)

    cluster_assignments = torch.argmax(log_prob, dim=1).cpu().numpy()
    return cluster_assignments

def calculate_percentages(assignment_list, reference_list, num_assignments, num_references, assignment_name):
    """
    Computes the distribution of reference labels for each predicted cluster.

    Note:
        This function prints warnings to the console for clusters that are
        empty or contain fewer than 1000 observations.

    Args:
        assignment_list (np.ndarray): Predicted cluster assignments.
        reference_list (np.ndarray): True reference labels.
        num_assignments (int): The total number of unique clusters.
        num_references (int): The total number of unique reference labels.
        assignment_name (str): The name of the assignment type for logging purposes.

    Returns:
        np.ndarray: A matrix where `percentages[i, j]` is the percentage of
            items in cluster `i` that have a true label of `j`.
            Shape is [num_assignments, num_references].
    """
    percentages = np.zeros((num_assignments, num_references))
    for assignment in range(num_assignments):
        assignment_indices = np.where(assignment_list == assignment)[0]
        if len(assignment_indices) == 0:
            print(f"{assignment_name} {assignment} is empty.")
            continue
        elif len(assignment_indices) <= 1000:
            print(f"{assignment_name} {assignment} is very small (less than 1000 observations).")
            continue
        assignment_references = reference_list[assignment_indices]
        for reference in range(num_references):
            percentages[assignment, reference] = np.sum(assignment_references == reference) / len(assignment_references) * 100
    return percentages

def adjusted_rand_index(Y_pred, Y):
    """
    Compute the Adjusted Rand Index (ARI) between two sets of cluster assignments.

    Args:
        Y_pred (torch.Tensor): The predicted cluster assignments.
        Y (torch.Tensor): The true cluster assignments.

    Returns:
        float: The Adjusted Rand Index, a value between -1 and 1.
    """
    assert Y_pred.shape == Y.shape
    
    # Convert tensors to numpy arrays for compatibility with sklearn
    Y_pred_np = Y_pred.numpy()
    Y_np = Y.numpy()
    
    # Compute the Adjusted Rand Index using sklearn
    ari = adjusted_rand_score(Y_np, Y_pred_np)
    
    return ari

def color_blue(val):
    """
    Creates a CSS style string to color a cell background blue based on its value.

    This function is intended for use with the `Styler.applymap` or `Styler.map`
    methods in a pandas DataFrame to create a heatmap-like visualization.
    It assumes the input value is a percentage from 0 to 100.

    Args:
        val (float): The cell value, expected to be in the range [0, 100].

    Returns:
        str: A CSS 'background-color' style string.
    """
    # Clamp the value between 0 and 100 for safety
    val = max(0, min(100, val))
    blue_intensity = int(255 * ((val / 100)))  #+((100-val)/1000)))  # Scale the value to 0-255
    return f'background-color: rgb(0,0,{blue_intensity})'