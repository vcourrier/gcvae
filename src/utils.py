import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score
import random


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