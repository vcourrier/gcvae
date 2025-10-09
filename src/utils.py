import numpy as np
import torch
from sklearn.metrics.cluster import adjusted_rand_score
import random


def set_seed(seed_value=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def assign_clusters_using_gmm(z, pi_c, mu_c, log_sigma_c):
    """

    Parameters:
    - z (torch.Tensor): The latent space vectors. [num_samples, latent_dim]
    - pi_c (torch.Tensor): The mixing coefficients of the GMM. [n_clusters]
    - mu_c (torch.Tensor): The means of the GMM. [latent_dim, n_clusters]
    - sigma_c (torch.Tensor): The variances of the GMM. [latent_dim, n_clusters]

    Returns:
    - cluster_assignments (np.ndarray): The cluster assignments for each data point. [num_samples]
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
    Calculate the percentage of each reference class in each assignment group.

    Parameters:
    - assignment_list (np.ndarray): The list of assignments for each data point. [num_samples]
    - reference_list (np.ndarray): The list of reference labels for each data point. [num_samples]
    - num_assignments (int): The number of unique assignments.
    - num_references (int): The number of unique reference labels.
    - assignment_name (str): The name of the assignment type (for logging purposes).

    Returns:
    - percentages (np.ndarray): The percentage matrix. [num_assignments, num_references]
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
    Compute the Adjusted Rand Index (ARI).

    Parameters:
    - Y_pred (torch.Tensor): The predicted cluster assignments. [num_samples]
    - Y (torch.Tensor): The true cluster assignments. [num_samples]

    Returns:
    - float: The Adjusted Rand Index.
    """
    assert Y_pred.shape == Y.shape
    
    # Convert tensors to numpy arrays for compatibility with sklearn
    Y_pred_np = Y_pred.numpy()
    Y_np = Y.numpy()
    
    # Compute the Adjusted Rand Index using sklearn
    ari = adjusted_rand_score(Y_np, Y_pred_np)
    
    return ari

def color_blue(val):
    blue_intensity = int(255 * ((val / 100)))  #+((100-val)/1000)))  # Scale the value to 0-255
    return f'background-color: rgb(0,0,{blue_intensity})'