import numpy as np
import torch
import torch.nn.functional as F

from gcvae.trainer import get_q_c_x


def encode(model, x, device='cpu', batch_size=512):
    """Latent means of the inputs, i.e. the mean of q(z|x).

    Args:
        model (GCVAE): A trained model.
        x (torch.Tensor or np.ndarray): The inputs, first dimension = observations.
        device (str or torch.device): Device of the model.
        batch_size (int): Number of rows encoded at once.

    Returns:
        np.ndarray: The latent means z, shape [n, latent_dim].
    """
    x = torch.as_tensor(x)
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i:i + batch_size].to(device)
            mean, _ = model.encoder_forward(xb)
            zs.append(mean.reshape(xb.shape[0], -1).cpu().numpy())
    return np.concatenate(zs, axis=0)


def assign(model, mu_c, log_sigma_c, pi_c_logits, x, device='cpu', batch_size=512):
    """Assign observations to clusters with the trained mixture prior.

    The latent mean z of each observation is computed with `encode`, then the
    cluster posterior q(c|x) = p(c|z) under the mixture, and the most probable
    cluster.

    Args:
        model (GCVAE): A trained model.
        mu_c (torch.Tensor): Component means [latent_dim, K].
        log_sigma_c (torch.Tensor): Component log standard deviations [latent_dim, K].
        pi_c_logits (torch.Tensor): Logits of the mixing weights [K].
        x (torch.Tensor or np.ndarray): The inputs.
        device (str or torch.device): Device of the model.
        batch_size (int): Number of rows encoded at once.

    Returns:
        dict: `z` [n, latent_dim] latent means, `q` [n, K] cluster posteriors
            (rows sum to 1) and `c` [n] the argmax cluster, all numpy arrays.
    """
    z = encode(model, x, device=device, batch_size=batch_size)
    with torch.no_grad():
        pi_c = F.softmax(pi_c_logits.detach().to(device), dim=0)
        q = get_q_c_x(pi_c, mu_c.detach().to(device), 2 * log_sigma_c.detach().to(device),
                      torch.from_numpy(z).to(device)).cpu().numpy()
    return {"z": z, "q": q, "c": q.argmax(axis=1)}
