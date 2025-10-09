import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

def get_q_c_x(pi_c, mean_c, log_var_c, z, epsilon=1e-5):
    """
    Compute the posterior distribution of the cluster given the latent space vector, i.e. q(c|x).

    Args:
    - pi_c (torch.Tensor): The mixing coefficients of the GMM. [n_clusters]
    - mean_c (torch.Tensor): The means of the GMM. [latent_dim, n_clusters]
    - log_var_c (torch.Tensor): The logarithm of the variance of the GMM. [latent_dim, n_clusters]
    - z (torch.Tensor): The latent space vector. [batch_size, latent_dim]
    - epsilon (float): A small value to ensure numerical stability.

    Returns:
    - posterior (torch.Tensor): The posterior distribution of the cluster given the latent space vector. [batch_size, n_clusters]
    """

    batch_size, latent_dim = z.shape
    n_clusters = pi_c.shape[0]

    sigma_2_c = torch.exp(log_var_c)
    
    mean_c_exp = mean_c.T.unsqueeze(0).expand(batch_size, n_clusters, latent_dim) 
    sigma_2_c_exp = sigma_2_c.T.unsqueeze(0).expand(batch_size, n_clusters, latent_dim)
    pi_c_exp = pi_c.unsqueeze(0).expand(batch_size, n_clusters)  

    log_pi_c_exp = torch.log(pi_c_exp + epsilon)
    log_sigma_2_c_exp = torch.log(sigma_2_c_exp + epsilon)
    
    log_numerator = log_pi_c_exp - 0.5 * torch.sum(log_sigma_2_c_exp, dim=2) - \
                    0.5 * torch.sum((z.unsqueeze(1) - mean_c_exp) ** 2 / sigma_2_c_exp, dim=2)
    
    log_denominator = torch.logsumexp(log_numerator, dim=1, keepdim=True)
    
    log_posterior = log_numerator - log_denominator
    posterior = torch.exp(log_posterior)

    return posterior



def loss(y, y_decoded, z, mean_encoder, log_var_encoder, pi_c, mean_c, log_sigma_c, \
         epsilon=1e-10, continuous = True, reconstruction_loss_fn=None, beta = 1, weights_y=[0.5,0.5]):
    """
    Compute the loss of the GCAVE model.

    Args:
        y (torch.Tensor): The ground truth target data.
        y_decoded (torch.Tensor): The reconstructed data from the decoder.
        z (torch.Tensor): The latent vectors.
        mean_encoder (torch.Tensor): The mean of the encoder's distribution q(z|x).
        log_var_encoder (torch.Tensor): The log variance of the encoder's distribution q(z|x).
        pi_c (torch.Tensor): The mixing coefficients of the GMM.
        mean_c (torch.Tensor): The means of the GMM components.
        log_sigma_c (torch.Tensor): The log standard deviation of the GMM components.
        epsilon (float): A small constant for numerical stability.
        continuous (bool): Flag indicating if the output `y` is continuous or categorical.
        reconstruction_loss_fn (callable, optional): The loss function for the
            reconstruction term. Defaults to L1Loss.
        beta (float): The weight for the KL divergence term.
        weights_y (list, optional): Class weights for the CrossEntropyLoss if `continuous`
            is False.

    Returns:
        tuple[torch.Tensor, float, float]: A tuple containing:
            - The mean ELBO loss for the batch.
            - The mean reconstruction loss component.
            - The mean KL divergence component.
    """
    batch_size = z.shape[0]
    n_clusters = pi_c.shape[0]
    log_var_c = 2*log_sigma_c

    q_c_x = get_q_c_x(pi_c, mean_c, log_var_c, z, epsilon) # Posterior distribution of the cluster given the data [batch_size, n_clusters]
    sigma_2_c = torch.exp(log_var_c)
    sigma_2_encoder = torch.exp(log_var_encoder) 

    # Assign to only one cluster but stay differentiable
    TEMPERATURE = 0.1
    logits = torch.log(q_c_x + 1e-9)
    sharp_differentiable_q = F.gumbel_softmax(logits, tau=TEMPERATURE, hard=True)
    q_c_x = sharp_differentiable_q

    # Ensure numerical stability
    q_c_x = torch.clamp(q_c_x, min=epsilon)
    pi_c = torch.clamp(pi_c, min=epsilon)

    # To match the shape of q_c_x
    sigma_2_encoder_expanded = sigma_2_encoder.unsqueeze(1).expand(-1, n_clusters, -1) 
    sigma_2_c_expanded = sigma_2_c.T.unsqueeze(0).expand(batch_size, -1, -1)
    mean_encoder_expanded = mean_encoder.unsqueeze(1).expand(-1, n_clusters, -1)
    mean_c_expanded = mean_c.T.unsqueeze(0).expand(batch_size, -1, -1) 
    pi_c_expanded = pi_c.unsqueeze(0).expand(batch_size, -1)
    log_var_c_expanded = log_var_c.T.unsqueeze(0).expand(batch_size, -1, -1)  

    if reconstruction_loss_fn is None:
        reconstruction_loss_fn = nn.L1Loss(reduction='none')

    if continuous is True :
        term1_y = reconstruction_loss_fn(y_decoded, y).view(y.shape[0], -1).sum(dim=1)
    else : 
        if y_decoded.dim() == 1:
            y_decoded = y_decoded.unsqueeze(1)
            y_decoded = torch.cat((1 - y_decoded, y_decoded), dim=1)
        term1_y = nn.CrossEntropyLoss(weight=weights_y,reduction='none')(y_decoded, y)

    term2 = 0.5 * torch.sum(q_c_x.unsqueeze(2) * (
                        log_var_c_expanded + 
                        sigma_2_encoder_expanded / sigma_2_c_expanded + 
                        (mean_encoder_expanded - mean_c_expanded) ** 2 / sigma_2_c_expanded), 
                    dim=(1, 2))
    
    term3 = - torch.sum(q_c_x * torch.log(pi_c_expanded), dim=1)
    term4 = - 0.5 * torch.sum(1 + log_var_encoder, dim=1)
    term5 = torch.sum(q_c_x * torch.log(q_c_x), dim=1)
    kl_term = term2 + term3 + term4 + term5
    ELBO = term1_y + beta * kl_term

    return ELBO.mean(), term1_y.mean().item(), kl_term.mean().item()



def compute_kl(locs_q, log_var_q, locs_p=None, scale_p=None):
    """
    Computes the KL[q||p]

    Args:
        locs_q (torch.Tensor): The mean of the first distribution, `q`.
        log_var_q (torch.Tensor): The log variance of the first distribution, `q`.
        locs_p (torch.Tensor, optional): The mean of the second distribution, `p`.
            Defaults to a tensor of zeros.
        scale_p (torch.Tensor, optional): The standard deviation of the second
            distribution, `p`. Defaults to a tensor of ones.

    Returns:
        torch.Tensor: The summed KL divergence for each item in the batch.
    """
    var_q = torch.exp(log_var_q)
    scale_q = torch.sqrt(var_q)

    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)


def loss_pretrain_kl(y_decoded, y, mean_tilde, log_var_tilde, reconstruction_loss_fn=None, beta = 1):
    """
    Compute the loss of the VAE during pretrain.

    Args:
        y_decoded (torch.Tensor): The reconstructed data from the decoder.
        y (torch.Tensor): The ground truth target data.
        mean_tilde (torch.Tensor): The mean of the encoder's distribution q(z|x).
        log_var_tilde (torch.Tensor): The log variance of the encoder's distribution q(z|x).
        reconstruction_loss_fn (callable, optional): The loss function for the
            reconstruction term. Defaults to L1Loss.
        beta (float): The weight for the KL divergence term.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - The ELBO loss for each sample in the batch.
            - The reconstruction loss component for each sample.
            - The KL divergence component for each sample.
    """
    if y.dim() == 1:
        y = y.view(-1, 1) # [batch_size, 1]
    
    if reconstruction_loss_fn is None:
        reconstruction_loss_fn = nn.L1Loss(reduction='none')

    term1_y = reconstruction_loss_fn(y_decoded, y).view(y.size(0), -1).sum(dim=1)

    latent_loss = compute_kl(mean_tilde, log_var_tilde, locs_p=None, scale_p=None)
    ELBO_pretrain = term1_y + beta*latent_loss
    return ELBO_pretrain, term1_y, latent_loss


def pretrain_fn(model, train_loader, val_loader, optimizer_nn, beta, pretrain_epoch, reconstruction_loss_fn=None, device='cpu'):
    """
    Executes the pre-training loop.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        optimizer_nn (torch.optim.Optimizer): The optimizer for the model's parameters.
        beta (float): The weight for the KL divergence term during pre-training.
        pretrain_epoch (int): The number of epochs for pre-training.
        reconstruction_loss_fn (callable, optional): The loss function for the
            reconstruction term. Defaults to L1Loss.
        device (str or torch.device): The device to run training on (e.g., 'cpu' or 'cuda').

    Returns:
        tuple[list, list, dict]: A tuple containing:
            - A list of training loss metrics per epoch.
            - A list of validation loss metrics per epoch.
            - The state dictionary of the pre-trained model.
    """
    pretrain_loss = []
    pretrain_y_loss = []
    pretrain_x_loss = []
    pretrain_kl_loss = []
    pretrain_val_loss = []
    pretrain_val_y_loss = []
    pretrain_val_x_loss = []
    pretrain_val_kl_loss = []

    for e in range(pretrain_epoch):
        model.train()
        running_train_loss = 0.0
        term_y_loss  = 0
        term_kl_loss = 0
        for batch in train_loader:
            data_x_batch = batch[0].to(device)
            data_y_batch = batch[1].to(device)

            y_decoded, _, mean_tilde, log_var_tilde = model(data_x_batch)
            loss_value, term_y, term_kl = loss_pretrain_kl(y_decoded, data_y_batch, mean_tilde, log_var_tilde, 
                                                           reconstruction_loss_fn, beta)
            loss_value = loss_value.mean()

            optimizer_nn.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_nn.step()
            optimizer_nn.zero_grad()  

            running_train_loss += loss_value.item() * data_x_batch.size(0)  
            term_y_loss  += term_y.mean().item()
            term_kl_loss += term_kl.mean().item()

        pretrain_loss.append(running_train_loss / len(train_loader.dataset))
        pretrain_y_loss.append(term_y_loss / len(train_loader) ) 
        pretrain_kl_loss.append(term_kl_loss / len(train_loader) )
        # Validation phase
        model.eval()  
        val_loss = 0.0
        val_term_y_loss  = 0
        val_term_kl_loss = 0
        with torch.no_grad(): 
            for batch in val_loader:
                val_x_batch = batch[0].to(device)
                val_y_batch = batch[1].to(device)

                val_y_decoded, _, val_mean_tilde, val_log_var_tilde = model(val_x_batch)
                val_loss_value, val_term_y, val_term_kl = loss_pretrain_kl(val_y_decoded, val_y_batch, \
                                               val_mean_tilde, val_log_var_tilde, reconstruction_loss_fn, beta)
                 
                val_loss += val_loss_value.mean().item()
                val_term_y_loss  += val_term_y.mean().item()
                val_term_kl_loss += val_term_kl.mean().item()

        val_loss /= len(val_loader)  
        pretrain_val_loss.append(val_loss)
        pretrain_val_y_loss.append(val_term_y_loss / len(val_loader)  )
        pretrain_val_kl_loss.append(val_term_kl_loss / len(val_loader)  )

    pretrained_state = model.state_dict()
    pretrain_losses = [pretrain_loss, pretrain_y_loss, pretrain_x_loss, pretrain_kl_loss]
    pretrain_val_losses = [pretrain_val_loss, pretrain_val_y_loss, pretrain_val_x_loss, pretrain_val_kl_loss]
    return pretrain_losses, pretrain_val_losses, pretrained_state


def train_fn(model, train_loader, val_loader, optimizer_nn, optimizer_gmm, beta, epochs, mu_c, log_sigma_c, pi_c_logits, reconstruction_loss_fn=None, device='cpu'): 
    """
    Executes the main training loop.

    Args:
        model (nn.Module): The GCVAE model.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        optimizer_nn (torch.optim.Optimizer): Optimizer for the VAE network parameters.
        optimizer_gmm (torch.optim.Optimizer): Optimizer for the GMM parameters.
        beta (float): Weight for the KL divergence term in the loss.
        epochs (int): The number of epochs for training.
        mu_c (nn.Parameter): The GMM component means.
        log_sigma_c (nn.Parameter): The GMM component log standard deviations.
        pi_c_logits (nn.Parameter): The logits for the GMM mixing coefficients.
        reconstruction_loss_fn (callable, optional): Loss function for reconstruction.
        device (str or torch.device): The device to run training on.

    Returns:
        dict: A dictionary containing lists of training and validation loss
            metrics for each epoch.
    """
    loss_metrics = {
        'train_loss': [],
        'train_y_loss'  : [],
        'train_kl_loss' : [],
        'train_val_loss': [], 
        'val_y_loss'  : [],
        'val_kl_loss' : []
    }

    for e in range(epochs):
        model.train() 
        train_loss = 0
        term_y_loss  = 0
        term_kl_loss = 0
        for batch in train_loader:
            data_x_batch = batch[0].to(device)
            data_y_batch = batch[1].to(device)

            pi_c = F.softmax(pi_c_logits, dim=0) 
            y_decoded, z, mean, log_var = model(data_x_batch)
            loss_value, term_y, term_kl = loss(data_y_batch, y_decoded, z, mean, log_var, \
                              pi_c, mu_c, log_sigma_c, reconstruction_loss_fn=reconstruction_loss_fn, beta=beta)

            loss_value = loss_value.mean()

            if torch.isnan(loss_value):
                print("NaN detected in loss")
                break

            optimizer_nn.zero_grad()
            optimizer_gmm.zero_grad()
            loss_value.backward()

            if torch.isnan(pi_c_logits.grad).any():
                print("NaN detected in gradients")
                break
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([pi_c_logits, mu_c, log_sigma_c], max_norm=1.0)

            optimizer_nn.step()
            optimizer_gmm.step()

            train_loss   += loss_value.item()
            term_y_loss  += term_y
            term_kl_loss += term_kl

        train_loss /= len(train_loader)  
        term_y_loss /= len(train_loader)  
        term_kl_loss /= len(train_loader)  
        loss_metrics['train_loss'].append(train_loss)
        loss_metrics['train_y_loss'].append(term_y_loss)
        loss_metrics['train_kl_loss'].append(term_kl_loss)

        # Validation phase
        model.eval()  
        val_loss = 0.0
        val_term_y_loss  = 0
        val_term_kl_loss = 0
        with torch.no_grad():  
            for batch in val_loader:
                val_x_batch = batch[0].to(device)
                val_y_batch = batch[1].to(device)

                val_y_decoded,  z_val, mean_val, log_var_val = model(val_x_batch)
                val_loss_value, val_term_y, val_term_kl = loss(val_y_batch, val_y_decoded, z_val, mean_val, \
                                               log_var_val, pi_c, mu_c, log_sigma_c, epsilon=1e-6,  \
                                                reconstruction_loss_fn=reconstruction_loss_fn, beta=beta)
                val_loss += val_loss_value.item()
                val_term_y_loss  += val_term_y
                val_term_kl_loss += val_term_kl

        val_loss /= len(val_loader) 
        val_term_y_loss  /= len(val_loader)  
        val_term_kl_loss /= len(val_loader)  

        print("Epoch: {}/{}, Validation loss: {:.4f}, Val Y: {:.4f}, Val KL: {:.4f}, Loss: {:.4f}, Y: {:.4f}, KL:{:.4f}".format(e + 1, \
                    epochs, val_loss, val_term_y_loss, val_term_kl_loss, train_loss, term_y_loss, term_kl_loss))
        loss_metrics['train_val_loss'].append(val_loss)
        loss_metrics['val_y_loss'].append(val_term_y_loss)
        loss_metrics['val_kl_loss'].append(val_term_kl_loss)
    
    return loss_metrics

def reinitialize_gmm(mu_c_pretrained, sigma_c_pretrained, pi_c_pretrained, device):
    """
    Converts pre-trained GMM parameters into trainable `nn.Parameter` tensors.

    Args:
        mu_c_pretrained (torch.Tensor): The pre-trained component means.
        sigma_c_pretrained (torch.Tensor): The pre-trained component variances.
        pi_c_pretrained (torch.Tensor): The pre-trained mixing coefficients.
        device (str or torch.device): The device to move the new parameters to.

    Returns:
        tuple[nn.Parameter, nn.Parameter, nn.Parameter]: A tuple containing the
            trainable GMM parameters:
            - mu_c: Component means.
            - log_sigma_c: Component log standard deviations.
            - pi_c_logits: Logits for the mixing coefficients.
    """
    mu_c = nn.Parameter(mu_c_pretrained.clone(), requires_grad=True).to(device)
    log_sigma_c = nn.Parameter(torch.log(torch.sqrt(sigma_c_pretrained).clone()), requires_grad=True).to(device)
    pi_c_logits = nn.Parameter(torch.log(pi_c_pretrained.clone()), requires_grad=True).to(device)
    return mu_c, log_sigma_c, pi_c_logits