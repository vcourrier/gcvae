import torch
import torch.nn as nn

def sample(mean, log_var):
    """
    Generate a random sample from a given distribution using the reparameterization trick.

    Args:
        mean (torch.Tensor): The mean of the Gaussian distribution.
        log_var (torch.Tensor): The logarithm of the variance of the distribution.

    Returns:
        torch.Tensor: A random sample from the distribution, with the same shape
            as `mean` and `log_var`.
    """
    device = mean.device
    epsilon = torch.randn(mean.size(), device=device)
    return mean + torch.exp(log_var / 2) * epsilon

class View(nn.Module):
    """A utility module that reshapes a tensor within an nn.Sequential pipeline.

    This class wraps the `torch.Tensor.view` method, allowing it to be used as a
    layer in a sequence of model operations.

    Args:
        size (tuple[int, ...]): The desired new shape for the tensor. One dimension
            can be -1 to be inferred from the other dimensions.
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    

class GCVAE(nn.Module):
    """A Guided Clustering Variational Autoencoder model.
    """
    def __init__(self, encoder, decoder, mean_layer, log_var_layer, continuous_output=True):
        """
        Initialize a GCVAE.

        Args:
            encoder (nn.Module): The feature extraction part of the encoder. It should
                output a feature map or vector.
            decoder (nn.Module): The decoder network that reconstructs data from a
                latent sample `z`.
            mean_layer (nn.Module): The layer that computes the mean of the latent
                distribution from the encoder's features.
            log_var_layer (nn.Module): The layer that computes the log-variance of
                the latent distribution from the encoder's features.
            continuous_output (bool): If True, applies a sigmoid activation to the
                decoder's output to scale it between 0 and 1.
        """
        super(GCVAE, self).__init__()

        self.continuous_output = continuous_output
        
        self.encoder = encoder
        self.decoder_z_y = decoder
        self.mean = mean_layer
        self.log_var = log_var_layer

        self.sigmoid = nn.Sigmoid()
    
    def encoder_forward(self, x):
        """Performs the forward pass of the encoder.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The mean of the latent distribution.
                - The log-variance of the latent distribution.
        """
        h = self.encoder(x)
        mean_tilde = self.mean(h)
        log_var_tilde = self.log_var(h)
        return mean_tilde.squeeze(), log_var_tilde.squeeze()
    
    def decoder_z_y_forward(self, z):
        """Performs the forward pass of the decoder.

        Args:
            z (torch.Tensor): A sample from the latent space.

        Returns:
            torch.Tensor: The reconstructed output data.
        """
        y_decoded = self.decoder_z_y(z)
        if self.continuous_output:
            y_decoded = self.sigmoid(y_decoded)
        return y_decoded

    def forward(self, x):
        """Performs a full forward pass.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - y_decoded: The reconstructed output data.
                - z: The sample from the latent space.
                - mean_tilde: The mean of the latent distribution.
                - log_var_tilde: The log-variance of the latent distribution.
        """
        mean_tilde, log_var_tilde = self.encoder_forward(x)
        z = sample(mean_tilde, log_var_tilde)
        y_decoded = self.decoder_z_y_forward(z)
        return y_decoded, z, mean_tilde, log_var_tilde
    
    def init_parameters(self):
        """
        Initialize the parameters of the linear layers for mean and log_var.
        """
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)