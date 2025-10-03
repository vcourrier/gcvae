import torch
import torch.nn as nn

def sample(mean, log_var):
    """
    Generate a random sample from a given distribution using the reparameterization trick.
    """
    device = mean.device
    epsilon = torch.randn(mean.size(), device=device)
    return mean + torch.exp(log_var / 2) * epsilon

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)
    

class VAE_GMM(nn.Module):
    def __init__(self, encoder, decoder, mean_layer, log_var_layer, continuous_output=True):
        """
        Initialize a fully generalized VAE-GMM.

        Parameters:
        - encoder (nn.Module): The feature extraction part of the encoder.
        - decoder (nn.Module): The decoder network.
        - mean_layer (nn.Module): The layer that computes the mean from the encoder's features.
        - log_var_layer (nn.Module): The layer that computes the log-variance from the encoder's features.
        - continuous_output (bool): If true, applies a sigmoid to the decoder's output.
        """
        super(VAE_GMM, self).__init__()

        self.continuous_output = continuous_output
        
        self.encoder = encoder
        self.decoder_z_y = decoder
        self.mean = mean_layer
        self.log_var = log_var_layer

        self.sigmoid = nn.Sigmoid()
    
    def encoder_forward(self, x):
        """
        Forward pass of the encoder.
        """
        h = self.encoder(x)
        mean_tilde = self.mean(h)
        log_var_tilde = self.log_var(h)
        return mean_tilde.squeeze(), log_var_tilde.squeeze()
    
    def decoder_z_y_forward(self, z):
        """
        Forward pass of the decoder.
        """
        y_decoded = self.decoder_z_y(z)
        if self.continuous_output:
            y_decoded = self.sigmoid(y_decoded)
        return y_decoded

    def forward(self, x):
        """
        Full forward pass of the model.
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


