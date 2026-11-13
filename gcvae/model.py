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
    def __init__(self, encoder, decoder, mean_layer, log_var_layer, continuous_output=True,
                 logvar_clamp=None):
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
            continuous_output (bool): Despite its name, this flag only controls a
                final sigmoid. If True, a sigmoid is applied to the decoder output,
                which is then in (0, 1): use it with `nn.BCELoss` (e.g. image guides
                such as MNIST). If False, the decoder returns raw values: use it
                with `nn.BCEWithLogitsLoss` for a binary guide (the output is a
                logit), or with an L1/MSE loss for an unbounded continuous guide.
            logvar_clamp (tuple[float, float] | None): If given, the encoder
                log-variance is clamped to (lo, hi). Default None: no clamp. A wide
                clamp such as (-10, 10) only stabilises runs where the log-variance
                explodes and has no effect on healthy runs.
        """
        super(GCVAE, self).__init__()

        self.continuous_output = continuous_output
        self.logvar_clamp = logvar_clamp
        
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
        if self.logvar_clamp is not None:
            log_var_tilde = log_var_tilde.clamp(self.logvar_clamp[0], self.logvar_clamp[1])
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


def build_conv_gcvae(x_shape, y_shape, latent_dim, hidden_dim):
    """Build the convolutional GCVAE used for MNIST-SVHN.

    Encoder: three `Conv2d(4, stride 2) -> ReLU` blocks whose base width is the
    image side, then two `Conv2d(4)` heads for the latent mean and log-variance.
    Decoder: `Linear -> ReLU` blocks of widths `reversed(hidden_dim)`, a linear
    output of size prod(y_shape[1:]) reshaped to `y_shape`, and the sigmoid of
    `continuous_output=True` (use `nn.BCELoss`).

    Args:
        x_shape (tuple[int, int, int]): Shape of one input image (channels, side, side).
        y_shape (tuple[int, int, int]): Shape of one guide image (channels, side, side).
        latent_dim (int): Dimension of the latent space.
        hidden_dim (list[int]): Widths of the decoder hidden layers.

    Returns:
        GCVAE: The initialised model, on CPU.
    """
    x_depth, x_square_dim, _ = x_shape
    y_depth, y_square_dim, _ = y_shape
    y_dim = y_square_dim**2

    encoder = nn.Sequential(
        nn.Conv2d(x_depth, x_square_dim, 4, 2, 1, bias=True), nn.ReLU(True),
        nn.Conv2d(x_square_dim, x_square_dim * 2, 4, 2, 1, bias=True), nn.ReLU(True),
        nn.Conv2d(x_square_dim * 2, x_square_dim * 4, 4, 2, 1, bias=True), nn.ReLU(True),
    )

    mean_layer = nn.Conv2d(x_square_dim * 4, latent_dim, 4, 1, 0, bias=True)
    log_var_layer = nn.Conv2d(x_square_dim * 4, latent_dim, 4, 1, 0, bias=True)

    decoder_layers = []
    in_dim = latent_dim
    for out_dim in reversed(hidden_dim):
        decoder_layers.append(nn.Linear(in_dim, out_dim))
        decoder_layers.append(nn.ReLU())
        in_dim = out_dim
    decoder_layers.append(nn.Linear(hidden_dim[0], y_dim))
    decoder_layers.append(View((-1, *y_shape)))
    decoder = nn.Sequential(*decoder_layers)

    model = GCVAE(encoder, decoder, mean_layer, log_var_layer)
    model.init_parameters()
    return model


def build_mlp_gcvae(x_dim, y_dim, w_e, w_d, J, continuous_output=False,
                    enc_depth=1, dec_depth=1, logvar_clamp=None):
    """Build a GCVAE with MLP encoder and decoder, for tabular data.

    Encoder: `enc_depth` blocks of `Linear -> ReLU -> BatchNorm1d(w_e)` (the first
    maps x_dim to w_e, the others w_e to w_e), then two `Linear(w_e, J)` heads for
    the latent mean and log-variance.
    Decoder: `dec_depth` blocks of `Linear -> ReLU` (the first maps J to w_d, the
    others w_d to w_d), then a linear output `Linear(w_d, y_dim)`.

    With the default `continuous_output=False` the decoder returns raw values.
    For a binary guide, they are logits: train with `nn.BCEWithLogitsLoss`. With
    `continuous_output=True` a sigmoid is applied to the output (see `GCVAE`).

    The encoder uses BatchNorm, so a training batch of size 1 fails: drop the
    last batch of the training loader if it has a single row.

    Args:
        x_dim (int): Number of input features.
        y_dim (int): Dimension of the guide.
        w_e (int): Width of the encoder hidden layers.
        w_d (int): Width of the decoder hidden layers.
        J (int): Dimension of the latent space.
        continuous_output (bool): Apply a sigmoid to the decoder output.
        enc_depth (int): Number of encoder hidden blocks (>= 1).
        dec_depth (int): Number of decoder hidden blocks (>= 1).
        logvar_clamp (tuple[float, float] | None): See `GCVAE`.

    Returns:
        GCVAE: The initialised model, on CPU.
    """
    if enc_depth < 1 or dec_depth < 1:
        raise ValueError(f"depths must be >= 1; got enc={enc_depth}, dec={dec_depth}")

    enc_layers = []
    for i in range(enc_depth):
        in_dim = x_dim if i == 0 else w_e
        enc_layers += [nn.Linear(in_dim, w_e), nn.ReLU(), nn.BatchNorm1d(w_e)]
    encoder = nn.Sequential(*enc_layers)

    mean_layer = nn.Linear(w_e, J)
    log_var_layer = nn.Linear(w_e, J)

    dec_layers = []
    for i in range(dec_depth):
        in_dim = J if i == 0 else w_d
        dec_layers += [nn.Linear(in_dim, w_d), nn.ReLU()]
    dec_layers += [nn.Linear(w_d, y_dim)]
    decoder = nn.Sequential(*dec_layers)

    model = GCVAE(encoder, decoder, mean_layer, log_var_layer,
                  continuous_output=continuous_output, logvar_clamp=logvar_clamp)
    model.init_parameters()
    return model
