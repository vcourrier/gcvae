import yaml
import torch
import torch.nn as nn
from sklearn import mixture
import argparse
import os

# Import from your new package structure
from src.mnist_data_loader import load_svhn_mnist
from src.model import GCVAE, View
from src.trainer import pretrain_fn, train_fn
from src.utils import set_seed


def initialize_gmm_parameters(model, train_loader, n_clusters, device):
    """Initializes GMM parameters by fitting on the latent space."""
    print("Fitting GMM to initialize parameters...")
    with torch.no_grad():
        z_list = []
        for batch in train_loader:
            x_batch = batch[0].to(device)
            _, z, _, _ = model(x_batch)
            z_list.append(z)
        z_all = torch.cat(z_list, dim=0).cpu().numpy()

    gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)
    gmm.fit(z_all)
    
    mu_c = torch.tensor(gmm.means_.T, dtype=torch.float32, device=device)
    variances_c = torch.tensor(gmm.covariances_.T, dtype=torch.float32, device=device)
    log_sigma_c = torch.log(torch.sqrt(variances_c))
    pi_c_logits = torch.log(torch.tensor(gmm.weights_, dtype=torch.float32, device=device))
    
    print("GMM initialized successfully.")
    return nn.Parameter(mu_c, requires_grad=True), \
           nn.Parameter(log_sigma_c, requires_grad=True), \
           nn.Parameter(pi_c_logits, requires_grad=True)


def main(config_path):
    # --- 1. SETUP ---
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create a directory to save results
    results_dir = os.path.join('results', config['experiment_name'])
    os.makedirs(results_dir, exist_ok=True)

    # Set seed for reproducibility and select device
    set_seed(config['general']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. DATA LOADING ---
    print("Loading data...")
    train_loader, val_loader, test_loader = load_svhn_mnist(
        data_path=config['data']['path'],
        batch_size=config['data']['batch_size'],
        shuffle=config['data']['shuffle']
    )
    
    # --- 3. MODEL BUILDING ---
    x_shape = train_loader.dataset.tensors[0].shape[1:]
    y_shape = train_loader.dataset.tensors[1].shape[1:]

    x_depth, x_square_dim, _ = x_shape
    y_depth, y_square_dim, _ = y_shape
    y_dim = y_square_dim**2
    
    latent_dim = config['model']['latent_dim']
    hidden_dim = config['model']['hidden_dim']

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

    model = GCVAE(encoder, decoder, mean_layer, log_var_layer).to(device)
    model.init_parameters()
    print("Model built and initialized successfully.")


    # --- 4. PRE-TRAINING PHASE ---
    print("\n--- Starting Pre-training Phase ---")
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=config['pretraining']['lr'])
    recon_loss_fn = nn.BCELoss(reduction='none') # Or BCEWithLogitsLoss if you remove sigmoid from model

    _, _, pretrained_state = pretrain_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_nn=optimizer_pretrain,
        beta=config['pretraining']['beta'],
        pretrain_epoch=config['pretraining']['epochs'],
        reconstruction_loss_fn=recon_loss_fn,
        device=device
    )
    model.load_state_dict(pretrained_state)
    torch.save(model.state_dict(), os.path.join(results_dir, 'pretrained_model.pth'))
    print("Pre-training finished and model saved.")

    # --- 5. GMM INITIALIZATION ---
    mu_c, log_sigma_c, pi_c_logits = initialize_gmm_parameters(
        model, train_loader, config['model']['n_clusters'], device
    )

    # --- 6. FULL TRAINING PHASE ---
    print("\n--- Starting Full Training Phase ---")
    optimizer_nn = torch.optim.Adam(model.parameters(), lr=config['training']['lr_nn'])
    optimizer_gmm = torch.optim.Adam([mu_c, log_sigma_c, pi_c_logits], lr=config['training']['lr_gmm'])

    loss_metrics = train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_nn=optimizer_nn,
        optimizer_gmm=optimizer_gmm,
        beta=config['training']['beta'],
        epochs=config['training']['epochs'],
        mu_c=mu_c,
        log_sigma_c=log_sigma_c, 
        pi_c_logits=pi_c_logits,
        reconstruction_loss_fn=recon_loss_fn,
        device=device
    )
    torch.save(model.state_dict(), os.path.join(results_dir, 'final_model.pth'))
    print("Full training finished and final model saved.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a VAE-GMM model.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)