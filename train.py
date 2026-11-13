import yaml
import torch
import torch.nn as nn
import argparse
import os

from gcvae.mnist_data_loader import load_svhn_mnist
from gcvae.model import build_conv_gcvae
from gcvae.trainer import pretrain_fn, train_fn, init_gmm_from_latent
from gcvae.utils import set_seed


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
        shuffle=config['data']['shuffle'],
        train_batches=config['data'].get('train_batches', 450),
        val_batches=config['data'].get('val_batches', 90),
        test_batches=config['data'].get('test_batches', 150)
    )
    
    # --- 3. MODEL BUILDING ---
    x_shape = train_loader.dataset.tensors[0].shape[1:]
    y_shape = train_loader.dataset.tensors[1].shape[1:]

    model = build_conv_gcvae(x_shape, y_shape,
                             latent_dim=config['model']['latent_dim'],
                             hidden_dim=config['model']['hidden_dim']).to(device)
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
    print("Fitting GMM to initialize parameters...")
    mu_c, log_sigma_c, pi_c_logits = init_gmm_from_latent(
        model, train_loader, config['model']['n_clusters'], seed=42, device=device
    )
    print("GMM initialized successfully.")

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