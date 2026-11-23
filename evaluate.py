"""Evaluate a GCVAE trained with train.py on MNIST-SVHN.

Loads results/<experiment_name>/final_model.pth and gmm_params.pth, rebuilds
the same data split as train.py (same config and seed), assigns the validation
and test sets to clusters with gcvae.inference.assign, and reports:
  - label-free metrics: eta^2 of the guide, cluster sizes;
  - if the split has labels: ARI and Hungarian accuracy against the digit.
The results are written to results/<experiment_name>/metrics.json.

Usage:
  python evaluate.py --config configs/svhn_mnist_config.yaml
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import adjusted_rand_score

from gcvae.inference import assign
from gcvae.mnist_data_loader import load_svhn_mnist
from gcvae.model import build_conv_gcvae
from gcvae.selection import eta2
from gcvae.utils import hungarian_accuracy, set_seed


def split_metrics(model, gmm, tensors, n_clusters, device):
    x, y = tensors[0], tensors[1]
    c = assign(model, gmm['mu_c'], gmm['log_sigma_c'], gmm['pi_c_logits'], x, device=device)['c']
    sizes = np.bincount(c, minlength=n_clusters)
    out = {'n': int(len(c)), 'eta2': eta2(y.numpy(), c), 'cluster_sizes': sizes.tolist(),
           'n_clusters_used': int((sizes > 0).sum())}
    if len(tensors) > 2:
        labels = tensors[2].numpy()
        out['ari'] = float(adjusted_rand_score(labels, c))
        out['acc'] = hungarian_accuracy(labels, c)
    return out


def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    results_dir = os.path.join('results', config['experiment_name'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same seed before loading as train.py, hence the same train/val/test split.
    set_seed(config['general']['seed'])
    d = config['data']
    train_loader, val_loader, test_loader = load_svhn_mnist(
        data_path=d['path'], batch_size=d['batch_size'], shuffle=d['shuffle'],
        train_batches=d.get('train_batches', 450), val_batches=d.get('val_batches', 90),
        test_batches=d.get('test_batches', 150))

    model = build_conv_gcvae(train_loader.dataset.tensors[0].shape[1:],
                             train_loader.dataset.tensors[1].shape[1:],
                             latent_dim=config['model']['latent_dim'],
                             hidden_dim=config['model']['hidden_dim'])
    model.load_state_dict(torch.load(os.path.join(results_dir, 'final_model.pth'),
                                     map_location='cpu', weights_only=True))
    model.to(device)
    gmm = torch.load(os.path.join(results_dir, 'gmm_params.pth'), map_location=device, weights_only=True)

    K = config['model']['n_clusters']
    metrics = {split: split_metrics(model, gmm, loader.dataset.tensors, K, device)
               for split, loader in (('val', val_loader), ('test', test_loader))}
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    for split, m in metrics.items():
        line = f"{split}: n = {m['n']}, eta2 = {m['eta2']:.4f}, clusters used = {m['n_clusters_used']}/{K}"
        if 'acc' in m:
            line += f", ARI = {m['ari']:.4f}, ACC = {m['acc']:.4f}"
        print(line)
        print(f"  cluster sizes: {m['cluster_sizes']}")
    print(f"Saved to {os.path.join(results_dir, 'metrics.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained GCVAE on MNIST-SVHN.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
