"""GCVAE: Guided Clustering Variational AutoEncoder.

The MNIST-SVHN data loader (`gcvae.mnist_data_loader`) is not imported here:
it needs the `mnist` extra (torchvision).
"""
from gcvae.inference import assign, encode
from gcvae.model import GCVAE, View, build_conv_gcvae, build_mlp_gcvae
from gcvae.selection import agreement, eta2, select
from gcvae.trainer import (get_q_c_x, init_gmm_from_latent, linear_annealing,
                           pretrain_fn, train_fn)
from gcvae.utils import (ResilientPool, hungarian_accuracy, pool_worker_init,
                         resolve_device, set_seed)

__version__ = "0.2.0"

__all__ = [
    "GCVAE", "View", "build_conv_gcvae", "build_mlp_gcvae",
    "pretrain_fn", "train_fn", "linear_annealing", "init_gmm_from_latent", "get_q_c_x",
    "encode", "assign",
    "eta2", "agreement", "select",
    "set_seed", "resolve_device", "ResilientPool", "pool_worker_init", "hungarian_accuracy",
]
