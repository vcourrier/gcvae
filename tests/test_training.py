import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from gcvae import (assign, build_mlp_gcvae, encode, init_gmm_from_latent, linear_annealing,
                   pretrain_fn, train_fn)

K, J = 3, 4


@pytest.fixture
def tabular():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 6)).astype(np.float32)
    y = (x[:, :1] > 0).astype(np.float32)  # binary guide, shape (n, 1)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=32, shuffle=True, drop_last=True)
    return x, y, loader


def _fit(loader, x, beta):
    model = build_mlp_gcvae(x.shape[1], 1, w_e=16, w_d=8, J=J)
    recon = nn.BCEWithLogitsLoss(reduction="none")
    pretrain_fn(model, loader, loader, torch.optim.Adam(model.parameters(), lr=1e-3),
                beta=0.1, pretrain_epoch=1, reconstruction_loss_fn=recon)
    mu_c, log_sigma_c, pi_c_logits = init_gmm_from_latent(model, x, K, seed=0, latent="mean")
    metrics = train_fn(model, loader, loader, torch.optim.Adam(model.parameters(), lr=1e-3),
                       torch.optim.Adam([mu_c, log_sigma_c, pi_c_logits], lr=1e-3),
                       beta=beta, epochs=3, mu_c=mu_c, log_sigma_c=log_sigma_c,
                       pi_c_logits=pi_c_logits, reconstruction_loss_fn=recon)
    return model, (mu_c, log_sigma_c, pi_c_logits), metrics


KEYS = {"train_loss", "train_y_loss", "train_kl_loss", "train_val_loss", "val_y_loss", "val_kl_loss"}


@pytest.mark.parametrize("beta", [0.1, linear_annealing(0.1, 2)])
def test_train_fn_float_and_callable_beta(tabular, beta):
    x, _, loader = tabular
    _, _, metrics = _fit(loader, x, beta)
    assert set(metrics) == KEYS
    assert all(len(v) == 3 and np.isfinite(v).all() for v in metrics.values())


def test_callable_beta_is_used(tabular):
    # beta = 0 at every epoch: the loss is the reconstruction term alone
    x, _, loader = tabular
    _, _, metrics = _fit(loader, x, lambda epoch: 0.0)
    np.testing.assert_allclose(metrics["train_loss"], metrics["train_y_loss"], rtol=1e-5)


def test_linear_annealing():
    f = linear_annealing(0.5, 4)
    assert [f(e) for e in range(6)] == [0.0, 0.125, 0.25, 0.375, 0.5, 0.5]


def test_mlp_assign_shapes(tabular):
    x, _, loader = tabular
    model, gmm, _ = _fit(loader, x, 0.1)
    out = assign(model, *gmm, x)
    assert out["z"].shape == (len(x), J)
    assert out["q"].shape == (len(x), K)
    np.testing.assert_allclose(out["q"].sum(axis=1), 1.0, rtol=1e-5)
    assert out["c"].shape == (len(x),) and out["c"].max() < K
    np.testing.assert_allclose(encode(model, torch.from_numpy(x)), out["z"])


def test_logvar_clamp():
    model = build_mlp_gcvae(6, 1, 16, 8, J, logvar_clamp=(-1.0, 1.0))
    with torch.no_grad():
        model.log_var.bias.fill_(50.0)
    model.eval()
    _, log_var = model.encoder_forward(torch.randn(5, 6))
    assert log_var.max() <= 1.0
