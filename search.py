"""Label-free hyperparameter search for GCVAE on MNIST-SVHN.

Bayesian optimisation (gcvae.optimization.optimize_eta2, Optuna TPE) proposes
configurations from the search space of the YAML file. Each one is trained once
per seed (pretrain, GMM initialisation, training) and its validation set is
assigned to clusters. The optimisation maximises the mean eta^2 of the
validation guide over the seeds, minus a penalty when a cluster empties; a
configuration with too many diverged seeds gets the worst score.

The final configuration is chosen among the trials by the procedure of the
thesis (gcvae.selection.select): shortlist the most relevant by eta^2, then pick
the highest across-seed agreement. The digit labels are never used for the
search or the selection; they are only logged in `oracle_*` columns.

Outputs, in results/<search_name>/:
  study.log      the Optuna journal; the search resumes from it
  runs.csv       one row per (configuration, seed); runs already there are not retrained
  preds/         validation assignments of each run
  trials.csv     one row per trial (hyperparameters, score, eta2, agreement, ...)
  selected.json  the selected configuration and the shortlist

Usage:
  python search.py --config configs/svhn_mnist_search.yaml
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import adjusted_rand_score

from gcvae.inference import assign
from gcvae.mnist_data_loader import load_svhn_mnist
from gcvae.model import build_conv_gcvae
from gcvae.optimization import optimize_eta2
from gcvae.selection import eta2, select
from gcvae.trainer import init_gmm_from_latent, pretrain_fn, train_fn
from gcvae.utils import hungarian_accuracy, resolve_device, set_seed

KEY = ["latent_dim", "hidden_dim", "beta", "beta_pretrain"]


def merge(base, overrides):
    """Recursively replace the values of `base` by those of `overrides`."""
    out = copy.deepcopy(base)
    for k, v in overrides.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cell_name(cfg):
    return f"J{cfg['latent_dim']}_h{cfg['hidden_dim']}_b{cfg['beta']:g}_bp{cfg['beta_pretrain']:g}"


def make_suggest(space):
    """Draw one configuration from the search space of the YAML file.

    Each entry is {type: categorical, choices: [...]}, {type: int, low, high[, step, log]}
    or {type: float, low, high[, log]}. Floats are rounded to 4 significant digits,
    so that run names stay readable and cached runs are found again.
    """
    def suggest(trial):
        cfg = {}
        for name in KEY:
            spec = space[name]
            if spec['type'] == 'categorical':
                cfg[name] = trial.suggest_categorical(name, spec['choices'])
            elif spec['type'] == 'int':
                cfg[name] = trial.suggest_int(name, spec['low'], spec['high'],
                                              step=spec.get('step', 1), log=spec.get('log', False))
            elif spec['type'] == 'float':
                v = trial.suggest_float(name, spec['low'], spec['high'], log=spec.get('log', False))
                cfg[name] = float(f"{v:.4g}")
            else:
                raise ValueError(f"unknown type {spec['type']!r} for {name}")
        return cfg
    return suggest


class MnistRuns:
    """`train_eval(cfg, seed)` for optimize_eta2, with every run cached in runs.csv."""

    def __init__(self, search, base, out_dir):
        self.search, self.base = search, base
        self.preds_dir = os.path.join(out_dir, 'preds')
        self.runs_csv = os.path.join(out_dir, 'runs.csv')
        os.makedirs(self.preds_dir, exist_ok=True)
        self.device = resolve_device()
        self.data = None

    def load_data(self):
        if self.data is None:
            # The split and the pairing are drawn once, so all runs see the same samples.
            set_seed(self.search.get('data_seed', self.base['general']['seed']))
            d = self.base['data']
            self.data = load_svhn_mnist(data_path=d['path'], batch_size=d['batch_size'],
                                        shuffle=d['shuffle'],
                                        train_batches=d.get('train_batches', 450),
                                        val_batches=d.get('val_batches', 90),
                                        test_batches=d.get('test_batches', 150))
        return self.data

    def pred_path(self, cfg, seed):
        return os.path.join(self.preds_dir, f"{cell_name(cfg)}_s{seed}_val.npy")

    def cached(self, cfg, seed):
        if not os.path.exists(self.runs_csv):
            return None
        runs = pd.read_csv(self.runs_csv)
        mask = (runs['seed'] == seed)
        for k in KEY:
            mask &= np.isclose(runs[k], cfg[k], rtol=1e-9, atol=0)
        return runs[mask].iloc[-1].to_dict() if mask.any() else None

    def __call__(self, cfg, seed):
        row = self.cached(cfg, seed)
        if row is None:
            print(f"    {cell_name(cfg)} seed {seed}: training", flush=True)
            row = self.train(cfg, seed)
            pd.DataFrame([row]).to_csv(self.runs_csv, mode='a', index=False,
                                       header=not os.path.exists(self.runs_csv))
        else:
            print(f"    {cell_name(cfg)} seed {seed}: already trained", flush=True)
        if row['diverged']:
            return None
        out = {k: row[k] for k in ('eta2', 'pi_min', 'min_cluster_frac')}
        out.update({k: row[k] for k in row if k.startswith('oracle_')})
        out['pred'] = np.load(self.pred_path(cfg, seed))
        return out

    def train(self, cfg, seed):
        """Train one configuration with one seed and evaluate it on the validation split."""
        train_loader, val_loader, test_loader = self.load_data()
        base, device = self.base, self.device
        K = base['model']['n_clusters']
        set_seed(seed)
        model = build_conv_gcvae(train_loader.dataset.tensors[0].shape[1:],
                                 train_loader.dataset.tensors[1].shape[1:],
                                 latent_dim=cfg['latent_dim'], hidden_dim=[cfg['hidden_dim']]).to(device)
        recon = nn.BCELoss(reduction='none')
        _, _, state = pretrain_fn(model, train_loader, val_loader,
                                  torch.optim.Adam(model.parameters(), lr=base['pretraining']['lr']),
                                  beta=cfg['beta_pretrain'], pretrain_epoch=base['pretraining']['epochs'],
                                  reconstruction_loss_fn=recon, device=device)
        model.load_state_dict(state)
        mu_c, log_sigma_c, pi_c_logits = init_gmm_from_latent(model, train_loader, K, seed=42, device=device)
        hist = train_fn(model, train_loader, val_loader,
                        torch.optim.Adam(model.parameters(), lr=base['training']['lr_nn']),
                        torch.optim.Adam([mu_c, log_sigma_c, pi_c_logits], lr=base['training']['lr_gmm']),
                        beta=cfg['beta'], epochs=base['training']['epochs'],
                        mu_c=mu_c, log_sigma_c=log_sigma_c, pi_c_logits=pi_c_logits,
                        reconstruction_loss_fn=recon, device=device)

        params = list(model.parameters()) + [mu_c, log_sigma_c, pi_c_logits]
        diverged = not all(torch.isfinite(p).all() for p in params) \
            or not np.isfinite(hist['train_loss']).all()
        row = {**cfg, 'seed': seed, 'diverged': bool(diverged),
               'final_train_loss': hist['train_loss'][-1], 'final_val_y_loss': hist['val_y_loss'][-1],
               'final_val_kl_loss': hist['val_kl_loss'][-1]}
        if diverged:
            return row

        x_val, y_val, lab_val = val_loader.dataset.tensors
        out = assign(model, mu_c, log_sigma_c, pi_c_logits, x_val, device=device)
        counts = np.bincount(out['c'], minlength=K)
        np.save(self.pred_path(cfg, seed), out['c'])
        row.update({'eta2': eta2(y_val.numpy(), out['c']),
                    'pi_min': float(F.softmax(pi_c_logits.detach(), dim=0).min()),
                    'min_cluster_frac': float(counts.min() / counts.sum()),
                    'n_clusters_used': int((counts > 0).sum())})

        # Oracle metrics: labels are logged for checking only, never used to search or select.
        x_test, _, lab_test = test_loader.dataset.tensors
        c_test = assign(model, mu_c, log_sigma_c, pi_c_logits, x_test, device=device)['c']
        row.update({'oracle_val_ari': adjusted_rand_score(lab_val.numpy(), out['c']),
                    'oracle_val_acc': hungarian_accuracy(lab_val.numpy(), out['c']),
                    'oracle_test_ari': adjusted_rand_score(lab_test.numpy(), c_test),
                    'oracle_test_acc': hungarian_accuracy(lab_test.numpy(), c_test)})
        return row


def main(config_path):
    with open(config_path, 'r') as file:
        search = yaml.safe_load(file)
    with open(search['base_config'], 'r') as file:
        base = merge(yaml.safe_load(file), search.get('base_overrides') or {})
    bo, sel = search.get('optimization', {}), search.get('selection', {})
    out_dir = os.path.join('results', search['search_name'])
    os.makedirs(out_dir, exist_ok=True)
    runs = MnistRuns(search, base, out_dir)
    print(f"Search '{search['search_name']}': {bo.get('n_trials', 40)} trials x "
          f"{len(search['seeds'])} seeds. Device: {runs.device}", flush=True)

    _, trials = optimize_eta2(
        runs, make_suggest(search['space']), n_trials=bo.get('n_trials', 40), seeds=search['seeds'],
        max_diverged_seeds=bo.get('max_diverged_seeds', 1),
        collapse_floor=bo.get('collapse_floor', 0.05), collapse_penalty=bo.get('collapse_penalty', 1.0),
        sampler_seed=bo.get('sampler_seed', 42), n_startup_trials=bo.get('n_startup_trials', 10),
        storage=os.path.join(out_dir, 'study.log'), study_name=search['search_name'],
        save_path=os.path.join(out_dir, 'trials.csv'))

    print("\nTrials (oracle_* columns use labels and are NOT used for search or selection):")
    print(trials.round(4).to_string(index=False))

    row, shortlist = select(trials, top_k=sel.get('top_k', 5), tol=sel.get('tol'),
                            min_cluster_frac=sel.get('min_cluster_frac'))
    chosen = {k: row[k].item() if hasattr(row[k], 'item') else row[k] for k in KEY}
    cols = ['trial'] + KEY + ['eta2', 'agreement', 'n_ok']
    result = {'selected': chosen, 'trial': int(row['trial']),
              'eta2': float(row['eta2']), 'agreement': float(row['agreement']),
              'shortlist': shortlist[cols].to_dict(orient='records'),
              'optimization': bo, 'selection_rule': sel}
    with open(os.path.join(out_dir, 'selected.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print("\nShortlist (by eta2):")
    print(shortlist[cols].round(4).to_string(index=False))
    print(f"\nSelected configuration: {chosen} "
          f"(eta2 = {row['eta2']:.4f}, agreement = {row['agreement']:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Label-free hyperparameter search for GCVAE on MNIST-SVHN.")
    parser.add_argument('--config', type=str, required=True, help="Path to the search YAML file.")
    args = parser.parse_args()
    main(args.config)
