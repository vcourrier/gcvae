"""Label-free hyperparameter search for GCVAE on MNIST-SVHN.

For each configuration of the grid and each seed: pretrain, initialise the GMM,
train, then assign the validation set to clusters. Per run, the guide-explained
variance eta^2 (validation guide y, assigned clusters) and the cluster sizes
are recorded. Per configuration, the agreement is the mean pairwise ARI between
the seeds' validation assignments. The selection (gcvae.selection.select) keeps
the configurations whose seeds did not diverge, shortlists the most relevant by
eta^2 and picks the most reproducible inside the shortlist.

The digit labels are never used for the selection. They are only logged in
`oracle_*` columns, to check the procedure afterwards.

Outputs, in results/<search_name>/:
  runs.csv       one row per (configuration, seed); the search resumes from it
  preds/         validation assignments of each run (for the agreement)
  summary.csv    one row per configuration
  selected.json  the selected configuration and the shortlist

Usage:
  python search.py --config configs/svhn_mnist_search.yaml
"""
import argparse
import copy
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import adjusted_rand_score

from gcvae.inference import assign
from gcvae.mnist_data_loader import load_svhn_mnist
from gcvae.model import build_conv_gcvae
from gcvae.selection import agreement, eta2, select
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


def run_one(cfg, seed, base, data, device, preds_dir):
    """Train one configuration with one seed and evaluate it on the validation split."""
    train_loader, val_loader, test_loader = data
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
    np.save(os.path.join(preds_dir, f"{cell_name(cfg)}_s{seed}_val.npy"), out['c'])
    row.update({'eta2': eta2(y_val.numpy(), out['c']),
                'min_cluster_frac': float(counts.min() / counts.sum()),
                'n_clusters_used': int((counts > 0).sum())})

    # Oracle metrics: labels are logged for checking only, never used to select.
    x_test, _, lab_test = test_loader.dataset.tensors
    c_test = assign(model, mu_c, log_sigma_c, pi_c_logits, x_test, device=device)['c']
    row.update({'oracle_val_ari': adjusted_rand_score(lab_val.numpy(), out['c']),
                'oracle_val_acc': hungarian_accuracy(lab_val.numpy(), out['c']),
                'oracle_test_ari': adjusted_rand_score(lab_test.numpy(), c_test),
                'oracle_test_acc': hungarian_accuracy(lab_test.numpy(), c_test)})
    return row


def summarize(runs, preds_dir, max_diverged_seeds):
    """One row per configuration: label-free signals, then oracle means."""
    rows = []
    for key, g in runs.groupby(KEY, sort=False):
        cfg = dict(zip(KEY, key))
        ok = g[~g['diverged'].astype(bool)]
        preds = [np.load(os.path.join(preds_dir, f"{cell_name(cfg)}_s{s}_val.npy")) for s in ok['seed']]
        row = {**cfg, 'n_seeds': len(g), 'n_ok': len(ok),
               'diverged': len(g) - len(ok) > max_diverged_seeds,
               'eta2': ok['eta2'].mean() if len(ok) else np.nan,
               'eta2_sd': ok['eta2'].std() if len(ok) > 1 else np.nan,
               'agreement': agreement(preds),
               'min_cluster_frac': ok['min_cluster_frac'].min() if len(ok) else np.nan}
        for col in ('oracle_val_ari', 'oracle_val_acc', 'oracle_test_ari', 'oracle_test_acc'):
            row[col] = ok[col].mean() if len(ok) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main(config_path):
    with open(config_path, 'r') as file:
        search = yaml.safe_load(file)
    with open(search['base_config'], 'r') as file:
        base = merge(yaml.safe_load(file), search.get('base_overrides') or {})
    sel = search.get('selection', {})

    out_dir = os.path.join('results', search['search_name'])
    preds_dir = os.path.join(out_dir, 'preds')
    os.makedirs(preds_dir, exist_ok=True)
    runs_csv = os.path.join(out_dir, 'runs.csv')
    device = resolve_device()

    grid = search['grid']
    configs = [dict(zip(KEY, v)) for v in itertools.product(*(grid[k] for k in KEY))]
    jobs = [(cfg, seed) for cfg in configs for seed in search['seeds']]
    done = set()
    if os.path.exists(runs_csv):
        prev = pd.read_csv(runs_csv)
        done = {(int(r.latent_dim), int(r.hidden_dim), round(r.beta, 10), round(r.beta_pretrain, 10),
                 int(r.seed)) for r in prev.itertuples()}
    todo = [(c, s) for c, s in jobs if (c['latent_dim'], c['hidden_dim'], round(c['beta'], 10),
                                        round(c['beta_pretrain'], 10), s) not in done]
    print(f"Search '{search['search_name']}': {len(configs)} configurations x "
          f"{len(search['seeds'])} seeds = {len(jobs)} runs, {len(jobs) - len(todo)} already done. "
          f"Device: {device}")

    if todo:
        # The split and the pairing are drawn once, so all runs see the same samples.
        set_seed(search.get('data_seed', base['general']['seed']))
        d = base['data']
        data = load_svhn_mnist(data_path=d['path'], batch_size=d['batch_size'], shuffle=d['shuffle'],
                               train_batches=d.get('train_batches', 450),
                               val_batches=d.get('val_batches', 90),
                               test_batches=d.get('test_batches', 150))
        for i, (cfg, seed) in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {cell_name(cfg)} seed={seed}")
            row = run_one(cfg, seed, base, data, device, preds_dir)
            pd.DataFrame([row]).to_csv(runs_csv, mode='a', index=False,
                                       header=not os.path.exists(runs_csv))

    runs = pd.read_csv(runs_csv)
    runs = runs[[tuple(r) in {tuple(c[k] for k in KEY) for c in configs}
                 for r in runs[KEY].itertuples(index=False)]]
    summary = summarize(runs, preds_dir, sel.get('max_diverged_seeds', 0))
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    print("\nPer-configuration summary (oracle_* columns use labels and are NOT used for selection):")
    print(summary.round(4).to_string(index=False))

    row, shortlist = select(summary, top_k=sel.get('top_k', 5), tol=sel.get('tol'),
                            min_cluster_frac=sel.get('min_cluster_frac'))
    chosen = {k: row[k].item() if hasattr(row[k], 'item') else row[k] for k in KEY}
    result = {'selected': chosen,
              'eta2': float(row['eta2']), 'agreement': float(row['agreement']),
              'shortlist': shortlist[KEY + ['eta2', 'agreement', 'n_ok']].to_dict(orient='records'),
              'selection_rule': sel}
    with open(os.path.join(out_dir, 'selected.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print("\nShortlist (by eta2):")
    print(shortlist[KEY + ['eta2', 'agreement', 'n_ok']].round(4).to_string(index=False))
    print(f"\nSelected configuration: {chosen} "
          f"(eta2 = {row['eta2']:.4f}, agreement = {row['agreement']:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Label-free hyperparameter search for GCVAE on MNIST-SVHN.")
    parser.add_argument('--config', type=str, required=True, help="Path to the search YAML file.")
    args = parser.parse_args()
    main(args.config)
