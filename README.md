# gcvae
PyTorch implementation of the GCVAE model.

This repository contains a PyTorch implementation of the Guided Clustering Variational AutoEncoder (GCVAE). GCVAE is a guided clustering model: the clusters are learned without labels, but they are guided by an auxiliary variable y. The encoder maps the input x to a latent variable z with a Gaussian mixture prior (one component per cluster), and the decoder reconstructs the guide y from z. Only y is reconstructed, so the clusters capture the part of x that is relevant to y.

The repository is a small Python package (`gcvae`) and the MNIST-SVHN experiment of the thesis.


## Installation

```bash
git clone https://github.com/vcourrier/gcvae.git
cd gcvae
pip install -e ".[mnist]"
```

The core package needs only `numpy`, `pandas`, `scikit-learn`, `scipy` and `torch`. The optional extras are:
- `mnist`: `torchvision` and `PyYAML`, for the MNIST-SVHN scripts (`train.py`, `evaluate.py`, `search.py`);
- `search`: `optuna`, for Bayesian hyperparameter searches;
- `examples`: `matplotlib`;
- `dev`: `pytest` and `nbstripout`.

`requirements.txt` pins the exact versions used for the MNIST-SVHN experiment (`pip install -r requirements.txt && pip install -e . --no-deps`).

Run the tests with `pytest`.

## Using GCVAE on your own data

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from gcvae import build_mlp_gcvae, pretrain_fn, init_gmm_from_latent, train_fn, linear_annealing, assign

# x: float32 array [n, d]; y: binary guide, float32 array [n, 1]
loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=64, shuffle=True, drop_last=True)
model = build_mlp_gcvae(x_dim=x.shape[1], y_dim=1, w_e=64, w_d=32, J=8)   # raw output = logit
recon = nn.BCEWithLogitsLoss(reduction="none")
pretrain_fn(model, loader, loader, torch.optim.Adam(model.parameters(), lr=1e-3),
            beta=0.1, pretrain_epoch=10, reconstruction_loss_fn=recon)
mu_c, log_sigma_c, pi_c_logits = init_gmm_from_latent(model, x, n_clusters=4, seed=0, latent="mean")
train_fn(model, loader, loader, torch.optim.Adam(model.parameters(), lr=1e-3),
         torch.optim.Adam([mu_c, log_sigma_c, pi_c_logits], lr=1e-3),
         beta=linear_annealing(0.1, 20), epochs=50, mu_c=mu_c, log_sigma_c=log_sigma_c,
         pi_c_logits=pi_c_logits, reconstruction_loss_fn=recon)
out = assign(model, mu_c, log_sigma_c, pi_c_logits, x)   # out["z"], out["q"] = q(c|x), out["c"]
```

Inside the KL term of the training loss, the cluster posterior q(c|x) is replaced by a Gumbel-softmax sample (temperature `gumbel_tau=0.1` of `train_fn`), a sharp but differentiable cluster assignment, as described in the thesis (Chapter 3).

Note on `continuous_output`: despite its name, it only adds a sigmoid on the decoder output. Use `continuous_output=True` with `nn.BCELoss` for guides in [0, 1] such as images (MNIST-SVHN), and `continuous_output=False` (the default of `build_mlp_gcvae`) with `nn.BCEWithLogitsLoss` for a binary guide, or with an L1/MSE loss for an unbounded guide.

## Package contents

| Module | Contents |
|---|---|
| `gcvae.model` | `GCVAE` (optional `logvar_clamp`), `build_conv_gcvae` (MNIST-SVHN), `build_mlp_gcvae` (tabular) |
| `gcvae.trainer` | `pretrain_fn`, `train_fn` (β constant or a function of the epoch), `linear_annealing`, `init_gmm_from_latent`, the loss and `get_q_c_x` |
| `gcvae.inference` | `encode` (latent means), `assign` (z, q(c\|x), cluster) |
| `gcvae.selection` | `eta2`, `agreement`, `select`: label-free hyperparameter selection |
| `gcvae.optimization` | `optimize_eta2`: Bayesian optimisation of η² (needs the `search` extra) |
| `gcvae.utils` | `set_seed`, `resolve_device`, `ResilientPool` (process pool that survives a worker crash), `hungarian_accuracy` (oracle metric) |
| `gcvae.mnist_data_loader` | paired MNIST-SVHN data (needs the `mnist` extra) |

```
gcvae/
├── configs/svhn_mnist_config.yaml
├── data/data_mnist_svhn/
├── gcvae/            # the package
├── tests/
├── train.py          # MNIST-SVHN experiment
├── examples/         # applications of the package
├── evaluate.py       # MNIST-SVHN evaluation (metrics.json)
├── search.py         # label-free hyperparameter search, Bayesian optimisation (MNIST-SVHN)
├── pyproject.toml
└── requirements.txt
```

## MNIST-SVHN experiment

The input x is an SVHN image, the guide y is the paired MNIST image of the same digit, and K = 10.

**Data.** The MNIST files and the MNIST-SVHN pairing indices (`*-ms-*-idx.pt`) are included in `data/data_mnist_svhn/`. The SVHN `.mat` files are stored with [Git LFS](https://git-lfs.com). Either run `git lfs pull` after cloning, or do nothing: if the files are missing or are LFS pointer files, `torchvision` detects the wrong checksum and downloads SVHN into the data directory on the first run.

**Training.** The configuration in `configs/svhn_mnist_config.yaml` holds the values used in the thesis.
```bash
python train.py --config configs/svhn_mnist_config.yaml
```
The network weights (`final_model.pth`) and the mixture parameters (`gmm_params.pth`) are saved in `results/<experiment_name>/`.

**Evaluation.**
```bash
python evaluate.py --config configs/svhn_mnist_config.yaml
```
It rebuilds the same data split, assigns the validation and test images to clusters, and writes `results/<experiment_name>/metrics.json`: the label-free metrics (η² of the guide, cluster sizes) and, since MNIST-SVHN has digit labels, the ARI and the clustering accuracy (Hungarian matching of clusters to digits). The thesis reports 62.1 % test accuracy with this configuration.

## Examples, notebooks and private data

`examples/` holds applications of the package, each with its own small utils module (see `examples/README.md`). Data that must stay private goes in `data/private/`, which git ignores.

Notebook outputs must never be committed, because they can print rows of private data. After cloning, run once:

```bash
pip install nbstripout
nbstripout --install
```

Git then strips the outputs of every `.ipynb` file when it is added (`.gitattributes` already declares the filter, but each clone has to install it).

## Hyperparameter selection without labels

GCVAE is meant for data without labels, so its hyperparameters (latent size, network widths, β) cannot be tuned on a clustering score. The thesis (Chapter 5, "ranking by relevance, then selecting the most reproducible") uses two label-free signals:
- relevance: η², the share of the guide variance explained by the clusters, η² = Σ_c n_c ‖ȳ_c − ȳ‖² / Σ_i ‖y_i − ȳ‖²;
- reproducibility: the agreement, the mean pairwise ARI between the cluster assignments of independently seeded runs.

The procedure:
1. A Bayesian optimisation (Optuna TPE, `gcvae.optimization.optimize_eta2`) proposes configurations, trains each with a few seeds, and maximises the mean η² over the seeds, minus a penalty when a cluster empties.
2. A configuration whose seeds diverge too often (by default 2 of 5) is discarded.
3. Among the trials, keep a shortlist of the most relevant by η² (about 5).
4. Within the shortlist, select the configuration whose seeds agree most (`gcvae.selection.select`).
5. The order matters: never combine the two signals into one score, and never rank by agreement first (a configuration can reproducibly find a partition unrelated to y). If a few labels are available, prefer them for the final choice.

`optimize_eta2` takes your own `train_eval(params, seed)` and `suggest(trial)` functions, so it applies to any data. For MNIST-SVHN, `search.py` runs it:

```bash
pip install -e ".[mnist,search]"
python search.py --config configs/svhn_mnist_search.yaml
```

The search YAML gives the base config, the search space, the seeds, the optimisation budget and the selection rule. Outputs in `results/<search_name>/`:
- `study.log`: the Optuna study; relaunching the command resumes it up to `n_trials`;
- `runs.csv`: one row per configuration and seed (a run already there is never retrained);
- `trials.csv`: one row per trial, with η², its standard deviation over seeds, the agreement, the smallest mixing weight and the number of non-diverged seeds;
- `selected.json`: the selected configuration and the shortlist.

The digit labels are never used for the search or the selection. ARI and accuracy against the labels are only logged, in the `oracle_*` columns, to check the procedure.

## Citation

If you use this code, please cite:

```bibtex
@misc{courrier2026guided,
  author        = {Courrier, Violaine and Biernacki, Christophe},
  title         = {From Unsupervised to Guided Clustering: A Variational Implementation},
  year          = {2026},
  eprint        = {2604.05513},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  doi           = {10.48550/arXiv.2604.05513},
  url           = {https://arxiv.org/abs/2604.05513}
}
```

The model and the label-free hyperparameter selection are also described in the PhD thesis:

```bibtex
@phdthesis{courrier2026thesis,
  author = {Courrier, Violaine},
  title  = {Exploring a General Guided Clustering Principle and its Deep Generative Implementations for Static and Switching Clusters},
  school = {Universit{\'e} de Lille},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
