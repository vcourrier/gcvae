# gcvae
PyTorch implementation of the GCVAE model.

This repository contains a PyTorch implementation of the Guided Clustering Variational AutoEncoder (GCVAE). GCVAE is a guided clustering model: the clusters are learned without labels, but they are guided by an auxiliary variable y. The encoder maps the input x to a latent variable z with a Gaussian mixture prior (one component per cluster), and the decoder reconstructs the guide y from z. Only y is reconstructed, so the clusters capture the part of x that is relevant to y.

The model is demonstrated on paired MNIST-SVHN data: the input x is an SVHN image, the guide y is the paired MNIST image of the same digit, and the clusters are formed in the latent space (K = 10).


## Project Structure

```
gcvae/
├── configs/
│   └── svhn_mnist_config.yaml
├── data/
│   └── data_mnist_svhn/
├── src/
│   ├── mnist_data_loader.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── train.py
```

## Setup and Installation

**1. Clone the repository:**
```bash
git clone https://github.com/vcourrier/gcvae.git
cd gcvae
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Prepare the data:**
The MNIST files and the MNIST-SVHN pairing indices (`*-ms-*-idx.pt`) are included in `data/data_mnist_svhn/`. The SVHN `.mat` files are stored with [Git LFS](https://git-lfs.com). Either:
- run `git lfs pull` after cloning to get them from the repository, or
- do nothing: if the files are missing or are LFS pointer files, `torchvision` detects the wrong checksum and downloads SVHN automatically into the data directory on the first run.

## How to Run
The training pipeline is controlled by the `train.py` script, which is configured using a YAML file.

**1. Configure your experiment:**
Modify the parameters in `configs/svhn_mnist_config.yaml`. The default values are the ones used in the thesis.

**2. Run the training script:**
```bash
python train.py --config configs/svhn_mnist_config.yaml
```

## Code Components
- `train.py`: the main script to run an experiment.
- `configs/`: YAML files for configuring experiments.
- `src/mnist_data_loader.py`: data loading and preprocessing of the MNIST-SVHN dataset.
- `src/model.py`: the GCVAE architecture.
- `src/trainer.py`: the pretraining and training loops and the loss functions.
- `src/utils.py`: helper functions for metrics (ARI), GMM assignment and seeding.
- `results/`: the default output directory. Each experiment creates a sub-folder named after `experiment_name` in the config.

## Citation

If you use this code, please cite:

```bibtex
@phdthesis{courrier2026guided,
  author = {Courrier, Violaine},
  title  = {Exploring a General Guided Clustering Principle and its Deep Generative Implementations for Static and Switching Clusters},
  school = {Universit{\'e} de Lille},
  year   = {2026}
}
```

The journal article presenting GCVAE is under review at the *Journal of Classification*. Its reference will be added here once it is published.

## License

MIT, see [LICENSE](LICENSE).
