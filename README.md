# gcvae
Pytorch implementation of the GCVAE model.
This repository contains a PyTorch implementation of the Guided Clustering Variational AutoEncoder (GCVAE). The model is designed for unsupervised deep clustering and is demonstrated on an image-to-image task, learning to map SVHN images to MNIST images while clustering them in the latent space.


## Project Structure

The project is organized into a modular structure to separate concerns like data loading, model architecture, and training logic.
```
gcvae/
├── configs/
│   └── svhn_mnist_config.yaml
├── data/
├── results/
├── src/
│   ├── init.py
│   ├── mnist_data_loader.py
│   ├── models.py
│   ├── trainer.py
│   └── utils.py
├── .gitignore
├── requirements.txt
└── train.py
```

## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the repository:**
```bash
git clone https://github.com/vcourrier/gcvae.git
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Prepare the data:**
The script will automatically download the MNIST and SVHN datasets into the directory specified in your configuration file. Ensure the path is correct.

## How to Run
The entire training and evaluation pipeline is controlled by the train.py script, which is configured using a YAML file.

**1. Configure your experiment:**
Modify the parameters in configs/svhn_mnist_config.yaml to suit your needs.

**2. Run the training script:**
Execute the following command from the root directory of the project:
```bash
python train.py --config configs/svhn_mnist_config.yaml
```

## Code Components
- train.py: The main executable script to run an experiment.

- configs/: Contains YAML files for configuring experiments.

- src/mnist_data_loader.py: Handles data loading and preprocessing of the mnist-SVHN dataset.

- src/models.py: Defines the GCVAE architecture.

- src/trainer.py: Contains the pretrain and train loops and the loss functions.

- src/utils.py: Includes helper functions for metrics (ARI), GMM assignment, and seeding.

- results/: The default directory where trained models, plots, and other artifacts are saved. Each experiment will create a sub-folder named after the experiment_name in the config.

