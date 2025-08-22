#!/bin/bash

set -e  # Exit on any error

ENV_NAME="unimol_env"
PYTHON_VERSION="3.12"

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch 2.6 with CUDA 12.6 support"
pip install torch

echo "Installing dependencies with exact versions"
pip install \
    numpy==2.2.6 \
    numba==0.61.2 \
    scipy==1.16.1 \
    scikit-learn==1.7.1
    pandas==2.3.1 \
    lmdb==1.7.3 \
    tqdm==4.67.1 \
    tokenizers==0.21.4 \
    wandb==0.21.0 \
    tensorboardX==2.6.4 \
    ml_collections==1.1.0 \
    rdkit==2025.3.5 \

echo "Verifying PyTorch CUDA installation"
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()} - GPUs: {torch.cuda.device_count()}')"

echo "Installing Unicore from source"
cd Uni-Core
pip install .

echo "Installation complete!"