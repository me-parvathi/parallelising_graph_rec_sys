#!/bin/bash

set -e
ENV_DIR="venv"
DATA_DIR="data/movielens"
DATASET_URL="http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_FILE="ml-100k.zip"

if [ ! -d "$ENV_DIR" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
echo "[INFO] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[INFO] Installing PyTorch (built for CUDA 11.8, compatible with 11.4 runtime)..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118



echo "[INFO] Installing PyG dependencies for torch 1.10.2+cu114..."
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html



echo "[INFO] Installing torch-geometric..."
pip install torch-geometric==2.5.0

echo "[INFO] Installing remaining utilities from requirements.txt..."
pip install -r requirements.txt

echo "[INFO] Creating project directories..."
mkdir -p pinsage ppnp benchmarking notebooks "$DATA_DIR"

if [ ! -f "$DATA_DIR/u.data" ]; then
  echo "[INFO] Downloading MovieLens 100k dataset..."
  wget -q "$DATASET_URL" -O "$ZIP_FILE"
  unzip -q "$ZIP_FILE" -d "$DATA_DIR"
  rm "$ZIP_FILE"
else
  echo "[INFO] Dataset already present."
fi

echo "[INFO] Writing .gitignore..."
cat <<EOL > .gitignore
__pycache__/
*.py[cod]
*.so
.env/
.venv/
venv/
.DS_Store
Thumbs.db
*.log
*.out
*.err
.cache/
*.pkl
*.pt
EOL

echo "[INFO] Writing README.md..."
cat <<EOL > README.md
# Graph-Based Recommendation System

This project explores scalable and parallelized implementations of graph-based recommendation models: **PinSage** and **PPNP**.

## How to Run
(TBD)
EOL

echo "[SUCCESS] Setup complete."
