#!/usr/bin/env bash
set -e

# Where to install Conda
CONDA_DIR=${CONDA_DIR:-/opt/conda}

echo "Installing Miniconda to $CONDA_DIR..."

# ====== Install dependencies ======
apt update && apt install -y wget bzip2 curl git

# ====== Download latest Miniconda ======
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# Create target dir if needed
mkdir -p "$CONDA_DIR"

# Run installer silently
bash /tmp/miniconda.sh -b -u -p "$CONDA_DIR"

# Clean up installer
rm -f /tmp/miniconda.sh

# Add conda to PATH (persist for future sessions)
echo "export PATH=\"$CONDA_DIR/bin:\$PATH\"" >> ~/.bashrc
export PATH="$CONDA_DIR/bin:$PATH"

# Initialize conda for bash
"$CONDA_DIR/bin/conda" init bash

# ====== Accept Anaconda Terms of Service to avoid non-interactive errors ======
"$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$CONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "Miniconda installed successfully. Restart shell or run:"
echo "source ~/.bashrc"

# ====== Create conda environment & install dependencies ======
conda create -n puppeteer python==3.10.13 -y
conda activate puppeteer

pip install cython==0.29.36

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html

# ====== Install npm and Node.js ======
echo "Installing Node.js (v20.x) and npm..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# ====== Install Claude Code ======
echo "Installing Claude Code (Anthropic) via npm..."
npm install -g @anthropic-ai/claude-code

echo "Installation complete. Restart your shell or run 'source ~/.bashrc' before using conda."

source ~/.bashrc

# ====== Other ======
apt update && apt install -y \
    xvfb \
    libx11-dev \
    libxrender1 \
    libxext6 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libosmesa6

pip install pyglet pyrender PyOpenGL




