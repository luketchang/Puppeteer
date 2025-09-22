#!/usr/bin/env bash
set -e

# Where to install Conda
CONDA_DIR=${CONDA_DIR:-/opt/conda}

echo "Installing Miniconda to $CONDA_DIR..."

# ====== Install dependencies ======
apt update && apt install -y wget bzip2 curl git ffmpeg

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
pip install huggingface_hub

# ====== Install npm and Node.js ======
echo "Installing Node.js (v20.x) and npm..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# ====== Install Claude Code ======
echo "Installing Claude Code (Anthropic) via npm..."
npm install -g @anthropic-ai/claude-code

source ~/.bashrc

# ====== Other dependencies for rendering and EGL ======
apt update && apt install -y \
    xvfb \
    libx11-dev \
    libxrender1 \
    libxext6 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libosmesa6 \
    libegl1-mesa \
    libegl1-mesa-dev

pip install pyglet pyrender PyOpenGL

# ====== Activate conda environment and install additional packages ======
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate puppeteer
pip install huggingface_hub

# ====== Download model checkpoints ======
echo "Downloading model checkpoints..."

# Download skeleton checkpoints
cd skeleton
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('third_partys/Michelangelo/checkpoints/aligned_shape_latents', exist_ok=True)
os.makedirs('skeleton_ckpts', exist_ok=True)

print('Downloading Michelangelo checkpoint...')
file_path = hf_hub_download(
    repo_id='Maikou/Michelangelo',
    filename='checkpoints/aligned_shape_latents/shapevae-256.ckpt',
    local_dir='third_partys/Michelangelo'
)
print(f'Downloaded: {file_path}')

print('Downloading skeleton checkpoints...')
file_path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth',
    local_dir='.'
)
print(f'Downloaded: {file_path}')

file_path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skeleton_ckpts/puppeteer_skeleton_wo_diverse_pose.pth',
    local_dir='.'
)
print(f'Downloaded: {file_path}')

file_path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skeleton_ckpts/puppeteer_skeleton_w_diverse_pose_bone_token.pth',
    local_dir='.'
)
print(f'Downloaded: {file_path}')
"
cd ..

# Download skinning checkpoints
cd skinning
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('third_partys/PartField/ckpt', exist_ok=True)
os.makedirs('skinning_ckpts', exist_ok=True)

print('Downloading PartField checkpoint...')
file_path = hf_hub_download(
    repo_id='mikaelaangel/partfield-ckpt',
    filename='model_objaverse.ckpt',
    local_dir='third_partys/PartField/ckpt'
)
print(f'Downloaded: {file_path}')

print('Downloading skinning checkpoints...')
file_path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth',
    local_dir='.'
)
print(f'Downloaded: {file_path}')

file_path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skinning_ckpts/puppeteer_skin_w_diverse_pose_depth2.pth',
    local_dir='.'
)
print(f'Downloaded: {file_path}')

file_path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skinning_ckpts/puppeteer_skin_wo_diverse_pose_depth1.pth',
    local_dir='.'
)
print(f'Downloaded: {file_path}')
"
cd ..

# ====== Setup third party dependencies ======
echo "Setting up third party dependencies..."

# Copy Michelangelo to skinning (create symlink to save space)
cp -r skeleton/third_partys/Michelangelo skinning/third_partys/ || true
rm -f skinning/third_partys/Michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt
ln -s /workspace/Puppeteer/skeleton/third_partys/Michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt skinning/third_partys/Michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt

# Download co_tracker checkpoint for animation
echo "Downloading co_tracker checkpoint..."
mkdir -p animation/third_partys/co_tracker/ckpt
cd animation/third_partys/co_tracker/ckpt
wget -q https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ../../../..

# Download Video_Depth_Anything checkpoint for animation
echo "Downloading Video_Depth_Anything checkpoint..."
mkdir -p animation/third_partys/Video_Depth_Anything/ckpt
cd animation/third_partys/Video_Depth_Anything/ckpt
wget -q https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
cd ../../../..

# Make demo scripts executable
chmod +x demo_rigging.sh
chmod +x demo_animation.sh

# ====== Setup SSH key for Git access ======
echo "Setting up SSH key for Git access..."
SSH_KEY_DIR="/workspace/ssh_keys"

# Check if SSH keys exist in the project
if [ -f "$SSH_KEY_DIR/id_rsa" ] && [ -f "$SSH_KEY_DIR/id_rsa.pub" ]; then
    echo "Found SSH keys in $SSH_KEY_DIR/"

    # Create .ssh directory if it doesn't exist
    mkdir -p ~/.ssh

    # Copy SSH keys to .ssh directory
    cp "$SSH_KEY_DIR/id_rsa" ~/.ssh/
    cp "$SSH_KEY_DIR/id_rsa.pub" ~/.ssh/

    # Set correct permissions
    chmod 600 ~/.ssh/id_rsa
    chmod 644 ~/.ssh/id_rsa.pub

    # Add to ssh-agent if running
    if [ -n "$SSH_AUTH_SOCK" ]; then
        ssh-add ~/.ssh/id_rsa 2>/dev/null || true
    fi

    echo "SSH key installed to ~/.ssh/"
    echo "Public key content (should be added to GitHub/GitLab):"
    cat "$SSH_KEY_DIR/id_rsa.pub"
    echo ""
else
    echo "Warning: SSH keys not found in $SSH_KEY_DIR/"
    echo "Expected files: $SSH_KEY_DIR/id_rsa and $SSH_KEY_DIR/id_rsa.pub"
fi

echo "Setup completed! You can now run:"
echo "PYOPENGL_PLATFORM=egl ./demo_rigging.sh"
echo "PYOPENGL_PLATFORM=egl ./demo_animation.sh"

echo "Installation complete. Restart your shell or run 'source ~/.bashrc' before using conda."



