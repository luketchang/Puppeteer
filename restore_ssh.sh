#!/usr/bin/env bash
set -e

echo "====== Restoring SSH Keys from Workspace ======"

# Check if keys exist in workspace
if [ ! -f /workspace/ssh_keys/id_ed25519 ]; then
    echo "ERROR: No SSH keys found in /workspace/ssh_keys/"
    echo "Please run setup_ssh.sh first to generate keys."
    exit 1
fi

# Copy keys to .ssh
echo "Copying keys from /workspace/ssh_keys to ~/.ssh/..."
mkdir -p ~/.ssh
cp /workspace/ssh_keys/id_ed25519 ~/.ssh/
cp /workspace/ssh_keys/id_ed25519.pub ~/.ssh/

# Set correct permissions
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Start ssh-agent and add key
echo "Adding key to ssh-agent..."
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Set git config
echo "Setting git config..."
git config --global user.email "luketchang00@gmail.com"
git config --global user.name "Luke Tchang"

echo ""
echo "✓ SSH keys restored successfully!"
echo "✓ Test with: ssh -T git@github.com"

