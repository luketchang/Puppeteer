#!/usr/bin/env bash
set -e

echo "====== SSH Key Setup for GitHub ======"

# Check if SSH key already exists
if [ -f ~/.ssh/id_ed25519 ]; then
    echo "SSH key already exists at ~/.ssh/id_ed25519"
    read -p "Do you want to overwrite it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing key. Displaying public key:"
        cat ~/.ssh/id_ed25519.pub
        exit 0
    fi
fi

# Generate new SSH key
echo "Generating new SSH key..."
ssh-keygen -t ed25519 -C "luketchang00@gmail.com" -f ~/.ssh/id_ed25519 -N ""

# Set correct permissions
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# Start ssh-agent and add key
echo "Adding key to ssh-agent..."
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Display public key
echo ""
echo "====== Your SSH Public Key ======"
echo "Copy this key and add it to GitHub:"
echo "https://github.com/settings/keys"
echo ""
cat ~/.ssh/id_ed25519.pub
echo ""
echo "======================================"

# Set git config
echo "Setting git config..."
git config --global user.email "luketchang00@gmail.com"
git config --global user.name "Luke Tchang"

echo ""
echo "After adding the key to GitHub, test with:"
echo "ssh -T git@github.com"

