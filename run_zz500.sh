#!/bin/bash
# Exit immediately if any command fails
set -e

echo "=== [1/4] Setting up environment paths ==="
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/4] Syncing environment via uv ==="
uv sync

echo "=== [3/4] Building the preprocessed zz500 dataset ==="
uv run python gen_data/build_dataset.py --market zz500

echo "=== [4/4] Running the zz500 HGAT model training ==="
uv run python main.py --market zz500 --policy HGAT
