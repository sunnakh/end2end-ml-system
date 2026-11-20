#!/usr/bin/env bash
# Download the Olist Brazilian E-Commerce dataset into data/raw/olist (gitignored).
# Requires a Kaggle API token in ~/.kaggle (see https://www.kaggle.com/docs/api).
set -euo pipefail

DEST="${1:-data/raw/olist}"
mkdir -p "$DEST"
uvx --with six kaggle datasets download -d olistbr/brazilian-ecommerce -p "$DEST" --unzip
ls -1 "$DEST"
