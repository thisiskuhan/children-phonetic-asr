#!/usr/bin/env bash
# Quick-zip submission/src → submission/<name>.zip
# Usage: bash zip_submission.sh [name]   (default: submission)
# Assumes model.safetensors, config.json, tokenizer/ already in place.
set -euo pipefail

NAME="${1:-submission}"

cd "$(dirname "$0")/../submission"
rm -f "${NAME}.zip"
cd src
zip -r "../${NAME}.zip" . -x "__pycache__/*" "*.pyc" ".gitkeep"
echo ""
du -h "../${NAME}.zip"
