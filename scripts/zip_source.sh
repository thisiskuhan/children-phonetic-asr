#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
zip -r src.zip \
    src/ \
    submission/src/ \
    submission/scripts/ \
    scripts/ \
    requirements.txt \
    .env.example
