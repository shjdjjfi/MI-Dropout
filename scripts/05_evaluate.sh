#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/default.yaml}
MODEL_PATH=${2:-}
TAG=${3:-manual_eval}
python -m src.main --config "$CFG" evaluate --model-path "$MODEL_PATH" --tag "$TAG"
