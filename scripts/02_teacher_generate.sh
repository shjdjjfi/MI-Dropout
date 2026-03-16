#!/usr/bin/env bash
set -euo pipefail
python -m src.main --config ${1:-configs/default.yaml} teacher-generate
