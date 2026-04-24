#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Serving repository at http://0.0.0.0:8000"
echo "Open: http://<host>:8000/web/download.html"
python -m http.server 8000
