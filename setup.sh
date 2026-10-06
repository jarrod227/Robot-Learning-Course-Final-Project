#!/usr/bin/env bash
# Clone the official implementation (pinned) + pretrained models, and install dependencies.
set -euo pipefail
cd "$(dirname "$0")"

OFFICIAL_COMMIT=d5931d88f33c31570fe7675234d3cd818f4b11c7
if [ ! -d third_party/diffusion-for-shared-autonomy ]; then
  git clone https://github.com/ripl/diffusion-for-shared-autonomy.git third_party/diffusion-for-shared-autonomy
fi
git -C third_party/diffusion-for-shared-autonomy checkout -q "$OFFICIAL_COMMIT"

PY=${PYTHON:-python3.10}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
echo "done. activate with: source .venv/bin/activate"
