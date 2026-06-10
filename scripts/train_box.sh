#!/usr/bin/env bash
# Box-side launcher for VAE training / smoke test.
#
# Syncs the GPU + imaging extras into the venv (torch, lightning, gcsfs, monai,
# antspyx) then runs scripts/train.py with whatever Hydra overrides are passed.
#
#   fleet sync
#   fleet train "bash scripts/train_box.sh experiment=vae_v0 trainer.fast_dev_run=true"
#   fleet logs -f
#
# Pass Hydra overrides as args; they are forwarded verbatim to train.py.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# Fresh GPU images ship python3 + gcloud only — bootstrap uv if missing.
if ! command -v uv >/dev/null 2>&1; then
  echo "[train_box] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$HOME/neurodrift"

echo "[train_box] uv sync --extra gpu --extra imaging"
uv sync --extra gpu --extra imaging

echo "[train_box] train.py $*"
uv run python scripts/train.py "$@"
