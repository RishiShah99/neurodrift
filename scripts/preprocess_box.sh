#!/usr/bin/env bash
# Box-side bootstrap for raw NIfTI -> Zarr preprocessing.
#
# Fresh GPU images ship python3 + gcloud only. This installs uv, syncs the
# project venv with the imaging extra (antspyx / antspynet), warms the
# antspynet brain-extraction weights once (so parallel workers don't race the
# same download), then runs the CPU-bound preprocess driver.
#
#   fleet sync
#   fleet train "bash scripts/preprocess_box.sh"
#   fleet logs -f
#
# Idempotent: uv sync is a no-op once satisfied, and preprocess.py skips any
# subject whose <stem>.zarr already exists in GCS.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

# Preprocess is CPU-bound and must leave the GPU free for training. Forcing
# CPU also sidesteps antspynet/TensorFlow's broken GPU cuBLAS init on this
# image ("Cannot load symbol cublasLtCreate").
export CUDA_VISIBLE_DEVICES=-1

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[preprocess_box] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$HOME/neurodrift"

# 2. project venv + imaging extra (antspyx, antspynet, nibabel, zarr, monai, torch)
echo "[preprocess_box] uv sync --extra imaging"
uv sync --extra imaging

# 3. warm antspynet brain-extraction weights (single process; avoids a 24-way
#    download race into ~/.keras when the pool starts)
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
echo "[preprocess_box] warming antspynet weights"
uv run python - <<'PY' || true
import ants
import antspynet

img = ants.image_read(ants.get_ants_data("mni"))
antspynet.utilities.brain_extraction(img, modality="t1")
print("antspynet brain-extraction weights warmed")
PY

# 4. raw -> Zarr. ITK threads pinned to 1 per worker so 24 procs don't
#    oversubscribe the 208 cores via ANTs' internal threading.
echo "[preprocess_box] preprocess.py --cohorts abide,openneuro --workers 24"
SCRATCH="$HOME/scratch" uv run python scripts/preprocess.py \
  --cohorts abide,openneuro --workers 24
