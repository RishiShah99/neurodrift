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

# --- Single-node multi-GPU NCCL on GCP A3 -----------------------------------
# The image's /etc/profile.d/nccl_env.sh sources the GPUDirect "gib"/TCPXO stack
# and prepends /usr/local/gib/lib64 to LD_LIBRARY_PATH. That dir ships an external
# NCCL net plugin (libnccl-net.so) built against NCCL 2.27.5, but torch's venv
# uses NCCL 2.29.7 -> ABI mismatch -> "Failed to initialize any NET plugin" at
# init_process_group. NCCL_NET_PLUGIN=none doesn't stop discovery via the loader
# path, and a plain a3-highgpu-8g has one NIC + no RxDM daemon so TCPXO can't work
# anyway. Strip gib from the loader path so the plugin is unfindable, drop the env
# the gib script injected, and force NCCL's built-in socket net. The 8 H100s are a
# full NVLink (NV18) mesh, so intra-node collectives run over NVLink/SHM.
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/usr/local/gib' | paste -sd: -)"
unset LD_PRELOAD NCCL_NET_PLUGIN NCCL_NET NCCL_TUNER_PLUGIN NCCL_PROFILER_PLUGIN NCCL_TUNER_CONFIG_PATH NCCL_TOPO_FILE
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -1)"
export NCCL_DEBUG=WARN

echo "[train_box] uv sync --extra gpu --extra imaging"
uv sync --extra gpu --extra imaging

echo "[train_box] train.py $*"
uv run python scripts/train.py "$@"
