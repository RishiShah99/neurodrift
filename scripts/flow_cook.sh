#!/usr/bin/env bash
# Flow cook + finalize for the Phase-2 lifespan flow on the latent dataset.
#
# train flow_v0 (resumes from ckpt_dir/last.ckpt via the config default
# ckpt_path=last) -> on CLEAN completion archive last.ckpt to GCS -> power off the
# box to end spend. On train FAILURE the box stays up for inspection.
#
# Spot-preemption-safe: a preemption STOPs the box (kills this process); an external
# watchdog restarts the box and re-invokes this script, which resumes from last.ckpt.
# Finalize is idempotent (re-archives the same ckpt). Distinguish "done" from
# "preempted" by the presence of gs://.../flow_v0.ckpt.
#
#   NEURODRIFT_LATENT_ROOT=$HOME/latents_local bash scripts/flow_cook.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/neurodrift"

LATENT_ROOT="${NEURODRIFT_LATENT_ROOT:-$HOME/latents_local}"
BUCKET="${GCS_BUCKET:-neurodrift-data}"
EXP="${EXP:-flow_v0}"
stamp() { date -u +%FT%TZ; }

echo "[flow-cook] $(stamp) START exp=$EXP latents=$LATENT_ROOT"
NEURODRIFT_LATENT_ROOT="$LATENT_ROOT" bash scripts/train_box.sh experiment="$EXP"
status=$?
echo "[flow-cook] $(stamp) train exit $status"
if [ "$status" -ne 0 ]; then
  echo "[flow-cook] $(stamp) train failed — box stays up for inspection"
  exit "$status"
fi

CKPT="$HOME/neurodrift/out/$EXP/ckpt/last.ckpt"
if [ -f "$CKPT" ]; then
  echo "[flow-cook] $(stamp) ARCHIVE -> gs://$BUCKET/checkpoints/_archive/${EXP}.ckpt"
  gcloud storage cp "$CKPT" "gs://$BUCKET/checkpoints/_archive/${EXP}.ckpt"
fi
echo "[flow-cook] $(stamp) DONE — powering off in 2 min (cancel: sudo shutdown -c)"
sudo shutdown -h +2 "neurodrift flow cook finalized" \
  || echo "[flow-cook] shutdown failed — STOP THE BOX MANUALLY"
