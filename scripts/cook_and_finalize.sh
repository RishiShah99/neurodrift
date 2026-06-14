#!/usr/bin/env bash
# Cook a VAE experiment, then on CLEAN completion eval + archive + power off the box.
#
# Why this exists: the previous training monitor was a session-only cron that died
# with the Claude session, so a finished cook would idle-burn (8xB200) until noticed.
# This folds finalize into the detached training command itself:
#   train -> (on success) canonical eval.py -> archive ckpt+metrics to GCS -> guest
#   shutdown (which transitions the instance to STOPPED and ends VM spend).
# On training FAILURE the && chain stops and the box stays UP for inspection.
# Spot-preemption-safe: the box STOPs on preemption, killing this chain; on a manual
# `fleet start` + relaunch the cook resumes from last.ckpt (ckpt_path=last).
#
#   ZARR_LOCAL=/home/USER/zarr_local_coreg EXP=vae_v0_disentangled_noadv \
#   ARCHIVE_TAG=disentangled_noadv_coreg bash scripts/cook_and_finalize.sh
#
# Env:
#   ZARR_LOCAL    (required) local zarr corpus root for train + eval
#   EXP           (required) Hydra experiment name
#   ARCHIVE_TAG   (default: $EXP)  archive basename -> vae_v0_<tag>.ckpt
#   EVAL_BATCHES  (default: 80)    val batches for the canonical eval
#   NO_POWEROFF   (default: 0)     set 1 to skip the shutdown (eval+archive only)
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/neurodrift"

: "${ZARR_LOCAL:?set ZARR_LOCAL to the local zarr corpus root}"
: "${EXP:?set EXP to the Hydra experiment name}"
ARCHIVE_TAG="${ARCHIVE_TAG:-$EXP}"
EVAL_BATCHES="${EVAL_BATCHES:-80}"
BUCKET="${GCS_BUCKET:-neurodrift-data}"
stamp() { date -u +%FT%TZ; }

echo "[finalize] $(stamp) COOK START exp=$EXP root=$ZARR_LOCAL"
NEURODRIFT_ZARR_ROOT="$ZARR_LOCAL" PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/train_box.sh experiment="$EXP"
status=$?
if [ "$status" -ne 0 ]; then
  echo "[finalize] $(stamp) train exited $status — NOT finalizing; box stays up for inspection"
  exit "$status"
fi

CKPT="$HOME/neurodrift/out/$EXP/ckpt/last.ckpt"
if [ ! -f "$CKPT" ]; then
  echo "[finalize] $(stamp) no ckpt at $CKPT — NOT finalizing"
  exit 1
fi

echo "[finalize] $(stamp) EVAL $CKPT on $ZARR_LOCAL (max_batches=$EVAL_BATCHES)"
NEURODRIFT_ZARR_ROOT="$ZARR_LOCAL" CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/eval.py experiment="$EXP" eval.ckpt="$CKPT" \
  eval.max_batches="$EVAL_BATCHES" \
  || echo "[finalize] $(stamp) eval FAILED (continuing to archive the ckpt)"

EVAL_JSON="$(find out/"$EXP" -name eval_metrics.json -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2-)"
echo "[finalize] $(stamp) ARCHIVE -> gs://$BUCKET/checkpoints/_archive/"
gcloud storage cp "$CKPT" "gs://$BUCKET/checkpoints/_archive/vae_v0_${ARCHIVE_TAG}.ckpt"
if [ -n "$EVAL_JSON" ]; then
  gcloud storage cp "$EVAL_JSON" \
    "gs://$BUCKET/checkpoints/_archive/vae_v0_${ARCHIVE_TAG}_eval.json"
  echo "[finalize] eval metrics:"; cat "$EVAL_JSON"
fi

echo "[finalize] $(stamp) DONE."
if [ "${NO_POWEROFF:-0}" = "1" ]; then
  echo "[finalize] NO_POWEROFF=1 — leaving box up"; exit 0
fi
echo "[finalize] powering off in 2 min to end spend (cancel with: sudo shutdown -c)"
sudo shutdown -h +2 "neurodrift cook finalized — ending spend" || \
  echo "[finalize] shutdown failed (no sudo?) — STOP THE BOX MANUALLY: fleet down"
