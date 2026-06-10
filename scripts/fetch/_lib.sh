#!/usr/bin/env bash
# Shared helpers for per-cohort fetchers: GCS sentinel + atomic upload+clean.
# Sourced by ixi.sh / abide.sh / openneuro.sh — never executed directly.
#
# Expects GCS_RAW (`gs://bucket/raw/<cohort>`) and FORCE (0/1) in env.

set -euo pipefail

: "${GCS_RAW:?missing GCS_RAW}"
: "${FORCE:=0}"

gcs_unit_done() {
  # Has this unit already been pushed?
  local unit="$1"
  if (( FORCE == 1 )); then return 1; fi
  gcloud storage ls "${GCS_RAW}/.done.${unit}" >/dev/null 2>&1
}

gcs_mark_done() {
  local unit="$1"
  local tmp
  tmp="$(mktemp)"
  gcloud storage cp "$tmp" "${GCS_RAW}/.done.${unit}" >/dev/null
  rm -f "$tmp"
}

# upload_unit <local_path> <gcs_subpath> <unit_name>
#   Pushes <local_path> (file or dir) to ${GCS_RAW}/<gcs_subpath>/, marks the
#   unit done in GCS, then deletes the local copy. Idempotent: if the unit is
#   already marked done, no-op.
upload_unit() {
  local src="$1" subpath="$2" unit="$3"
  if gcs_unit_done "$unit"; then
    echo "  [${unit}] already in GCS — skip upload"
    rm -rf "$src"
    return 0
  fi
  if [[ ! -e "$src" ]]; then
    echo "  [${unit}] nothing local to upload at $src"
    return 0
  fi
  local dst="${GCS_RAW}/${subpath}"
  echo "  [${unit}] uploading $src -> $dst"
  if [[ -d "$src" ]]; then
    gcloud storage cp --recursive --gzip-in-flight-all "${src}/" "${dst}/" --quiet
  else
    gcloud storage cp --gzip-in-flight-all "$src" "${dst}" --quiet
  fi
  gcs_mark_done "$unit"
  echo "  [${unit}] cleaning local"
  rm -rf "$src"
}
