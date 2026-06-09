#!/usr/bin/env bash
# Driver: download brain-MRI cohorts from each source's CDN straight to GCS.
#
# Run on the fleet box, not locally:
#   fleet up h100
#   fleet sync
#   fleet train "bash scripts/fetch_data.sh --cohorts ixi,openbhb,oasis3,hcp_d,hcp_a"
#   fleet logs -f
#
# Per cohort: download to a scratch dir on the box, push to GCS, clear the
# scratch dir, move on. Idempotent — re-runs skip cohorts whose GCS prefix
# already has data.

set -euo pipefail

# ---------- defaults ----------
GCS_BUCKET="${GCS_BUCKET:-neurodrift-data}"
SCRATCH="${SCRATCH:-/mnt/scratch/raw}"
COHORTS=""
FORCE=0

usage() {
  cat <<EOF
Usage: $0 [--cohorts list] [--scratch dir] [--bucket name] [--force]

  --cohorts   Comma-sep subset of: ixi,abide,openneuro,openbhb,oasis3,hcp_d,hcp_a
              (default: ixi,abide,openneuro — v0 walk-up corpus)
  --scratch   Local scratch dir on the fleet box (default: /mnt/scratch/raw)
  --bucket    GCS bucket (default: \$GCS_BUCKET or neurodrift-data)
  --force     Re-download cohorts even if their GCS prefix is non-empty

Env vars needed:
  GCS_BUCKET             target bucket (or pass --bucket)
  IEEE_DATAPORT_TOKEN    OpenBHB (deferred — DataPort signups currently flaky)
  XNAT_USER, XNAT_PASS   OASIS-3 (v1 only)
  HCP_AWS_KEY, HCP_AWS_SECRET   HCP-Aging + HCP-Development (v1 only)

v0 default cohorts (ixi, abide, openneuro) require no credentials.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohorts) COHORTS="$2"; shift 2 ;;
    --scratch) SCRATCH="$2"; shift 2 ;;
    --bucket)  GCS_BUCKET="$2"; shift 2 ;;
    --force)   FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         echo "unknown arg: $1"; usage; exit 1 ;;
  esac
done

COHORTS="${COHORTS:-ixi,abide,openneuro}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH"

# ---------- helpers ----------
log() { printf '[fetch_data %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

gcs_has_data() {
  local cohort="$1"
  local count
  count=$(gcloud storage ls "gs://${GCS_BUCKET}/raw/${cohort}/**" 2>/dev/null | head -1 | wc -l)
  [[ "$count" -gt 0 ]]
}

run_cohort() {
  local cohort="$1"
  local fetcher="${SCRIPT_DIR}/fetch/${cohort}.sh"
  if [[ ! -x "$fetcher" ]]; then
    log "ERROR: no fetcher at $fetcher"
    return 1
  fi
  if (( FORCE == 0 )) && gcs_has_data "$cohort"; then
    log "skip $cohort — gs://${GCS_BUCKET}/raw/${cohort}/ already populated (use --force to redo)"
    return 0
  fi

  log "==== $cohort: download → GCS ===="
  local local_dir="${SCRATCH}/${cohort}"
  mkdir -p "$local_dir"

  GCS_BUCKET="$GCS_BUCKET" \
  COHORT_DIR="$local_dir" \
  bash "$fetcher"

  log "uploading $cohort to gs://${GCS_BUCKET}/raw/${cohort}/"
  gcloud storage cp --recursive --gzip-in-flight-all \
    "${local_dir}/" \
    "gs://${GCS_BUCKET}/raw/${cohort}/"

  log "clearing scratch ${local_dir}"
  rm -rf "$local_dir"
  log "done $cohort"
}

# ---------- main ----------
log "bucket=gs://${GCS_BUCKET} scratch=${SCRATCH} cohorts=${COHORTS}"

IFS=',' read -ra LIST <<< "$COHORTS"
for c in "${LIST[@]}"; do
  c="${c// /}"
  run_cohort "$c"
done

log "all done. bucket layout:"
gcloud storage ls "gs://${GCS_BUCKET}/raw/" || true
