#!/usr/bin/env bash
# Driver: dispatch per-cohort fetchers that incrementally push to GCS.
#
# Each cohort fetcher uploads its own data in small units (per archive, per
# site, per dataset) and writes `.done.<unit>` sentinels into the cohort's
# GCS prefix. That way a spot-preempted fetch resumes from the last
# completed unit instead of restarting the whole cohort.
#
# Run on the fleet box:
#   fleet up h100
#   fleet sync
#   fleet train "SCRATCH=\$HOME/scratch bash scripts/fetch_data.sh"
#   fleet logs -f

set -euo pipefail

# ---------- defaults ----------
GCS_BUCKET="${GCS_BUCKET:-neurodrift-data}"
SCRATCH="${SCRATCH:-$HOME/scratch}"
COHORTS=""
FORCE=0

usage() {
  cat <<EOF
Usage: $0 [--cohorts list] [--scratch dir] [--bucket name] [--force]

  --cohorts   Comma-sep subset of: ixi,abide,openneuro,openbhb,oasis3,hcp_d,hcp_a
              (default: ixi,abide,openneuro — v0 walk-up corpus)
  --scratch   Local scratch dir on the fleet box (default: \$HOME/scratch)
  --bucket    GCS bucket (default: \$GCS_BUCKET or neurodrift-data)
  --force     Re-upload units even if their GCS .done sentinel exists

Each fetcher checkpoints per unit into gs://\$GCS_BUCKET/raw/<cohort>/
so a spot preemption only loses the in-flight unit.
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

# IXI off the default path: brain-development.org mirror is 403, HF mirror is
# T1-only. LEMON (ds000221, pulled via openneuro) now supplies multimodal.
COHORTS="${COHORTS:-abide,openneuro}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$SCRATCH"

log() { printf '[fetch_data %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

bootstrap_tooling() {
  # Bare GCP image: python3 + gcloud only. Bring in pip + aws + openneuro-py.
  if ! command -v pip >/dev/null && ! python3 -m pip --version >/dev/null 2>&1; then
    log "installing python3-pip"
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip
  fi
  if ! python3 -c "import boto3, google.cloud.storage" >/dev/null 2>&1; then
    log "installing boto3 + google-cloud-storage"
    python3 -m pip install --quiet --user --break-system-packages \
      boto3 google-cloud-storage
  fi
  if ! python3 -c "import openneuro" >/dev/null 2>&1; then
    log "installing openneuro-py"
    python3 -m pip install --quiet --user --break-system-packages openneuro-py
  fi
  export PATH="$HOME/.local/bin:$PATH"
}

bootstrap_tooling

run_cohort() {
  local cohort="$1"
  local fetcher="${SCRIPT_DIR}/fetch/${cohort}.sh"
  if [[ ! -f "$fetcher" ]]; then
    log "ERROR: no fetcher at $fetcher"
    return 1
  fi

  log "==== $cohort: per-unit fetch + GCS checkpoint ===="
  local local_dir="${SCRATCH}/${cohort}"
  mkdir -p "$local_dir"

  local rc=0
  GCS_BUCKET="$GCS_BUCKET" \
  COHORT="$cohort" \
  COHORT_DIR="$local_dir" \
  GCS_RAW="gs://${GCS_BUCKET}/raw/${cohort}" \
  FORCE="$FORCE" \
  bash "$fetcher" || rc=$?

  log "clearing scratch ${local_dir}"
  rm -rf "$local_dir"
  if (( rc != 0 )); then
    log "WARN: $cohort exited rc=$rc — moving on"
  else
    log "done $cohort"
  fi
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
