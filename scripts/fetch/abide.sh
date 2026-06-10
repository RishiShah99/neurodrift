#!/usr/bin/env bash
# ABIDE I + II T1w — public AWS Open Data. Delegates to abide_parallel.py
# which runs a threaded boto3 → google-cloud-storage pipeline. Resumable via
# per-object .done.<unit> markers in GCS.
#
# Driver provides: COHORT_DIR, GCS_RAW, GCS_BUCKET, FORCE.

set -euo pipefail
cd "${COHORT_DIR:?missing COHORT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GCS_BUCKET GCS_RAW FORCE WORKERS="${WORKERS:-128}"
exec python3 "${SCRIPT_DIR}/abide_parallel.py"
