#!/usr/bin/env bash
# OpenNeuro bulk fetcher — delegates to openneuro_parallel.py which streams
# directly from s3://openneuro.org with boto3 thread pool. Per-file .done.<unit>
# markers in GCS make this resumable through preemption.
#
# Driver provides: COHORT_DIR, GCS_RAW, GCS_BUCKET, FORCE.

set -euo pipefail
cd "${COHORT_DIR:?missing COHORT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GCS_BUCKET GCS_RAW FORCE WORKERS="${WORKERS:-128}"
exec python3 "${SCRIPT_DIR}/openneuro_parallel.py"
