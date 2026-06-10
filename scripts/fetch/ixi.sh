#!/usr/bin/env bash
# IXI — public, hosted on the Imperial College brain-development mirror.
# Five archives: T1, T2, PD, MRA, DTI. Each is downloaded, extracted, and
# pushed to GCS as its own unit before the next one starts — preemption only
# loses the in-flight archive.
#
# Driver provides: COHORT_DIR, GCS_RAW, FORCE.

set -euo pipefail

cd "${COHORT_DIR:?missing COHORT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

BASE="https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
ARCHIVES=(
  "IXI-T1.tar"
  "IXI-T2.tar"
  "IXI-PD.tar"
  "IXI-MRA.tar"
  "IXI-DTI.tar"
)

for tar in "${ARCHIVES[@]}"; do
  unit="${tar%.tar}"
  if gcs_unit_done "$unit"; then
    echo "skip $tar (already in GCS)"
    continue
  fi

  echo "fetching $tar"
  curl -fSL --retry 5 --retry-delay 10 -o "$tar" "${BASE}/${tar}"

  echo "extracting $tar"
  rm -rf "$unit"
  mkdir -p "$unit"
  tar -xf "$tar" -C "$unit"
  rm "$tar"

  upload_unit "$unit" "$unit" "$unit"
done

echo "ixi: all archives in ${GCS_RAW}"
