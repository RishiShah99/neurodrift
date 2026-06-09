#!/usr/bin/env bash
# IXI — fully public, direct HTTP from brain-development.org.
# Five archives: T1, T2, PD, MRA, DTI. Untar in place.
#
# Driver provides: $COHORT_DIR (where to write).

set -euo pipefail

cd "${COHORT_DIR:?missing COHORT_DIR}"

BASE="https://brain-development.org/ixi-dataset/IXI-Dataset"
ARCHIVES=(
  "IXI-T1.tar"
  "IXI-T2.tar"
  "IXI-PD.tar"
  "IXI-MRA.tar"
  "IXI-DTI.tar"
)

for tar in "${ARCHIVES[@]}"; do
  if [[ -f ".done.${tar}" ]]; then
    echo "skip $tar"
    continue
  fi
  echo "fetching $tar"
  curl -fsSLO --retry 5 --retry-delay 10 "${BASE}/${tar}"
  echo "extracting $tar"
  mkdir -p "${tar%.tar}"
  tar -xf "$tar" -C "${tar%.tar}"
  rm "$tar"
  touch ".done.${tar}"
done

echo "ixi: ready in ${COHORT_DIR}"
