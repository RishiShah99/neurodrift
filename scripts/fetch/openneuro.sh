#!/usr/bin/env bash
# Bulk OpenNeuro lifespan / healthy T1 puller.
# Reads dataset IDs from scripts/fetch/openneuro_datasets.txt.
# Each dataset lands in $COHORT_DIR/<dataset_id>/ with its own .done sentinel.
#
# Driver provides: $COHORT_DIR.

set -euo pipefail
cd "${COHORT_DIR:?missing COHORT_DIR}"

if ! python3 -c "import openneuro" 2>/dev/null; then
  pip install --quiet openneuro-py
fi

LIST_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/openneuro_datasets.txt"
if [[ ! -f "$LIST_FILE" ]]; then
  echo "ERROR: missing $LIST_FILE"
  exit 1
fi

# Strip comments / blanks, take first whitespace-separated token per line.
mapfile -t DATASETS < <(
  sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "$LIST_FILE" \
    | awk 'NF{print $1}'
)

if (( ${#DATASETS[@]} == 0 )); then
  echo "no datasets listed in $LIST_FILE"
  exit 0
fi

echo "openneuro: pulling ${#DATASETS[@]} dataset(s): ${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
  if [[ -f "${ds}/.done" ]]; then
    echo "  skip ${ds} (already done)"
    continue
  fi
  echo "  fetching ${ds}"
  mkdir -p "${ds}"
  DATASET_ID="$ds" python3 - <<'PY'
import os
from pathlib import Path
import openneuro

ds  = os.environ["DATASET_ID"]
out = Path(ds).resolve()
openneuro.download(
    dataset=ds,
    target_dir=out,
    include=["sub-*/anat/*", "sub-*/dwi/*", "participants.tsv", "dataset_description.json"],
    exclude=["*func*", "*meg*", "*beh*", "*eeg*"],
)
PY
  touch "${ds}/.done"
done

echo "openneuro: ready in ${COHORT_DIR}"
