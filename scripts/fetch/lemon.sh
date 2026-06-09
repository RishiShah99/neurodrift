#!/usr/bin/env bash
# OpenNeuro: LEMON (ds000221) — adult lifespan 20-77, T1 + T2 + dMRI.
# https://openneuro.org/datasets/ds000221
#
# Fully public, no account needed. Use openneuro-py to pull.
#
# Driver provides: $COHORT_DIR.

set -euo pipefail
cd "${COHORT_DIR:?missing COHORT_DIR}"

if ! python3 -c "import openneuro" 2>/dev/null; then
  pip install --quiet openneuro-py
fi

# Pull only anat + dwi — drop the rs-fMRI to save ~60% of dataset size.
python3 - <<'PY'
import os
from pathlib import Path
import openneuro

out = Path(os.environ["COHORT_DIR"]).resolve()
openneuro.download(
    dataset="ds000221",
    target_dir=out,
    include=["sub-*/anat/*", "sub-*/dwi/*", "participants.tsv", "dataset_description.json"],
    exclude=["*func*", "*meg*"],
)
print(f"lemon: downloaded into {out}")
PY

echo "lemon: ready in ${COHORT_DIR}"
