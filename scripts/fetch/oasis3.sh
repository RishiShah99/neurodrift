#!/usr/bin/env bash
# OASIS-3 — XNAT (central.xnat.org) project OASIS3.
# Uses NRG's xnat_python (https://github.com/NrgXnat/xnatpy). Pulls in BIDS layout.
#
# Required env: XNAT_USER, XNAT_PASS (your central.xnat.org credentials).
# Driver provides: $COHORT_DIR.

set -euo pipefail

: "${XNAT_USER:?set XNAT_USER in .env}"
: "${XNAT_PASS:?set XNAT_PASS in .env}"

cd "${COHORT_DIR:?missing COHORT_DIR}"

# xnatpy is on PyPI. Install into the box's uv env if missing.
if ! python3 -c "import xnat" 2>/dev/null; then
  pip install --quiet xnat
fi

cat > pull.py <<'PY'
import os
import xnat
from pathlib import Path

out = Path(os.environ["COHORT_DIR"])
user = os.environ["XNAT_USER"]
pw   = os.environ["XNAT_PASS"]
proj = os.environ.get("OASIS_PROJECT", "OASIS3")

with xnat.connect("https://central.xnat.org", user=user, password=pw) as session:
    project = session.projects[proj]
    subjects = list(project.subjects.values())
    print(f"OASIS-3: {len(subjects)} subjects")
    for i, sub in enumerate(subjects, 1):
        target = out / sub.label
        if (target / ".done").exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        try:
            sub.download_dir(str(target))
            (target / ".done").touch()
        except Exception as e:
            print(f"  subject {sub.label} failed: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(subjects)}")
PY

COHORT_DIR="$COHORT_DIR" \
XNAT_USER="$XNAT_USER" \
XNAT_PASS="$XNAT_PASS" \
python3 pull.py

rm pull.py
echo "oasis3: ready in ${COHORT_DIR}"
