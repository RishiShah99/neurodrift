#!/usr/bin/env bash
# OpenBHB — IEEE DataPort, requires an API token.
# https://baobablab.github.io/bhb/
#
# Set IEEE_DATAPORT_TOKEN in the fleet box's env (fleet sync pushes .env).
# Driver provides: $COHORT_DIR.

set -euo pipefail

: "${IEEE_DATAPORT_TOKEN:?set IEEE_DATAPORT_TOKEN in .env (IEEE DataPort profile -> API)}"

cd "${COHORT_DIR:?missing COHORT_DIR}"

# IEEE DataPort dataset ID for OpenBHB. Confirm against
# https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset
# at fetch time — IDs are stable but verify before a 50 GB pull.
DATASET_ID="${OPENBHB_DATASET_ID:-DSC03100}"
BASE="https://ieee-dataport.org/api/v2/files/${DATASET_ID}"

echo "querying file list for OpenBHB (DataPort id=${DATASET_ID})"
curl -fsSL \
  -H "Authorization: Bearer ${IEEE_DATAPORT_TOKEN}" \
  "${BASE}" > files.json

# The actual download URLs come back in files.json. We jq them out and curl.
mapfile -t URLS < <(python3 -c "
import json, sys
for f in json.load(open('files.json'))['data']:
    print(f['download_url'])
")

for url in "${URLS[@]}"; do
  fname="$(basename "$url" | sed 's/[?&].*//')"
  if [[ -f ".done.${fname}" ]]; then
    echo "skip ${fname}"
    continue
  fi
  echo "fetching ${fname}"
  curl -fsSLO --retry 5 --retry-delay 10 \
    -H "Authorization: Bearer ${IEEE_DATAPORT_TOKEN}" \
    "$url"
  case "$fname" in
    *.tar.gz|*.tgz) tar -xzf "$fname" && rm "$fname" ;;
    *.tar)          tar -xf  "$fname" && rm "$fname" ;;
    *.zip)          unzip -q "$fname" && rm "$fname" ;;
  esac
  touch ".done.${fname}"
done

rm -f files.json
echo "openbhb: ready in ${COHORT_DIR}"
