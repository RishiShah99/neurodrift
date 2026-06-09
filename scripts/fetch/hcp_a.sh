#!/usr/bin/env bash
# HCP-Aging 2.0 release — same ConnectomeDB AWS credentials as HCP-D.
# https://www.humanconnectome.org/study/hcp-lifespan-aging
#
# Required env: HCP_AWS_KEY, HCP_AWS_SECRET.
# Driver provides: $COHORT_DIR.

set -euo pipefail

: "${HCP_AWS_KEY:?set HCP_AWS_KEY in .env (ConnectomeDB -> AWS Connection)}"
: "${HCP_AWS_SECRET:?set HCP_AWS_SECRET in .env}"

cd "${COHORT_DIR:?missing COHORT_DIR}"

if ! command -v aws >/dev/null; then
  echo "installing awscli"
  pip install --quiet awscli
fi

export AWS_ACCESS_KEY_ID="$HCP_AWS_KEY"
export AWS_SECRET_ACCESS_KEY="$HCP_AWS_SECRET"
export AWS_DEFAULT_REGION="us-east-1"

SOURCE="s3://hcp-openaccess/HCA/"
INCLUDE=(
  "--include" "*T1w*"
  "--include" "*T2w*"
  "--include" "*dMRI*"
  "--include" "*.json"
  "--include" "*bvals*"
  "--include" "*bvecs*"
)

aws s3 sync "$SOURCE" . \
  --exclude "*" "${INCLUDE[@]}" \
  --only-show-errors

echo "hcp_a: ready in ${COHORT_DIR}"
