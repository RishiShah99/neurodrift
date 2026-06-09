#!/usr/bin/env bash
# HCP-Development 2.0 release — ConnectomeDB serves it via AWS S3.
# https://www.humanconnectome.org/study/hcp-lifespan-development
#
# Required env: HCP_AWS_KEY, HCP_AWS_SECRET (from ConnectomeDB "AWS Connection").
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

# HCP-D 2.0 lives under s3://hcp-openaccess/HCD/
SOURCE="s3://hcp-openaccess/HCD/"
# Restrict to T1, T2, dMRI — drop fMRI to save ~70% bandwidth.
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

echo "hcp_d: ready in ${COHORT_DIR}"
