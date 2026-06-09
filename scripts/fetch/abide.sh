#!/usr/bin/env bash
# ABIDE I + II — public, no signup. Hosted on NITRC + Amazon Public Datasets.
# https://fcon_1000.projects.nitrc.org/indi/abide/
#
# We pull preprocessed T1 (CPAC pipeline) which is what brain-MRI groups
# typically use. Skip the rs-fMRI to save bandwidth.
#
# Driver provides: $COHORT_DIR.

set -euo pipefail
cd "${COHORT_DIR:?missing COHORT_DIR}"

if ! command -v aws >/dev/null; then
  pip install --quiet awscli
fi

# ABIDE is on the AWS Open Data registry as a public bucket — no creds needed.
ABIDE_I="s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/func_preproc/"
ABIDE_I_T1="s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/nofilt_noglobal/anat/"
ABIDE_II="s3://fcp-indi/data/Projects/ABIDE_II/Outputs/cpac/nofilt_noglobal/anat/"

mkdir -p abide_i abide_ii

echo "syncing ABIDE I anat (T1) from public AWS bucket"
aws s3 sync --no-sign-request \
  "$ABIDE_I_T1" ./abide_i/ \
  --exclude "*" --include "*_T1w*" \
  --only-show-errors

echo "syncing ABIDE II anat (T1)"
aws s3 sync --no-sign-request \
  "$ABIDE_II" ./abide_ii/ \
  --exclude "*" --include "*_T1w*" \
  --only-show-errors

echo "abide: ready in ${COHORT_DIR}"
