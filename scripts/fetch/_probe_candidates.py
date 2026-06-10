"""Verify candidate OpenNeuro datasets on s3://openneuro.org — count anat + dwi files."""

from __future__ import annotations

import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client(
    "s3",
    config=Config(signature_version=UNSIGNED, region_name="us-east-1"),
)

CANDIDATES = [
    # Already in (sanity).
    "ds000221",  # LEMON — verified
    # Candidates worth checking. Mix of lifespan / healthy adult MRI.
    "ds000228",  # MIT children + adults
    "ds002785",  # NKI Rockland sample
    "ds003097",  # lifespan T1
    "ds004215",  # ?
    "ds000030",  # UCLA Consortium for Neuropsychiatric Phenomics
    "ds002790",  # Aomic ID1000
    "ds001734",  # IXI mirror? unlikely
    "ds001226",  # OpenNeuro multimodal lifespan
    "ds002338",  # ?
    "ds004302",  # ?
    "ds004215",  # ?
]

for ds in CANDIDATES:
    n_anat = 0
    n_dwi = 0
    n_t1 = 0
    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    try:
        for page in paginator.paginate(Bucket="openneuro.org", Prefix=f"{ds}/sub-"):
            for obj in page.get("Contents") or []:
                found_any = True
                k = obj["Key"]
                if "/anat/" in k and (k.endswith(".nii.gz") or k.endswith(".nii")):
                    n_anat += 1
                    if "T1w" in k:
                        n_t1 += 1
                if "/dwi/" in k and (k.endswith(".nii.gz") or k.endswith(".nii")):
                    n_dwi += 1
        status = "OK" if found_any else "MISSING"
    except Exception as e:
        status = f"ERROR {e}"
    print(f"{ds:10s}  anat={n_anat:5d}  T1={n_t1:5d}  dwi={n_dwi:5d}  [{status}]")
