"""Drill into LEMON to find anat + dwi paths."""

from __future__ import annotations

import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, region_name="us-east-1"))


def list_all(prefix: str, limit: int = 30) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket="openneuro.org", Prefix=prefix):
        for obj in page.get("Contents") or []:
            keys.append(obj["Key"])
            if len(keys) >= limit:
                return keys
    return keys


print("== ds000221/sub-010001 (full recursive, first 30 files) ==")
for k in list_all("ds000221/sub-010001/", 30):
    print(f"  {k}")

print("\n== ds000221 anat counts (sample first page) ==")
paginator = s3.get_paginator("list_objects_v2")
n_anat = 0
n_dwi = 0
n_total = 0
for page in paginator.paginate(Bucket="openneuro.org", Prefix="ds000221/sub-"):
    for obj in page.get("Contents") or []:
        n_total += 1
        if "/anat/" in obj["Key"]:
            n_anat += 1
        if "/dwi/" in obj["Key"]:
            n_dwi += 1
print(f"  total sub- files: {n_total}, anat: {n_anat}, dwi: {n_dwi}")
