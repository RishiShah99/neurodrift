"""Probe ABIDE II layout + one sample anat dir from ABIDE I."""

from __future__ import annotations

import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

print("== ABIDE I sample anat dir ==")
r = s3.list_objects_v2(
    Bucket="fcp-indi",
    Prefix="data/Projects/ABIDE_Initiative/RawDataBIDS/Yale/sub-0050612/anat/",
)
for k in r.get("Contents") or []:
    print(f"  {k['Key']}  ({k['Size']} bytes)")

print("\n== ABIDE I sites ==")
r = s3.list_objects_v2(
    Bucket="fcp-indi",
    Prefix="data/Projects/ABIDE_Initiative/RawDataBIDS/",
    Delimiter="/",
)
sites = [c["Prefix"] for c in r.get("CommonPrefixes") or []]
print(f"  count: {len(sites)}")
for s in sites[:5]:
    print(f"    {s}")

print("\n== ABIDE II top-level ==")
r = s3.list_objects_v2(
    Bucket="fcp-indi",
    Prefix="data/Projects/ABIDE_II/",
    Delimiter="/",
)
for c in r.get("CommonPrefixes") or []:
    print(f"  DIR  {c['Prefix']}")
for k in (r.get("Contents") or [])[:5]:
    print(f"  FILE {k['Key']}")
