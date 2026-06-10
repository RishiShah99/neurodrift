"""One-off probe — list ABIDE I + II top-level S3 layout to find the real T1 path."""

from __future__ import annotations

import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def walk(prefix: str, depth: int = 0, max_depth: int = 4) -> None:
    if depth > max_depth:
        return
    r = s3.list_objects_v2(Bucket="fcp-indi", Prefix=prefix, Delimiter="/")
    for c in r.get("CommonPrefixes") or []:
        p = c["Prefix"]
        print("  " * depth + "DIR " + p)
        walk(p, depth + 1, max_depth)
    for k in (r.get("Contents") or [])[:3]:
        print("  " * depth + "FILE " + k["Key"])


for top in (
    "data/Projects/ABIDE_Initiative/",
    "data/Projects/ABIDE_II/",
):
    print("=" * 40)
    print(top)
    walk(top, max_depth=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
