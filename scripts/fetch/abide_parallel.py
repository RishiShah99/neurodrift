"""Parallel ABIDE I T1w fetcher (raw BIDS).

Lists s3://fcp-indi/data/Projects/ABIDE_Initiative/RawDataBIDS/<site>/sub-*/anat/*_T1w.nii.gz,
streams each object through memory to gs://${GCS_BUCKET}/raw/abide/, and
writes a .done.<unit> marker in GCS per object. Resumable through any
preemption: a fresh run lists the existing markers and skips them.

ABIDE II isn't on this S3 bucket — dropped from v0.

Env (provided by scripts/fetch_data.sh):
    GCS_BUCKET   target GCS bucket
    GCS_RAW      gs://<bucket>/raw/abide
    FORCE        "1" to re-upload even if marker exists
    WORKERS      thread pool size (default 128)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from google.cloud import storage as gcs

BUCKET = os.environ["GCS_BUCKET"]
GCS_RAW = os.environ["GCS_RAW"]
FORCE = os.environ.get("FORCE", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", "128"))

PREFIX = GCS_RAW.replace(f"gs://{BUCKET}/", "")  # 'raw/abide'

S3_BUCKET = "fcp-indi"
ABIDE_I_ROOT = "data/Projects/ABIDE_Initiative/RawDataBIDS/"


def log(msg: str) -> None:
    print(msg, flush=True)


def make_s3() -> object:
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            max_pool_connections=WORKERS * 2,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def list_done(gcs_client: gcs.Client) -> set[str]:
    if FORCE:
        return set()
    done = set()
    for blob in gcs_client.list_blobs(BUCKET, prefix=f"{PREFIX}/.done."):
        done.add(blob.name.rsplit("/.done.", 1)[1])
    return done


def list_t1_keys(s3: object) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")  # type: ignore[attr-defined]
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=ABIDE_I_ROOT):
        for obj in page.get("Contents") or []:
            k = obj["Key"]
            if k.endswith("_T1w.nii.gz") and "/anat/" in k:
                keys.append(k)
    return keys


def main() -> int:
    s3 = make_s3()
    gcs_client = gcs.Client()
    bucket = gcs_client.bucket(BUCKET)

    log("abide: listing ABIDE I T1w on s3://fcp-indi …")
    keys = list_t1_keys(s3)
    log(f"abide: S3 listing — {len(keys)} T1w volumes")

    done = list_done(gcs_client)
    log(f"abide: {len(done)} units already in GCS (skip)")

    items: list[tuple[str, str, str]] = []
    for k in keys:
        rel = k[len(ABIDE_I_ROOT) :]
        unit = f"abide_{rel.replace('/', '_')}"
        if unit not in done:
            items.append((k, rel, unit))

    log(f"abide: {len(items)} new objects to fetch ({WORKERS} threads)")
    if not items:
        log("abide: nothing to do")
        return 0

    def transfer(item: tuple[str, str, str]) -> tuple[str, bool, str]:
        key, rel, unit = item
        try:
            body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()  # type: ignore[attr-defined]
            bucket.blob(f"{PREFIX}/{rel}").upload_from_string(body)
            bucket.blob(f"{PREFIX}/.done.{unit}").upload_from_string(b"")
            return (unit, True, "")
        except Exception as e:
            return (unit, False, repr(e))

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(transfer, it) for it in items]
        for i, fut in enumerate(as_completed(futures), 1):
            unit, success, err = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                log(f"  FAIL {unit}: {err}")
            if i % 100 == 0:
                log(f"  abide progress {i}/{len(items)}  ok={ok}  fail={fail}")

    log(f"abide: done  ok={ok}  fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
