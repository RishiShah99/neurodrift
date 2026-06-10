"""Parallel OpenNeuro fetcher via direct S3 (s3://openneuro.org).

Bypasses openneuro-py (GraphQL/HTTP, brittle). Lists each dataset listed in
openneuro_datasets.txt, filters to anat/* + dwi/*, and streams each object
through memory to gs://${GCS_BUCKET}/raw/openneuro/<dataset>/<rel>. Per-file
.done.<unit> markers in GCS make the run resumable through preemption.

Env (provided by scripts/fetch_data.sh):
    GCS_BUCKET   target GCS bucket
    GCS_RAW      gs://<bucket>/raw/openneuro
    FORCE        "1" to re-upload even if marker exists
    WORKERS      thread pool size (default 128)
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from google.cloud import storage as gcs

BUCKET = os.environ["GCS_BUCKET"]
GCS_RAW = os.environ["GCS_RAW"]
FORCE = os.environ.get("FORCE", "0") == "1"
WORKERS = int(os.environ.get("WORKERS", "128"))

PREFIX = GCS_RAW.replace(f"gs://{BUCKET}/", "")  # 'raw/openneuro'

S3_BUCKET = "openneuro.org"
DATASETS_TXT = Path(__file__).parent / "openneuro_datasets.txt"

# Per-file path filter: keep anat + dwi, drop the bulky modalities.
KEEP_SUBPATHS = ("/anat/", "/dwi/")
DROP_SUFFIXES = (".json",)  # we just need NIfTI for v0
INCLUDE_EXTS = (".nii.gz", ".nii", ".bval", ".bvec")


def log(msg: str) -> None:
    print(msg, flush=True)


def read_dataset_ids() -> list[str]:
    ids: list[str] = []
    for raw in DATASETS_TXT.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        ids.append(line.split()[0])
    return ids


def make_s3() -> object:
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            region_name="us-east-1",
            max_pool_connections=WORKERS * 2,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def list_done(gcs_client: gcs.Client) -> set[str]:
    if FORCE:
        return set()
    done: set[str] = set()
    for blob in gcs_client.list_blobs(BUCKET, prefix=f"{PREFIX}/.done."):
        done.add(blob.name.rsplit("/.done.", 1)[1])
    return done


def list_dataset_keys(s3: object, dataset: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")  # type: ignore[attr-defined]
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{dataset}/sub-"):
        for obj in page.get("Contents") or []:
            k = obj["Key"]
            if not any(sub in k for sub in KEEP_SUBPATHS):
                continue
            if not any(k.endswith(ext) for ext in INCLUDE_EXTS):
                continue
            keys.append(k)
    return keys


def main() -> int:
    datasets = read_dataset_ids()
    if not datasets:
        log("openneuro: no datasets listed; nothing to do")
        return 0
    log(f"openneuro: datasets = {datasets}")

    s3 = make_s3()
    gcs_client = gcs.Client()
    bucket = gcs_client.bucket(BUCKET)

    done = list_done(gcs_client)
    log(f"openneuro: {len(done)} units already in GCS (skip)")

    items: list[tuple[str, str, str]] = []
    for ds in datasets:
        log(f"openneuro: listing {ds} on s3://openneuro.org …")
        keys = list_dataset_keys(s3, ds)
        log(f"openneuro: {ds} — {len(keys)} candidate files")
        for k in keys:
            rel = k  # keep dataset/ prefix in GCS layout
            unit = rel.replace("/", "_")
            if unit not in done:
                items.append((k, rel, unit))

    log(f"openneuro: {len(items)} new objects to fetch ({WORKERS} threads)")
    if not items:
        log("openneuro: nothing to do")
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
            if i % 200 == 0:
                log(f"  openneuro progress {i}/{len(items)}  ok={ok}  fail={fail}")

    log(f"openneuro: done  ok={ok}  fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
