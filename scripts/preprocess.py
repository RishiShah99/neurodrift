"""Raw NIfTI → Zarr preprocessing driver.

Walks `gs://${GCS_BUCKET}/raw/<cohort>/` (BIDS-style layout), runs the
`PreprocessPipeline.default` (register → skull-strip → N4 → harmonize → Zarr)
on each scan locally, and pushes the resulting Zarr store back to
`gs://${GCS_BUCKET}/zarr/<cohort>/<stem>.zarr/`.

CPU-bound. Run on the fleet box alongside the fetch (won't fight the GPU):

    fleet train "uv run python scripts/preprocess.py --cohorts ixi,abide,openneuro"

Skips any subject whose `<stem>.zarr` already exists in GCS unless `--force`.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from neurodrift.data.bids import Scan, iter_bids
from neurodrift.data.preprocess import PreprocessPipeline

log = logging.getLogger("preprocess")


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=capture, text=True)


def _existing_zarr_stems(bucket: str, cohort: str) -> set[str]:
    """Stems of every `<stem>.zarr` already in GCS for a cohort.

    One listing call instead of one-ls-per-scan — at 7k+ scans the per-scan
    probe added ~an hour of serial `gcloud` overhead before processing began.
    """
    prefix = f"gs://{bucket}/zarr/{cohort}/"
    res = subprocess.run(
        ["gcloud", "storage", "ls", prefix], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        return set()  # prefix doesn't exist yet → nothing cached
    stems: set[str] = set()
    for line in res.stdout.splitlines():
        name = line.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            stems.add(name[: -len(".zarr")])
    return stems


def _download_raw_cohort(bucket: str, cohort: str, dest: Path) -> Path:
    cohort_dst = dest / cohort
    cohort_dst.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            "--no-clobber",
            f"gs://{bucket}/raw/{cohort}/*",
            str(cohort_dst) + "/",
        ]
    )
    return cohort_dst


def _upload_zarr(bucket: str, cohort: str, zarr_dir: Path) -> None:
    # Copy the store INTO the cohort prefix: `cp --recursive foo.zarr gs://.../cohort/`
    # yields `cohort/foo.zarr/...`. Naming the destination `cohort/foo.zarr/`
    # makes gcloud append the basename again -> `foo.zarr/foo.zarr/...`.
    dst = f"gs://{bucket}/zarr/{cohort}/"
    _run(["gcloud", "storage", "cp", "--recursive", str(zarr_dir), dst])


def _process_one(scan: Scan, work_dir: Path, template: Path) -> Path:
    pipeline = PreprocessPipeline.default(work_dir=work_dir, template=template)
    return pipeline.run(scan)


def _process_cohort(
    bucket: str,
    cohort: str,
    scratch: Path,
    template: Path,
    workers: int,
    force: bool,
) -> tuple[int, int]:
    log.info("=== %s: download raw → scratch ===", cohort)
    raw_root = _download_raw_cohort(bucket, cohort, scratch / "raw")

    work_dir = scratch / "work" / cohort
    work_dir.mkdir(parents=True, exist_ok=True)

    scans = list(iter_bids(raw_root))
    log.info("%s: found %d scans in BIDS layout", cohort, len(scans))

    cached = set() if force else _existing_zarr_stems(bucket, cohort)
    todo: list[Scan] = [scan for scan in scans if scan.stem not in cached]
    log.info("%s: %d scans need processing (%d cached)", cohort, len(todo), len(scans) - len(todo))

    n_ok = 0
    n_fail = 0
    if workers <= 1:
        for scan in todo:
            try:
                out = _process_one(scan, work_dir, template)
                _upload_zarr(bucket, cohort, out)
                n_ok += 1
            except Exception:
                log.exception("%s: %s failed", cohort, scan.stem)
                n_fail += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, scan, work_dir, template): scan for scan in todo}
            for fut in as_completed(futures):
                scan = futures[fut]
                try:
                    out = fut.result()
                    _upload_zarr(bucket, cohort, out)
                    n_ok += 1
                except Exception:
                    log.exception("%s: %s failed", cohort, scan.stem)
                    n_fail += 1
    log.info("%s: done — %d ok, %d failed", cohort, n_ok, n_fail)
    return n_ok, n_fail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", default="ixi,abide,openneuro")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "neurodrift-data"))
    parser.add_argument("--scratch", default=os.environ.get("SCRATCH", "/mnt/scratch"))
    parser.add_argument(
        "--template",
        default=os.environ.get("MNI_TEMPLATE", "/data/templates/MNI152_T1_1mm.nii.gz"),
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    template = Path(args.template)
    if not template.exists():
        log.warning("MNI template not found at %s — registration may be a no-op", template)

    total_ok = 0
    total_fail = 0
    for cohort in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
        ok, fail = _process_cohort(
            bucket=args.bucket,
            cohort=cohort,
            scratch=scratch,
            template=template,
            workers=args.workers,
            force=args.force,
        )
        total_ok += ok
        total_fail += fail
    log.info("ALL done — %d ok, %d failed across cohorts", total_ok, total_fail)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
