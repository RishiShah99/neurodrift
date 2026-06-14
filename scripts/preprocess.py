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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from neurodrift.data.bids import Scan, iter_bids
from neurodrift.data.preprocess import PreprocessPipeline, coregister_subject_group

log = logging.getLogger("preprocess")

# Structural modalities the VAE trains on; dwi and other anat variants are skipped.
_COOK_MODALITIES = frozenset({"T1w", "T2w", "FLAIR"})


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=capture, text=True)


def _existing_zarr_stems(bucket: str, cohort: str, zarr_prefix: str = "zarr") -> set[str]:
    """Stems of every `<stem>.zarr` already in GCS for a cohort.

    One listing call instead of one-ls-per-scan — at 7k+ scans the per-scan
    probe added ~an hour of serial `gcloud` overhead before processing began.
    `zarr_prefix` selects the corpus root (e.g. `zarr_coreg` for the E7 rebuild).
    """
    prefix = f"gs://{bucket}/{zarr_prefix}/{cohort}/"
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


def _upload_zarr(bucket: str, cohort: str, zarr_dir: Path, zarr_prefix: str = "zarr") -> None:
    # Copy the store INTO the cohort prefix: `cp --recursive foo.zarr gs://.../cohort/`
    # yields `cohort/foo.zarr/...`. Naming the destination `cohort/foo.zarr/`
    # makes gcloud append the basename again -> `foo.zarr/foo.zarr/...`.
    dst = f"gs://{bucket}/{zarr_prefix}/{cohort}/"
    _run(["gcloud", "storage", "cp", "--recursive", str(zarr_dir), dst])


def _process_one(scan: Scan, work_dir: Path, template: Path) -> Path:
    pipeline = PreprocessPipeline.default(work_dir=work_dir, template=template)
    return pipeline.run(scan)


def _process_group(
    scans: list[Scan], work_dir: Path, template: Path, cached: frozenset[str]
) -> list[Path]:
    """E7 path: co-register a subject-session's modalities THROUGH its T1, then run the
    per-scan pipeline for the not-yet-cached scans. Returns the zarr stores to upload.

    The whole group is co-registered (the T1 anchors its siblings even when the T1's own
    zarr is already cached), but only uncached scans run the full pipeline + upload.
    RegisterStep then idempotently skips the 01_register output the pre-pass wrote.
    """
    coregister_subject_group(scans, work_dir, template)
    pipeline = PreprocessPipeline.default(work_dir=work_dir, template=template)
    return [pipeline.run(scan) for scan in scans if scan.stem not in cached]


def _process_cohort(
    bucket: str,
    cohort: str,
    scratch: Path,
    template: Path,
    workers: int,
    force: bool,
    modalities: frozenset[str] = _COOK_MODALITIES,
    coregister: bool = False,
    zarr_prefix: str = "zarr",
) -> tuple[int, int]:
    log.info("=== %s: download raw → scratch ===", cohort)
    raw_root = _download_raw_cohort(bucket, cohort, scratch / "raw")

    work_dir = scratch / "work" / cohort
    work_dir.mkdir(parents=True, exist_ok=True)

    # Only preprocess the requested structural modalities. The fetcher also pulls
    # dwi/ (large 4D DTI) which the model never reads; and `--modalities` lets us
    # regenerate just T2w/FLAIR (e.g. after a skull-strip fix) without redoing T1.
    scans = [s for s in iter_bids(raw_root) if s.modality in modalities]
    log.info("%s: found %d scans for %s in BIDS layout", cohort, len(scans), sorted(modalities))

    cached = set() if force else _existing_zarr_stems(bucket, cohort, zarr_prefix)
    todo: list[Scan] = [scan for scan in scans if scan.stem not in cached]
    log.info("%s: %d scans need processing (%d cached)", cohort, len(todo), len(scans) - len(todo))

    n_ok = 0
    n_fail = 0

    # --- E7 path: process whole subject-session GROUPS so siblings co-register through
    # the same T1 in one worker (avoids cross-process races on the shared 01_register
    # T1 output). Writes a SEPARATE zarr root so the current corpus stays intact for A/B.
    if coregister:
        groups: dict[tuple[str, str | None], list[Scan]] = defaultdict(list)
        for scan in scans:
            groups[(scan.subject, scan.session)].append(scan)
        todo_groups = [g for g in groups.values() if any(s.stem not in cached for s in g)]
        log.info(
            "%s: %d subject-session groups to co-register (%d cached scans skipped)",
            cohort,
            len(todo_groups),
            len(scans) - len(todo),
        )
        cached_fz = frozenset(cached)

        def _handle(group: list[Scan], outs: list[Path]) -> None:
            nonlocal n_ok
            for out in outs:
                _upload_zarr(bucket, cohort, out, zarr_prefix)
                n_ok += 1

        if workers <= 1:
            for group in todo_groups:
                try:
                    _handle(group, _process_group(group, work_dir, template, cached_fz))
                except Exception:
                    log.exception("%s: group %s failed", cohort, group[0].stem)
                    n_fail += sum(1 for s in group if s.stem not in cached)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process_group, g, work_dir, template, cached_fz): g
                    for g in todo_groups
                }
                for fut in as_completed(futures):
                    group = futures[fut]
                    try:
                        _handle(group, fut.result())
                    except Exception:
                        log.exception("%s: group %s failed", cohort, group[0].stem)
                        n_fail += sum(1 for s in group if s.stem not in cached)
        log.info("%s: done — %d ok, %d failed", cohort, n_ok, n_fail)
        return n_ok, n_fail

    # --- default path: independent per-scan registration (unchanged) ---
    if workers <= 1:
        for scan in todo:
            try:
                out = _process_one(scan, work_dir, template)
                _upload_zarr(bucket, cohort, out, zarr_prefix)
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
                    _upload_zarr(bucket, cohort, out, zarr_prefix)
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
    parser.add_argument(
        "--modalities",
        default="T1w,T2w,FLAIR",
        help="Comma-sep subset of T1w,T2w,FLAIR to process (default: all).",
    )
    parser.add_argument(
        "--coregister",
        action="store_true",
        help="E7: rigid-register each subject's other modalities to its OWN T1, then "
        "compose with T1->MNI (one resampling), instead of registering each to MNI "
        "independently. Tightens intra-subject cross-modal alignment.",
    )
    parser.add_argument(
        "--zarr-prefix",
        default=None,
        help="GCS corpus root under the bucket. Defaults to 'zarr_coreg' when "
        "--coregister (keeps the baseline 'zarr' corpus intact for the A/B), else 'zarr'.",
    )
    args = parser.parse_args()
    modalities = frozenset(m.strip() for m in args.modalities.split(",") if m.strip())
    zarr_prefix = args.zarr_prefix or ("zarr_coreg" if args.coregister else "zarr")

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
            modalities=modalities,
            coregister=args.coregister,
            zarr_prefix=zarr_prefix,
        )
        total_ok += ok
        total_fail += fail
    log.info("ALL done — %d ok, %d failed across cohorts", total_ok, total_fail)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
