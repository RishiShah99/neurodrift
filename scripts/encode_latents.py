"""Frozen-VAE latent encoding driver — the Phase-2 latent-dataset prereq.

Walks the preprocessed Zarr corpus at `gs://${GCS_BUCKET}/<zarr_prefix>/<cohort>/`,
groups stores by (cohort, subject, session), encodes each group's present
modalities through a FROZEN `DisentangledVAE3D` to the canonical fused content
latent `z = encode(x, mask)[0]` -> (C, d, d, d), and writes one latent store per
subject-session to `gs://${GCS_BUCKET}/<latent_prefix>/<cohort>/<stem>.zarr/`.

AGE WIRING (the hidden critical path): age is NaN in the corpus today. This
script reads each cohort's BIDS `participants.tsv` from
`gs://${GCS_BUCKET}/raw/<cohort>/participants.tsv` and stamps the per-subject age
into each latent store's attrs (NaN where the cohort has none).

CPU-bound and runnable on the box alongside the GPU (won't fight it):

    python scripts/encode_latents.py --cohorts abide,openneuro \
        --ckpt out/vae_v0/ckpt/last.ckpt

Idempotent: skips any subject-session whose `<stem>.zarr` already exists under
the latent prefix unless `--force` (mirrors scripts/preprocess.py).

References:
  - scripts/preprocess.py — argparse + GCS download/upload + idempotent skip.
  - scripts/eval.py:_load_weights — strip the Lightning `model.` prefix.
  - neurodrift/train/data_module.py — `_list_zarr_stems` / `_group_by_subject`
    enumeration and `_zscore` / `_random_crop_or_pad` voxel prep.
  - neurodrift/models/vae3d.py — DisentangledVAE3D.encode(x, mask)[0] == z.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from neurodrift.data.latents import parse_participants_tsv, write_latent_store
from neurodrift.models.vae3d import DisentangledVAE3D
from neurodrift.train.data_module import (
    SubjectGroup,
    _group_by_subject,
    _list_zarr_stems,
    _random_crop_or_pad,
    _zscore,
)

log = logging.getLogger("encode_latents")

# v0 frozen-VAE defaults (configs/model/vae3d.yaml). The encoder must be built
# with the SAME geometry the checkpoint was trained with or the weights won't load.
_DEFAULT_MODALITIES = ("T1w", "T2w", "FLAIR")


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _existing_latent_stems(bucket: str, cohort: str, latent_prefix: str) -> set[str]:
    """Stems of every `<stem>.zarr` already under the latent prefix for a cohort.

    One listing instead of one-ls-per-subject (mirrors
    preprocess._existing_zarr_stems). Stems are `sub-X[_ses-Y]` (no modality).
    """
    prefix = f"gs://{bucket}/{latent_prefix}/{cohort}/"
    res = subprocess.run(
        ["gcloud", "storage", "ls", prefix], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        return set()
    stems: set[str] = set()
    for line in res.stdout.splitlines():
        name = line.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".zarr"):
            stems.add(name[: -len(".zarr")])
    return stems


def _download_participants_tsv(bucket: str, cohort: str, dest: Path) -> dict[str, float]:
    """Fetch + parse `gs://<bucket>/raw/<cohort>/participants.tsv` -> {sub: age}.

    Tolerates absence: a cohort without a participants.tsv (or a failed copy)
    yields `{}`, so every latent in that cohort gets a NaN age.
    """
    local = dest / f"{cohort}_participants.tsv"
    src = f"gs://{bucket}/raw/{cohort}/participants.tsv"
    res = subprocess.run(
        ["gcloud", "storage", "cp", src, str(local)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        log.warning("%s: no participants.tsv (%s) -> all ages NaN", cohort, src.split("/")[-1])
        return {}
    ages = parse_participants_tsv(local)
    log.info("%s: participants.tsv -> %d subject ages", cohort, len(ages))
    return ages


def _stem_for_group(group: SubjectGroup) -> str:
    """`sub-X[_ses-Y]` — the latent stem (no modality suffix)."""
    ses = f"_{group.session}" if group.session else ""
    return f"{group.subject}{ses}"


def _load_group_volume(
    group: SubjectGroup, modalities: tuple[str, ...], image_size: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stack a subject-session's present modalities into (1, M, D, H, W) + mask.

    Mirrors data_module's per-subject crop/z-score: ONE crop window shared across
    modalities (they're co-registered on the same grid). Deterministic crop (this
    is inference, not augmentation). Returns None if no modality loaded.
    """
    import random

    import zarr

    m = len(modalities)
    x = np.zeros((m, image_size, image_size, image_size), dtype=np.float32)
    present = np.zeros(m, dtype=np.float32)
    crop_rng = random.Random(0)  # deterministic, shared across this group's slots
    crop_seed = crop_rng.randrange(2**31)
    for i, modality in enumerate(modalities):
        url = group.scans_by_modality.get(modality)
        if url is None:
            continue
        try:
            root = zarr.open(url, mode="r")
            vol = np.asarray(root["data"], dtype=np.float32)
        except (KeyError, ValueError, OSError, RuntimeError):
            continue  # partial/corrupt store -> treat modality as absent
        vol = _random_crop_or_pad(vol, image_size, random.Random(crop_seed))
        x[i] = _zscore(vol)
        present[i] = 1.0
    if present.sum() == 0:
        return None
    return (
        torch.from_numpy(x).unsqueeze(0),  # (1, M, D, H, W)
        torch.from_numpy(present).unsqueeze(0),  # (1, M)
    )


def _build_model(args: argparse.Namespace) -> DisentangledVAE3D:
    modalities = tuple(m.strip() for m in args.modalities.split(",") if m.strip())
    return DisentangledVAE3D(
        modalities=modalities,
        latent_channels=args.latent_channels,
        style_dim=args.style_dim,
        base_channels=args.base_channels,
        channel_mults=tuple(int(c) for c in args.channel_mults.split(",") if c.strip()),
        num_res_blocks=args.num_res_blocks,
    )


def _load_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load a Lightning or bare state-dict checkpoint (copied from scripts/eval.py).

    Lightning saves under `state_dict` with a `model.` prefix (the LitModule wraps
    the VAE as `self.model`); strip it so the weights land on the bare module.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        cleaned[key] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning("missing keys when loading ckpt: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])


@torch.no_grad()
def _encode_cohort(
    args: argparse.Namespace,
    model: DisentangledVAE3D,
    device: torch.device,
    cohort: str,
    scratch: Path,
) -> tuple[int, int]:
    log.info("=== %s: enumerate corpus + read participants.tsv ===", cohort)
    ages = _download_participants_tsv(args.bucket, cohort, scratch)

    zarr_root = f"gs://{args.bucket}/{args.zarr_prefix}"
    refs = _list_zarr_stems(zarr_root, cohort)
    groups = _group_by_subject(refs)
    groups.sort(key=lambda g: (g.cohort, g.subject, g.session or ""))
    log.info("%s: %d subject-session groups (%d scans)", cohort, len(groups), len(refs))

    cached = (
        set() if args.force else _existing_latent_stems(args.bucket, cohort, args.latent_prefix)
    )
    todo = [g for g in groups if _stem_for_group(g) not in cached]
    log.info("%s: %d groups to encode (%d cached)", cohort, len(todo), len(groups) - len(todo))

    modalities = model.modalities
    out_dir = scratch / "latents" / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_fail = 0
    for group in todo:
        stem = _stem_for_group(group)
        try:
            loaded = _load_group_volume(group, modalities, args.image_size)
            if loaded is None:
                log.warning("%s: %s has no loadable modality, skipping", cohort, stem)
                n_fail += 1
                continue
            x, mask = loaded
            x, mask = x.to(device), mask.to(device)
            z = model.encode(x, mask)[0][0]  # (C, d, d, d) — drop the batch dim

            store_path = out_dir / f"{stem}.zarr"
            write_latent_store(
                store_path,
                z.cpu().numpy(),
                age=ages.get(group.subject, float("nan")),
                cohort=cohort,
                subject=group.subject,
                session=group.session or "",
            )
            _upload_latent(args.bucket, cohort, store_path, args.latent_prefix)
            n_ok += 1
        except Exception:
            log.exception("%s: %s failed", cohort, stem)
            n_fail += 1
    log.info("%s: done — %d ok, %d failed", cohort, n_ok, n_fail)
    return n_ok, n_fail


def _upload_latent(bucket: str, cohort: str, store: Path, latent_prefix: str) -> None:
    # Copy the store INTO the cohort prefix (mirrors preprocess._upload_zarr): name
    # the destination `<latent_prefix>/<cohort>/` so gcloud appends the basename
    # once -> `<cohort>/<stem>.zarr/...`.
    dst = f"gs://{bucket}/{latent_prefix}/{cohort}/"
    _run(["gcloud", "storage", "cp", "--recursive", str(store), dst])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", default="abide,openneuro")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "neurodrift-data"))
    parser.add_argument("--ckpt", required=True, help="Frozen VAE checkpoint (Lightning or bare).")
    parser.add_argument("--scratch", default=os.environ.get("SCRATCH", "/mnt/scratch"))
    parser.add_argument(
        "--zarr-prefix", default="zarr", help="Source corpus root under the bucket."
    )
    parser.add_argument("--latent-prefix", default="latents", help="Latent store root.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    # VAE geometry overrides (must match the checkpoint). Defaults = v0.
    parser.add_argument("--modalities", default=",".join(_DEFAULT_MODALITIES))
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--style-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", default="1,2,4")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    model = _build_model(args)
    _load_weights(model, Path(args.ckpt))
    model.eval().to(device)

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_fail = 0
    with tempfile.TemporaryDirectory(dir=scratch, prefix="encode_latents_") as tmp:
        tmp_path = Path(tmp)
        for cohort in [c.strip() for c in args.cohorts.split(",") if c.strip()]:
            ok, fail = _encode_cohort(args, model, device, cohort, tmp_path)
            total_ok += ok
            total_fail += fail
    log.info("ALL done — %d ok, %d failed across cohorts", total_ok, total_fail)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
