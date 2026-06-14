"""Latent store: cache one VAE content latent per subject-session + its metadata.

The Phase-2 flow / SAE models train on the FROZEN VAE's latent codes, not on
voxels — encoding 128^3 trimodal volumes every step is the bottleneck, and the
latent (B, 16, 32, 32, 32) is ~2000x smaller than the (B, 3, 128, 128, 128)
input. `scripts/encode_latents.py` precomputes the canonical subject latent
`z = DisentangledVAE3D.encode(x, mask)[0]` once and writes it here; this module
is the read side + the Lightning datamodule those models consume.

Schema (frozen shared contract — other agents depend on it):
    zarr store, array key "data" = z float32 shape (C, d, d, d);
    attrs = {age: float (NaN if unknown), sex: int (-1 unknown), dx: int (-1),
             apoe: int (-1), treatment: int (0=placebo/null), cohort: str,
             subject: str, session: str}.
    Path layout: <latent_root>/<cohort>/<stem>.zarr ,
    stem = "sub-X[_ses-Y]" (NO modality suffix — one latent per subject-session).

Age is the hidden critical-path field: it is NaN in the source corpus today and
is populated from each cohort's BIDS `participants.tsv` at encode time. NaN ages
are preserved here unchanged — the downstream model imputes them.

References:
  - `neurodrift/train/data_module.py` (`_list_zarr_stems` / `_group_by_subject`
    enumeration, `_load_volume` corrupt-store tolerance, `_loader_kwargs`
    fork-vs-spawn rule, deterministic subject-level split) — mirrored here.
  - `neurodrift/data/io.py` (`zarr.open(..., create_dataset("data", ...))` write).
  - BIDS participants.tsv spec (`participant_id` column + an age column);
    https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html
"""

from __future__ import annotations

import csv
import logging
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# Latent stems carry NO modality suffix (one fused latent per subject-session),
# unlike the voxel corpus stems in data_module._STEM_RE. Match `sub-X[_ses-Y]`.
_LATENT_STEM_RE = re.compile(r"^(?P<subject>sub-[^_]+)(?:_(?P<session>ses-[^_]+))?$")

# BIDS participants.tsv age column, in preference order. Cohorts in the wild use
# any of these spellings; the first present non-empty one wins.
_AGE_COLUMNS: tuple[str, ...] = ("age", "Age", "age_at_scan", "Age_at_scan")

# Categorical attrs and their "unknown"/null sentinels (the frozen schema).
_CATEGORICAL_DEFAULTS: dict[str, int] = {
    "sex": -1,
    "dx": -1,
    "apoe": -1,
    "treatment": 0,
}


# ---------------------------------------------------------------------------
# Store I/O
# ---------------------------------------------------------------------------
def write_latent_store(
    path: str | Path,
    z: np.ndarray | torch.Tensor,
    *,
    age: float,
    sex: int = -1,
    dx: int = -1,
    apoe: int = -1,
    treatment: int = 0,
    cohort: str,
    subject: str,
    session: str,
) -> None:
    """Write one subject-session latent `z` + metadata to a zarr store.

    `z` is the canonical fused content latent (C, d, d, d) — typically
    `DisentangledVAE3D.encode(x, mask)[0][0]`. Idempotent: opening with mode "w"
    truncates any prior store at `path`, so a re-encode cleanly overwrites a
    partial/stale store (mirrors `data.io.nifti_to_zarr`).
    """
    if isinstance(z, torch.Tensor):
        arr = z.detach().to("cpu", torch.float32).numpy()
    else:
        arr = np.asarray(z, dtype=np.float32)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(path), mode="w")
    data = root.create_dataset(
        "data", shape=arr.shape, chunks=arr.shape, dtype=np.float32, overwrite=True
    )
    data[...] = arr
    # float() / int() coerce numpy scalars to JSON-serializable Python scalars
    # (zarr's .zattrs is JSON); a np.float32 NaN would otherwise raise on write.
    root.attrs["age"] = float(age)
    root.attrs["sex"] = int(sex)
    root.attrs["dx"] = int(dx)
    root.attrs["apoe"] = int(apoe)
    root.attrs["treatment"] = int(treatment)
    root.attrs["cohort"] = str(cohort)
    root.attrs["subject"] = str(subject)
    root.attrs["session"] = str(session)


def read_latent_store(path: str | Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Load a latent store into (z, attrs), or None if missing/partial/corrupt.

    A store whose write was interrupted (spot preemption mid-upload) can exist
    without its `data` array; reading it raises KeyError. Tolerate that the same
    way `data_module.ZarrMultimodalDataset._load_volume` does — return None so a
    single bad store can't crash a dataloader worker (which kills a DDP rank).
    """
    try:
        root = zarr.open(str(path), mode="r")
        z = np.asarray(root["data"], dtype=np.float32)
        return z, dict(root.attrs)
    except (KeyError, ValueError, OSError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# participants.tsv parsing (age wiring)
# ---------------------------------------------------------------------------
def _coerce_age(raw: str | None) -> float:
    """BIDS age cell -> float, with NaN for missing / 'n/a' / unparseable."""
    if raw is None:
        return float("nan")
    raw = raw.strip()
    if raw == "" or raw.lower() in {"n/a", "na", "nan", "none", "null"}:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _normalize_participant_id(raw: str) -> str | None:
    """Canonicalize a participant_id cell to a `sub-XXX` key, or None if empty."""
    pid = raw.strip()
    if not pid:
        return None
    return pid if pid.startswith("sub-") else f"sub-{pid}"


def parse_participants_tsv(path: str | Path) -> dict[str, float]:
    """Parse a BIDS `participants.tsv` into `{sub-XXX: age}`.

    Tolerant by design: a missing file, a missing `participant_id` column, or a
    missing age column all yield `{}` (callers fall back to NaN ages); individual
    "n/a"/blank/unparseable age cells map to NaN. The age column is the first of
    `_AGE_COLUMNS` present in the header.

    BIDS participants.tsv is tab-separated with a `participant_id` column; see
    https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if "participant_id" not in fields:
            return {}
        age_col = next((c for c in _AGE_COLUMNS if c in fields), None)
        out: dict[str, float] = {}
        for row in reader:
            key = _normalize_participant_id(row.get("participant_id", ""))
            if key is None:
                continue
            out[key] = _coerce_age(row.get(age_col) if age_col else None)
    return out


def parse_sessions_tsv(path: str | Path) -> dict[str, float]:
    """Parse a BIDS `sub-XXX_sessions.tsv` into `{ses-YY: age}` (session-level age).

    Optional: many cohorts record per-session age (e.g. longitudinal OpenNeuro)
    in a `session_id` + age column file beside the subject. Same tolerance rules
    as `parse_participants_tsv`. Returns `{}` if the file or required column is
    absent; keys are canonical `ses-YY`.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        if "session_id" not in fields:
            return {}
        age_col = next((c for c in _AGE_COLUMNS if c in fields), None)
        out: dict[str, float] = {}
        for row in reader:
            sid = (row.get("session_id") or "").strip()
            if not sid:
                continue
            key = sid if sid.startswith("ses-") else f"ses-{sid}"
            out[key] = _coerce_age(row.get(age_col) if age_col else None)
    return out


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatentRef:
    """One latent store: a subject-session and its zarr url."""

    cohort: str
    subject: str
    session: str | None
    url: str


def list_latent_refs(root: str, cohort: str) -> list[LatentRef]:
    """List every `<stem>.zarr` directly under `${root}/${cohort}/`.

    Works for `gs://` (via gcsfs) and local paths (via os), mirroring
    `data_module._list_zarr_stems` — but the stem regex has NO modality suffix
    (one latent per subject-session).
    """
    import fsspec

    base = f"{root.rstrip('/')}/{cohort}"
    fs, base_path = fsspec.core.url_to_fs(base)
    refs: list[LatentRef] = []
    try:
        entries = fs.ls(base_path, detail=False)
    except FileNotFoundError:
        return refs
    for entry in entries:
        name = entry.rsplit("/", 1)[-1]
        if not name.endswith(".zarr"):
            continue
        stem = name[: -len(".zarr")]
        m = _LATENT_STEM_RE.match(stem)
        if m is None:
            continue
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        url = entry if protocol == "file" else f"{protocol}://{entry}"
        refs.append(LatentRef(cohort=cohort, subject=m["subject"], session=m["session"], url=url))
    return refs


# ---------------------------------------------------------------------------
# Dataset + DataModule
# ---------------------------------------------------------------------------
def _attr_int(attrs: dict[str, Any], key: str, default: int) -> int:
    """Read a categorical attr as int, falling back to its sentinel default."""
    val = attrs.get(key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _attr_age(attrs: dict[str, Any]) -> float:
    """Read the age attr as float; absent / unparseable -> NaN (model imputes)."""
    val = attrs.get("age")
    if val is None or val == "":
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


class LatentZarrDataset(Dataset[dict[str, Any]]):
    """Yield one subject-session latent per index as a batch-item dict.

    Robust to a partial/corrupt store: `read_latent_store` returns None, which is
    surfaced as a zero latent with NaN age and sentinel categoricals so a bad
    store degrades a sample instead of crashing the worker (DDP-safe). NaN age is
    left NaN here — the downstream model imputes it.
    """

    def __init__(self, refs: Sequence[LatentRef]) -> None:
        self.refs = list(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ref = self.refs[idx]
        loaded = read_latent_store(ref.url)
        if loaded is None:
            log.warning("latent store missing/corrupt, emitting empty sample: %s", ref.url)
            return {
                "z": torch.zeros(1, dtype=torch.float32),
                "age": torch.tensor(float("nan"), dtype=torch.float32),
                "sex": torch.tensor(_CATEGORICAL_DEFAULTS["sex"], dtype=torch.long),
                "dx": torch.tensor(_CATEGORICAL_DEFAULTS["dx"], dtype=torch.long),
                "apoe": torch.tensor(_CATEGORICAL_DEFAULTS["apoe"], dtype=torch.long),
                "treatment": torch.tensor(_CATEGORICAL_DEFAULTS["treatment"], dtype=torch.long),
                "cohort": ref.cohort,
                "subject": ref.subject,
                "session": ref.session or "",
            }
        z, attrs = loaded
        return {
            "z": torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)),
            "age": torch.tensor(_attr_age(attrs), dtype=torch.float32),
            "sex": torch.tensor(_attr_int(attrs, "sex", -1), dtype=torch.long),
            "dx": torch.tensor(_attr_int(attrs, "dx", -1), dtype=torch.long),
            "apoe": torch.tensor(_attr_int(attrs, "apoe", -1), dtype=torch.long),
            "treatment": torch.tensor(_attr_int(attrs, "treatment", 0), dtype=torch.long),
            "cohort": str(attrs.get("cohort", ref.cohort)),
            "subject": str(attrs.get("subject", ref.subject)),
            "session": str(attrs.get("session", ref.session or "")),
        }


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors; keep the string metadata as per-sample lists."""
    return {
        "z": torch.stack([s["z"] for s in samples]),
        "age": torch.stack([s["age"] for s in samples]),
        "sex": torch.stack([s["sex"] for s in samples]),
        "dx": torch.stack([s["dx"] for s in samples]),
        "apoe": torch.stack([s["apoe"] for s in samples]),
        "treatment": torch.stack([s["treatment"] for s in samples]),
        "cohort": [s["cohort"] for s in samples],
        "subject": [s["subject"] for s in samples],
        "session": [s["session"] for s in samples],
    }


class LatentDataModule(L.LightningDataModule):
    """Lightning datamodule over the cached latent store.

    Walks `${latent_root}/${cohort}/*.zarr` for every cohort, splits on
    (cohort, subject) so every session of a subject lands wholly in train or val
    (no cross-session leakage), and emits the frozen batch dict:

        {"z": (B,C,d,d,d) float, "age": (B,) float, "sex"/"dx"/"apoe"/"treatment":
         (B,) long, "cohort"/"subject"/"session": list[str]}.
    """

    def __init__(
        self,
        latent_root: str,
        cohorts: Sequence[str],
        batch_size: int = 8,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        seed: int = 1337,
        **_: Any,
    ) -> None:
        super().__init__()
        self.latent_root = latent_root
        self.cohorts = list(cohorts)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.train_ds: LatentZarrDataset | None = None
        self.val_ds: LatentZarrDataset | None = None

    def _discover(self) -> list[LatentRef]:
        refs: list[LatentRef] = []
        for cohort in self.cohorts:
            refs.extend(list_latent_refs(self.latent_root, cohort))
        # Canonical sort BEFORE any seeded shuffle: fs.ls is unordered (scandir
        # order locally, lexicographic on gcsfs), so a seeded shuffle over an
        # unordered list is not reproducible across machines — the downstream
        # model could rebuild a DIFFERENT val split than a checkpoint was selected
        # against. Sorting first makes the split identical everywhere. Mirrors
        # data_module._discover.
        refs.sort(key=lambda r: (r.cohort, r.subject, r.session or ""))
        return refs

    def setup(self, stage: str | None = None) -> None:
        refs = self._discover()
        if not refs:
            raise RuntimeError(
                f"no latent stores found under {self.latent_root} for cohorts "
                f"{self.cohorts}; run scripts/encode_latents.py first"
            )
        # Split on (cohort, subject), NOT per-session: a subject with >=2 sessions
        # (OpenNeuro repeats) must land ENTIRELY in train or val, else the same
        # anatomy/scanner sits on both sides and inflates val. Mirrors
        # data_module.setup's subject-level split (no multimodality stratum — a
        # latent is one fused code per session, modality structure is gone).
        rng = random.Random(self.seed)
        subjects = sorted({(r.cohort, r.subject) for r in refs})
        rng.shuffle(subjects)
        n_val = max(1, int(len(subjects) * self.val_fraction)) if len(subjects) > 1 else 0
        val_subjects = set(subjects[:n_val])
        val_refs = [r for r in refs if (r.cohort, r.subject) in val_subjects]
        train_refs = [r for r in refs if (r.cohort, r.subject) not in val_subjects]
        log.info(
            "latent split: %d val subjects, %d train / %d val stores",
            len(val_subjects),
            len(train_refs),
            len(val_refs),
        )
        self.train_ds = LatentZarrDataset(train_refs)
        self.val_ds = LatentZarrDataset(val_refs)

    def _loader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": _collate,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        # gcsfs opens a gs:// store via an asyncio loop + background thread that do
        # not survive a fork, so a gs:// root needs spawned workers; a local cache
        # (plain files) forks correctly and lighter. Only force spawn when remote.
        # Copied from data_module._loader_kwargs.
        if self.num_workers > 0 and str(self.latent_root).startswith("gs://"):
            kwargs["multiprocessing_context"] = "spawn"
        return kwargs

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        assert self.train_ds is not None
        return DataLoader(self.train_ds, shuffle=True, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        assert self.val_ds is not None
        return DataLoader(self.val_ds, shuffle=False, **self._loader_kwargs())
