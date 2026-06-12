"""Lightning DataModule over preprocessed Zarr volumes.

Reads the v0 corpus from `${zarr_root}/${cohort}/<stem>.zarr`, groups Zarr
stores by (cohort, subject, session) so a single batch sees all modalities
that exist for one scan, applies modality dropout, and emits:

    {
        "image":         (B, M, D, H, W) tensor — model INPUT; zero-filled for dropped slots
        "target":        (B, M, D, H, W) tensor — clean recon TARGET; every acquired
                                                  modality kept intact (never zeroed),
                                                  so dropped-input slots are a real
                                                  cross-modal synthesis objective
        "modality_mask": (B, M) float tensor — 1 = fed to the encoder, 0 = dropped/missing
        "present_mask":  (B, M) float tensor — 1 = acquired for this scan (drives the
                                               recon loss), 0 = never acquired
        "age":           (B,) float tensor — scan age from Zarr attrs, NaN if unknown
                                             (Phase-2 conditioning hook)
        "cohort":        list[str] of length B
    }

The order of modality slots is fixed by `modalities` so the VAE's per-modality
stems line up with the right channel.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset

_STEM_RE = re.compile(
    r"^(?P<subject>sub-[^_]+)(?:_(?P<session>ses-[^_]+))?_(?P<modality>[A-Za-z0-9]+)$"
)


@dataclass(frozen=True)
class ScanRef:
    cohort: str
    subject: str
    session: str | None
    modality: str
    url: str


@dataclass(frozen=True)
class SubjectGroup:
    cohort: str
    subject: str
    session: str | None
    scans_by_modality: dict[str, str]  # modality -> zarr url


def _list_zarr_stems(root: str, cohort: str) -> list[ScanRef]:
    """List every `<stem>.zarr` directly under `${root}/${cohort}/`.

    Works for `gs://` (via gcsfs) and local paths (via os).
    """
    import fsspec

    base = f"{root.rstrip('/')}/{cohort}"
    fs, base_path = fsspec.core.url_to_fs(base)
    refs: list[ScanRef] = []
    try:
        entries = fs.ls(base_path, detail=False)
    except FileNotFoundError:
        return refs
    for entry in entries:
        name = entry.rsplit("/", 1)[-1]
        if not name.endswith(".zarr"):
            continue
        stem = name[: -len(".zarr")]
        m = _STEM_RE.match(stem)
        if m is None:
            continue
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        url = entry if protocol == "file" else f"{protocol}://{entry}"
        refs.append(
            ScanRef(
                cohort=cohort,
                subject=m["subject"],
                session=m["session"],
                modality=m["modality"],
                url=url,
            )
        )
    return refs


def _group_by_subject(refs: Sequence[ScanRef]) -> list[SubjectGroup]:
    grouped: dict[tuple[str, str, str | None], dict[str, str]] = defaultdict(dict)
    for r in refs:
        grouped[(r.cohort, r.subject, r.session)][r.modality] = r.url
    return [
        SubjectGroup(cohort=c, subject=s, session=sess, scans_by_modality=mods)
        for (c, s, sess), mods in grouped.items()
    ]


def _random_crop_or_pad(volume: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    """Random crop to (size, size, size); pad with zeros where smaller."""
    out = np.zeros((size, size, size), dtype=np.float32)
    src = volume
    src_shape = src.shape
    src_starts = []
    out_starts = []
    extents = []
    for s in src_shape:
        if s >= size:
            start = rng.randint(0, s - size)
            src_starts.append(start)
            out_starts.append(0)
            extents.append(size)
        else:
            src_starts.append(0)
            out_starts.append((size - s) // 2)
            extents.append(s)
    sl_src = tuple(slice(a, a + e) for a, e in zip(src_starts, extents, strict=True))
    sl_out = tuple(slice(a, a + e) for a, e in zip(out_starts, extents, strict=True))
    out[sl_out] = src[sl_src].astype(np.float32, copy=False)
    return out


_ZSCORE_CLIP = 10.0  # std devs; real brain signal sits well within ±6


def _zscore(x: np.ndarray) -> np.ndarray:
    # Source volumes can carry NaN/inf voxels (out-of-FOV / masked regions) AND
    # finite-but-huge intensities. nan_to_num handles the former; the latter
    # overflow a float32 mean/variance reduction (a single voxel above ~1.8e19
    # squares past the float32 max), producing an inf/NaN sd and hence NaN
    # z-scores that survive into the target and poison the shared encoder ->
    # NaN loss -> NaN gradients. Reduce in float64 and sanitize the output so
    # neither path can inject a non-finite value downstream.
    #
    # Sanitizing to finite is necessary but not sufficient: an extreme finite
    # voxel z-scores to a huge finite value, which dominates the recon L1 and
    # produces giant per-batch loss spikes (grad-clipping contains them, but the
    # batch is wasted). Clip to ±_ZSCORE_CLIP std so pathological outlier voxels
    # can't swamp the loss; genuine anatomy is unaffected at 10 std.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    nonzero = x[x > 0].astype(np.float64)
    if nonzero.size < 100:
        return x
    mu = nonzero.mean()
    sd = nonzero.std() + 1e-6
    out = (x.astype(np.float64) - mu) / sd
    out = np.clip(out, -_ZSCORE_CLIP, _ZSCORE_CLIP)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class ZarrMultimodalDataset(Dataset[dict[str, Any]]):
    """Yield one subject per index: stack present modalities into (M, D, H, W)."""

    def __init__(
        self,
        groups: Sequence[SubjectGroup],
        modalities: Sequence[str],
        image_size: int,
        modality_dropout_p: float = 0.3,
        synth_dropout_p: float = 0.0,
        train: bool = True,
        seed: int = 1337,
    ) -> None:
        self.groups = list(groups)
        self.modalities = list(modalities)
        self.image_size = image_size
        self.modality_dropout_p = modality_dropout_p
        # Probability that a multimodal subject is put in the "synthesis regime":
        # keep exactly ONE input modality and supervise the rest. This trains the
        # 1->N case the cross-modal eval actually scores (feed T1 only, predict
        # T2/FLAIR); plain per-modality dropout at p=0.3 almost always leaves >=2
        # inputs, so the model never practises single-modality synthesis.
        self.synth_dropout_p = synth_dropout_p
        self.train = train
        self._seed = seed

    def __len__(self) -> int:
        return len(self.groups)

    def _load_volume(self, url: str) -> tuple[np.ndarray, dict[str, Any]] | None:
        """Load a zarr store, or return None if it is missing/partial/corrupt.

        A store whose write was interrupted (spot preemption mid-upload) can exist
        without its `data` array; reading it raises KeyError. One such store must
        not crash a dataloader worker (which kills a DDP rank and hangs the whole
        run on the NCCL barrier) — treat it as an absent modality instead.
        """
        try:
            root = zarr.open(url, mode="r")
            return np.asarray(root["data"], dtype=np.float32), dict(root.attrs)
        except (KeyError, ValueError, OSError, RuntimeError):
            return None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.groups[idx]
        rng = random.Random(self._seed + idx if not self.train else None)

        m = len(self.modalities)
        shape = (m, self.image_size, self.image_size, self.image_size)
        # `target` holds every acquired modality at full fidelity; `image` is the
        # encoder input, which gets dropped slots zeroed below. Keeping them
        # separate is what makes modality dropout a cross-modal *synthesis*
        # objective (reconstruct the dropped modality's true volume) instead of
        # silently training the decoder to emit zeros for dropped slots.
        target = np.zeros(shape, dtype=np.float32)
        present_mask = np.zeros(m, dtype=np.float32)
        age = float(
            "nan"
        )  # Phase-2 conditioning hook; read from Zarr attrs if preprocessing wrote it.

        # One crop window per subject: every modality is MNI-registered onto the
        # same grid, so they must be cropped with the IDENTICAL window or the
        # cross-modal pairs are voxel-misaligned (T2 target spatially shifted vs
        # the T1 input) and the 1->N synthesis objective trains on shifted targets.
        # Seed drawn from `rng`, so train still re-randomizes the window per epoch
        # (shared across modalities) while val stays deterministic.
        crop_seed = rng.randrange(2**31)

        for i, modality in enumerate(self.modalities):
            url = group.scans_by_modality.get(modality)
            if url is None:
                continue
            loaded = self._load_volume(url)
            if loaded is None:
                continue  # partial/corrupt store -> treat this modality as absent
            vol, attrs = loaded
            vol = _random_crop_or_pad(vol, self.image_size, random.Random(crop_seed))
            target[i] = _zscore(vol)
            present_mask[i] = 1.0
            if math.isnan(age) and attrs.get("age") not in (None, ""):
                try:
                    age = float(attrs["age"])
                except (TypeError, ValueError):
                    age = float("nan")

        retain_mask = present_mask.copy()
        present_idx = [i for i in range(m) if present_mask[i] == 1.0]
        if self.train and len(present_idx) >= 2 and rng.random() < self.synth_dropout_p:
            # Synthesis regime: keep exactly one modality, supervise the rest.
            keep = rng.choice(present_idx)
            retain_mask[:] = 0.0
            retain_mask[keep] = 1.0
        elif self.train and self.modality_dropout_p > 0:
            for i in range(m):
                if retain_mask[i] == 1.0 and rng.random() < self.modality_dropout_p:
                    retain_mask[i] = 0.0
            if retain_mask.sum() == 0 and present_idx:
                retain_mask[rng.choice(present_idx)] = 1.0

        image = target.copy()
        for i in range(m):
            if retain_mask[i] == 0.0:
                image[i] = 0.0

        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "modality_mask": torch.from_numpy(retain_mask),
            "present_mask": torch.from_numpy(present_mask),
            "age": torch.tensor(age, dtype=torch.float32),
            "cohort": group.cohort,
            "subject": group.subject,
            "session": group.session or "",
        }


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "image": torch.stack([s["image"] for s in samples]),
        "target": torch.stack([s["target"] for s in samples]),
        "modality_mask": torch.stack([s["modality_mask"] for s in samples]),
        "present_mask": torch.stack([s["present_mask"] for s in samples]),
        "age": torch.stack([s["age"] for s in samples]),
        "cohort": [s["cohort"] for s in samples],
        "subject": [s["subject"] for s in samples],
        "session": [s["session"] for s in samples],
    }
    return batch


class ZarrMultimodalDataModule(L.LightningDataModule):
    """v0 multimodal datamodule. Walks `${zarr_root}/${cohort}/` for every cohort."""

    def __init__(
        self,
        zarr_root: str,
        cohorts: Sequence[str],
        modalities: Sequence[str] = ("T1w", "T2w", "PDw", "dwi"),
        image_size: int = 128,
        batch_size: int = 2,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        modality_dropout_p: float = 0.3,
        synth_dropout_p: float = 0.0,
        multimodal_oversample: float = 1.0,
        seed: int = 1337,
        **_: Any,
    ) -> None:
        super().__init__()
        self.zarr_root = zarr_root
        self.cohorts = list(cohorts)
        self.modalities = list(modalities)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.modality_dropout_p = modality_dropout_p
        self.synth_dropout_p = synth_dropout_p
        # Sampling weight for subjects with >=2 acquired modalities. Cross-modal
        # synthesis can only be learned from multimodal subjects, which are a
        # minority of the corpus (most are T1-only); oversampling them keeps the
        # cross-modal gradient from being drowned out. 1.0 = no oversampling.
        self.multimodal_oversample = multimodal_oversample
        self.seed = seed
        self.train_ds: ZarrMultimodalDataset | None = None
        self.val_ds: ZarrMultimodalDataset | None = None

    def _discover(self) -> list[SubjectGroup]:
        refs: list[ScanRef] = []
        for cohort in self.cohorts:
            refs.extend(_list_zarr_stems(self.zarr_root, cohort))
        return _group_by_subject(refs)

    def setup(self, stage: str | None = None) -> None:
        groups = self._discover()
        if not groups:
            raise RuntimeError(
                f"no zarr stores found under {self.zarr_root} for cohorts {self.cohorts}; "
                "run scripts/preprocess.py first"
            )
        rng = random.Random(self.seed)
        rng.shuffle(groups)
        n_val = max(1, int(len(groups) * self.val_fraction))
        val_groups = groups[:n_val]
        train_groups = groups[n_val:]
        self.train_ds = ZarrMultimodalDataset(
            train_groups,
            self.modalities,
            self.image_size,
            self.modality_dropout_p,
            synth_dropout_p=self.synth_dropout_p,
            train=True,
            seed=self.seed,
        )
        self.val_ds = ZarrMultimodalDataset(
            val_groups,
            self.modalities,
            self.image_size,
            modality_dropout_p=0.0,
            synth_dropout_p=0.0,
            train=False,
            seed=self.seed,
        )

    def _multimodal_count(self, group: SubjectGroup) -> int:
        return sum(1 for mod in self.modalities if mod in group.scans_by_modality)

    def _train_sampler(self) -> "torch.utils.data.Sampler[int] | None":
        """WeightedRandomSampler oversampling multimodal subjects, or None."""
        if self.multimodal_oversample == 1.0 or self.train_ds is None:
            return None
        from torch.utils.data import WeightedRandomSampler

        weights = [
            self.multimodal_oversample if self._multimodal_count(g) >= 2 else 1.0
            for g in self.train_ds.groups
        ]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def _loader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "collate_fn": _collate,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        # gcsfs opens a gs:// store via an asyncio loop + background thread that
        # do not survive a fork; forked workers crash on the first zarr.open.
        # Spawned workers start clean and build their own loop per process.
        if self.num_workers > 0:
            kwargs["multiprocessing_context"] = "spawn"
        return kwargs

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        assert self.train_ds is not None
        sampler = self._train_sampler()
        # A sampler and shuffle are mutually exclusive; the weighted sampler
        # already randomizes order.
        shuffle = sampler is None
        return DataLoader(self.train_ds, shuffle=shuffle, sampler=sampler, **self._loader_kwargs())

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        assert self.val_ds is not None
        return DataLoader(self.val_ds, shuffle=False, **self._loader_kwargs())


NiftiDataModule = ZarrMultimodalDataModule
