"""Local-Zarr smoke test for the v0 multimodal DataModule.

Writes a couple of synthetic Zarr stores in BIDS-style stems and verifies
discovery, grouping, batch dict shape, and modality-mask semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from neurodrift.train.data_module import ZarrMultimodalDataModule  # noqa: E402


def _write_zarr(path: Path, shape: tuple[int, int, int], attrs: dict[str, str]) -> None:
    root = zarr.open(str(path), mode="w")
    arr = root.create_dataset("data", shape=shape, dtype=np.float32, overwrite=True)
    arr[...] = np.random.RandomState(0).randn(*shape).astype(np.float32)
    root.create_dataset("affine", shape=(4, 4), dtype=np.float64, overwrite=True)[...] = np.eye(4)
    for k, v in attrs.items():
        root.attrs[k] = v


@pytest.fixture
def fake_zarr_root(tmp_path: Path) -> Path:
    cohort = tmp_path / "ixi"
    cohort.mkdir()
    for subj in ("sub-001", "sub-002"):
        for mod in ("T1w", "T2w"):
            _write_zarr(
                cohort / f"{subj}_ses-01_{mod}.zarr",
                shape=(40, 40, 40),
                attrs={"subject": subj, "modality": mod},
            )
    return tmp_path


def test_discover_groups_modalities_by_subject(fake_zarr_root: Path) -> None:
    dm = ZarrMultimodalDataModule(
        zarr_root=str(fake_zarr_root),
        cohorts=["ixi"],
        modalities=["T1w", "T2w", "PDw", "dwi"],
        image_size=32,
        batch_size=2,
        num_workers=0,
        val_fraction=0.5,
        modality_dropout_p=0.0,
    )
    dm.setup()
    assert dm.train_ds is not None and dm.val_ds is not None
    assert len(dm.train_ds) + len(dm.val_ds) == 2


def test_batch_shape_and_mask(fake_zarr_root: Path) -> None:
    dm = ZarrMultimodalDataModule(
        zarr_root=str(fake_zarr_root),
        cohorts=["ixi"],
        modalities=["T1w", "T2w", "PDw", "dwi"],
        image_size=32,
        batch_size=1,
        num_workers=0,
        val_fraction=0.5,
        modality_dropout_p=0.0,
    )
    dm.setup()
    loader = dm.val_dataloader()
    batch = next(iter(loader))
    assert batch["image"].shape == (1, 4, 32, 32, 32)
    assert batch["modality_mask"].shape == (1, 4)
    # T1w + T2w present, PDw + dwi missing.
    assert batch["present_mask"][0, 0].item() == 1.0
    assert batch["present_mask"][0, 1].item() == 1.0
    assert batch["present_mask"][0, 2].item() == 0.0
    assert batch["present_mask"][0, 3].item() == 0.0
