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


def test_dropout_preserves_clean_target(fake_zarr_root: Path) -> None:
    """Modality dropout must zero the INPUT slot but keep the TARGET intact.

    Regression for the reconstruct-against-zero bug: if a dropped-but-acquired
    modality had its target zeroed, the masked-L1 (driven by present_mask) would
    train the decoder to emit zeros for dropped modalities, destroying cross-modal
    synthesis — and invisibly, since validation runs with dropout_p=0.
    """
    from neurodrift.train.data_module import ZarrMultimodalDataModule

    dm = ZarrMultimodalDataModule(
        zarr_root=str(fake_zarr_root),
        cohorts=["ixi"],
        modalities=["T1w", "T2w", "PDw", "dwi"],
        image_size=32,
        batch_size=2,
        num_workers=0,
        val_fraction=0.5,
        modality_dropout_p=1.0,  # drop as aggressively as the keep-one guard allows
    )
    dm.setup()
    ds = dm.train_ds
    assert ds is not None
    saw_dropped_but_present = False
    for idx in range(len(ds)):
        s = ds[idx]
        image, target = s["image"], s["target"]
        present, retain = s["present_mask"], s["modality_mask"]
        for j in range(present.shape[0]):
            if present[j] == 1.0 and retain[j] == 0.0:
                saw_dropped_but_present = True
                assert image[j].abs().sum() == 0.0, "dropped modality must be zeroed in the INPUT"
                assert target[j].abs().sum() > 0.0, "acquired modality must stay intact in TARGET"
            if retain[j] == 1.0:
                assert torch.equal(image[j], target[j]), "kept slot: input must equal target"
    assert saw_dropped_but_present, "test setup never produced a dropped-but-acquired modality"


def test_synth_dropout_keeps_single_input_modality(fake_zarr_root: Path) -> None:
    """synth_dropout_p=1.0 must leave exactly one input modality for multimodal subjects.

    This is what trains the 1->N cross-modal case the eval scores; without it the
    model only ever sees >=2 inputs and never learns single-modality synthesis.
    """
    from neurodrift.train.data_module import ZarrMultimodalDataModule

    dm = ZarrMultimodalDataModule(
        zarr_root=str(fake_zarr_root),
        cohorts=["ixi"],
        modalities=["T1w", "T2w", "PDw", "dwi"],
        image_size=32,
        batch_size=2,
        num_workers=0,
        val_fraction=0.5,
        modality_dropout_p=0.3,
        synth_dropout_p=1.0,  # always enter synthesis regime
    )
    dm.setup()
    ds = dm.train_ds
    assert ds is not None
    for idx in range(len(ds)):
        s = ds[idx]
        present, retain = s["present_mask"], s["modality_mask"]
        if present.sum() >= 2:  # both T1w + T2w acquired in the fixture
            assert retain.sum().item() == 1.0, "synthesis regime must keep exactly one input"
            # the kept modality must be one that was actually acquired
            kept = int(retain.argmax())
            assert present[kept] == 1.0
            assert s["target"][kept].abs().sum() > 0.0, "held-out modalities stay in target"


def test_corrupt_zarr_store_treated_as_absent(tmp_path: Path) -> None:
    """A partial/corrupt store (no `data` array) must not crash the worker.

    Regression for a DDP-killing crash: a write interrupted mid-upload leaves a
    zarr group without its `data` array; reading raised KeyError in a dataloader
    worker, killing a rank and hanging the run on the NCCL barrier. The store
    must instead be treated as an absent modality.
    """
    from neurodrift.train.data_module import SubjectGroup, ZarrMultimodalDataset

    good = tmp_path / "sub-001_T1w.zarr"
    r = zarr.open(str(good), mode="w")
    # non-constant volume so the z-score isn't degenerate (constant -> std 0 -> all 0)
    ramp = (np.arange(40 * 40 * 40, dtype=np.float32).reshape(40, 40, 40) + 1.0)
    r.create_dataset("data", shape=ramp.shape, dtype=np.float32, overwrite=True)[...] = ramp
    bad = tmp_path / "sub-001_T2w.zarr"
    zarr.open(str(bad), mode="w")  # group with NO `data` array — the partial-store case

    grp = SubjectGroup(
        cohort="c",
        subject="sub-001",
        session=None,
        scans_by_modality={"T1w": str(good), "T2w": str(bad)},
    )
    ds = ZarrMultimodalDataset([grp], ["T1w", "T2w", "PDw", "dwi"], image_size=32, train=False)
    s = ds[0]
    assert s["present_mask"][0].item() == 1.0, "good T1w store must load"
    assert s["present_mask"][1].item() == 0.0, "corrupt T2w store must be treated as absent"
    assert s["target"][0].abs().sum() > 0.0


def test_crop_window_shared_across_modalities(fake_zarr_root: Path) -> None:
    """Every modality of a subject must be cropped with the IDENTICAL window.

    Regression for cross-modal voxel-misalignment: the fixture writes the same
    RandomState(0) volume for both T1w and T2w, so an aligned crop yields equal
    target slots. A per-modality crop offset (the bug) would shift them apart and
    train the 1->N synthesis objective on misaligned pairs. Uses train_ds to
    exercise the train-mode (seed=None) crop path.
    """
    from neurodrift.train.data_module import ZarrMultimodalDataModule

    dm = ZarrMultimodalDataModule(
        zarr_root=str(fake_zarr_root),
        cohorts=["ixi"],
        modalities=["T1w", "T2w", "PDw", "dwi"],
        image_size=32,  # < the 40^3 fixture volume, so a real crop happens
        batch_size=2,
        num_workers=0,
        val_fraction=0.5,
        modality_dropout_p=0.0,
    )
    dm.setup()
    ds = dm.train_ds
    assert ds is not None
    checked = False
    for idx in range(len(ds)):
        s = ds[idx]
        present, target = s["present_mask"], s["target"]
        if present[0] == 1.0 and present[1] == 1.0:
            assert torch.equal(target[0], target[1]), (
                "identical-content modalities must share one crop window (aligned)"
            )
            checked = True
    assert checked, "fixture never produced a subject with both T1w and T2w"


def test_perceptual_loss_carries_gradient() -> None:
    """The perceptual term must backprop into recon (it was dead under no_grad)."""
    pytest.importorskip("torchvision")
    from neurodrift.train.lightning_module import _TriOrthoVGGPerceptual

    perc = _TriOrthoVGGPerceptual()
    recon = torch.randn(1, 3, 16, 16, 16, requires_grad=True)
    target = torch.randn(1, 3, 16, 16, 16)
    loss = perc(recon, target)
    loss.backward()
    assert recon.grad is not None and recon.grad.abs().sum().item() > 0.0, (
        "perceptual loss produced no gradient on recon"
    )


def test_target_drives_masked_l1_for_dropped_modality() -> None:
    """A dropped-but-acquired modality must yield a real (nonzero-target) recon loss.

    With the old behaviour the target slot was zero, so a decoder that emitted
    zeros would score a perfect loss on dropped modalities. Here the masked-L1 is
    computed against the clean target, so emitting zeros is penalised.
    """
    from neurodrift.train.lightning_module import _masked_l1

    b, m, d = 1, 2, 8
    recon = torch.zeros(b, m, d, d, d)  # decoder that emits zeros everywhere
    target = torch.ones(b, m, d, d, d)  # both modalities acquired, nonzero truth
    present = torch.ones(b, m)
    loss = _masked_l1(recon, target, present)
    assert loss.item() == pytest.approx(1.0), "zeros-vs-clean-target must incur real L1"


def test_zscore_sanitizes_overflow_and_nonfinite() -> None:
    """_zscore must never emit a non-finite value, even for pathological volumes.

    Regression for a silent NaN-injection path: nan_to_num removed inf/NaN voxels
    but the float32 mean/variance reduction still overflowed on finite-but-huge
    intensities (a single voxel > ~1.8e19 squares past the float32 max -> inf/NaN
    sd -> NaN z-scores), which then flowed into the loss and killed the encoder.
    """
    from neurodrift.train.data_module import _zscore

    base = np.abs(np.random.RandomState(0).randn(40, 40, 40)).astype(np.float32) + 1.0
    # one finite voxel large enough to overflow a float32 sum-of-squares
    spike = base.copy()
    spike[0, 0, 0] = 1e25
    # and a volume carrying inf/NaN voxels
    dirty = base.copy()
    dirty[1, 1, 1] = np.inf
    dirty[2, 2, 2] = np.nan
    for vol in (spike, dirty):
        out = _zscore(vol)
        assert np.isfinite(out).all(), "z-scored volume must be entirely finite"
        assert out.dtype == np.float32
        # extreme finite voxels must be clipped, not just finite, so a single
        # outlier can't dominate the recon L1 and spike the per-batch loss
        assert np.abs(out).max() <= 10.0 + 1e-3, "z-scored output must be clipped to ±10 std"


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
