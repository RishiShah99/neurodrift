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


@pytest.fixture
def multisession_zarr_root(tmp_path: Path) -> Path:
    """10 subjects, each with TWO sessions (T1w+T2w) — the repeat-session case."""
    cohort = tmp_path / "openneuro"
    cohort.mkdir()
    for i in range(10):
        subj = f"sub-{i:03d}"
        for sess in ("ses-01", "ses-02"):
            for mod in ("T1w", "T2w"):
                _write_zarr(
                    cohort / f"{subj}_{sess}_{mod}.zarr",
                    shape=(40, 40, 40),
                    attrs={"subject": subj, "modality": mod},
                )
    return tmp_path


def _val_subjects(dm: ZarrMultimodalDataModule) -> set[tuple[str, str]]:
    assert dm.val_ds is not None
    return {(g.cohort, g.subject) for g in dm.val_ds.groups}


def _train_subjects(dm: ZarrMultimodalDataModule) -> set[tuple[str, str]]:
    assert dm.train_ds is not None
    return {(g.cohort, g.subject) for g in dm.train_ds.groups}


def test_split_keeps_all_sessions_of_a_subject_on_one_side(
    multisession_zarr_root: Path,
) -> None:
    """No subject may appear in BOTH train and val (cross-session leakage).

    Regression for an inflated cross-modal dB: keying the split on
    (cohort, subject, session) put one session of a repeat-scanned subject in train
    and another in val — same anatomy, same scanner — so val scored partly on
    training subjects. The split must be on (cohort, subject).
    """
    dm = ZarrMultimodalDataModule(
        zarr_root=str(multisession_zarr_root),
        cohorts=["openneuro"],
        modalities=["T1w", "T2w", "PDw", "dwi"],
        image_size=32,
        batch_size=2,
        num_workers=0,
        val_fraction=0.3,
        modality_dropout_p=0.0,
    )
    dm.setup()
    val_subj, train_subj = _val_subjects(dm), _train_subjects(dm)
    assert val_subj and train_subj, "both splits must be non-empty"
    assert val_subj.isdisjoint(train_subj), "a subject leaked across train/val"
    # every subject's BOTH sessions must be present on its assigned side
    assert dm.val_ds is not None
    val_sessions = [(g.subject, g.session) for g in dm.val_ds.groups]
    for subj in {s for _, s in val_subj}:
        assert sum(1 for s, _ in val_sessions if s == subj) == 2


def test_val_split_is_deterministic_across_instances(
    multisession_zarr_root: Path,
) -> None:
    """Same seed + config must yield the IDENTICAL val membership every time.

    Regression for a non-reproducible split: the seeded shuffle ran over an
    unordered fs.ls() listing, so eval.py could rebuild a different val set than
    training used. Discovery now sorts to a canonical order first.
    """

    def build() -> ZarrMultimodalDataModule:
        dm = ZarrMultimodalDataModule(
            zarr_root=str(multisession_zarr_root),
            cohorts=["openneuro"],
            modalities=["T1w", "T2w", "PDw", "dwi"],
            image_size=32,
            batch_size=2,
            num_workers=0,
            val_fraction=0.3,
            seed=1337,
        )
        dm.setup()
        return dm

    assert _val_subjects(build()) == _val_subjects(build())


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
    ramp = np.arange(40 * 40 * 40, dtype=np.float32).reshape(40, 40, 40) + 1.0
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

    # weights=None: this test only checks the gradient path, and the random VGG keeps
    # CI offline (no 550 MB ImageNet download).
    perc = _TriOrthoVGGPerceptual(weights=None)
    recon = torch.randn(1, 3, 16, 16, 16, requires_grad=True)
    target = torch.randn(1, 3, 16, 16, 16)
    loss = perc(recon, target)
    loss.backward()
    assert recon.grad is not None and recon.grad.abs().sum().item() > 0.0, (
        "perceptual loss produced no gradient on recon"
    )


def test_perceptual_defaults_to_pretrained_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """The perceptual loss must request PRETRAINED VGG weights by default.

    Regression for the silent quality bug where vgg19(weights=None) shipped a
    randomly-initialized 'perceptual' loss (~a random projection). Asserted via a
    monkeypatched vgg19 so the test needs no network.
    """
    pytest.importorskip("torchvision")
    import torchvision
    from neurodrift.train.lightning_module import _TriOrthoVGGPerceptual

    seen: dict[str, object] = {}
    real_vgg19 = torchvision.models.vgg19

    def spy_vgg19(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen["weights"] = kwargs.get("weights", "MISSING")
        return real_vgg19(weights=None)  # never download in the test

    monkeypatch.setattr(torchvision.models, "vgg19", spy_vgg19)
    _TriOrthoVGGPerceptual()  # default weights
    assert seen["weights"] == "DEFAULT", "perceptual VGG must default to pretrained weights"


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


def test_zscore_rezeros_background() -> None:
    """Background (outside the brain) must stay exactly 0 after z-scoring.

    Regression for a nonzero background plane: applying (x-mu)/sd to every voxel
    maps background 0 -> -mu/sd, so a present slot's background disagreed with a
    dropped/absent slot's 0 and widened the per-volume PSNR data_range. _zscore now
    masks to the brain (x>0) and re-zeros background.
    """
    from neurodrift.train.data_module import _zscore

    vol = np.zeros((40, 40, 40), dtype=np.float32)
    brain = (np.abs(np.random.RandomState(1).randn(20, 20, 20)).astype(np.float32)) + 1.0
    vol[10:30, 10:30, 10:30] = brain
    out = _zscore(vol)
    bg = out.copy()
    bg[10:30, 10:30, 10:30] = 0.0
    assert np.all(bg == 0.0), "background must remain exactly zero after z-score"
    assert np.abs(out[10:30, 10:30, 10:30]).sum() > 0.0, (
        "brain region must be normalized, not zeroed"
    )


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
