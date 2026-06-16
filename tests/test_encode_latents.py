"""CPU tests for the latent store + LatentDataModule + age wiring.

Exercises neurodrift/data/latents.py end to end with a tiny DisentangledVAE3D and
tmp zarr stores — no GCS, no GPU, seconds on CPU. scripts/encode_latents.py is not
run against GCS here; its load-bearing pieces (the encode->store path and
participants.tsv parsing) are covered through latents.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")
pytest.importorskip("lightning")

from neurodrift.data.latents import (  # noqa: E402
    LatentDataModule,
    parse_participants_tsv,
    parse_sessions_tsv,
    read_latent_store,
    write_latent_store,
)
from neurodrift.models.vae3d import DisentangledVAE3D  # noqa: E402

MODS = ("T1w", "T2w", "FLAIR")


# ---------------------------------------------------------------------------
# Store roundtrip
# ---------------------------------------------------------------------------
def test_store_roundtrip_preserves_z_and_attrs(tmp_path: Path) -> None:
    """write -> read must preserve z exactly and every attr in the frozen schema."""
    z = np.random.RandomState(0).randn(4, 8, 8, 8).astype(np.float32)
    path = tmp_path / "sub-001_ses-01.zarr"
    write_latent_store(
        path,
        z,
        age=42.5,
        sex=1,
        dx=2,
        apoe=4,
        treatment=1,
        cohort="abide",
        subject="sub-001",
        session="ses-01",
    )
    loaded = read_latent_store(path)
    assert loaded is not None
    z_out, attrs = loaded
    assert z_out.dtype == np.float32
    assert np.allclose(z_out, z)
    assert attrs["age"] == pytest.approx(42.5)
    assert attrs["sex"] == 1
    assert attrs["dx"] == 2
    assert attrs["apoe"] == 4
    assert attrs["treatment"] == 1
    assert attrs["cohort"] == "abide"
    assert attrs["subject"] == "sub-001"
    assert attrs["session"] == "ses-01"


def test_store_roundtrip_from_tensor_and_nan_age(tmp_path: Path) -> None:
    """A torch tensor input and a NaN age must round-trip (NaN preserved, not coerced)."""
    z = torch.randn(2, 4, 4, 4)
    path = tmp_path / "sub-002.zarr"
    write_latent_store(path, z, age=float("nan"), cohort="openneuro", subject="sub-002", session="")
    loaded = read_latent_store(path)
    assert loaded is not None
    z_out, attrs = loaded
    assert np.allclose(z_out, z.numpy())
    assert math.isnan(attrs["age"]), "NaN age must survive the store roundtrip"
    assert attrs["sex"] == -1 and attrs["treatment"] == 0  # schema defaults


def test_read_corrupt_store_returns_none(tmp_path: Path) -> None:
    """A group with no `data` array (interrupted write) must read as None, not crash."""
    import zarr

    bad = tmp_path / "sub-bad.zarr"
    zarr.open(str(bad), mode="w")  # group with NO `data` array
    assert read_latent_store(bad) is None
    assert read_latent_store(tmp_path / "does_not_exist.zarr") is None


# ---------------------------------------------------------------------------
# participants.tsv age wiring
# ---------------------------------------------------------------------------
def test_parse_participants_tsv(tmp_path: Path) -> None:
    """participant_id -> age, with n/a and a bare numeric id handled."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(
        "participant_id\tage\tsex\n"
        "sub-001\t34.0\tM\n"
        "sub-002\tn/a\tF\n"
        "sub-003\t\tM\n"
        "004\t56\tF\n",  # bare id (no sub- prefix) -> normalized
        encoding="utf-8",
    )
    ages = parse_participants_tsv(tsv)
    assert ages["sub-001"] == pytest.approx(34.0)
    assert math.isnan(ages["sub-002"]), "'n/a' age -> NaN"
    assert math.isnan(ages["sub-003"]), "blank age -> NaN"
    assert ages["sub-004"] == pytest.approx(56.0), "bare participant_id must gain sub- prefix"


def test_parse_participants_tsv_alt_age_column(tmp_path: Path) -> None:
    """An 'age_at_scan' column (no plain 'age') must be picked up."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id\tage_at_scan\nsub-001\t71.2\n", encoding="utf-8")
    assert parse_participants_tsv(tsv)["sub-001"] == pytest.approx(71.2)


def test_parse_participants_tsv_missing_and_no_age_column(tmp_path: Path) -> None:
    """Missing file -> {}; a tsv with no age column -> NaN for each subject."""
    assert parse_participants_tsv(tmp_path / "nope.tsv") == {}
    tsv = tmp_path / "participants.tsv"
    tsv.write_text("participant_id\tsex\nsub-001\tM\n", encoding="utf-8")
    ages = parse_participants_tsv(tsv)
    assert set(ages) == {"sub-001"}
    assert math.isnan(ages["sub-001"]), "no age column -> NaN age for every subject"


def test_parse_sessions_tsv(tmp_path: Path) -> None:
    """Optional session-level age parsing: session_id -> age."""
    tsv = tmp_path / "sub-001_sessions.tsv"
    tsv.write_text("session_id\tage\nses-01\t60.0\nses-02\tn/a\n", encoding="utf-8")
    ages = parse_sessions_tsv(tsv)
    assert ages["ses-01"] == pytest.approx(60.0)
    assert math.isnan(ages["ses-02"])
    assert parse_sessions_tsv(tmp_path / "nope.tsv") == {}


# ---------------------------------------------------------------------------
# dummy VAE -> store
# ---------------------------------------------------------------------------
def test_dummy_vae_encode_to_store(tmp_path: Path) -> None:
    """A tiny VAE's encode()[0] latent must write + read back at the right shape.

    This is the encode_latents.py path in miniature: encode(x, mask)[0] is the
    canonical fused content latent; drop the batch dim and store (C, d, d, d).
    """
    model = DisentangledVAE3D(
        modalities=MODS,
        latent_channels=4,
        style_dim=8,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
    )
    model.eval()
    x = torch.randn(1, 3, 16, 16, 16)
    mask = torch.ones(1, 3)
    with torch.no_grad():
        z = model.encode(x, mask)[0][0]  # (4, 4, 4, 4)
    assert z.shape == (4, 4, 4, 4)

    path = tmp_path / "sub-007_ses-01.zarr"
    write_latent_store(path, z, age=50.0, cohort="abide", subject="sub-007", session="ses-01")
    loaded = read_latent_store(path)
    assert loaded is not None
    z_out, attrs = loaded
    assert z_out.shape == (4, 4, 4, 4)
    assert np.allclose(z_out, z.numpy())
    assert attrs["age"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# LatentDataModule
# ---------------------------------------------------------------------------
def _seed_latents(root: Path, cohort: str, c: int = 4, d: int = 8) -> None:
    """Write a handful of tmp latent stores under root/cohort/."""
    (root / cohort).mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(1)
    specs = [
        ("sub-001", "ses-01", 30.0, 1),
        ("sub-002", "ses-01", 45.0, 0),
        ("sub-003", "ses-01", float("nan"), -1),  # NaN age must survive to the batch
        ("sub-004", "ses-01", 60.0, 1),
    ]
    for subject, session, age, sex in specs:
        write_latent_store(
            root / cohort / f"{subject}_{session}.zarr",
            rs.randn(c, d, d, d).astype(np.float32),
            age=age,
            sex=sex,
            cohort=cohort,
            subject=subject,
            session=session,
        )


def test_datamodule_batch_contract(tmp_path: Path) -> None:
    """setup() then one train batch must satisfy the frozen batch dict contract."""
    _seed_latents(tmp_path, "abide", c=4, d=8)
    dm = LatentDataModule(
        latent_root=str(tmp_path),
        cohorts=["abide"],
        batch_size=2,
        num_workers=0,
        val_fraction=0.25,
        seed=1337,
    )
    dm.setup()
    assert dm.train_ds is not None and dm.val_ds is not None
    assert len(dm.train_ds) + len(dm.val_ds) == 4

    batch = next(iter(dm.train_dataloader()))
    b = batch["z"].shape[0]
    assert batch["z"].shape == (b, 4, 8, 8, 8)
    assert batch["age"].shape == (b,)
    assert batch["age"].dtype == torch.float32
    for key in ("sex", "dx", "apoe", "treatment"):
        assert batch[key].shape == (b,)
        assert batch[key].dtype == torch.long
    assert isinstance(batch["cohort"], list) and len(batch["cohort"]) == b
    assert isinstance(batch["subject"], list) and isinstance(batch["session"], list)
    assert all(c == "abide" for c in batch["cohort"])


def test_datamodule_nan_age_reaches_batch(tmp_path: Path) -> None:
    """A NaN-age store must arrive as NaN in the batch (the model imputes downstream)."""
    _seed_latents(tmp_path, "abide", c=4, d=8)
    dm = LatentDataModule(
        latent_root=str(tmp_path),
        cohorts=["abide"],
        batch_size=4,
        num_workers=0,
        val_fraction=0.25,
        seed=1337,
    )
    dm.setup()
    seen_ages = []
    for loader in (dm.train_dataloader(), dm.val_dataloader()):
        for batch in loader:
            seen_ages.extend(batch["age"].tolist())
    assert any(math.isnan(a) for a in seen_ages), "the NaN-age subject must reach a batch as NaN"
    assert any(a == pytest.approx(30.0) for a in seen_ages), "finite ages must pass through"


def test_datamodule_split_is_subject_level_and_deterministic(tmp_path: Path) -> None:
    """Split must be on (cohort, subject) and identical across instances (same seed)."""
    _seed_latents(tmp_path, "abide", c=4, d=8)

    def build() -> LatentDataModule:
        dm = LatentDataModule(
            latent_root=str(tmp_path),
            cohorts=["abide"],
            batch_size=2,
            num_workers=0,
            val_fraction=0.25,
            seed=1337,
        )
        dm.setup()
        return dm

    a, b = build(), build()
    assert a.val_ds is not None and b.val_ds is not None
    val_a = {(r.cohort, r.subject) for r in a.val_ds.refs}
    val_b = {(r.cohort, r.subject) for r in b.val_ds.refs}
    train_a = {(r.cohort, r.subject) for r in a.train_ds.refs}  # type: ignore[union-attr]
    assert val_a == val_b, "same seed must yield identical val membership"
    assert val_a and train_a and val_a.isdisjoint(train_a), "subject must not leak across split"


def test_datamodule_empty_root_raises(tmp_path: Path) -> None:
    """An empty latent root must raise a clear error (pointing at encode_latents)."""
    dm = LatentDataModule(latent_root=str(tmp_path), cohorts=["abide"], num_workers=0)
    with pytest.raises(RuntimeError, match="encode_latents"):
        dm.setup()


# ---------------------------------------------------------------------------
# encode_latents.py — local corpus -> local latents + local age wiring
# ---------------------------------------------------------------------------
def _write_voxel_store(path: Path, vol: np.ndarray) -> None:
    """Write a (D,H,W) voxel zarr store with a `data` array (the corpus schema)."""
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(path), mode="w")
    data = root.create_dataset(
        "data", shape=vol.shape, chunks=vol.shape, dtype=np.float32, overwrite=True
    )
    data[...] = vol.astype(np.float32)


def test_encode_latents_local_end_to_end(tmp_path: Path) -> None:
    """The local path: --zarr-root + --latent-root + --participants-dir, no GCS.

    Drives the real CLI over a tiny voxel corpus through a tiny frozen VAE, and
    checks each subject-session latent lands locally with the recovered age (and
    NaN where the TSV has none). Idempotent re-run must encode nothing new.
    """
    import subprocess
    import sys

    rs = np.random.RandomState(0)
    # Tiny frozen VAE -> bare state-dict ckpt (geometry must match the CLI overrides).
    model = DisentangledVAE3D(
        modalities=MODS,
        latent_channels=4,
        style_dim=8,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
    )
    ckpt = tmp_path / "vae.ckpt"
    torch.save(model.state_dict(), ckpt)

    # Voxel corpus: sub-001 (T1+T2, no session), sub-002 (T1, ses-01), sub-003 (T1, no age).
    corpus = tmp_path / "corpus"
    for stem in ("sub-001_T1w", "sub-001_T2w", "sub-002_ses-01_T1w", "sub-003_T1w"):
        _write_voxel_store(corpus / "openneuro" / f"{stem}.zarr", rs.randn(16, 16, 16))

    # Recovered ages: sub-001 + sub-002 present, sub-003 absent -> NaN downstream.
    pdir = tmp_path / "participants"
    pdir.mkdir()
    (pdir / "openneuro_participants.tsv").write_text(
        "participant_id\tage\nsub-001\t30.0\nsub-002\t55.0\n", encoding="utf-8"
    )

    latents = tmp_path / "latents"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cmd = [
        sys.executable,
        str(Path("scripts/encode_latents.py").resolve()),
        "--cohorts",
        "openneuro",
        "--ckpt",
        str(ckpt),
        "--zarr-root",
        str(corpus),
        "--latent-root",
        str(latents),
        "--participants-dir",
        str(pdir),
        "--device",
        "cpu",
        "--scratch",
        str(scratch),
        "--image-size",
        "16",
        "--latent-channels",
        "4",
        "--style-dim",
        "8",
        "--base-channels",
        "8",
        "--channel-mults",
        "1,2,4",
        "--num-res-blocks",
        "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    # Three subject-session latents, no GCS upload, correct shape + age wiring.
    out = latents / "openneuro"
    stems = {p.name[: -len(".zarr")] for p in out.glob("*.zarr")}
    assert stems == {"sub-001", "sub-002_ses-01", "sub-003"}, stems

    by_stem = {s: read_latent_store(out / f"{s}.zarr") for s in stems}
    for s, loaded in by_stem.items():
        assert loaded is not None, s
        z, _ = loaded
        assert z.shape == (4, 4, 4, 4), (s, z.shape)
    assert by_stem["sub-001"][1]["age"] == pytest.approx(30.0)
    assert by_stem["sub-002_ses-01"][1]["age"] == pytest.approx(55.0)
    assert math.isnan(by_stem["sub-003"][1]["age"]), "missing age -> NaN in the store"

    # Idempotent: a second run finds all three cached and encodes nothing new.
    res2 = subprocess.run(cmd, capture_output=True, text=True)
    assert res2.returncode == 0, res2.stderr
    assert "0 groups to encode (3 cached)" in (res2.stderr + res2.stdout)
