"""End-to-end smoke test for the preprocessing pipeline.

Uses a 64³ synthetic NIfTI ellipsoid in a tmp_path. External CLIs (ANTs,
SynthStrip) are monkey-patched to passthrough so the test runs on any CPU
without imaging toolchains installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest
from neurodrift.data import cli as ext_cli
from neurodrift.data.bids import iter_bids
from neurodrift.data.io import zarr_to_array
from neurodrift.data.preprocess import PreprocessPipeline

# ----- fixtures -----------------------------------------------------------------


@pytest.fixture
def synth_t1(tmp_path: Path) -> Path:
    """Synthesize a 64³ ellipsoid NIfTI in a BIDS-like layout."""
    bids_root = tmp_path / "bids"
    anat = bids_root / "sub-001" / "ses-01" / "anat"
    anat.mkdir(parents=True)

    grid = np.indices((64, 64, 64)) - 32
    r2 = (grid[0] / 28) ** 2 + (grid[1] / 24) ** 2 + (grid[2] / 20) ** 2
    volume = np.where(r2 < 1, 1.0 - r2, 0.0).astype(np.float32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    out = anat / "sub-001_ses-01_T1w.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), str(out))
    return bids_root


@pytest.fixture
def template(tmp_path: Path) -> Path:
    """A dummy 'MNI template'; passthrough wrappers never actually read it."""
    p = tmp_path / "template.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4)), str(p))
    return p


@pytest.fixture
def stub_external_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every external imaging CLI with a passthrough copy."""

    def passthrough(input_nifti: Path, output_nifti: Path, **_: Any) -> Path:
        return ext_cli.passthrough_copy(input_nifti, output_nifti)

    monkeypatch.setattr(ext_cli, "ants_register_to_mni", passthrough)
    monkeypatch.setattr(ext_cli, "synthstrip", passthrough)
    monkeypatch.setattr(ext_cli, "n4_bias_correct", passthrough)


# ----- tests --------------------------------------------------------------------


def test_iter_bids_finds_t1(synth_t1: Path) -> None:
    scans = list(iter_bids(synth_t1))
    assert len(scans) == 1
    scan = scans[0]
    assert scan.subject == "sub-001"
    assert scan.session == "ses-01"
    assert scan.modality == "T1w"
    assert scan.path.name == "sub-001_ses-01_T1w.nii.gz"


def test_iter_bids_grouped_layout(tmp_path: Path) -> None:
    """Raw mirrors land as <cohort>/<site|dataset>/sub-XXX/[ses-YY/]anat/...

    iter_bids must find scans below an extra grouping level and ignore the
    flat `.done.*` sentinels (which end in .nii.gz but live at the root).
    """
    root = tmp_path / "raw" / "abide"
    vol = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))

    # site/sub/anat (no session) — ABIDE shape
    a = root / "CMU_a" / "sub-0050642" / "anat"
    a.mkdir(parents=True)
    nib.save(vol, str(a / "sub-0050642_T1w.nii.gz"))

    # dataset/sub/ses/anat — OpenNeuro shape
    b = root / "ds000221" / "sub-010002" / "ses-01" / "anat"
    b.mkdir(parents=True)
    nib.save(vol, str(b / "sub-010002_ses-01_T2w.nii.gz"))

    # flat sentinel at the root: ends in .nii.gz but must be ignored
    (root / ".done.abide_CMU_a_sub-0050642_anat_sub-0050642_T1w.nii.gz").touch()

    scans = sorted(iter_bids(root), key=lambda s: s.stem)
    assert [(s.subject, s.session, s.modality) for s in scans] == [
        ("sub-0050642", None, "T1w"),
        ("sub-010002", "ses-01", "T2w"),
    ]


def test_pipeline_end_to_end(
    synth_t1: Path, template: Path, tmp_path: Path, stub_external_cli: None
) -> None:
    """Pipeline produces a Zarr that roundtrips back to the input shape."""
    work = tmp_path / "work"
    pipeline = PreprocessPipeline.default(work_dir=work, template=template)

    scans = list(iter_bids(synth_t1))
    out_paths = [pipeline.run(scan) for scan in scans]

    assert len(out_paths) == 1
    zarr_out = out_paths[0]
    assert zarr_out.suffix == ".zarr"
    assert zarr_out.exists()

    data, affine, attrs = zarr_to_array(zarr_out)
    assert data.shape == (64, 64, 64)
    assert affine.shape == (4, 4)
    assert attrs["subject"] == "sub-001"
    assert attrs["modality"] == "T1w"


def test_pipeline_idempotent(
    synth_t1: Path, template: Path, tmp_path: Path, stub_external_cli: None
) -> None:
    """A second run is a no-op: each step short-circuits on its output sentinel."""
    work = tmp_path / "work"
    pipeline = PreprocessPipeline.default(work_dir=work, template=template)
    scan = next(iter(iter_bids(synth_t1)))

    first = pipeline.run(scan)
    first_mtimes = {p: p.stat().st_mtime_ns for p in _all_intermediates(work)}

    pipeline.run(scan)
    second_mtimes = {p: p.stat().st_mtime_ns for p in _all_intermediates(work)}

    assert first.exists()
    assert first_mtimes == second_mtimes, "intermediate files were rewritten on second run"


def test_skullstrip_uses_scan_modality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The skull-strip step must pass the scan's modality to brain extraction.

    Regression for a silent quality bug: brain_extraction was hardcoded to the
    T1 model, so T2w/FLAIR volumes were masked with the wrong contrast model.
    """
    from neurodrift.data.bids import Scan
    from neurodrift.data.preprocess import RegisterStep, SkullStripStep

    seen: set[str] = set()

    def capture(input_nifti: Path, output_nifti: Path, **kw: Any) -> Path:
        seen.add(kw.get("modality", "MISSING"))
        return ext_cli.passthrough_copy(input_nifti, output_nifti)

    monkeypatch.setattr(ext_cli, "synthstrip", capture)

    work = tmp_path / "work"
    for mod in ("T1w", "T2w", "FLAIR"):
        src = tmp_path / f"sub-001_{mod}.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), str(src))
        scan = Scan(subject="sub-001", session=None, modality=mod, path=src)  # type: ignore[arg-type]
        # SkullStripStep now requires its predecessor (01_register) to exist
        # rather than silently falling back to the raw scan.
        reg_out = work / RegisterStep.name / f"{scan.stem}.nii.gz"
        reg_out.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), str(reg_out))
        SkullStripStep().run(scan, work)

    assert seen == {"T1w", "T2w", "FLAIR"}, "skull-strip must receive each scan's modality"


def _all_intermediates(work_dir: Path) -> list[Path]:
    return sorted(p for p in work_dir.rglob("*") if p.is_file())


# ----- E7 intra-subject co-registration -----------------------------------------


def _make_scan(tmp_path: Path, subject: str, modality: str, session: str | None = None) -> Any:
    from neurodrift.data.bids import Scan

    p = tmp_path / f"{subject}_{modality}.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), str(p))
    return Scan(subject=subject, session=session, modality=modality, path=p)  # type: ignore[arg-type]


def test_coregister_routes_siblings_through_t1(tmp_path: Path) -> None:
    """T1 registers straight to MNI; every sibling co-registers THROUGH the subject T1."""
    from neurodrift.data.preprocess import coregister_subject_group

    work = tmp_path / "work"
    scans = [
        _make_scan(tmp_path, "sub-001", "T1w"),
        _make_scan(tmp_path, "sub-001", "T2w"),
        _make_scan(tmp_path, "sub-001", "FLAIR"),
    ]
    reg_calls: list[Path] = []
    coreg_calls: list[tuple[Path, Path]] = []  # (input, t1 reference)

    def reg_stub(inp: Path, out: Path, **kw: Any) -> Path:
        reg_calls.append(Path(inp))
        return ext_cli.passthrough_copy(inp, out)

    def coreg_stub(inp: Path, out: Path, **kw: Any) -> Path:
        coreg_calls.append((Path(inp), Path(kw["t1_nifti"])))
        return ext_cli.passthrough_copy(inp, out)

    outputs = coregister_subject_group(
        scans,
        work,
        template=tmp_path / "tmpl.nii.gz",
        register_fn=reg_stub,
        coregister_fn=coreg_stub,
    )
    t1_path = scans[0].path
    assert reg_calls == [t1_path], "only the T1 goes straight to MNI"
    assert {c[0] for c in coreg_calls} == {scans[1].path, scans[2].path}
    assert all(ref == t1_path for _, ref in coreg_calls), "siblings must anchor to the subject T1"
    assert set(outputs) == {"T1w", "T2w", "FLAIR"}
    assert all(p.exists() for p in outputs.values())


def test_coregister_falls_back_to_mni_without_t1(tmp_path: Path) -> None:
    """A session with no T1 cannot anchor — every modality goes straight to MNI, none dropped."""
    from neurodrift.data.preprocess import coregister_subject_group

    work = tmp_path / "work"
    scans = [_make_scan(tmp_path, "sub-002", "T2w"), _make_scan(tmp_path, "sub-002", "FLAIR")]
    reg_calls: list[Path] = []
    coreg_calls: list[Path] = []

    def reg_stub(inp: Path, out: Path, **kw: Any) -> Path:
        reg_calls.append(Path(inp))
        return ext_cli.passthrough_copy(inp, out)

    def coreg_stub(inp: Path, out: Path, **kw: Any) -> Path:
        coreg_calls.append(Path(inp))
        return ext_cli.passthrough_copy(inp, out)

    coregister_subject_group(
        scans, work, template=tmp_path / "t.nii.gz", register_fn=reg_stub, coregister_fn=coreg_stub
    )
    assert set(reg_calls) == {scans[0].path, scans[1].path}, "no T1 -> all straight to MNI"
    assert coreg_calls == [], "no T1 -> nothing to co-register through"


def test_coregister_is_idempotent(tmp_path: Path) -> None:
    """A modality whose 01_register output already exists must be skipped."""
    from neurodrift.data.preprocess import RegisterStep, coregister_subject_group

    work = tmp_path / "work"
    scans = [_make_scan(tmp_path, "sub-003", "T1w"), _make_scan(tmp_path, "sub-003", "T2w")]
    t2_out = work / RegisterStep.name / f"{scans[1].stem}.nii.gz"
    t2_out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), str(t2_out))

    calls: list[tuple[str, Path]] = []

    def reg_stub(inp: Path, out: Path, **kw: Any) -> Path:
        calls.append(("reg", Path(inp)))
        return ext_cli.passthrough_copy(inp, out)

    def coreg_stub(inp: Path, out: Path, **kw: Any) -> Path:
        calls.append(("coreg", Path(inp)))
        return ext_cli.passthrough_copy(inp, out)

    coregister_subject_group(
        scans, work, template=tmp_path / "t.nii.gz", register_fn=reg_stub, coregister_fn=coreg_stub
    )
    assert ("reg", scans[0].path) in calls, "the T1 must still be registered"
    assert ("coreg", scans[1].path) not in calls, "a pre-existing output must be skipped"
