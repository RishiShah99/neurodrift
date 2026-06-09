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


def _all_intermediates(work_dir: Path) -> list[Path]:
    return sorted(p for p in work_dir.rglob("*") if p.is_file())
