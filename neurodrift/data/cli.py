"""Thin subprocess wrappers around the external imaging CLIs.

Each wrapper is a single function with a typed signature. Tests monkey-patch
these functions to passthrough-copy the input so the pipeline can run end-to-end
on a CPU laptop with no Freesurfer / ANTs installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class CLIError(RuntimeError):
    """Raised when an external preprocessing CLI exits non-zero."""


def synthstrip(input_nifti: Path, output_nifti: Path, *, no_csf: bool = False) -> Path:
    """Skull-strip with Freesurfer's SynthStrip (Hoopes et al. 2022).

    Assumes `mri_synthstrip` is on PATH (Freesurfer 7.4+).
    """
    cmd = ["mri_synthstrip", "-i", str(input_nifti), "-o", str(output_nifti)]
    if no_csf:
        cmd.append("--no-csf")
    _run(cmd)
    return output_nifti


def ants_register_to_mni(
    input_nifti: Path,
    output_nifti: Path,
    template_nifti: Path,
    *,
    transform: str = "Rigid",
) -> Path:
    """Affine / rigid registration to MNI152 1mm via ANTs `antsRegistrationSyNQuick.sh`."""
    cmd = [
        "antsRegistrationSyNQuick.sh",
        "-d",
        "3",
        "-f",
        str(template_nifti),
        "-m",
        str(input_nifti),
        "-o",
        str(output_nifti.with_suffix("")) + "_",
        "-t",
        transform[0].lower(),
    ]
    _run(cmd)
    warped = output_nifti.with_suffix("").with_name(output_nifti.stem + "_Warped.nii.gz")
    if not warped.exists():
        raise CLIError(f"ANTs did not produce expected output: {warped}")
    warped.rename(output_nifti)
    return output_nifti


def n4_bias_correct(input_nifti: Path, output_nifti: Path) -> Path:
    """N4 bias-field correction (ANTs `N4BiasFieldCorrection`)."""
    cmd = [
        "N4BiasFieldCorrection",
        "-d",
        "3",
        "-i",
        str(input_nifti),
        "-o",
        str(output_nifti),
    ]
    _run(cmd)
    return output_nifti


def _run(cmd: list[str]) -> None:
    """Execute `cmd`, raise `CLIError` on non-zero exit."""
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise CLIError(f"Executable not found: {cmd[0]}") from exc
    if result.returncode != 0:
        raise CLIError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}\nstderr:\n{result.stderr}"
        )


def passthrough_copy(input_nifti: Path, output_nifti: Path, **_: object) -> Path:
    """No-op stand-in used by tests to short-circuit external CLIs."""
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(input_nifti), str(output_nifti))
    return output_nifti
