"""In-process imaging primitives backed by the antspyx / antspynet Python APIs.

Earlier iterations shelled out to ANTs and Freesurfer binaries
(`antsRegistrationSyNQuick.sh`, `mri_synthstrip`, `N4BiasFieldCorrection`).
Those aren't present on a fresh GPU image and `antsRegistrationSyNQuick.sh`
resamples onto the template grid only as a side effect. We now call the
`antspyx` Python API directly: same scientific operations, no external
toolchain, and the registered output lands on the template grid so every
subject shares one shape.

`antspyx` / `antspynet` are Linux/Mac wheels (gated out on Windows in
`pyproject.toml`), so every heavy import is **lazy** — importing this module
stays cheap and dependency-free on Windows, and the test-suite keeps
monkey-patching these functions with `passthrough_copy`.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class CLIError(RuntimeError):
    """Raised when an imaging primitive cannot produce its expected output."""


def ants_register_to_mni(
    input_nifti: Path,
    output_nifti: Path,
    template_nifti: Path,
    *,
    transform: str = "Rigid",
) -> Path:
    """Register `input_nifti` to an MNI152 template via antspyx.

    The moving image is warped into the fixed (template) space, so the output
    is resampled onto the template grid — giving every subject a common shape.
    If `template_nifti` does not exist, fall back to the MNI152 template
    bundled with antspyx (`ants.get_ants_data('mni')`).
    """
    import ants

    template_path = Path(template_nifti)
    if not template_path.exists():
        template_path = Path(ants.get_ants_data("mni"))

    fixed = ants.image_read(str(template_path))
    moving = ants.image_read(str(input_nifti))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=transform)
    warped = reg.get("warpedmovout")
    if warped is None:
        raise CLIError(f"antspyx registration produced no warpedmovout for {input_nifti}")

    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(output_nifti))
    return output_nifti


def synthstrip(input_nifti: Path, output_nifti: Path, *, no_csf: bool = False) -> Path:
    """Brain-extract `input_nifti`.

    Primary path: antspynet deep brain extraction (`modality='t1'`). If
    antspynet/TensorFlow is unavailable or fails, fall back to an antspyx
    Otsu mask so the pipeline degrades instead of dying.
    """
    import ants

    img = ants.image_read(str(input_nifti))
    try:
        from antspynet.utilities import brain_extraction

        prob = brain_extraction(img, modality="t1")
        mask = ants.threshold_image(prob, 0.5, 1.0, 1, 0)
    except Exception:
        mask = ants.get_mask(img)

    brain = ants.mask_image(img, mask)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(brain, str(output_nifti))
    return output_nifti


def n4_bias_correct(input_nifti: Path, output_nifti: Path) -> Path:
    """N4 bias-field correction via antspyx (`ants.n4_bias_field_correction`)."""
    import ants

    img = ants.image_read(str(input_nifti))
    corrected = ants.n4_bias_field_correction(img)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(corrected, str(output_nifti))
    return output_nifti


def passthrough_copy(input_nifti: Path, output_nifti: Path, **_: object) -> Path:
    """No-op stand-in used by tests to short-circuit the imaging primitives."""
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(input_nifti), str(output_nifti))
    return output_nifti
