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


def ants_coregister_via_t1(
    input_nifti: Path,
    output_nifti: Path,
    t1_nifti: Path,
    template_nifti: Path,
    *,
    transform: str = "Rigid",
) -> Path:
    """Register `input_nifti` to MNI THROUGH the subject's own T1, resampling ONCE (E7).

    Registering each modality to MNI independently leaves small residual T1<->T2
    misalignment per subject — an irreducible cross-modal synthesis floor that
    masquerades as a model cap. Instead: rigid-register `input` to the subject's T1
    (intra-subject), then COMPOSE that with the T1->MNI transform and resample the
    input a SINGLE time through the concatenated transform (composing avoids a double
    interpolation that would re-blur). Result: `input` lands on the MNI template grid
    AND is voxel-aligned to the subject's T1.

    Falls back to the antspyx-bundled MNI template if `template_nifti` is absent.
    """
    import ants

    template_path = Path(template_nifti)
    if not template_path.exists():
        template_path = Path(ants.get_ants_data("mni"))

    template = ants.image_read(str(template_path))
    t1 = ants.image_read(str(t1_nifti))
    moving = ants.image_read(str(input_nifti))

    # T1 -> MNI (lands T1 on the template grid) and input -> T1 (intra-subject), rigid.
    reg_t1 = ants.registration(fixed=template, moving=t1, type_of_transform=transform)
    reg_mod = ants.registration(fixed=t1, moving=moving, type_of_transform=transform)

    # Resample `input` onto the template grid through input->T1->MNI in ONE pass.
    # apply_transforms composes the list LAST-element-first (the antspyx fwdtransforms
    # convention: an [warp, affine] list applies the affine first), so the transform
    # nearest the moving image (input->T1) must come FIRST and the T1->MNI transform
    # LAST. Concretely: a template-grid point is mapped template->T1 (reg_t1, applied
    # first) then T1->input (reg_mod, applied last) to pull the input intensity.
    warped = ants.apply_transforms(
        fixed=template,
        moving=moving,
        transformlist=[*reg_mod["fwdtransforms"], *reg_t1["fwdtransforms"]],
    )
    if warped is None:
        raise CLIError(f"antspyx co-registration produced no output for {input_nifti}")

    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(warped, str(output_nifti))
    return output_nifti


# Map our BIDS modality labels to antspynet brain-extraction model names.
# Using the T1 model on a T2/FLAIR volume yields a wrong brain mask (the
# contrast it was trained on is inverted), which silently degrades exactly the
# non-T1 modalities. antspynet ships dedicated "t2"/"flair" models — use them.
_BRAIN_EXTRACT_MODALITY: dict[str, str] = {"T1w": "t1", "T2w": "t2", "FLAIR": "flair"}


def synthstrip(
    input_nifti: Path, output_nifti: Path, *, modality: str = "T1w", no_csf: bool = False
) -> Path:
    """Brain-extract `input_nifti` with a contrast-matched model.

    Primary path: antspynet deep brain extraction, selecting the model for the
    scan's `modality` (t1/t2/flair). If antspynet/TensorFlow is unavailable or
    fails, fall back to an antspyx Otsu mask so the pipeline degrades instead of
    dying.
    """
    import ants

    img = ants.image_read(str(input_nifti))
    try:
        from antspynet.utilities import brain_extraction

        be_modality = _BRAIN_EXTRACT_MODALITY.get(modality, "t1")
        prob = brain_extraction(img, modality=be_modality)
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
