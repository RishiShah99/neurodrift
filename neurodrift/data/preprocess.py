"""Preprocessing pipeline scaffold.

Each `Step` is idempotent: if the output exists, it is a no-op. The pipeline
runs steps in declared order; each step reads its predecessor's output from that
step's well-known work-dir location (raising if it's missing), and the last step
emits the final Zarr cache.

External CLIs (ANTs, SynthStrip) are reached via `neurodrift.data.cli`. Tests
monkey-patch those functions with `passthrough_copy` so the whole pipeline can
run on a CI box without imaging toolchains installed.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from neurodrift.data import cli
from neurodrift.data.bids import Scan
from neurodrift.data.io import nifti_to_zarr

CLIFn = Callable[..., Path]


def _require_prev(work_dir: Path, prev_step: str, scan: Scan) -> Path:
    """Resolve a predecessor step's output, raising if it's missing.

    Steps must NOT silently fall back to the raw scan when an upstream output is
    absent — that emits an unregistered/unstripped volume into the corpus that
    looks valid. If the declared predecessor ran, its output must exist.
    """
    prev = work_dir / prev_step / f"{scan.stem}.nii.gz"
    if not prev.exists():
        raise cli.CLIError(
            f"{prev_step} output missing for {scan.stem}; refusing to fall back to the "
            "raw scan (would inject an unprocessed volume into the corpus)"
        )
    return prev


@dataclass
class Step(ABC):
    """Abstract idempotent preprocessing step. Subclasses set `name` as a ClassVar."""

    name: ClassVar[str] = "step"

    def run(self, scan: Scan, work_dir: Path) -> Path:
        """Run the step, skipping if its output sentinel already exists."""
        out = self.output_path(scan, work_dir)
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        return self._do(scan, work_dir, out)

    def output_path(self, scan: Scan, work_dir: Path) -> Path:
        return work_dir / self.name / f"{scan.stem}.nii.gz"

    @abstractmethod
    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path: ...


@dataclass
class RegisterStep(Step):
    """Affine / rigid registration to MNI152 1mm.

    `cli_fn=None` resolves to `cli.ants_register_to_mni` at call time, so tests
    that monkey-patch `cli.ants_register_to_mni` after construction are honoured.
    """

    name: ClassVar[str] = "01_register"
    template: Path  # required: a default of Path('.') would silently register to cwd
    cli_fn: CLIFn | None = None
    transform: str = "Rigid"

    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path:
        fn = self.cli_fn or cli.ants_register_to_mni
        return fn(scan.path, out, template_nifti=self.template, transform=self.transform)


@dataclass
class SkullStripStep(Step):
    """SynthStrip skull strip."""

    name: ClassVar[str] = "02_skullstrip"
    cli_fn: CLIFn | None = None

    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path:
        fn = self.cli_fn or cli.synthstrip
        src = _require_prev(work_dir, RegisterStep.name, scan)
        # Pass the scan modality so brain extraction uses the contrast-matched
        # model (t1/t2/flair) instead of always assuming T1.
        return fn(src, out, modality=scan.modality)


@dataclass
class N4Step(Step):
    """N4 bias-field correction."""

    name: ClassVar[str] = "03_n4"
    cli_fn: CLIFn | None = None

    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path:
        fn = self.cli_fn or cli.n4_bias_correct
        src = _require_prev(work_dir, SkullStripStep.name, scan)
        return fn(src, out)


@dataclass
class HarmonizeStep(Step):
    """Cross-site harmonization (NeuroHarmonize / COMBAT).

    NeuroHarmonize fits across a cohort, so this step is a placeholder that
    cohorts can override with a fitted transformer. Default impl passes through.
    """

    name: ClassVar[str] = "04_harmonize"
    cli_fn: CLIFn | None = None

    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path:
        fn = self.cli_fn or cli.passthrough_copy
        src = _require_prev(work_dir, N4Step.name, scan)
        return fn(src, out)


@dataclass
class ZarrCacheStep(Step):
    """Write the final NIfTI into a Zarr store for the dataloader."""

    name: ClassVar[str] = "05_zarr"

    def output_path(self, scan: Scan, work_dir: Path) -> Path:
        return work_dir / self.name / f"{scan.stem}.zarr"

    def run(self, scan: Scan, work_dir: Path) -> Path:
        # A .zarr is a directory; a write interrupted (e.g. spot preemption) leaves
        # a partial dir that bare exists() reports as complete, so a re-run would
        # trust a corrupt store. Treat "has the `data` array" as the completeness
        # sentinel; otherwise discard the partial store and rebuild.
        out = self.output_path(scan, work_dir)
        if out.exists():
            if (out / "data").exists():
                return out
            shutil.rmtree(out, ignore_errors=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        return self._do(scan, work_dir, out)

    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path:
        src = _require_prev(work_dir, HarmonizeStep.name, scan)
        return nifti_to_zarr(
            src,
            out,
            attrs={
                "subject": scan.subject,
                "session": scan.session or "",
                "modality": scan.modality,
            },
        )


@dataclass
class PreprocessPipeline:
    """Ordered sequence of preprocessing steps with idempotent execution."""

    steps: list[Step]
    work_dir: Path

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, scan: Scan) -> Path:
        """Run every step on `scan` and return the final Zarr cache path."""
        out: Path = scan.path
        for step in self.steps:
            out = step.run(scan, self.work_dir)
        return out

    @classmethod
    def default(cls, work_dir: Path, template: Path) -> PreprocessPipeline:
        """Default pipeline: register → skull-strip → N4 → harmonize → Zarr."""
        return cls(
            steps=[
                RegisterStep(template=template),
                SkullStripStep(),
                N4Step(),
                HarmonizeStep(),
                ZarrCacheStep(),
            ],
            work_dir=work_dir,
        )
