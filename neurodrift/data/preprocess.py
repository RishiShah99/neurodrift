"""Preprocessing pipeline scaffold.

Each `Step` is idempotent: if the output exists, it is a no-op. The pipeline
runs steps in declared order, threading paths from one to the next, and emits
a final Zarr cache.

External CLIs (ANTs, SynthStrip) are reached via `neurodrift.data.cli`. Tests
monkey-patch those functions with `passthrough_copy` so the whole pipeline can
run on a CI box without imaging toolchains installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from neurodrift.data import cli
from neurodrift.data.bids import Scan
from neurodrift.data.io import nifti_to_zarr

CLIFn = Callable[..., Path]


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
    template: Path = field(default_factory=Path)
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
        prev = work_dir / RegisterStep.name / f"{scan.stem}.nii.gz"
        src = prev if prev.exists() else scan.path
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
        prev = work_dir / SkullStripStep.name / f"{scan.stem}.nii.gz"
        src = prev if prev.exists() else scan.path
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
        prev = work_dir / N4Step.name / f"{scan.stem}.nii.gz"
        src = prev if prev.exists() else scan.path
        return fn(src, out)


@dataclass
class ZarrCacheStep(Step):
    """Write the final NIfTI into a Zarr store for the dataloader."""

    name: ClassVar[str] = "05_zarr"

    def output_path(self, scan: Scan, work_dir: Path) -> Path:
        return work_dir / self.name / f"{scan.stem}.zarr"

    def _do(self, scan: Scan, work_dir: Path, out: Path) -> Path:
        prev = work_dir / HarmonizeStep.name / f"{scan.stem}.nii.gz"
        src = prev if prev.exists() else scan.path
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
