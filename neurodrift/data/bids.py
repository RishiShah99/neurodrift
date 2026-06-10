"""Minimal BIDS-style dataset traversal.

Full BIDS is a bigger spec than we need. We only iterate `sub-XXX/ses-XXX/anat/*.nii.gz`
and `.../dwi/*.nii.gz` and treat anything else as out-of-scope for Phase 0 ingest.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Modality = Literal["T1w", "T2w", "FLAIR", "dwi", "amyloid_pet", "tau_pet"]


@dataclass(frozen=True)
class Scan:
    """One scan in a BIDS-ish layout."""

    subject: str
    session: str | None
    modality: Modality
    path: Path

    @property
    def stem(self) -> str:
        ses = f"_{self.session}" if self.session else ""
        return f"{self.subject}{ses}_{self.modality}"


_SUFFIX_TO_MODALITY: dict[str, Modality] = {
    "T1w": "T1w",
    "T2w": "T2w",
    "FLAIR": "FLAIR",
    "dwi": "dwi",
}


def iter_bids(root: Path) -> Iterator[Scan]:
    """Yield `Scan` objects for every anat/dwi NIfTI under a BIDS-style root.

    Depth-agnostic: finds every `anat/` or `dwi/` directory at any depth and
    derives subject/session from the nearest `sub-*` / `ses-*` ancestor. This
    tolerates an extra grouping level above `sub-*` — our raw mirrors land as
    `<cohort>/<site|dataset>/sub-XXX/[ses-YY/]{anat,dwi}/*.nii.gz` — as well as
    the canonical `sub-XXX/[ses-YY/]{anat,dwi}/...` tree. Flat sentinel files at
    the root are skipped because their parent is not an `anat`/`dwi` directory.

    Recognized suffixes: `T1w`, `T2w`, `FLAIR`, `dwi`.
    """
    root = Path(root)
    seen: set[Path] = set()
    for kind in ("anat", "dwi"):
        for kind_dir in sorted(root.rglob(kind)):
            if not kind_dir.is_dir() or kind_dir in seen:
                continue
            seen.add(kind_dir)
            subject = _ancestor_token(kind_dir, "sub-")
            if subject is None:
                continue
            session = _ancestor_token(kind_dir, "ses-")
            for nii in sorted(kind_dir.glob("*.nii*")):
                modality = _modality_from_filename(nii.name)
                if modality is None:
                    continue
                yield Scan(subject=subject, session=session, modality=modality, path=nii)


def _ancestor_token(path: Path, prefix: str) -> str | None:
    """Return the nearest ancestor directory name starting with `prefix`."""
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name
    return None


def _modality_from_filename(name: str) -> Modality | None:
    """Pull the BIDS suffix off the filename before the `.nii(.gz)` extension."""
    base = name
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    suffix = base.rsplit("_", 1)[-1]
    return _SUFFIX_TO_MODALITY.get(suffix)
