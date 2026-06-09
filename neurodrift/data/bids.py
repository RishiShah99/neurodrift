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

    Recognized suffixes: `T1w`, `T2w`, `FLAIR`, `dwi`. Files outside `anat/`
    or `dwi/` are skipped.
    """
    root = Path(root)
    for sub_dir in sorted(root.glob("sub-*")):
        subject = sub_dir.name
        session_dirs = sorted(sub_dir.glob("ses-*")) or [sub_dir]
        for ses_dir in session_dirs:
            session = ses_dir.name if ses_dir.name.startswith("ses-") else None
            for kind in ("anat", "dwi"):
                kind_dir = ses_dir / kind
                if not kind_dir.is_dir():
                    continue
                for nii in sorted(kind_dir.glob("*.nii*")):
                    modality = _modality_from_filename(nii.name)
                    if modality is None:
                        continue
                    yield Scan(subject=subject, session=session, modality=modality, path=nii)


def _modality_from_filename(name: str) -> Modality | None:
    """Pull the BIDS suffix off the filename before the `.nii(.gz)` extension."""
    base = name
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    suffix = base.rsplit("_", 1)[-1]
    return _SUFFIX_TO_MODALITY.get(suffix)
