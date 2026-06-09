"""NIfTI ↔ Zarr round-trip used by the preprocessing cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import zarr


def nifti_to_zarr(
    nifti_path: Path,
    zarr_path: Path,
    chunks: tuple[int, int, int] | None = (64, 64, 64),
    attrs: dict[str, Any] | None = None,
) -> Path:
    """Write a NIfTI volume into a Zarr store, preserving affine + extra attrs."""
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)

    zarr_path = Path(zarr_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(zarr_path), mode="w")
    arr = root.create_dataset(
        "data",
        shape=data.shape,
        chunks=chunks if data.ndim == 3 else None,
        dtype=data.dtype,
        overwrite=True,
    )
    arr[...] = data
    root.create_dataset("affine", shape=affine.shape, dtype=affine.dtype, overwrite=True)[...] = (
        affine
    )

    root.attrs["source_nifti"] = str(nifti_path)
    root.attrs["shape"] = list(data.shape)
    if attrs:
        for k, v in attrs.items():
            root.attrs[k] = v
    return zarr_path


def zarr_to_array(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a Zarr store back into (data, affine, attrs)."""
    root = zarr.open(str(zarr_path), mode="r")
    data = np.asarray(root["data"])
    affine = np.asarray(root["affine"])
    attrs = dict(root.attrs)
    return data, affine, attrs
