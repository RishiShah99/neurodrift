"""Lightning DataModule over preprocessed Zarr volumes.

Phase 0 placeholder: emits random tensors at the configured resolution so the
trainer scaffold can run before any real data lands. Phase 1 will swap this
for a Zarr-backed dataset using `neurodrift.data.io.zarr_to_array`.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset


class _RandomVolumes(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, n: int, image_size: int, modalities: int) -> None:
        self.n = n
        self.image_size = image_size
        self.modalities = modalities

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = torch.randn(self.modalities, self.image_size, self.image_size, self.image_size)
        return {"image": x, "age": torch.tensor(60.0)}


class NiftiDataModule(L.LightningDataModule):
    """Stand-in DataModule. Replace with a Zarr-backed dataset once data lands."""

    def __init__(
        self,
        bids_root: str,
        work_dir: str,
        template: str,
        modalities: list[str] = ("T1w",),
        image_size: int = 128,
        batch_size: int = 2,
        num_workers: int = 4,
        val_fraction: float = 0.05,
        **_: Any,
    ) -> None:
        super().__init__()
        self.bids_root = bids_root
        self.work_dir = work_dir
        self.template = template
        self.modalities = list(modalities)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

    def setup(self, stage: str | None = None) -> None:
        n_train = 16
        n_val = max(2, int(n_train * self.val_fraction * 10))
        self.train_ds = _RandomVolumes(n_train, self.image_size, len(self.modalities))
        self.val_ds = _RandomVolumes(n_val, self.image_size, len(self.modalities))

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
