"""LightningModule wrapper for the Phase 1 VAE training loop."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from neurodrift.models.vae3d import VAE3D


class VAELitModule(L.LightningModule):
    """Wraps `VAE3D` with the L1 + KL Phase 1 objective."""

    def __init__(
        self,
        model: VAE3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).recon

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["image"]
        out = self.model(x)

        recon_loss = F.l1_loss(out.recon, x)
        kl = -0.5 * torch.mean(1 + out.logvar - out.mu.pow(2) - out.logvar.exp())
        loss = recon_loss + self.model.beta_kl * kl

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}/recon_l1", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/kl", kl, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
