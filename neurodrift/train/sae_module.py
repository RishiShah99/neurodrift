"""LightningModule training a TopK-SAE on frozen VAE latents — Phase 5.

Reads the cached latent batch (`z = (B, C, d, d, d)` + age + categoricals) from
neurodrift.data.latents.LatentDataModule, reshapes each latent into one token per
spatial voxel (B*d^3 tokens of dim C), and trains the dictionary with

    loss = MSE(x_hat, tokens) + aux_coef * aux_loss

where `aux_loss` is the SAE's own AuxK dead-latent term (already coefficient-free
inside the model; `aux_coef` here is an extra LitModule-level knob, default 1.0).

Automatic optimization + plain DDP — the same rock-solid path the no-GAN VAE cook
ran. The decoder is unit-norm-renormalised and the dead-latent clock advanced
once per step (on_train_batch_end). The non-finite-grad guard is imported from the
VAE module so the SAE inherits the identical DDP-safe skip.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from neurodrift.models.sae import SAEOutput, TopKSAE
from neurodrift.train.lightning_module import _skip_step_if_nonfinite


def _tokens(z: torch.Tensor) -> torch.Tensor:
    """Latent (B, C, d, d, d) -> tokens (B*d^3, C): one C-vector per spatial voxel.

    permute moves the channel axis last; reshape flattens batch + spatial into the
    token axis. contiguous() so the reshape is a view-safe copy under autograd/DDP.
    """
    c = z.shape[1]
    return z.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, c)


class SAELitModule(L.LightningModule):
    """Train a TopK-SAE over the frozen VAE content latent."""

    def __init__(
        self,
        model: TopKSAE,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        aux_coef: float = 1.0,
    ) -> None:
        super().__init__()
        self.automatic_optimization = True
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.aux_coef = aux_coef

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out: SAEOutput = self.model(tokens)
        return out.x_hat

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        tokens = _tokens(batch["z"])
        out: SAEOutput = self.model(tokens)
        mse = F.mse_loss(out.x_hat, tokens)
        # out.aux_loss already carries the model's internal aux_coef; aux_coef here
        # is an additional LitModule-level weight (default 1.0, no double-scaling
        # surprise — set the model's aux_coef for the OpenAI 1/k default).
        loss = mse + self.aux_coef * out.aux_loss

        dead_frac = self.model.dead_mask.float().mean()
        bs = tokens.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/mse", mse, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/aux", out.aux_loss, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/dead_frac", dead_frac, on_epoch=True, batch_size=bs)
        # L0 is fixed by construction (exactly k per token); log the realised mean
        # so a regression in the TopK path (e.g. duplicate indices) shows up.
        l0 = (out.acts != 0).float().sum(dim=-1).mean()
        self.log(f"{stage}/l0", l0, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        # AFTER the optimizer step: renormalise the dictionary and advance the
        # dead-latent clock from the latents that just fired. Recompute the TopK
        # indices under no_grad (cheap) rather than threading them out of the step.
        self.model.normalize_decoder()
        batch = args[1] if len(args) > 1 else kwargs.get("batch")
        if batch is None:
            return
        with torch.no_grad():
            _, indices = self.model.encode(_tokens(batch["z"]))
        self.model.update_dead_tracker(indices)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        _skip_step_if_nonfinite(self, optimizer)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        # SINGLE SOURCE OF TRUTH for the cosine half-period (mirrors VAELitModule):
        # the configured T_max is a placeholder; pin it to the real planned step
        # count so the LR anneals to eta_min exactly at the end. Editing the YAML
        # T_max does nothing — change max_epochs (and the corpus) instead.
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
