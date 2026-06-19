"""LightningModule for the v0 Phase 2 lifespan flow backbone.

Trains `neurodrift.models.flow.MMDiT3D` with the rectified-flow / linear stochastic
interpolant objective over the frozen VAE latents emitted by Agent D's
`LatentDataModule` (batch: z + age + the categorical conditioning slots). The loss
is plain MSE between the predicted velocity and the interpolant target `x1 - x0`.

This deliberately uses the rock-solid AUTOMATIC-optimization + plain-DDP path the v0
VAE cook validated (no manual-opt / GAN / find_unused — that deadlocked the GAN
variant twice). The optimizer/scheduler wiring and the non-finite-gradient guard are
copied from `VAELitModule` so the cosine half-period is pinned to the real planned
step count and a NaN step is skipped identically on every rank.
"""

from __future__ import annotations

import copy
from typing import Any, cast

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurodrift.models.flow import (
    CATEGORICAL_FIELDS,
    FlowMatchingObjective,
    MMDiT3D,
)
from neurodrift.train.lightning_module import _skip_step_if_nonfinite


def build_cond_from_batch(model: MMDiT3D, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Assemble the COND dict from a batch, defaulting missing slots to NULL/0.

    `age` is the only real v0 signal (NaN-safe handling lives in the embedder). Every
    absent or non-tensor categorical defaults to a zeros (B,) long tensor — the
    reserved NULL index — so a corpus with no labels trains the unconditional path.
    The latent store emits `cohort` as a list[str] (and may omit labelled fields), so
    any non-tensor slot trains NULL. Known ids are clamped into [0, cardinality] so an
    "unknown" sentinel (-1) maps to NULL instead of indexing the wrong embedding row.

    Shared by the flow trainer and the distillation trainer so the student is
    conditioned through the byte-identical contract as the teacher.
    """
    z = batch["z"]
    b = z.shape[0]
    age = batch.get("age")
    if age is None:
        age = torch.zeros(b, device=z.device)
    cond: dict[str, torch.Tensor] = {"age": age.to(z.device).float()}
    cat_embed = model.cond_embed.cat_embed
    for field in CATEGORICAL_FIELDS:
        val = batch.get(field)
        if not isinstance(val, torch.Tensor):
            cond[field] = torch.zeros(b, dtype=torch.long, device=z.device)
            continue
        ids = val.to(z.device).long()
        if field in cat_embed:
            ids = ids.clamp(0, cast(nn.Embedding, cat_embed[field]).num_embeddings - 1)
        else:
            ids = ids.clamp_min(0)
        cond[field] = ids
    return cond


class FlowLitModule(L.LightningModule):
    """Flow-matching trainer for the lifespan velocity model (automatic optimization)."""

    def __init__(
        self,
        model: MMDiT3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        cfg_dropout_p: float = 0.1,
        ema_decay: float | None = 0.999,
    ) -> None:
        super().__init__()
        self.automatic_optimization = True
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        # The CFG-dropout probability lives on the model's ConditioningEmbedder (that
        # is where the null-substitution happens); keep them in sync so the config's
        # litmodule.cfg_dropout_p is authoritative regardless of the model default.
        self.cfg_dropout_p = cfg_dropout_p
        self.model.cond_embed.cfg_dropout_p = cfg_dropout_p
        self.objective = FlowMatchingObjective()
        # Optional EMA of the weights — the canonical eval target for diffusion/flow
        # models. Kept simple: a frozen deep copy updated in-place each train batch.
        self.ema_decay = ema_decay
        self.ema_model: MMDiT3D | None = None
        if ema_decay is not None:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        velocity: torch.Tensor = self.model(z_t, t, cond)
        return velocity

    def _build_cond(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Assemble the COND dict from a batch (see `build_cond_from_batch`)."""
        return build_cond_from_batch(self.model, batch)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x1 = batch["z"]
        cond = self._build_cond(batch)
        interp = self.objective.sample_interpolant(x1)
        v = self.model(interp.x_t, interp.t, cond)
        loss = F.mse_loss(v, interp.target)

        bs = x1.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/fm_mse", loss, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    @torch.no_grad()
    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        if self.ema_model is None:
            return
        decay = self.ema_decay if self.ema_decay is not None else 0.0
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters(), strict=True):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        # Buffers (e.g. none currently learnable, but RoPE caches / future stats)
        # are copied straight through so the EMA model stays runnable.
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            ema_b.copy_(b)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        _skip_step_if_nonfinite(self, optimizer)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        # SINGLE SOURCE OF TRUTH for the cosine half-period (copied from VAELitModule):
        # the configured scheduler.T_max is only a placeholder; the true step count
        # depends on corpus size, devices, batch size and accumulation, so pin T_max
        # to the actual planned step count here. Editing the YAML T_max does NOT change
        # training; change max_epochs (and the data) instead.
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
