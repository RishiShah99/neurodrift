"""LightningModule wrapping the v0 Phase 1 VAE training loop.

Loss = masked L1 (reconstruct every present-in-ground-truth modality)
     + β·KL (linear warmup so the encoder learns recon before regularizing)
     + λ_p · perceptual (tri-orthogonal VGG19 on axial / coronal / sagittal slices)
     + λ_c · cross-modal cycle (re-encoding recon with a different mask
                                 must land in the same latent neighbourhood)

Per-cohort PSNR is logged separately so a single weak cohort never gets buried
under the pooled average.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from neurodrift.models.vae3d import VAE3D


def _masked_l1(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 averaged only over modality slots where `mask` is 1.

    `recon`, `target`: (B, M, D, H, W); `mask`: (B, M).
    """
    err = (recon - target).abs().mean(dim=(2, 3, 4))
    denom = mask.sum().clamp_min(1.0)
    return (err * mask).sum() / denom


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    if mse.item() <= 0:
        return torch.tensor(99.0, device=pred.device)
    data_range = target.abs().max().clamp_min(1e-6)
    return 10.0 * torch.log10(data_range.pow(2) / mse)


class _TriOrthoVGGPerceptual(nn.Module):
    """Tri-orthogonal mid-slice VGG19 feature distance.

    Cheap perceptual loss that works without a med-imaging-specific backbone.
    Swap in Med3D-VGG once weights are downloaded by setting `model` to
    something with the same `features` interface.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as err:
            raise RuntimeError("torchvision required for perceptual loss") from err
        vgg = models.vgg19(weights=None)
        self.features = vgg.features[:16]
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    @torch.no_grad()
    def _to_rgb_slice(self, vol: torch.Tensor, axis: int) -> torch.Tensor:
        mid = vol.shape[axis + 2] // 2
        if axis == 0:
            s = vol[:, :, mid, :, :]
        elif axis == 1:
            s = vol[:, :, :, mid, :]
        else:
            s = vol[:, :, :, :, mid]
        s = s.repeat(1, 3 // s.shape[1] + 1, 1, 1)[:, :3]
        s = (s - s.amin(dim=(2, 3), keepdim=True)) / (
            s.amax(dim=(2, 3), keepdim=True) - s.amin(dim=(2, 3), keepdim=True) + 1e-6
        )
        return s

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = recon.new_zeros(())
        for axis in (0, 1, 2):
            s_r = self._to_rgb_slice(recon, axis)
            s_t = self._to_rgb_slice(target, axis)
            f_r = self.features(s_r)
            f_t = self.features(s_t)
            loss = loss + F.l1_loss(f_r, f_t)
        return loss / 3.0


class VAELitModule(L.LightningModule):
    """v0 Phase 1: VAE with masked L1 + KL + perceptual + cross-modal cycle loss."""

    def __init__(
        self,
        model: VAE3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        kl_warmup_steps: int = 5000,
        perceptual_weight: float = 0.1,
        cycle_weight: float = 0.5,
        use_perceptual: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.kl_warmup_steps = kl_warmup_steps
        self.perceptual_weight = perceptual_weight
        self.cycle_weight = cycle_weight
        self.use_perceptual = use_perceptual
        self.perceptual: nn.Module | None = _TriOrthoVGGPerceptual() if use_perceptual else None

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(x, modality_mask).recon

    def _kl_beta(self) -> float:
        if self.kl_warmup_steps <= 0:
            return 1.0
        frac = min(1.0, self.global_step / float(self.kl_warmup_steps))
        return frac

    def _cycle_loss(
        self,
        recon: torch.Tensor,
        mu: torch.Tensor,
        present_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode recon under a different mask; match its mu to original mu (detached)."""
        b, m = present_mask.shape
        alt_mask = present_mask.clone()
        if m > 1:
            for i in range(b):
                present_idx = present_mask[i].nonzero(as_tuple=False).flatten()
                if present_idx.numel() < 2:
                    continue
                drop = present_idx[torch.randint(present_idx.numel(), (1,))].item()
                alt_mask[i, drop] = 0.0
        if alt_mask.sum() == 0:
            return mu.new_zeros(())
        recon_masked = recon * alt_mask.view(b, m, 1, 1, 1)
        mu2, _ = self.model.encode(recon_masked, alt_mask)
        return F.l1_loss(mu2, mu.detach())

    def _log_per_cohort_psnr(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        present_mask: torch.Tensor,
        cohorts: list[str],
        stage: str,
    ) -> None:
        per_cohort: dict[str, list[torch.Tensor]] = defaultdict(list)
        b, m = present_mask.shape
        for i in range(b):
            for j in range(m):
                if present_mask[i, j] == 1.0:
                    per_cohort[cohorts[i]].append(_psnr(recon[i, j], target[i, j]))
        for cohort, vals in per_cohort.items():
            mean_psnr = torch.stack(vals).mean()
            self.log(
                f"{stage}/psnr_{cohort}",
                mean_psnr,
                on_step=False,
                on_epoch=True,
                batch_size=b,
            )

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x = batch["image"]
        # `target` is the clean, never-dropped volume. Reconstructing against it
        # (not the zero-filled input `x`) is what turns modality dropout into a
        # cross-modal synthesis objective. Falls back to `x` for any caller that
        # predates the target split.
        target = batch.get("target", x)
        modality_mask = batch["modality_mask"]
        present_mask = batch["present_mask"]
        cohorts = batch["cohort"]

        out = self.model(x, modality_mask)

        recon_loss = _masked_l1(out.recon, target, present_mask)
        kl = _kl_divergence(out.mu, out.logvar)
        beta = self._kl_beta()
        loss = recon_loss + beta * self.model.beta_kl * kl

        perc_loss = x.new_zeros(())
        if self.perceptual is not None and stage == "train":
            # Zero recon's never-acquired slots so they match `target` (also zero
            # there) and contribute no spurious perceptual gradient.
            recon_perc = out.recon * present_mask.view(*present_mask.shape, 1, 1, 1)
            perc_loss = self.perceptual(recon_perc, target)
            loss = loss + self.perceptual_weight * perc_loss

        cycle = x.new_zeros(())
        if self.cycle_weight > 0 and stage == "train":
            cycle = self._cycle_loss(out.recon, out.mu, present_mask)
            loss = loss + self.cycle_weight * cycle

        bs = x.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/recon_l1", recon_loss, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/kl", kl, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/kl_beta", beta, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/perceptual", perc_loss, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/cycle", cycle, on_epoch=True, batch_size=bs)

        if not math.isnan(loss.item()):
            self._log_per_cohort_psnr(out.recon, target, present_mask, cohorts, stage)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        # The configured T_max is a guess; the true step count depends on corpus
        # size, devices, batch size and accumulation — all of which have churned
        # on this project. Pin the cosine half-period to the actual planned step
        # count so the LR genuinely anneals to eta_min instead of stalling halfway.
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
