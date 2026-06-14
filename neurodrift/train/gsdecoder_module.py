"""LightningModule for the Phase 4 feed-forward Gaussian-splat decoder (v0 curriculum).

Trains the decoder on FROZEN ground-truth latents: each batch supplies ``z`` plus
the three orthogonal GT mid-slices the rendered Gaussians must match. The loss is

    L = L1(render, slices)
      + dssim_weight · DSSIM(render, slices)     (structural, 2D Gaussian-window)
      + tv_weight   · TV(render)                  (anisotropic total variation prior)
      [+ optional slice-discriminator slot — OFF in v0]

Rendering uses the pure-torch reference rasteriser on CPU (and in tests); a
``use_gsplat`` flag swaps in the CUDA path on the box. Automatic optimisation +
plain DDP, mirroring the rock-solid no-GAN VAE path. The non-finite-gradient guard
is shared with the VAE module (``_skip_step_if_nonfinite``).
"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F

from neurodrift.models.gsdecoder import GSDecoder3D
from neurodrift.train.lightning_module import _skip_step_if_nonfinite


def _gaussian_window_2d(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Separable 2D Gaussian window ``(1, 1, k, k)`` (reuses the ssim3d idea in 2D)."""
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    return w.view(1, 1, window_size, window_size)


def dssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
    sigma: float = 1.5,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Structural dissimilarity ``(1 - SSIM) / 2`` over batched 2D slice stacks.

    ``pred``/``target`` are ``(B, S, H, W)`` (S orthogonal slices). The Gaussian-window
    SSIM mirrors ``neurodrift.eval.metrics.ssim3d`` in 2D. ``data_range`` defaults to 2
    (slices live in the decoder's normalised ``[-1, 1]`` intensity range). Returns a
    scalar in ``[0, 1]`` (0 = identical), differentiable so it can drive the loss.
    """
    b, s, h, w = pred.shape
    x = pred.reshape(b * s, 1, h, w)
    y = target.reshape(b * s, 1, h, w)
    win = _gaussian_window_2d(window_size, sigma, pred.device).to(pred.dtype)
    pad = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(x, win, padding=pad)
    mu_y = F.conv2d(y, win, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, win, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, win, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, win, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return (1.0 - ssim_map.mean()) * 0.5


def total_variation(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV over batched 2D slice stacks ``(B, S, H, W)`` -> scalar.

    A mild smoothness prior that suppresses the salt-and-pepper a sparse splat set can
    leave between footprints, without the staircase a strong TV would impose.
    """
    dh = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
    dw = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
    return dh + dw


class GSDecoderLitModule(L.LightningModule):
    """v0 Phase 4: feed-forward Gaussian-splat decoder over frozen GT latents."""

    def __init__(
        self,
        model: GSDecoder3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        dssim_weight: float = 0.5,
        tv_weight: float = 0.01,
        data_range: float = 2.0,
        use_gsplat: bool = False,
    ) -> None:
        super().__init__()
        # Automatic optimization + plain DDP (no GAN, fully-used graph): the same
        # rock-solid path the no-GAN VAE flagship cooked on. The non-finite guard runs
        # in on_before_optimizer_step.
        self.automatic_optimization = True
        self.model = model
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.dssim_weight = dssim_weight
        self.tv_weight = tv_weight
        self.data_range = data_range
        # Picks the CUDA rasteriser on the box; CPU/tests keep the reference path.
        self.use_gsplat = use_gsplat

    def _render(self, params: Any, image_size: int) -> torch.Tensor:
        if self.use_gsplat:
            return self.model.render_slices_gsplat(params, image_size)
        return self.model.render_slices_reference(params, image_size)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        z = batch["z"]
        slices = batch["slices"]  # (B, 3, H, W) GT orthogonal mid-slices
        mask = batch.get("mask")
        image_size = slices.shape[-1]

        params = self.model(z, mask)
        rendered = self._render(params, image_size)

        l1 = F.l1_loss(rendered, slices)
        ds = dssim(rendered, slices, data_range=self.data_range)
        tv = total_variation(rendered)
        loss = l1 + self.dssim_weight * ds + self.tv_weight * tv

        bs = z.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/l1", l1, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/dssim", ds, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/tv", tv, on_epoch=True, batch_size=bs)
        return loss

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        _skip_step_if_nonfinite(self, optimizer)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        # SINGLE SOURCE OF TRUTH for the cosine half-period (mirrors VAELitModule): the
        # configured scheduler.T_max is a placeholder, unconditionally overridden with
        # the real planned step count so the LR anneals to eta_min exactly at the end.
        # Editing the YAML T_max does nothing; change max_epochs instead.
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
