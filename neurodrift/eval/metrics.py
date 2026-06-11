"""Volumetric image-quality metrics for the v0 validation portfolio.

Dependency-light (torch only) so the eval runs single-process on any box. PSNR
and a 3D Gaussian-window SSIM, both operating on a single (D, H, W) volume.

`data_range` defaults to the target's peak-to-peak span. The corpus is z-scored
(mean 0, unit std over the brain), so a fixed data range is not meaningful across
subjects — deriving it per volume keeps PSNR/SSIM comparable to the way the
training target is normalised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _data_range(target: torch.Tensor) -> torch.Tensor:
    return (target.amax() - target.amin()).clamp_min(1e-6)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio over one volume (any shape; reduced over all elements)."""
    mse = F.mse_loss(pred, target)
    if mse.item() <= 0:
        return 99.0
    dr = torch.as_tensor(data_range, device=target.device) if data_range else _data_range(target)
    return float(10.0 * torch.log10(dr.pow(2) / mse))


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # separable 3D kernel via outer products
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return w.view(1, 1, window_size, window_size, window_size)


def ssim3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
    window_size: int = 7,
    sigma: float = 1.5,
) -> float:
    """3D SSIM with a Gaussian window. Inputs are (D, H, W) single-channel volumes."""
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError(
            f"ssim3d expects (D, H, W); got {tuple(pred.shape)} / {tuple(target.shape)}"
        )
    device = target.device
    x = pred.view(1, 1, *pred.shape).float()
    y = target.view(1, 1, *target.shape).float()
    win = _gaussian_window(window_size, sigma, device)
    pad = window_size // 2

    dr = torch.as_tensor(data_range, device=device) if data_range else _data_range(target)
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2

    mu_x = F.conv3d(x, win, padding=pad)
    mu_y = F.conv3d(y, win, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv3d(x * x, win, padding=pad) - mu_x2
    sigma_y2 = F.conv3d(y * y, win, padding=pad) - mu_y2
    sigma_xy = F.conv3d(x * y, win, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return float(ssim_map.mean())
