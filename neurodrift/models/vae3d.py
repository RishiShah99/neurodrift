"""Multimodal 3D VAE — Phase 1 placeholder.

Wire-only: forward returns zeros so the rest of the training stack (Lightning
module, Hydra config, W&B logger, checkpoint callback) can be exercised before
the real encoder/decoder lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VAEOutput:
    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class VAE3D(nn.Module):
    """Placeholder VAE with the public shape contract Phase 1 will fill in."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 16,
        base_channels: int = 64,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        beta_kl: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.beta_kl = beta_kl

        # Stand-in projection so the module has parameters and forward returns
        # something Lightning can backprop through during fast_dev_run.
        self.proj_in = nn.Conv3d(in_channels, latent_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(latent_channels, in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj_in(x)
        mu = z
        logvar = torch.zeros_like(z)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj_out(z)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar)
