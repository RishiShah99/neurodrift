"""Multimodal 3D VAE — v0 Phase 1.

Per-modality conv stems with learned modality embeddings feed a shared
3D-ResNet encoder; a 32^3 x 16 latent feeds per-modality 3D-ResNet decoders.
`forward(x, modality_mask)` accepts an (B, M, D, H, W) stack and returns
reconstructions for every modality slot, regardless of which ones were
present at the input — this is what the cross-modal cycle loss needs.

Downsample factor is fixed at 4x per spatial dim: 128 -> 32, 192 -> 48, 256 -> 64.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class VAEOutput:
    """recon: (B, M, D, H, W); mu/logvar: (B, latent_c, D/4, H/4, W/4)."""

    recon: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


def _safe_groupnorm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _conv3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class _ResBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.norm1 = _safe_groupnorm(in_c)
        self.conv1 = _conv3(in_c, out_c)
        self.norm2 = _safe_groupnorm(out_c)
        self.conv2 = _conv3(out_c, out_c)
        self.skip: nn.Module = (
            nn.Identity() if in_c == out_c else nn.Conv3d(in_c, out_c, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Downsample3D(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.op = nn.Conv3d(c, c, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class _Upsample3D(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.op = nn.Conv3d(c, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class _EncoderBody(nn.Module):
    """Resblock + downsample stack from fused base-channel features to latent."""

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        latent_channels: int,
    ) -> None:
        super().__init__()
        chans = [base_channels * m for m in channel_mults]
        blocks: list[nn.Module] = []
        c_in = base_channels
        for i, c_out in enumerate(chans):
            for _ in range(num_res_blocks):
                blocks.append(_ResBlock3D(c_in, c_out))
                c_in = c_out
            if i < len(chans) - 1:
                blocks.append(_Downsample3D(c_in))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = _safe_groupnorm(c_in)
        self.out_conv = nn.Conv3d(c_in, 2 * latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for b in self.blocks:
            h = b(h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar


class _DecoderBody(nn.Module):
    """Per-modality resblock + upsample stack from latent back to (1, D, H, W)."""

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        latent_channels: int,
    ) -> None:
        super().__init__()
        chans = [base_channels * m for m in channel_mults][::-1]
        self.in_conv = _conv3(latent_channels, chans[0])
        blocks: list[nn.Module] = []
        c_in = chans[0]
        for i, c_out in enumerate(chans):
            for _ in range(num_res_blocks):
                blocks.append(_ResBlock3D(c_in, c_out))
                c_in = c_out
            if i < len(chans) - 1:
                blocks.append(_Upsample3D(c_in))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = _safe_groupnorm(c_in)
        self.out_conv = nn.Conv3d(c_in, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        for b in self.blocks:
            h = b(h)
        return self.out_conv(F.silu(self.out_norm(h)))


DEFAULT_MODALITIES: tuple[str, ...] = ("T1w", "T2w", "PDw", "dwi")


class VAE3D(nn.Module):
    """Multimodal 3D VAE: per-modality stems → shared encoder → per-modality decoders."""

    def __init__(
        self,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        latent_channels: int = 16,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        beta_kl: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.num_modalities = len(self.modalities)
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.beta_kl = beta_kl

        self.stems = nn.ModuleList([_conv3(1, base_channels) for _ in range(self.num_modalities)])
        self.modality_embed = nn.Parameter(torch.zeros(self.num_modalities, base_channels))

        self.encoder = _EncoderBody(base_channels, channel_mults, num_res_blocks, latent_channels)
        self.decoders = nn.ModuleList(
            [
                _DecoderBody(base_channels, channel_mults, num_res_blocks, latent_channels)
                for _ in range(self.num_modalities)
            ]
        )

    def _fuse_stems(self, x: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """Stem each modality, add modality embedding, mask-average across modalities."""
        b, m = x.shape[0], x.shape[1]
        if m != self.num_modalities:
            raise ValueError(f"expected {self.num_modalities} modality channels, got {m}")
        stem_outs = []
        for i, stem in enumerate(self.stems):
            feats = stem(x[:, i : i + 1])
            feats = feats + self.modality_embed[i].view(1, -1, 1, 1, 1)
            stem_outs.append(feats)
        stacked = torch.stack(stem_outs, dim=1)
        mask = modality_mask.view(b, m, 1, 1, 1, 1).to(stacked.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (stacked * mask).sum(dim=1) / denom

    def encode(
        self, x: torch.Tensor, modality_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], self.num_modalities, device=x.device)
        fused = self._fuse_stems(x, modality_mask)
        return self.encoder(fused)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to every modality slot. Returns (B, M, D, H, W)."""
        recons = [dec(z) for dec in self.decoders]
        return torch.cat(recons, dim=1)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> VAEOutput:
        mu, logvar = self.encode(x, modality_mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(recon=recon, mu=mu, logvar=logvar)
