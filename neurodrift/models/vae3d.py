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
from torch.utils.checkpoint import checkpoint


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
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
            if self.use_checkpoint and torch.is_grad_enabled():
                h = checkpoint(b, h, use_reentrant=False)
            else:
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
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
            if self.use_checkpoint and torch.is_grad_enabled():
                h = checkpoint(b, h, use_reentrant=False)
            else:
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
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.num_modalities = len(self.modalities)
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.beta_kl = beta_kl
        self.use_checkpoint = use_checkpoint

        self.stems = nn.ModuleList([_conv3(1, base_channels) for _ in range(self.num_modalities)])
        self.modality_embed = nn.Parameter(torch.zeros(self.num_modalities, base_channels))

        self.encoder = _EncoderBody(
            base_channels, channel_mults, num_res_blocks, latent_channels, use_checkpoint
        )
        self.decoders = nn.ModuleList(
            [
                _DecoderBody(
                    base_channels, channel_mults, num_res_blocks, latent_channels, use_checkpoint
                )
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


# ---------------------------------------------------------------------------
# Content/style disentangled VAE (v0 improvement E1 — MUNIT / Hi-Net style)
# ---------------------------------------------------------------------------
#
# WHY this exists. The shared-latent `VAE3D` above mask-AVERAGES every modality
# into one 16-channel code, then every decoder reads that one code. Cross-modal
# synthesis (feed T1, read T2) is therefore capped (~19 dB observed): the single
# averaged latent discards the modality-specific contrast the T2 decoder needs,
# and L1-on-an-averaged-code regresses to the mean (blur). This model instead
# factors each scan into:
#
#   * a SPATIAL "content" code  c  (anatomy; meant to be modality-INVARIANT), and
#   * a GLOBAL  "style"   code  s  (contrast/appearance; per-modality).
#
# A SINGLE shared decoder takes (content, target-style) -> a modality. Because the
# decoder is shared and modality identity is carried ONLY by the style vector,
# nothing but the style can set the output contrast — which is exactly what forces
# `content` to become modality-invariant when the disentanglement losses pull on
# it. Cross-modal T1->T2 = decode(content(T1), style_prototype[T2]).
#
# The decoder is style-conditioned via FiLM (per-resblock affine modulation from a
# linear projection of the style vector), the established AdaIN/FiLM recipe.


class _StyleResBlock3D(nn.Module):
    """ResBlock whose second norm is FiLM-modulated by a global style vector."""

    def __init__(self, in_c: int, out_c: int, style_dim: int) -> None:
        super().__init__()
        self.norm1 = _safe_groupnorm(in_c)
        self.conv1 = _conv3(in_c, out_c)
        self.norm2 = _safe_groupnorm(out_c)
        self.conv2 = _conv3(out_c, out_c)
        # Predict (scale, shift) per output channel from the style code. Zero-init
        # so the block starts as a plain identity-style residual and the network
        # learns to *use* style gradually (stabler than random modulation at init).
        self.to_film = nn.Linear(style_dim, out_c * 2)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)
        self.skip: nn.Module = (
            nn.Identity() if in_c == out_c else nn.Conv3d(in_c, out_c, 1, bias=False)
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.to_film(style).chunk(2, dim=1)
        b = h.shape[0]
        h = self.norm2(h)
        h = h * (1.0 + scale.view(b, -1, 1, 1, 1)) + shift.view(b, -1, 1, 1, 1)
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class _ContentEncoder(nn.Module):
    """Single-modality stem features -> spatial content latent (mu, logvar)."""

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        latent_channels: int,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
            if self.use_checkpoint and torch.is_grad_enabled():
                h = checkpoint(b, h, use_reentrant=False)
            else:
                h = b(h)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h.chunk(2, dim=1)


class _StyleEncoder(nn.Module):
    """Single-modality volume -> global style vector (B, style_dim).

    Deliberately shallow and spatially aggressive (stride-2 convs to a global
    pool): style is the contrast/appearance summary, not anatomy, so it must not
    carry spatial structure (that would leak content into style and defeat the
    factorization).
    """

    def __init__(self, base_channels: int, style_dim: int, num_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv3d(1, base_channels, 4, stride=2, padding=1)]
        c = base_channels
        for _ in range(num_layers - 1):
            layers.append(nn.SiLU())
            layers.append(nn.Conv3d(c, min(c * 2, base_channels * 4), 4, stride=2, padding=1))
            c = min(c * 2, base_channels * 4)
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(c, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)


class _StyleDecoderBody(nn.Module):
    """Shared decoder: (content latent, style vector) -> single-modality volume."""

    def __init__(
        self,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        latent_channels: int,
        style_dim: int,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        chans = [base_channels * m for m in channel_mults][::-1]
        self.in_conv = _conv3(latent_channels, chans[0])
        blocks: list[nn.Module] = []
        c_in = chans[0]
        for i, c_out in enumerate(chans):
            for _ in range(num_res_blocks):
                blocks.append(_StyleResBlock3D(c_in, c_out, style_dim))
                c_in = c_out
            if i < len(chans) - 1:
                blocks.append(_Upsample3D(c_in))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = _safe_groupnorm(c_in)
        self.out_conv = nn.Conv3d(c_in, 1, kernel_size=1)

    def forward(self, z: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(z)
        for b in self.blocks:
            if isinstance(b, _StyleResBlock3D):
                if self.use_checkpoint and torch.is_grad_enabled():
                    h = checkpoint(b, h, style, use_reentrant=False)
                else:
                    h = b(h, style)
            else:
                h = b(h)
        return self.out_conv(F.silu(self.out_norm(h)))


@dataclass
class DisentangledOutput:
    """Structured per-modality content/style for the disentanglement losses.

    content_mu/content_logvar: (B, M, Cc, d, d, d) — per-modality content codes.
    style:                     (B, M, Sc)          — per-modality style codes.
    Entries are only meaningful where `present_mask[:, j] == 1`.
    """

    content_mu: torch.Tensor
    content_logvar: torch.Tensor
    style: torch.Tensor


class DisentangledVAE3D(nn.Module):
    """Content/style disentangled multimodal 3D VAE.

    Same external contract as `VAE3D` (`forward(x, mask) -> VAEOutput` with `.recon`
    over every modality slot, plus `encode`/`decode`) so `scripts/eval.py` and the
    cross-modal eval work unchanged. The disentanglement machinery
    (`encode_all`, `decode_one`, `style_prototype`, `translate`) is what the
    `DisentangledVAELitModule` drives for the translation / content-invariance /
    style-cycle losses.
    """

    def __init__(
        self,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        latent_channels: int = 16,
        style_dim: int = 16,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        beta_kl: float = 1.0e-4,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.num_modalities = len(self.modalities)
        self.latent_channels = latent_channels
        self.style_dim = style_dim
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.beta_kl = beta_kl
        self.use_checkpoint = use_checkpoint

        self.stems = nn.ModuleList([_conv3(1, base_channels) for _ in range(self.num_modalities)])
        self.modality_embed = nn.Parameter(torch.zeros(self.num_modalities, base_channels))

        self.content_encoder = _ContentEncoder(
            base_channels, channel_mults, num_res_blocks, latent_channels, use_checkpoint
        )
        self.style_encoder = _StyleEncoder(base_channels, style_dim)
        # Learned per-modality "prototype" style — the contrast to inject when the
        # target modality was NOT given as input (the cross-modal synthesis case).
        self.style_prototypes = nn.Parameter(torch.zeros(self.num_modalities, style_dim))
        self.decoder = _StyleDecoderBody(
            base_channels, channel_mults, num_res_blocks, latent_channels, style_dim, use_checkpoint
        )

    # -- low-level building blocks ------------------------------------------
    def _stem(self, x_single: torch.Tensor, idx: int) -> torch.Tensor:
        feats = self.stems[idx](x_single)
        return feats + self.modality_embed[idx].view(1, -1, 1, 1, 1)

    def encode_content_one(
        self, x_single: torch.Tensor, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single modality volume (B,1,D,H,W) -> content (mu, logvar)."""
        return self.content_encoder(self._stem(x_single, idx))

    def encode_style_one(self, x_single: torch.Tensor) -> torch.Tensor:
        """Single modality volume (B,1,D,H,W) -> style (B, style_dim)."""
        return self.style_encoder(x_single)

    def style_prototype(self, idx: int, batch: int, device: torch.device) -> torch.Tensor:
        return self.style_prototypes[idx].unsqueeze(0).expand(batch, -1).to(device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        return mu + torch.randn_like(std) * std

    def decode_one(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """(content latent, style vector) -> single-modality volume (B,1,D,H,W)."""
        return self.decoder(content, style)

    # -- structured encode for the disentanglement losses -------------------
    def encode_all(
        self, x: torch.Tensor, present_mask: torch.Tensor | None = None
    ) -> DisentangledOutput:
        """Per-modality content + style for EVERY slot (absent slots from zeros).

        `x`: (B, M, D, H, W). `present_mask` is accepted for call-site compatibility
        but intentionally does NOT gate computation: every slot's stem + content +
        style encoder runs every call so the autograd graph is IDENTICAL on every
        DDP rank regardless of each rank's modality mix. Gating on presence (the
        previous behaviour) made per-modality stems used on some ranks and not
        others, so `find_unused_parameters` could never agree across ranks and the
        all-reduce deadlocked (8 GPUs spinning at 100%, step frozen). The loss
        caller masks absent slots out, so always-encode is free of correctness cost.
        """
        b, m = x.shape[0], x.shape[1]
        if m != self.num_modalities:
            raise ValueError(f"expected {self.num_modalities} modality channels, got {m}")
        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        styles: list[torch.Tensor] = []
        for j in range(m):
            mu_j, logvar_j = self.encode_content_one(x[:, j : j + 1], j)
            s_j = self.encode_style_one(x[:, j : j + 1])
            mus.append(mu_j)
            logvars.append(logvar_j)
            styles.append(s_j)
        return DisentangledOutput(
            content_mu=torch.stack(mus, dim=1),
            content_logvar=torch.stack(logvars, dim=1),
            style=torch.stack(styles, dim=1),
        )

    def fuse_content(self, content: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask-average per-modality content codes into one shared anatomy code.

        `content`: (B, M, Cc, d, d, d); `mask`: (B, M). Averaging is sound HERE (it
        was the weak link in `VAE3D` only because there it averaged *contrast-laden*
        features) — once content is modality-invariant, averaging present anatomy
        codes is a clean multi-view fuse.
        """
        b, m = mask.shape
        w = mask.view(b, m, 1, 1, 1, 1).to(content.dtype)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (content * w).sum(dim=1) / denom

    # -- VAE3D-compatible contract (drives scripts/eval.py) -----------------
    def encode(
        self, x: torch.Tensor, modality_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the FUSED content (mu, logvar) — the canonical subject latent."""
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], self.num_modalities, device=x.device)
        enc = self.encode_all(x, modality_mask)
        mu = self.fuse_content(enc.content_mu, modality_mask)
        logvar = self.fuse_content(enc.content_logvar, modality_mask)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the fused content into every modality slot using prototype styles."""
        b = z.shape[0]
        recons = [
            self.decode_one(z, self.style_prototype(j, b, z.device))
            for j in range(self.num_modalities)
        ]
        return torch.cat(recons, dim=1)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> VAEOutput:
        """Self/cross-modal reconstruction over every slot.

        Content is fused from the modalities present in `modality_mask`; each output
        slot is decoded from that fused content plus the slot's style — the encoded
        style where the slot was given as input, the learned prototype where it was
        not (the synthesis case the cross-modal eval scores).
        """
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], self.num_modalities, device=x.device)
        b, m = x.shape[0], x.shape[1]
        enc = self.encode_all(x, modality_mask)
        mu = self.fuse_content(enc.content_mu, modality_mask)
        logvar = self.fuse_content(enc.content_logvar, modality_mask)
        z = self.reparameterize(mu, logvar)
        recons = []
        for j in range(m):
            given = modality_mask[:, j].view(b, 1)
            proto = self.style_prototype(j, b, x.device)
            style_j = torch.where(given > 0, enc.style[:, j], proto)
            recons.append(self.decode_one(z, style_j))
        return VAEOutput(recon=torch.cat(recons, dim=1), mu=mu, logvar=logvar)

    def translate(self, x: torch.Tensor, src_idx: int, dst_idx: int) -> torch.Tensor:
        """Synthesize modality `dst_idx` from a single source modality `src_idx`.

        `x`: (B, 1, D, H, W) — the source volume. Uses source content + the target
        modality's learned prototype style. This is the headline cross-modal path.
        """
        mu, logvar = self.encode_content_one(x, src_idx)
        z = self.reparameterize(mu, logvar)
        return self.decode_one(z, self.style_prototype(dst_idx, x.shape[0], x.device))


def _sn(conv: nn.Module) -> nn.Module:
    """Spectral-normalize a conv. SN-PatchGAN is the consensus stabilizer for 3D
    medical translation — it bounds the discriminator's Lipschitz constant so it
    can't overpower the generator, which removes the brittle D-learning-rate tuning
    that otherwise dominates 3D GAN training."""
    return nn.utils.spectral_norm(conv)


class PatchDiscriminator3D(nn.Module):
    """3D SN-PatchGAN critic (pix2pix-style) with per-modality conditioning.

    Outputs a grid of real/fake logits over volume patches rather than one scalar,
    which keeps the adversarial signal local (sharpen texture) instead of global
    (mode-collapse-prone). A 4-layer stride-2 stack gives a ~34³ receptive field at
    128³ — enough anatomy to judge local realism without collapsing to one scalar.
    Modality identity is injected as a learned bias on the first feature map so one
    critic can judge "is this a plausible T2 / FLAIR / T1".
    """

    def __init__(
        self,
        num_modalities: int,
        base_channels: int = 32,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.first = _sn(nn.Conv3d(1, base_channels, 4, stride=2, padding=1))
        self.modality_bias = nn.Parameter(torch.zeros(num_modalities, base_channels))
        body: list[nn.Module] = []
        c = base_channels
        for _ in range(num_layers - 1):
            c_out = min(c * 2, base_channels * 8)
            body.append(nn.LeakyReLU(0.2, inplace=True))
            body.append(_sn(nn.Conv3d(c, c_out, 4, stride=2, padding=1)))
            c = c_out
        self.body = nn.Sequential(*body)
        self.head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True), _sn(nn.Conv3d(c, 1, 3, padding=1))
        )

    def forward(self, x: torch.Tensor, modality_idx: int) -> torch.Tensor:
        h = self.first(x)
        h = h + self.modality_bias[modality_idx].view(1, -1, 1, 1, 1)
        h = self.body(h)  # body already opens with LeakyReLU
        return self.head(h)
