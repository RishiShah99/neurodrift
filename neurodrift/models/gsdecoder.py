"""Phase 4 — hierarchical feed-forward 3D Gaussian-splat decoder.

Maps a frozen content latent ``z = (B, C, d, d, d)`` to an explicit set of 3D
Gaussians (``mu, scale, quaternion, alpha, intensity``) in a single forward pass
and renders orthogonal mid-slices from them. This is a FEED-FORWARD amortised
decoder, NOT per-scene optimisation: a fixed bank of learned Gaussian *query*
tokens cross-attends to the flattened latent and a few attention layers regress
the per-Gaussian parameters (C4G-style compact query tokens, arXiv:2605.31595).

Hierarchy: a coarse head emits Gaussians over the whole volume; a fine head adds
detail Gaussians that are gated by a parcellation mask (cortical ribbon /
hippocampus / ventricles) so capacity concentrates where structure is dense.

Two render paths:
  * ``render_slices_gsplat`` — the real CUDA rasteriser (Kerbl 2023 3DGS via the
    ``gsplat`` library). ``gsplat`` is CUDA-only, so it is imported lazily INSIDE
    the function and is never touched at import / test-collection time.
  * ``render_slices_reference`` — a pure-torch, differentiable, CPU reference
    rasteriser used by the loss and the unit tests. It projects each Gaussian
    onto each orthogonal mid-plane (slice coordinate ~0), splats an
    (approximately) isotropic 2D footprint scaled by the in-plane covariance,
    and alpha-weights the intensity accumulation.

Fork references:
  * C4G compact query tokens — arXiv:2605.31595
  * Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Radiance Field
    Rendering" (3DGS) — the splatting / covariance formulation.
  * gsplat — https://github.com/nerfstudio-project/gsplat (CUDA rasteriser).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Level tags carried on every Gaussian so the merged set is introspectable and the
# fine tier can be masked/gated downstream without re-deriving which token it came from.
LEVEL_COARSE = 0
LEVEL_FINE = 1


@dataclass
class GaussianParams:
    """A batched set of 3D Gaussians emitted by the decoder.

    All tensors share batch ``B`` and Gaussian count ``N`` (= coarse + fine):
      * ``mu``        (B, N, 3)  — centres in the normalised cube ``[-1, 1]^3``.
      * ``scale``     (B, N, 3)  — per-axis std devs, strictly positive.
      * ``quat``      (B, N, 4)  — unit quaternions ``(w, x, y, z)`` (rotation).
      * ``alpha``     (B, N, 1)  — opacity in ``[0, 1]``.
      * ``intensity`` (B, N, 1)  — emitted scalar intensity (per modality if widened).
      * ``level``     (B, N)     — ``LEVEL_COARSE`` / ``LEVEL_FINE`` tag (long).
    """

    mu: torch.Tensor
    scale: torch.Tensor
    quat: torch.Tensor
    alpha: torch.Tensor
    intensity: torch.Tensor
    level: torch.Tensor

    @property
    def num_gaussians(self) -> int:
        return self.mu.shape[1]


def quaternion_to_rotation(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion ``(w, x, y, z)`` -> rotation matrix ``R`` ``(..., 3, 3)``.

    Input is L2-normalised here defensively so a not-quite-unit quaternion (e.g.
    straight off a regression head before its own normalisation) still yields an
    orthonormal ``R`` and hence a valid (PSD) covariance.
    """
    q = F.normalize(quat, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    r00 = 1.0 - (tyy + tzz)
    r01 = txy - twz
    r02 = txz + twy
    r10 = txy + twz
    r11 = 1.0 - (txx + tzz)
    r12 = tyz - twx
    r20 = txz - twy
    r21 = tyz + twx
    r22 = 1.0 - (txx + tyy)
    rows = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    return rows


def to_covariance(scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Build ``Σ = R diag(scale^2) R^T`` from per-axis scales and a quaternion.

    ``scale`` (..., 3), ``quat`` (..., 4) -> ``Σ`` (..., 3, 3), symmetric PSD.
    PSD is guaranteed by construction (a congruence of a non-negative diagonal),
    so the symmetrisation at the end only cleans against float round-off.
    """
    rot = quaternion_to_rotation(quat)  # (..., 3, 3)
    var = scale.pow(2).clamp_min(1e-8)  # (..., 3)
    # R @ diag(var) @ R^T without materialising the diagonal matrix:
    # scaled columns of R, then R @ (var * R)^T.
    scaled = rot * var.unsqueeze(-2)  # multiply each column j of R by var_j
    cov = scaled @ rot.transpose(-1, -2)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return cov


class _AttentionBlock(nn.Module):
    """Pre-norm cross-attention (queries <- context) then a small MLP.

    With ``context=None`` it degenerates to self-attention over the queries, so the
    same block serves both the cross-attn (queries read the latent) and self-attn
    (queries talk to each other) stages.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        q = self.norm_q(queries)
        kv = q if context is None else self.norm_kv(context)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm_mlp(queries))
        return queries


class _GaussianHead(nn.Module):
    """Per-token MLP heads mapping a feature vector to raw Gaussian parameters.

    Returns *activated* parameters: ``mu`` in ``[-1, 1]`` (tanh), positive ``scale``
    (softplus, small init via a negative bias), unit ``quat`` (L2-norm), ``alpha`` in
    ``[0, 1]`` (sigmoid), and a free ``intensity`` (optionally per modality).
    """

    def __init__(self, dim: int, intensity_channels: int = 1, scale_init: float = 0.02) -> None:
        super().__init__()
        self.intensity_channels = intensity_channels
        self.mu = nn.Linear(dim, 3)
        self.scale = nn.Linear(dim, 3)
        self.quat = nn.Linear(dim, 4)
        self.alpha = nn.Linear(dim, 1)
        self.intensity = nn.Linear(dim, intensity_channels)
        # Small initial scales: softplus(bias) ~= scale_init so Gaussians start tight
        # and grow as needed (3DGS-style), rather than blanketing the whole cube.
        with torch.no_grad():
            inv = math.log(math.expm1(scale_init)) if scale_init > 0 else -4.0
            self.scale.bias.fill_(inv)
            self.scale.weight.mul_(0.1)
            # Identity-ish quaternion (w=1) at init.
            self.quat.bias.zero_()
            self.quat.bias[0] = 1.0

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        mu = torch.tanh(self.mu(feats))
        scale = F.softplus(self.scale(feats)) + 1e-4
        quat = F.normalize(self.quat(feats), dim=-1, eps=1e-8)
        alpha = torch.sigmoid(self.alpha(feats))
        intensity = self.intensity(feats)
        return {"mu": mu, "scale": scale, "quat": quat, "alpha": alpha, "intensity": intensity}


class GSDecoder3D(nn.Module):
    """Hierarchical feed-forward latent -> 3D-Gaussians decoder.

    Args:
        latent_channels: channel dim ``C`` of the input latent ``z (B, C, d, d, d)``.
        model_dim: width of the query tokens / attention stack.
        num_coarse: number of coarse Gaussian query tokens (real target ~50k).
        num_fine: number of fine Gaussian query tokens (real target ~200k).
        num_heads: attention heads.
        depth: number of (cross-attn, self-attn) layer pairs.
        image_size: default render resolution ``H = W`` for ``render_slices_*``.
        intensity_channels: per-Gaussian intensity channels (1 = scalar / shared).
        mask_threshold: a continuous gate value above this counts the latent-cell as
            "active" when deriving the fine gate from a parcellation mask.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        model_dim: int = 256,
        num_coarse: int = 50000,
        num_fine: int = 200000,
        num_heads: int = 8,
        depth: int = 4,
        image_size: int = 128,
        intensity_channels: int = 1,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.model_dim = model_dim
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.image_size = image_size
        self.intensity_channels = intensity_channels
        self.mask_threshold = mask_threshold

        # Project flattened latent tokens (one per voxel of z) to the model dim, plus a
        # learned positional embedding generated on the fly from a normalised grid so
        # the same module handles any latent side length ``d`` (tiny in tests).
        self.latent_proj = nn.Linear(latent_channels, model_dim)
        self.pos_proj = nn.Linear(3, model_dim)

        # Learned query tokens — the compact C4G-style queries that become Gaussians.
        self.coarse_queries = nn.Parameter(torch.randn(num_coarse, model_dim) * 0.02)
        self.fine_queries = nn.Parameter(torch.randn(num_fine, model_dim) * 0.02)

        self.cross_blocks = nn.ModuleList(
            [_AttentionBlock(model_dim, num_heads) for _ in range(depth)]
        )
        self.self_blocks = nn.ModuleList(
            [_AttentionBlock(model_dim, num_heads) for _ in range(depth)]
        )

        self.coarse_head = _GaussianHead(model_dim, intensity_channels)
        self.fine_head = _GaussianHead(model_dim, intensity_channels)

    # -- latent tokenisation ------------------------------------------------
    def _latent_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, C, d, d, d)`` -> context tokens ``(B, d^3, model_dim)``."""
        b, c, d0, d1, d2 = z.shape
        if c != self.latent_channels:
            raise ValueError(
                f"latent channels {c} != configured latent_channels {self.latent_channels}"
            )
        tokens = z.reshape(b, c, d0 * d1 * d2).permute(0, 2, 1)  # (B, d^3, C)
        feats: torch.Tensor = self.latent_proj(tokens)
        feats = feats + self.pos_proj(self._grid_coords(d0, d1, d2, z.device, z.dtype))
        return feats

    @staticmethod
    def _grid_coords(
        d0: int, d1: int, d2: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Normalised ``[-1, 1]`` voxel coordinates, flattened to ``(1, d^3, 3)``."""
        axes = [
            torch.linspace(-1.0, 1.0, steps=n, device=device, dtype=dtype)
            if n > 1
            else torch.zeros(1, device=device, dtype=dtype)
            for n in (d0, d1, d2)
        ]
        gx, gy, gz = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        coords = torch.stack([gx, gy, gz], dim=-1).reshape(1, d0 * d1 * d2, 3)
        return coords

    # -- attention trunk ----------------------------------------------------
    def _decode_queries(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = queries
        for cross, slf in zip(self.cross_blocks, self.self_blocks, strict=True):
            h = cross(h, context)
            h = slf(h, None)
        return h

    # -- fine-tier gate -----------------------------------------------------
    def _fine_gate(self, mask: torch.Tensor | None, b: int, device: torch.device) -> torch.Tensor:
        """Per-(batch, fine-token) keep-gate in ``[0, 1]`` from a parcellation mask.

        ``mask`` is a region-id / probability volume ``(B, d, d, d)`` (or ``(B, ...)``
        broadcastable). It is pooled to a single fraction of *active* voxels per
        sample, and that fraction sets how many of the fine queries stay live
        (highest-index gating is deterministic so the same query slots are dropped).
        With ``mask=None`` the whole fine tier is active (gate all-ones); with an
        all-zero mask the fine tier is fully gated off (gate all-zeros) — the
        contract the hierarchy test checks.
        """
        n_fine = self.num_fine
        if n_fine == 0:
            return torch.zeros(b, 0, device=device)
        if mask is None:
            return torch.ones(b, n_fine, device=device)
        active = (mask.reshape(b, -1) > self.mask_threshold).float().mean(dim=1)  # (B,)
        keep = (active * n_fine).round().long().clamp(0, n_fine)  # (B,)
        idx = torch.arange(n_fine, device=device).unsqueeze(0)  # (1, N)
        gate = (idx < keep.unsqueeze(1)).float()  # (B, N)
        return gate

    # -- forward ------------------------------------------------------------
    def forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> GaussianParams:
        b = z.shape[0]
        context = self._latent_tokens(z)

        coarse_q = self.coarse_queries.unsqueeze(0).expand(b, -1, -1)
        coarse_feats = self._decode_queries(coarse_q, context)
        coarse = self.coarse_head(coarse_feats)

        if self.num_fine > 0:
            fine_q = self.fine_queries.unsqueeze(0).expand(b, -1, -1)
            fine_feats = self._decode_queries(fine_q, context)
            fine = self.fine_head(fine_feats)
            gate = self._fine_gate(mask, b, z.device).unsqueeze(-1)  # (B, N_fine, 1)
            # Gate suppresses a fine Gaussian's contribution by zeroing its opacity;
            # the Gaussian is kept in the set (fixed N for batching) but renders nothing.
            fine_alpha = fine["alpha"] * gate
            fine_level = torch.full((b, self.num_fine), LEVEL_FINE, device=z.device)
            coarse_level = torch.full((b, self.num_coarse), LEVEL_COARSE, device=z.device)
            return GaussianParams(
                mu=torch.cat([coarse["mu"], fine["mu"]], dim=1),
                scale=torch.cat([coarse["scale"], fine["scale"]], dim=1),
                quat=torch.cat([coarse["quat"], fine["quat"]], dim=1),
                alpha=torch.cat([coarse["alpha"], fine_alpha], dim=1),
                intensity=torch.cat([coarse["intensity"], fine["intensity"]], dim=1),
                level=torch.cat([coarse_level, fine_level], dim=1),
            )

        coarse_level = torch.full((b, self.num_coarse), LEVEL_COARSE, device=z.device)
        return GaussianParams(
            mu=coarse["mu"],
            scale=coarse["scale"],
            quat=coarse["quat"],
            alpha=coarse["alpha"],
            intensity=coarse["intensity"],
            level=coarse_level,
        )

    def merged_gaussians(self, params: GaussianParams) -> GaussianParams:
        """Identity passthrough — ``forward`` already returns the merged coarse+fine
        set with a per-Gaussian ``level`` tag. Kept as an explicit, named entry point
        for callers that want the merged set regardless of internal layout."""
        return params

    # -- rendering ----------------------------------------------------------
    def render_slices_reference(
        self,
        params: GaussianParams,
        image_size: int | None = None,
        axes: tuple[int, int, int] = (0, 1, 2),
        slice_coord: float = 0.0,
        slab: float = 0.15,
    ) -> torch.Tensor:
        """Pure-torch differentiable CPU rasteriser of three orthogonal mid-slices.

        For each requested axis we take the plane at ``slice_coord`` (default the
        mid-slice, normalised coord 0). A Gaussian contributes to a plane weighted by
        how close its centre is to the plane along the slice axis (a 1D Gaussian
        falloff using that axis' variance, widened by ``slab`` so thin Gaussians still
        register on a discrete plane). In-plane it splats an axis-aligned 2D Gaussian
        footprint from the marginal covariance of the two in-plane axes. Contributions
        are alpha-weighted and intensity is normalised by accumulated weight.

        Returns ``(B, len(axes), H, W)`` — by convention axis order ``(0, 1, 2)`` is
        axial / coronal / sagittal. Finite and small.
        """
        size = image_size if image_size is not None else self.image_size
        mu = params.mu
        device, dtype = mu.device, mu.dtype
        b = mu.shape[0]
        cov = to_covariance(params.scale, params.quat)  # (B, N, 3, 3)
        alpha = params.alpha[..., 0]  # (B, N)
        intensity = params.intensity[..., 0]  # (B, N) — first channel for the gray render

        # In-plane pixel grid in normalised [-1, 1] coords (shared by every axis).
        lin = torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")  # (H, W) each
        grid = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)  # (H*W, 2)

        out = torch.zeros(b, len(axes), size, size, device=device, dtype=dtype)
        var_floor = (2.0 / size) ** 2  # ~one-pixel minimum variance so a splat isn't a spike
        for out_idx, axis in enumerate(axes):
            plane_axes = [a for a in (0, 1, 2) if a != axis]
            pa0, pa1 = plane_axes
            # Distance of each Gaussian centre to the slice plane along the slice axis.
            d_axis = mu[..., axis] - slice_coord  # (B, N)
            var_axis = cov[..., axis, axis].clamp_min(var_floor) + slab**2
            w_plane = torch.exp(-0.5 * d_axis.pow(2) / var_axis)  # (B, N)

            # In-plane marginal: diagonal of the 2x2 in-plane covariance block.
            mu_plane = torch.stack([mu[..., pa0], mu[..., pa1]], dim=-1)  # (B, N, 2)
            var0 = cov[..., pa0, pa0].clamp_min(var_floor)
            var1 = cov[..., pa1, pa1].clamp_min(var_floor)

            # (B, N, 1, 2) - (1, 1, H*W, 2) -> per-pixel offset; isotropic-ish footprint.
            diff = mu_plane.unsqueeze(2) - grid.view(1, 1, -1, 2)  # (B, N, P, 2)
            quad = diff[..., 0].pow(2) / var0.unsqueeze(-1) + diff[..., 1].pow(2) / var1.unsqueeze(
                -1
            )
            foot = torch.exp(-0.5 * quad)  # (B, N, P)

            weight = (alpha * w_plane).unsqueeze(-1) * foot  # (B, N, P)
            num = (weight * intensity.unsqueeze(-1)).sum(dim=1)  # (B, P)
            den = weight.sum(dim=1).clamp_min(1e-6)  # (B, P)
            plane = (num / den).reshape(b, size, size)
            out[:, out_idx] = plane
        return out

    def render_slices_gsplat(
        self,
        params: GaussianParams,
        image_size: int | None = None,
        axes: tuple[int, int, int] = (0, 1, 2),
    ) -> torch.Tensor:
        """Real CUDA splatting path (Kerbl 2023 3DGS via ``gsplat``). GPU only.

        ``gsplat`` is CUDA-only, so it is imported HERE (never at module load). On a
        CPU box this raises a clear ``RuntimeError`` directing the caller to the
        reference rasteriser. The full projection/rasterisation wiring lands when the
        box-side GT-slice loader does; this stub guarantees the import contract.
        """
        try:
            import gsplat  # noqa: F401  (lazy: CUDA-only, import-guarded)
        except ImportError as err:  # pragma: no cover - exercised only off-box
            raise RuntimeError(
                "render_slices_gsplat requires the CUDA-only `gsplat` package "
                "(install the [splat] extra on a GPU box). Use render_slices_reference "
                "for CPU / tests."
            ) from err
        raise RuntimeError(
            "render_slices_gsplat is GPU-only and not wired in this CPU build; "
            "use render_slices_reference."
        )
