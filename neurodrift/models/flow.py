"""Lifespan flow backbone — v0 Phase 2 (the critical path).

An MM-DiT-style rectified-flow / stochastic-interpolant velocity model over the
frozen VAE latents from `neurodrift.models.vae3d.DisentangledVAE3D` (z = (B, 16,
32, 32, 32) at 128^3). The model predicts the flow velocity `v(z_t, t, cond)` and
is trained to match the linear interpolant target `x1 - x0`; sampling Euler-
integrates `dz/dt = v` from noise (t=0) to a data latent (t=1).

Conditioning. The ONLY real v0 signal is continuous AGE (a sinusoidal embedding +
MLP). Every categorical (sex, dx, apoe, treatment, cohort) is a wired slot trained
almost entirely on its reserved NULL index 0 (the corpus has no reliable labels at
v0), kept here so the same checkpoint can be conditioned on them later without a
re-architecture. Each categorical uses `nn.Embedding(cardinality + 1, hidden)` with
index 0 reserved for NULL; classifier-free-guidance dropout (training only) replaces
an id with 0 (null) with probability `cfg_dropout_p`.

Architecture (MMDiT3D, ~300M params at v0 defaults; the tests use tiny dims):
  * Patchify  : Conv3d(C, hidden, patch, stride=patch) -> tokens (B, T, hidden),
                T = (d / patch)^3. The (gd, gh, gw) token grid is carried for 3D
                RoPE and for unpatchify.
  * Register tokens (Darcet 2023): `num_register_tokens` learned tokens prepended
                to the sequence; they receive NO RoPE (only patch tokens are
                rotated) and are dropped before unpatchify.
  * Attention : 3D RoPE over (gd, gh, gw) — the head dim is split into three thirds,
                one rotary block per spatial axis — plus qk-norm (RMSNorm on q and k
                before the dot product). RMSNorm (not LayerNorm) for all block norms.
  * Cond      : a SINGLE cond vector c from `ConditioningEmbedder` drives per-block
                (shift, scale, gate) for both the attention and MLP sub-blocks via a
                zero-initialized Linear — standard DiT adaLN-zero, so every block
                starts as the identity and learns to use the conditioning gradually.
  * Head      : RMSNorm + adaLN-zero modulate -> zero-init Linear(hidden, patch^3 * C)
                -> unpatchify to a velocity field (B, C, d, d, d).

Fork references / what is adapted vs invented:
  * adaLN-zero conditioning + the zero-init output head: facebookresearch/DiT
    (Peebles & Xie 2023). Adapted from 2D image tokens to 3D latent patches.
  * Rectified-flow / linear stochastic interpolant (x_t = (1-t) x0 + t x1, target
    x1 - x0) + Euler sampling: willisma/SiT (Ma et al. 2024) and
    facebookresearch/flow_matching (Lipman et al. 2023). Used as-is.
  * Register tokens: Darcet et al. 2023 ("Vision Transformers Need Registers").
  * Rotary position embedding extended to 3D by splitting the head dim across the
    three spatial axes: lucidrains/rotary-embedding-torch, applied per-axis here.
  * NEW for this project: the 3D RoPE axis split over a (gd, gh, gw) latent-patch
    grid with register tokens excluded from rotation, and continuous-age sinusoidal
    conditioning summed into the DiT cond vector alongside CFG-droppable categoricals
    (the lifespan signal that makes this a *continuous-time* brain model rather than
    a class-conditioned generator).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Order is FROZEN: matches the COND contract and the data module batch keys. `age`
# is handled separately (continuous); these are the CFG-droppable categoricals.
CATEGORICAL_FIELDS: tuple[str, ...] = ("sex", "dx", "apoe", "treatment", "cohort")

# v0 cardinalities (number of real classes, excluding the reserved NULL index 0).
DEFAULT_CARDINALITIES: dict[str, int] = {
    "sex": 2,
    "dx": 4,
    "apoe": 6,
    "treatment": 6,
    "cohort": 8,
}


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a (B,) scalar into (B, dim), in float32.

    The standard transformer / DiT sinusoidal table. Used for both the flow time
    `t in [0, 1]` and continuous age. `dim` may be odd (a single zero column is
    appended), though the callers here always pass an even `dim`.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ConditioningEmbedder(nn.Module):
    """Build the single DiT cond vector c = t_emb + age_emb + sum(categorical_emb).

    Flow time and age each go sinusoidal -> 2-layer MLP -> hidden. Each categorical
    is an `nn.Embedding(cardinality + 1, hidden)` whose row 0 is the reserved NULL
    embedding. CFG-dropout (training only) replaces a sample's id with 0 (null) with
    probability `cfg_dropout_p`, which is exactly the unconditional path the sampler
    can later guide toward/away from.
    """

    def __init__(
        self,
        hidden: int,
        cardinalities: Mapping[str, int],
        cfg_dropout_p: float = 0.1,
        sinusoidal_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.cfg_dropout_p = cfg_dropout_p
        self.sinusoidal_dim = sinusoidal_dim
        # Field order is fixed by CATEGORICAL_FIELDS so embedding-module ordering is
        # deterministic regardless of dict insertion order in the config.
        self.fields: tuple[str, ...] = tuple(f for f in CATEGORICAL_FIELDS if f in cardinalities)

        self.time_mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.age_mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # +1 for the reserved NULL index 0. The padding_idx initializes row 0 to
        # zeros (it is still learnable — DiT's null class is trained — but starting
        # at zero keeps the unconditional path neutral at init).
        self.cat_embed = nn.ModuleDict(
            {f: nn.Embedding(cardinalities[f] + 1, hidden, padding_idx=0) for f in self.fields}
        )

    def _maybe_cfg_drop(self, ids: torch.Tensor) -> torch.Tensor:
        """Replace ids with 0 (null) w.p. cfg_dropout_p — TRAINING ONLY."""
        if not self.training or self.cfg_dropout_p <= 0.0:
            return ids
        drop = torch.rand(ids.shape[0], device=ids.device) < self.cfg_dropout_p
        return torch.where(drop, torch.zeros_like(ids), ids)

    def forward(
        self,
        t: torch.Tensor,
        cond: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """t: (B,) in [0, 1]; cond: dict with `age` (B,) float + categorical (B,) long.

        Returns the cond vector (B, hidden).
        """
        t_emb = self.time_mlp(_timestep_embedding(t, self.sinusoidal_dim))

        age = cond["age"]
        # NaN-safe: the corpus carries unknown ages as NaN. Map them to 0.0 BEFORE
        # the sinusoidal table (sin/cos of NaN is NaN, which would poison the whole
        # cond vector and hence every adaLN modulation).
        age = torch.nan_to_num(age.float(), nan=0.0)
        c = t_emb + self.age_mlp(_timestep_embedding(age, self.sinusoidal_dim))

        for f in self.fields:
            ids = cond[f].long()
            ids = self._maybe_cfg_drop(ids)
            embed = self.cat_embed[f]
            c = c + embed(ids)
        out: torch.Tensor = c
        return out


def _rope_freqs(seq_len: int, axis_dim: int, device: torch.device) -> torch.Tensor:
    """Rotary angles for one spatial axis: (seq_len, axis_dim // 2).

    `axis_dim` is the (even) number of head-dim channels allotted to this axis;
    rotary operates on `axis_dim // 2` 2D rotation planes.
    """
    half = axis_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1))
    )
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    return torch.outer(pos, inv_freq)  # (seq_len, half)


def build_axial_rope(
    grid: tuple[int, int, int], head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """3D RoPE cos/sin tables for a (gd, gh, gw) patch grid.

    The head dim is split into three CONTIGUOUS thirds (depth, height, width); each
    third is rotated by its own axis position. Any remainder channels (head_dim not
    divisible by 6 -> the three thirds don't tile evenly) are left UNROTATED, an
    identity rotation, so the split is always well defined. Returns (cos, sin), each
    (T, head_dim) with T = gd*gh*gw, ready to broadcast over (B, heads, T, head_dim).
    """
    gd, gh, gw = grid
    # Each axis gets a third of the head dim, rounded DOWN to an even number so each
    # third decomposes into 2D rotation planes.
    per_axis = head_dim // 3
    per_axis -= per_axis % 2
    sizes = [per_axis, per_axis, per_axis]
    rotated = per_axis * 3
    rest = head_dim - rotated  # unrotated tail channels

    # Per-token position index along each axis (row-major: d outermost, w innermost),
    # matching the flatten order of the patch grid in `MMDiT3D._patchify`.
    dd = torch.arange(gd, device=device).view(gd, 1, 1).expand(gd, gh, gw).reshape(-1)
    hh = torch.arange(gh, device=device).view(1, gh, 1).expand(gd, gh, gw).reshape(-1)
    ww = torch.arange(gw, device=device).view(1, 1, gw).expand(gd, gh, gw).reshape(-1)
    coords = [dd, hh, ww]

    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for axis_dim, pos in zip(sizes, coords, strict=True):
        if axis_dim == 0:
            continue
        half = axis_dim // 2
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / max(half, 1))
        )
        ang = torch.outer(pos.float(), inv_freq)  # (T, half)
        # Duplicate each angle so it covers the (x, y) pair of one rotation plane,
        # interleaved as [a0, a0, a1, a1, ...] to match the rotate-half convention.
        ang = ang.repeat_interleave(2, dim=-1)  # (T, axis_dim)
        cos_parts.append(torch.cos(ang))
        sin_parts.append(torch.sin(ang))

    if rest > 0:  # identity rotation on the leftover tail
        t = cos_parts[0].shape[0] if cos_parts else gd * gh * gw
        cos_parts.append(torch.ones(t, rest, device=device))
        sin_parts.append(torch.zeros(t, rest, device=device))

    cos = torch.cat(cos_parts, dim=-1)  # (T, head_dim)
    sin = torch.cat(sin_parts, dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[..., (x0, x1, x2, x3, ...)] -> [..., (-x1, x0, -x3, x2, ...)] (pairwise)."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary to x (B, heads, T, head_dim) with cos/sin (T, head_dim)."""
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class _Attention(nn.Module):
    """Multi-head self-attention with qk-norm and 3D RoPE on the patch tokens.

    The first `num_register_tokens` tokens are register tokens and are NOT rotated
    (RoPE applies only to the patch tokens); everything attends to everything.
    """

    def __init__(self, hidden: int, num_heads: int, num_register_tokens: int) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden {hidden} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.num_register_tokens = num_register_tokens
        self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)
        # qk-norm: RMSNorm over the per-head channel dim, applied to q and k before
        # the dot product (stabilizes attention logits — the qk-norm recipe).
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Rotate ONLY the patch tokens; register tokens (the leading r) get no RoPE.
        r = self.num_register_tokens
        if cos.shape[0] != t - r:
            raise ValueError(f"RoPE table len {cos.shape[0]} != patch tokens {t - r}")
        q_pat = apply_rope(q[:, :, r:], cos, sin)
        k_pat = apply_rope(k[:, :, r:], cos, sin)
        q = torch.cat([q[:, :, :r], q_pat], dim=2)
        k = torch.cat([k[:, :, :r], k_pat], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.head_dim)
        proj: torch.Tensor = self.proj(out)
        return proj


class _MLP(nn.Module):
    def __init__(self, hidden: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner = int(hidden * mlp_ratio)
        self.fc1 = nn.Linear(hidden, inner)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.fc2(self.act(self.fc1(x)))
        return out


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift, with (B, hidden) broadcast over T."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _DiTBlock(nn.Module):
    """A DiT block with adaLN-zero conditioning (Peebles & Xie 2023).

    A zero-initialized Linear maps the cond vector c to six (B, hidden) vectors
    (shift/scale/gate for the attention sub-block and for the MLP sub-block). Zero
    init means both gates start at 0, so the block is the identity at init and the
    residual stream is untouched until training learns to open the gates.
    """

    def __init__(
        self, hidden: int, num_heads: int, num_register_tokens: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden, elementwise_affine=False)
        self.attn = _Attention(hidden, num_heads, num_register_tokens)
        self.norm2 = nn.RMSNorm(hidden, elementwise_affine=False)
        self.mlp = _MLP(hidden, mlp_ratio)
        # adaLN-zero: SiLU(c) -> Linear -> 6 modulation vectors. Zero-init the Linear
        # so both gates start at 0 (block == identity at init). Kept as a named Linear
        # rather than indexing an nn.Sequential.
        self.ada_act = nn.SiLU()
        self.ada_linear = nn.Linear(hidden, 6 * hidden)
        nn.init.zeros_(self.ada_linear.weight)
        nn.init.zeros_(self.ada_linear.bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_linear(self.ada_act(c)).chunk(
            6, dim=-1
        )
        x = x + gate_a.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_a, scale_a), cos, sin
        )
        x = x + gate_m.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x


class _FinalLayer(nn.Module):
    """adaLN-zero final layer: RMSNorm + modulate -> zero-init Linear to patch pixels."""

    def __init__(self, hidden: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden, elementwise_affine=False)
        self.linear = nn.Linear(hidden, patch_size**3 * out_channels)
        self.ada_act = nn.SiLU()
        self.ada_linear = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.ada_linear.weight)
        nn.init.zeros_(self.ada_linear.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_linear(self.ada_act(c)).chunk(2, dim=-1)
        out: torch.Tensor = self.linear(_modulate(self.norm(x), shift, scale))
        return out


class MMDiT3D(nn.Module):
    """3D MM-DiT flow-velocity model over VAE latents (~300M at v0 defaults).

    `forward(z_t, t, cond) -> velocity (B, C, d, d, d)`. The latent is patchified
    into tokens, register tokens are prepended, depth DiT blocks (adaLN-zero, 3D
    RoPE, qk-norm) run, then an adaLN-zero head + unpatchify produce the velocity
    field in latent space.
    """

    def __init__(
        self,
        latent_channels: int = 16,
        latent_size: int = 32,
        patch_size: int = 2,
        hidden: int = 768,
        depth: int = 12,
        heads: int = 12,
        num_register_tokens: int = 4,
        mlp_ratio: float = 4.0,
        cardinalities: Mapping[str, int] | None = None,
        cfg_dropout_p: float = 0.1,
        cond_sinusoidal_dim: int = 256,
    ) -> None:
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError(f"latent_size {latent_size} not divisible by patch_size {patch_size}")
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.hidden = hidden
        self.depth = depth
        self.heads = heads
        self.num_register_tokens = num_register_tokens
        self.grid_size = latent_size // patch_size
        self.num_patches = self.grid_size**3

        cards: dict[str, int] = dict(
            DEFAULT_CARDINALITIES if cardinalities is None else cardinalities
        )

        self.patch_embed = nn.Conv3d(
            latent_channels, hidden, kernel_size=patch_size, stride=patch_size
        )
        # Register tokens (Darcet 2023): learned, prepended, no RoPE, dropped before
        # unpatchify. Small init so they don't dominate the residual stream at start.
        self.register_tokens = (
            nn.Parameter(torch.randn(1, num_register_tokens, hidden) * 0.02)
            if num_register_tokens > 0
            else None
        )
        self.cond_embed = ConditioningEmbedder(
            hidden, cards, cfg_dropout_p=cfg_dropout_p, sinusoidal_dim=cond_sinusoidal_dim
        )
        self.blocks = nn.ModuleList(
            [_DiTBlock(hidden, heads, num_register_tokens, mlp_ratio) for _ in range(depth)]
        )
        self.final = _FinalLayer(hidden, patch_size, latent_channels)
        # cos/sin RoPE tables depend only on the grid + head dim, so cache them per
        # (grid, device, dtype) instead of rebuilding every forward.
        self._rope_cache: dict[
            tuple[int, int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        # Patch-embed like a linear projection (DiT initializes the patchify conv as
        # if it were the equivalent Linear).
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

    def _rope(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.hidden // self.heads
        grid = (self.grid_size, self.grid_size, self.grid_size)
        key = (*grid, device, dtype)
        cached = self._rope_cache.get(key)
        if cached is None:
            cos, sin = build_axial_rope(grid, head_dim, device)
            cos, sin = cos.to(dtype), sin.to(dtype)
            self._rope_cache[key] = (cos, sin)
            return cos, sin
        return cached

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C, d, d, d) -> tokens (B, T, hidden), T = grid^3 (row-major d,h,w)."""
        h: torch.Tensor = self.patch_embed(z)  # (B, hidden, gd, gh, gw)
        return h.flatten(2).transpose(1, 2).contiguous()

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, patch^3 * C) -> velocity (B, C, d, d, d)."""
        b = tokens.shape[0]
        g = self.grid_size
        p = self.patch_size
        c = self.latent_channels
        x = tokens.reshape(b, g, g, g, c, p, p, p)
        # (B, gd, gh, gw, C, pd, ph, pw) -> (B, C, gd, pd, gh, ph, gw, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return x.reshape(b, c, g * p, g * p, g * p)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """z_t: (B, C, d, d, d); t: (B,) in [0, 1]; cond per the COND contract."""
        b = z_t.shape[0]
        x = self._patchify(z_t)  # (B, num_patches, hidden)
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(b, -1, -1)
            x = torch.cat([reg, x], dim=1)
        cos, sin = self._rope(z_t.device, x.dtype)
        c = self.cond_embed(t, cond)
        for block in self.blocks:
            x = block(x, c, cos, sin)
        x = x[:, self.num_register_tokens :]  # drop register tokens
        x = self.final(x, c)
        return self._unpatchify(x)


# ---------------------------------------------------------------------------
# Rectified-flow / linear stochastic interpolant objective + Euler sampler
# ---------------------------------------------------------------------------


@dataclass
class Interpolant:
    """One linear-interpolant training sample.

    x_t = (1 - t) * x0 + t * x1 ; target = x1 - x0. Predicting `target` from
    `(x_t, t)` is rectified flow (Liu 2022) / the linear stochastic interpolant
    (Albergo 2023) — the velocity that transports noise x0 (t=0) to data x1 (t=1).
    """

    x_t: torch.Tensor
    t: torch.Tensor
    target: torch.Tensor


class FlowMatchingObjective(nn.Module):
    """Sample the linear interpolant for a batch of data latents.

    Stateless (no parameters); a Module only so it composes with `.to(device)` and
    reads cleanly in the LitModule. `sample_interpolant(x1)` draws x0 ~ N(0, I) and
    t ~ U(0, 1) per sample and returns (x_t, t, target=x1-x0).
    """

    def sample_interpolant(
        self, x1: torch.Tensor, generator: torch.Generator | None = None
    ) -> Interpolant:
        b = x1.shape[0]
        x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        t = torch.rand(b, device=x1.device, dtype=x1.dtype, generator=generator)
        t_b = t.view(b, *([1] * (x1.dim() - 1)))
        x_t = (1.0 - t_b) * x0 + t_b * x1
        target = x1 - x0
        return Interpolant(x_t=x_t, t=t, target=target)

    def forward(self, x1: torch.Tensor, generator: torch.Generator | None = None) -> Interpolant:
        return self.sample_interpolant(x1, generator=generator)


@torch.no_grad()
def sample(
    model: MMDiT3D,
    shape: Sequence[int],
    cond: Mapping[str, torch.Tensor],
    num_steps: int = 50,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Euler-integrate dz/dt = v(z, t, cond) from noise (t=0) to a latent (t=1).

    `shape` is (B, C, d, d, d). Returns (B, C, d, d, d).

    v0 limitation (documented deliberately): for the per-subject lifespan proxy we
    fix the SAME initial noise x0 across the ages we sweep (pass a seeded
    `generator`), so that varying only `cond["age"]` moves a *single* subject along
    the age axis rather than resampling a new identity per age. This couples the
    sampled identity to the seed, not to any subject embedding — a true per-subject
    latent is Phase 3+ work, not v0. Without a fixed generator the identity drifts
    with every call.
    """
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    b = shape[0]
    z = torch.randn(tuple(shape), device=dev, generator=generator)
    cond = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}
    was_training = model.training
    model.eval()
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((b,), t_val, device=dev, dtype=z.dtype)
        v = model(z, t, cond)
        z = z + dt * v
    if was_training:
        model.train()
    return z
