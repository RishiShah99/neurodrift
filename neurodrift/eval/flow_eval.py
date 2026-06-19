"""Canonical v0 evaluation of the lifespan flow backbone (Phase 2).

`flow_v0.ckpt` is trained but, until this module, was *never evaluated* — the open
question is whether the flow learned real lifespan structure or just memorised the
marginal latent distribution. This is the single-process, deterministic check that
answers it, mirroring `neurodrift.eval.runner.evaluate` for the VAE: run it AFTER a
cook on one GPU (or CPU); never trust live DDP training loss for the verdict.

What it measures (maps to PLAN.md §6 v0, Phase 2 acceptance):

  * **Population-mean MAE** — sample N latents at a target age, compare the sampled
    population mean to the held-out REAL population mean at that age. PLAN target:
    per-region MAE < 2.5%. At v0 (no FreeSurfer parcellation on walk-up data) the
    "regions" are the 16 VAE content-latent channels, with a decoded-voxel variant
    when a VAE checkpoint is supplied.
  * **Age-trajectory** — fix the sampling noise (one identity) and sweep age; the
    decoded brain must move monotonically and smoothly with age (ventricles enlarge,
    tissue volume falls). Quantified by `trajectory_smoothness` (no jumps) and the
    Pearson correlation of each proxy with age (right direction).
  * **Uncertainty envelope coverage** — the stochastic-interpolant promise is a
    *calibrated* sampled envelope. The 90% per-element envelope from N samples at an
    age should contain ~90% of held-out real elements at that age. PLAN target:
    empirical coverage within 5% of nominal.

Everything here is pure tensor ops + thin wrappers over `neurodrift.models.flow.sample`,
so the whole module unit-tests on CPU with a tiny stand-in model — the gate that keeps a
silent eval bug from producing a false "the flow works" verdict on the B200s.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from neurodrift.models.flow import CATEGORICAL_FIELDS, sample

# ---------------------------------------------------------------------------
# Conditioning + sampling wrappers
# ---------------------------------------------------------------------------


def build_cond(
    age: float,
    batch: int,
    device: torch.device | str = "cpu",
    cond_overrides: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Flow COND dict for a constant age + NULL categoricals (the v0 contract).

    `age` fills the whole (B,) age vector; every categorical defaults to its reserved
    NULL index 0 (the corpus has no reliable labels at v0) unless given in
    `cond_overrides` (e.g. {"cohort": 3}), which sets that slot's id for all B.
    """
    dev = torch.device(device)
    cond: dict[str, torch.Tensor] = {
        "age": torch.full((batch,), float(age), dtype=torch.float32, device=dev)
    }
    overrides = dict(cond_overrides or {})
    for field in CATEGORICAL_FIELDS:
        val = int(overrides.get(field, 0))
        cond[field] = torch.full((batch,), val, dtype=torch.long, device=dev)
    return cond


def age_sweep_latents(
    model: Any,
    ages: Sequence[float],
    *,
    latent_shape: Sequence[int],
    seed: int = 0,
    num_steps: int = 50,
    device: torch.device | str = "cpu",
    cond_overrides: Mapping[str, int] | None = None,
) -> torch.Tensor:
    """Sample one latent per age with a FIXED initial noise (a single identity).

    Re-seeding the generator with the SAME `seed` before each age makes `sample`
    draw the identical x0 every time, so only `cond["age"]` changes — the per-subject
    lifespan-trajectory proxy documented on `flow.sample`. Returns (A, *latent_shape).
    """
    dev = torch.device(device)
    shape = (1, *latent_shape)
    out: list[torch.Tensor] = []
    for age in ages:
        # The generator must live on the SAME device as the noise `sample` draws,
        # else torch.randn(device=cuda, generator=cpu_gen) raises on a GPU run.
        gen = torch.Generator(device=dev).manual_seed(seed)
        cond = build_cond(age, 1, dev, cond_overrides)
        z = sample(model, shape, cond, num_steps=num_steps, generator=gen, device=dev)
        out.append(z[0])
    return torch.stack(out, dim=0)


def sample_population(
    model: Any,
    age: float,
    *,
    n: int,
    latent_shape: Sequence[int],
    base_seed: int = 0,
    num_steps: int = 50,
    device: torch.device | str = "cpu",
    cond_overrides: Mapping[str, int] | None = None,
) -> torch.Tensor:
    """Sample `n` DISTINCT identities at one age (seed varies per sample).

    Unlike `age_sweep_latents` (one identity across ages), here the seed advances per
    sample so the draws span the conditional population at `age` — the input to
    `population_mean_mae` and `envelope_coverage`. Returns (n, *latent_shape).
    """
    dev = torch.device(device)
    shape = (1, *latent_shape)
    out: list[torch.Tensor] = []
    for i in range(n):
        gen = torch.Generator(device=dev).manual_seed(base_seed + i)
        cond = build_cond(age, 1, dev, cond_overrides)
        z = sample(model, shape, cond, num_steps=num_steps, generator=gen, device=dev)
        out.append(z[0])
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """NaN-safe Pearson correlation over the finite pairs of two 1-D tensors.

    Returns NaN if fewer than 2 finite pairs remain or either side is constant
    (zero variance) — the same degenerate cases the age proxies can hit on a tiny
    or single-cohort slice.
    """
    x = x.flatten().float()
    y = y.flatten().float()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[mask], y[mask]
    if x.numel() < 2:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = xc.norm() * yc.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((xc @ yc) / denom)


def per_channel_age_correlation(
    swept: torch.Tensor, ages: Sequence[float] | torch.Tensor
) -> dict[str, Any]:
    """Per-latent-channel energy-vs-age correlation across an age sweep.

    `swept` is the fixed-identity age sweep — (A, C, *spatial) from `age_sweep_latents`.
    For each channel we take its mean |activation| over the spatial dims at each age
    (an (A,) energy curve), then Pearson-correlate it with age. The pooled
    `latent_energy_vs_age_r` (one number over ALL channels) washes out a strong signal
    carried by a few channels against many flat ones; this resolves it per channel.

    Returns:
      * `per_channel_r`  — list length C, Pearson r of each channel's energy vs age,
      * `max_abs_r`      — strongest channel |r| (the real age-signal strength),
      * `argmax_channel` — which channel carries it,
      * `mean_abs_r`     — mean |r| over channels (how broadly age is encoded),
      * `n_strong`       — channels with |r| > 0.6 (a sharper read than the blunt mean),
      * `n_channels`     — C.
    NaN channels (constant energy) are dropped from the summaries, never counted strong.
    """
    a = torch.as_tensor(ages, dtype=torch.float32).flatten()
    if swept.dim() == 2:
        energy = swept.abs().float()  # already (A, C)
    elif swept.dim() >= 3:
        energy = swept.abs().float().flatten(2).mean(dim=2)  # (A, C)
    else:
        raise ValueError(f"swept must be (A, C) or (A, C, *spatial), got {tuple(swept.shape)}")
    if energy.shape[0] != a.numel():
        raise ValueError(f"age count {a.numel()} != sweep length {energy.shape[0]}")
    c = energy.shape[1]
    per_channel = [pearson_r(energy[:, j], a) for j in range(c)]
    r_t = torch.tensor(per_channel, dtype=torch.float32)
    finite = torch.isfinite(r_t)
    abs_r = r_t.abs()
    if bool(finite.any()):
        masked = abs_r.masked_fill(~finite, -1.0)
        argmax = int(masked.argmax())
        max_abs_r = float(abs_r[finite].max())
        mean_abs_r = float(abs_r[finite].mean())
        n_strong = int((abs_r[finite] > 0.6).sum())
    else:
        argmax, max_abs_r, mean_abs_r, n_strong = -1, float("nan"), float("nan"), 0
    return {
        "per_channel_r": per_channel,
        "max_abs_r": max_abs_r,
        "argmax_channel": argmax,
        "mean_abs_r": mean_abs_r,
        "n_strong": n_strong,
        "n_channels": c,
    }


def population_mean_mae(pred_mean: torch.Tensor, real_mean: torch.Tensor) -> dict[str, float]:
    """MAE between a predicted and a real population-mean tensor (same shape).

    Returns the raw MAE plus two normalisations so the PLAN "< 2.5%" target is
    expressible without a parcellation:
      * `nmae_range` = MAE / (real max - real min)  — fraction of the real dynamic range,
      * `nmae_l1`    = MAE / mean(|real|)           — fraction of the real signal scale.
    Either is a defensible "%" reading; the script reports both.
    """
    pred = pred_mean.flatten().float()
    real = real_mean.flatten().float()
    if pred.shape != real.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs real {real.shape}")
    mae = float((pred - real).abs().mean())
    rng = float(real.max() - real.min())
    l1 = float(real.abs().mean())
    return {
        "mae": mae,
        "nmae_range": mae / rng if rng > 0 else float("nan"),
        "nmae_l1": mae / l1 if l1 > 0 else float("nan"),
    }


def envelope_coverage(samples: torch.Tensor, reals: torch.Tensor, *, level: float = 0.90) -> float:
    """Empirical coverage of a per-element `level` envelope built from `samples`.

    `samples` (N, *feat) are model draws at an age; `reals` (M, *feat) are held-out
    real latents at that age. The per-element [lo, hi] quantile band at the central
    `level` mass is computed from `samples`; coverage is the fraction of real
    elements (over M x feat) that fall inside it. A calibrated stochastic interpolant
    gives coverage ~= level. Returns a float in [0, 1].
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if samples.shape[1:] != reals.shape[1:]:
        raise ValueError(f"feature shape mismatch: {samples.shape[1:]} vs {reals.shape[1:]}")
    tail = (1.0 - level) / 2.0
    q = torch.tensor([tail, 1.0 - tail], dtype=torch.float32)
    s = samples.flatten(1).float()  # (N, F)
    bounds = torch.quantile(s, q, dim=0)  # (2, F)
    lo, hi = bounds[0], bounds[1]
    r = reals.flatten(1).float()  # (M, F)
    inside = (r >= lo.unsqueeze(0)) & (r <= hi.unsqueeze(0))
    return float(inside.float().mean())


def trajectory_smoothness(curve: torch.Tensor) -> dict[str, float]:
    """Smoothness / monotonicity of a 1-D metric-vs-age curve (length A >= 2).

    The PLAN smoothness bar is "continuous, no jumps across cohort boundaries".
    Returns:
      * `range`          — max - min of the curve,
      * `max_jump`       — largest |first difference| between adjacent ages,
      * `max_jump_frac`  — max_jump / range  (a single step swallowing the whole
                           trajectory signals a discontinuity; small == smooth),
      * `mean_roughness` — mean |second difference| (curvature / wobble),
      * `monotonic_frac` — fraction of steps moving in the curve's overall direction
                           (1.0 == perfectly monotonic).
    """
    c = curve.flatten().float()
    if c.numel() < 2:
        raise ValueError("trajectory_smoothness needs at least 2 points")
    d1 = c[1:] - c[:-1]
    rng = float(c.max() - c.min())
    max_jump = float(d1.abs().max())
    overall = float(c[-1] - c[0])
    direction = 1.0 if overall >= 0 else -1.0
    monotonic_frac = float((torch.sign(d1) == direction).float().mean())
    if c.numel() >= 3:
        d2 = d1[1:] - d1[:-1]
        mean_roughness = float(d2.abs().mean())
    else:
        mean_roughness = 0.0
    return {
        "range": rng,
        "max_jump": max_jump,
        "max_jump_frac": max_jump / rng if rng > 0 else float("nan"),
        "mean_roughness": mean_roughness,
        "monotonic_frac": monotonic_frac,
    }


# ---------------------------------------------------------------------------
# Decoded-volume proxies (no FreeSurfer at v0)
# ---------------------------------------------------------------------------


def _as_batched(vol: torch.Tensor) -> torch.Tensor:
    """Accept (D,H,W) or (B,D,H,W); return (B,D,H,W)."""
    if vol.dim() == 3:
        return vol.unsqueeze(0)
    if vol.dim() == 4:
        return vol
    raise ValueError(f"expected a (D,H,W) or (B,D,H,W) volume, got shape {tuple(vol.shape)}")


def foreground_fraction(vol: torch.Tensor, *, thresh: float = 0.1) -> torch.Tensor:
    """Brain-tissue volume proxy: fraction of voxels brighter than `thresh * max`.

    `thresh` is relative to each volume's own max so the proxy is robust to the VAE's
    intensity scaling. Decoded T1 tissue is bright, background ~0, so this tracks
    total tissue volume — expected to FALL with age (atrophy). Returns (B,).
    """
    v = _as_batched(vol).float()
    b = v.shape[0]
    flat = v.reshape(b, -1)
    peak = flat.max(dim=1, keepdim=True).values.clamp_min(1e-8)
    return (flat > thresh * peak).float().mean(dim=1)


def dark_core_fraction(
    vol: torch.Tensor, *, dark_thresh: float = 0.15, core: float = 0.5
) -> torch.Tensor:
    """Ventricle / CSF proxy: dark-voxel fraction inside the central core crop.

    Ventricles are dark (CSF) and central; they enlarge with age. We crop the central
    `core` fraction of each spatial axis (avoiding the dark background rim) and report
    the fraction of voxels DIMMER than `dark_thresh * max`. Expected to RISE with age.
    Returns (B,). `dark_thresh` is relative to each volume's max (scale-robust).
    """
    v = _as_batched(vol).float()
    b, d, h, w = v.shape

    def _slice(n: int) -> slice:
        lo = round(n * (1.0 - core) / 2.0)
        hi = n - lo
        return slice(lo, max(hi, lo + 1))

    crop = v[:, _slice(d), _slice(h), _slice(w)]
    peak = v.reshape(b, -1).max(dim=1).values.clamp_min(1e-8).view(b, 1, 1, 1)
    return (crop < dark_thresh * peak).float().reshape(b, -1).mean(dim=1)


def _central_cube_mask(d: int, h: int, w: int, frac: float, device: torch.device) -> torch.Tensor:
    """Boolean (D,H,W) mask: the central `frac` of each spatial axis (a centered cube).

    `frac` in (0, 1]; 1.0 selects the whole volume, 0.3 the central 30% per axis.
    Shared by the regional proxies so their crop geometry is defined in one place.
    """

    def _ax(n: int) -> torch.Tensor:
        lo = round(n * (1.0 - frac) / 2.0)
        hi = max(n - lo, lo + 1)
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[lo:hi] = True
        return m

    md, mh, mw = _ax(d), _ax(h), _ax(w)
    return md[:, None, None] & mh[None, :, None] & mw[None, None, :]


def central_slab_ventricle_fraction(
    vol: torch.Tensor, *, core: float = 0.3, dark_thresh: float = 0.15
) -> torch.Tensor:
    """Sharper ventricle proxy: dark-voxel fraction in a TIGHT central cube.

    The lateral/third-ventricle bodies are deep and central; the blunt
    `dark_core_fraction` (core=0.5) also sweeps in peripheral sulcal CSF that dilutes
    the trajectory. Cropping a tighter central cube (default 30% per axis vs 50%)
    isolates the deep periventricular CSF, so the dark fraction tracks ventricular
    enlargement more specifically. Threshold is relative to each volume's max
    (scale-robust). Expected to RISE with age. Returns (B,).
    """
    v = _as_batched(vol).float()
    b, d, h, w = v.shape
    mask = _central_cube_mask(d, h, w, core, v.device).reshape(1, -1)
    peak = v.reshape(b, -1).max(dim=1, keepdim=True).values.clamp_min(1e-8)
    dark = (v.reshape(b, -1) < dark_thresh * peak) & mask
    return dark.float().sum(dim=1) / mask.float().sum().clamp_min(1.0)


def cortical_rim_fraction(
    vol: torch.Tensor, *, inner: float = 0.5, outer: float = 0.9, thresh: float = 0.1
) -> torch.Tensor:
    """Cortical-thinning proxy: bright-tissue fraction in the peripheral brain shell.

    The cortical ribbon sits at the brain's outer surface, not its core (deep WM /
    ventricles) nor the cube corners (background on an MNI-registered, skull-stripped
    volume). We measure the foreground fraction in the SHELL between the central
    `inner` cube and the `outer` cube — peripheral tissue that recedes as cortex thins
    and the brain atrophies. Threshold is relative to each volume's max (scale-robust).
    Expected to FALL with age. Returns (B,).
    """
    if not 0.0 < inner < outer <= 1.0:
        raise ValueError(f"need 0 < inner < outer <= 1, got inner={inner} outer={outer}")
    v = _as_batched(vol).float()
    b, d, h, w = v.shape
    shell = _central_cube_mask(d, h, w, outer, v.device) & ~_central_cube_mask(
        d, h, w, inner, v.device
    )
    shell_flat = shell.reshape(1, -1)
    peak = v.reshape(b, -1).max(dim=1, keepdim=True).values.clamp_min(1e-8)
    bright = (v.reshape(b, -1) > thresh * peak) & shell_flat
    return bright.float().sum(dim=1) / shell_flat.float().sum().clamp_min(1.0)
