"""Tests for the lifespan-flow evaluation library (neurodrift.eval.flow_eval).

Tiny dims + a tiny stand-in flow model so the whole verdict path — age-sweep
sampling, population-mean MAE, envelope coverage, smoothness, and the decoded
proxies — runs on CPU in seconds. This is the gate that stops a silent eval bug
from manufacturing a false "the flow works" conclusion on the B200s.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurodrift.eval.flow_eval import (  # noqa: E402
    age_sweep_latents,
    build_cond,
    central_slab_ventricle_fraction,
    cortical_rim_fraction,
    dark_core_fraction,
    envelope_coverage,
    foreground_fraction,
    pearson_r,
    per_channel_age_correlation,
    population_mean_mae,
    sample_population,
    trajectory_smoothness,
)
from neurodrift.models.flow import CATEGORICAL_FIELDS, MMDiT3D  # noqa: E402

TINY = dict(
    latent_channels=4,
    latent_size=8,
    patch_size=2,
    hidden=32,
    depth=2,
    heads=4,
    num_register_tokens=2,
)
CARDS = {"sex": 2, "dx": 4, "apoe": 6, "treatment": 6, "cohort": 8}
SHAPE = (4, 8, 8, 8)  # (C, d, d, d) matching TINY


@pytest.fixture
def tiny_model() -> MMDiT3D:
    return MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def test_build_cond_age_fill_and_null_defaults() -> None:
    cond = build_cond(42.0, batch=3)
    assert cond["age"].shape == (3,)
    assert torch.allclose(cond["age"], torch.full((3,), 42.0))
    for field in CATEGORICAL_FIELDS:
        assert cond[field].shape == (3,)
        assert cond[field].dtype == torch.long
        assert int(cond[field].abs().sum()) == 0  # NULL index 0


def test_build_cond_overrides_one_slot() -> None:
    cond = build_cond(30.0, batch=2, cond_overrides={"cohort": 3})
    assert torch.equal(cond["cohort"], torch.full((2,), 3, dtype=torch.long))
    assert int(cond["sex"].abs().sum()) == 0  # others still NULL


# ---------------------------------------------------------------------------
# Sampling wrappers
# ---------------------------------------------------------------------------
def test_age_sweep_shape_and_finite(tiny_model: MMDiT3D) -> None:
    ages = [10.0, 30.0, 50.0, 70.0]
    out = age_sweep_latents(tiny_model, ages, latent_shape=SHAPE, seed=0, num_steps=3)
    assert out.shape == (4, *SHAPE)
    assert torch.isfinite(out).all()


def test_age_sweep_fixed_seed_is_reproducible(tiny_model: MMDiT3D) -> None:
    ages = [20.0, 60.0]
    a = age_sweep_latents(tiny_model, ages, latent_shape=SHAPE, seed=7, num_steps=3)
    b = age_sweep_latents(tiny_model, ages, latent_shape=SHAPE, seed=7, num_steps=3)
    assert torch.allclose(a, b, atol=1e-6)


def test_sample_population_distinct_identities(tiny_model: MMDiT3D) -> None:
    pop = sample_population(tiny_model, age=40.0, n=3, latent_shape=SHAPE, base_seed=0, num_steps=3)
    assert pop.shape == (3, *SHAPE)
    assert torch.isfinite(pop).all()
    # Different seeds -> different draws (the population spread, not one identity).
    assert not torch.allclose(pop[0], pop[1], atol=1e-4)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_pearson_perfect_and_anti() -> None:
    x = torch.arange(10.0)
    assert pytest.approx(pearson_r(x, 2 * x + 1), abs=1e-5) == 1.0
    assert pytest.approx(pearson_r(x, -3 * x), abs=1e-5) == -1.0


def test_pearson_constant_is_nan() -> None:
    x = torch.arange(5.0)
    assert torch.isnan(torch.tensor(pearson_r(x, torch.ones(5))))


def test_pearson_drops_nan_pairs() -> None:
    x = torch.tensor([1.0, 2.0, float("nan"), 4.0])
    y = torch.tensor([2.0, 4.0, 100.0, 8.0])  # y = 2x on the finite pairs
    assert pytest.approx(pearson_r(x, y), abs=1e-5) == 1.0


def test_population_mean_mae_zero_when_identical() -> None:
    real = torch.randn(4, 4, 4)
    m = population_mean_mae(real.clone(), real)
    assert m["mae"] == 0.0
    assert m["nmae_range"] == 0.0


def test_population_mean_mae_normalisations() -> None:
    real = torch.tensor([0.0, 2.0, 4.0])  # range 4, mean|.| = 2
    pred = real + 1.0  # MAE = 1
    m = population_mean_mae(pred, real)
    assert pytest.approx(m["mae"], abs=1e-6) == 1.0
    assert pytest.approx(m["nmae_range"], abs=1e-6) == 0.25
    assert pytest.approx(m["nmae_l1"], abs=1e-6) == 0.5


def test_population_mean_mae_shape_guard() -> None:
    with pytest.raises(ValueError):
        population_mean_mae(torch.randn(3), torch.randn(4))


def test_envelope_coverage_inside_and_outside() -> None:
    # samples element 0 spans 0..10 -> 90% band is [0.5, 9.5] (linear-interp quantiles).
    samples = torch.arange(11.0).view(11, 1)
    assert envelope_coverage(samples, torch.tensor([[5.0]]), level=0.90) == 1.0  # inside
    assert envelope_coverage(samples, torch.tensor([[100.0]]), level=0.90) == 0.0  # outside
    both = torch.tensor([[0.0], [10.0]])  # both beyond [0.5, 9.5]
    assert envelope_coverage(samples, both, level=0.90) == 0.0


def test_envelope_coverage_partial() -> None:
    samples = torch.arange(11.0).view(11, 1)
    reals = torch.tensor([[5.0], [100.0]])  # one in, one out -> 0.5
    assert pytest.approx(envelope_coverage(samples, reals, level=0.90), abs=1e-6) == 0.5


def test_envelope_coverage_feature_shape_guard() -> None:
    with pytest.raises(ValueError):
        envelope_coverage(torch.randn(5, 3), torch.randn(2, 4))


def test_trajectory_smoothness_monotone_line() -> None:
    curve = torch.linspace(0.0, 9.0, 10)  # perfectly increasing, equal steps
    s = trajectory_smoothness(curve)
    assert s["monotonic_frac"] == 1.0
    assert pytest.approx(s["mean_roughness"], abs=1e-5) == 0.0
    assert s["max_jump_frac"] < 0.2  # one step is a small slice of the whole range


def test_trajectory_smoothness_flags_a_jump() -> None:
    curve = torch.tensor([0.0, 0.0, 0.0, 10.0, 10.0])  # one big discontinuity
    s = trajectory_smoothness(curve)
    assert pytest.approx(s["max_jump"], abs=1e-6) == 10.0
    assert pytest.approx(s["max_jump_frac"], abs=1e-6) == 1.0


def test_trajectory_smoothness_needs_two_points() -> None:
    with pytest.raises(ValueError):
        trajectory_smoothness(torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# Channel-resolved age correlation
# ---------------------------------------------------------------------------
def test_per_channel_isolates_the_signal_channel() -> None:
    # Channel 1's energy rises linearly with age; channels 0 and 2 stay flat.
    a, c = 5, 3
    swept = torch.ones(a, c, 2, 2, 2)
    for age_i in range(a):
        swept[age_i, 1] = float(age_i + 1)
    out = per_channel_age_correlation(swept, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert out["n_channels"] == 3
    assert out["argmax_channel"] == 1
    assert pytest.approx(out["max_abs_r"], abs=1e-5) == 1.0
    assert out["n_strong"] == 1  # only the one real channel; flat channels are NaN, dropped


def test_per_channel_accepts_precomputed_energy_2d() -> None:
    energy = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])  # ch0 rises, ch1 flat
    out = per_channel_age_correlation(energy, [10.0, 20.0, 30.0])
    assert out["argmax_channel"] == 0
    assert pytest.approx(out["max_abs_r"], abs=1e-5) == 1.0


def test_per_channel_all_flat_is_graceful() -> None:
    out = per_channel_age_correlation(torch.ones(3, 2), [1.0, 2.0, 3.0])
    assert out["argmax_channel"] == -1
    assert out["n_strong"] == 0
    assert torch.isnan(torch.tensor(out["max_abs_r"]))


def test_per_channel_age_count_guard() -> None:
    with pytest.raises(ValueError):
        per_channel_age_correlation(torch.randn(4, 2), [1.0, 2.0])


# ---------------------------------------------------------------------------
# Decoded-volume proxies
# ---------------------------------------------------------------------------
def test_foreground_fraction_half_bright() -> None:
    vol = torch.zeros(1, 4, 4, 4)
    vol[:, :2] = 1.0  # half the voxels bright
    frac = foreground_fraction(vol, thresh=0.1)
    assert frac.shape == (1,)
    assert pytest.approx(float(frac[0]), abs=1e-6) == 0.5


def test_foreground_fraction_accepts_unbatched() -> None:
    vol = torch.ones(4, 4, 4)
    frac = foreground_fraction(vol, thresh=0.1)
    assert frac.shape == (1,)
    assert pytest.approx(float(frac[0]), abs=1e-6) == 1.0


def test_dark_core_fraction_dark_vs_bright_center() -> None:
    # Bright everywhere, then carve a dark central core -> high dark-core fraction.
    dark = torch.ones(1, 8, 8, 8)
    dark[:, 2:6, 2:6, 2:6] = 0.0
    bright = torch.ones(1, 8, 8, 8)
    fd = float(dark_core_fraction(dark, dark_thresh=0.15, core=0.5)[0])
    fb = float(dark_core_fraction(bright, dark_thresh=0.15, core=0.5)[0])
    assert fd > fb
    assert fb == 0.0  # nothing dark in the bright volume


def test_proxy_shape_guard() -> None:
    with pytest.raises(ValueError):
        foreground_fraction(torch.randn(2, 3))  # not 3- or 4-D


# ---------------------------------------------------------------------------
# Regional proxies (sharper than the gross whole-volume fractions)
# ---------------------------------------------------------------------------
def test_central_slab_localises_deep_csf_more_than_blunt_core() -> None:
    # A tiny dark cube exactly at the deep center: the tight central crop reads it as a
    # large fraction; the blunt core=0.5 crop dilutes it with surrounding bright tissue.
    vol = torch.ones(1, 10, 10, 10)
    vol[:, 4:6, 4:6, 4:6] = 0.0
    fs = float(central_slab_ventricle_fraction(vol, core=0.3, dark_thresh=0.15)[0])
    fd = float(dark_core_fraction(vol, core=0.5, dark_thresh=0.15)[0])
    assert fs > fd  # sharper localisation of the deep dark region


def test_central_slab_zero_when_no_dark() -> None:
    assert float(central_slab_ventricle_fraction(torch.ones(1, 8, 8, 8))[0]) == 0.0


def test_cortical_rim_falls_as_periphery_empties() -> None:
    young = torch.ones(1, 12, 12, 12)  # tissue out to the periphery
    old = torch.zeros(1, 12, 12, 12)
    old[:, 4:8, 4:8, 4:8] = 1.0  # tissue only in the core; rim emptied (atrophy)
    fy = float(cortical_rim_fraction(young)[0])
    fo = float(cortical_rim_fraction(old)[0])
    assert fy > fo
    assert pytest.approx(fy, abs=1e-6) == 1.0  # bright everywhere -> full rim


def test_cortical_rim_accepts_unbatched() -> None:
    assert cortical_rim_fraction(torch.ones(8, 8, 8)).shape == (1,)


def test_cortical_rim_inner_outer_guard() -> None:
    with pytest.raises(ValueError):
        cortical_rim_fraction(torch.ones(1, 8, 8, 8), inner=0.9, outer=0.5)


# ---------------------------------------------------------------------------
# Script helpers (scripts/flow_eval.py)
# ---------------------------------------------------------------------------
def test_quantile_bins_balanced_and_ordered() -> None:
    from scripts.flow_eval import _quantile_bins

    ages = torch.tensor([5.0, 80.0, 20.0, 60.0, 35.0, 70.0, 10.0, 90.0])
    bins = _quantile_bins(ages, k=4)
    assert len(bins) == 4
    # Equal-count membership and ascending, non-overlapping age ranges.
    assert all(idx.numel() == 2 for _, _, idx in bins)
    his = [hi for _, hi, _ in bins]
    los = [lo for lo, _, _ in bins]
    assert los == sorted(los) and his == sorted(his)


def test_quantile_bins_handles_few_samples() -> None:
    from scripts.flow_eval import _quantile_bins

    bins = _quantile_bins(torch.tensor([42.0]), k=4)
    assert len(bins) == 1
    assert bins[0][2].numel() == 1


def test_load_flow_weights_prefers_ema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A Lightning-style state with model.* AND ema_model.* loads the EMA by default."""
    from scripts.flow_eval import _load_flow_weights

    target = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    live = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    ema = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    # Make the EMA weights distinctive so we can prove which copy was loaded.
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(7.0)
    state = {f"model.{k}": v for k, v in live.state_dict().items()}
    state.update({f"ema_model.{k}": v for k, v in ema.state_dict().items()})
    ckpt = tmp_path / "flow.ckpt"
    torch.save({"state_dict": state}, ckpt)

    src = _load_flow_weights(target, ckpt, prefer_ema=True)
    assert src == "ema"
    for k, v in target.state_dict().items():
        assert torch.allclose(v, ema.state_dict()[k])

    assert _load_flow_weights(target, ckpt, prefer_ema=False) == "model"


def test_load_flow_weights_bare_state_dict(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A bare (un-prefixed) state-dict still loads."""
    from scripts.flow_eval import _load_flow_weights

    target = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    src = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    ckpt = tmp_path / "bare.ckpt"
    torch.save(src.state_dict(), ckpt)
    assert _load_flow_weights(target, ckpt, prefer_ema=True) == "bare"
