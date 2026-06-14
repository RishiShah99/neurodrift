"""Shape/contract tests for the TopK-SAE, its LitModule, and the probe math.

Tiny dims so the whole suite (including a real Lightning fast_dev_run over a
synthetic latent loader) runs on CPU in seconds — the gate that keeps a silent
SAE bug off the B200s. Synthetic in-memory tensors only; no zarr, no Agent D
datamodule import (referenced by _target_ in the experiment config only).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning as L  # noqa: E402
from neurodrift.models.sae import SAEOutput, TopKSAE  # noqa: E402
from neurodrift.train.sae_module import SAELitModule, _tokens  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

D_IN, D_HIDDEN, K, AUX_K = 8, 64, 4, 8


@pytest.fixture
def tiny_sae() -> TopKSAE:
    return TopKSAE(d_in=D_IN, d_hidden=D_HIDDEN, k=K, aux_k=AUX_K, dead_steps_threshold=2)


# ---------------------------------------------------------------------------
# Model: shapes
# ---------------------------------------------------------------------------
def test_encode_decode_shapes(tiny_sae: TopKSAE) -> None:
    x = torch.randn(10, D_IN)
    acts, indices = tiny_sae.encode(x)
    assert acts.shape == (10, D_HIDDEN)
    assert indices.shape == (10, K)
    x_hat = tiny_sae.decode(acts)
    assert x_hat.shape == x.shape


def test_forward_returns_dataclass(tiny_sae: TopKSAE) -> None:
    x = torch.randn(10, D_IN)
    out = tiny_sae(x)
    assert isinstance(out, SAEOutput)
    assert out.x_hat.shape == x.shape
    assert out.acts.shape == (10, D_HIDDEN)
    assert out.indices.shape == (10, K)
    assert out.aux_loss.shape == ()


# ---------------------------------------------------------------------------
# Model: EXACT TopK sparsity
# ---------------------------------------------------------------------------
def test_exact_topk_sparsity(tiny_sae: TopKSAE) -> None:
    """Every token row has EXACTLY k nonzero entries (the headline invariant)."""
    x = torch.randn(32, D_IN)
    out = tiny_sae(x)
    nnz = (out.acts != 0).sum(dim=1)
    assert torch.equal(nnz, torch.full((32,), K, dtype=nnz.dtype))


def test_topk_indices_are_the_nonzeros(tiny_sae: TopKSAE) -> None:
    """The returned indices index exactly the nonzero positions of each row."""
    x = torch.randn(5, D_IN)
    out = tiny_sae(x)
    for i in range(5):
        nz = set(out.acts[i].nonzero(as_tuple=False).flatten().tolist())
        assert nz.issubset(set(out.indices[i].tolist()))
        assert len(nz) == K


# ---------------------------------------------------------------------------
# Model: reconstruction improves on a zero baseline after a few steps
# ---------------------------------------------------------------------------
def test_recon_beats_zero_baseline_after_fit(tiny_sae: TopKSAE) -> None:
    """x_hat finite; a few manual opt steps drive MSE below the all-zero baseline."""
    torch.manual_seed(0)
    x = torch.randn(128, D_IN)
    opt = torch.optim.Adam(tiny_sae.parameters(), lr=1e-2)
    out0 = tiny_sae(x)
    assert torch.isfinite(out0.x_hat).all()
    mse0 = (out0.x_hat - x).pow(2).mean().item()
    for _ in range(50):
        opt.zero_grad()
        out = tiny_sae(x)
        loss = (out.x_hat - x).pow(2).mean() + out.aux_loss
        loss.backward()
        opt.step()
        tiny_sae.normalize_decoder()
    mse_fit = (tiny_sae(x).x_hat - x).pow(2).mean().item()
    zero_baseline = x.pow(2).mean().item()
    assert mse_fit < zero_baseline
    assert mse_fit < mse0


# ---------------------------------------------------------------------------
# Model: dead-feature handling + decoder normalisation
# ---------------------------------------------------------------------------
def test_aux_loss_is_finite_and_computed(tiny_sae: TopKSAE) -> None:
    x = torch.randn(16, D_IN)
    # No latent has fired yet, but steps_since_fired starts at 0 (<= threshold), so
    # nothing is dead and aux_loss is exactly 0.
    assert float(tiny_sae(x).aux_loss) == 0.0
    # Drive the dead clock past the threshold for latents that never fire.
    for _ in range(5):
        _, idx = tiny_sae.encode(x)
        tiny_sae.update_dead_tracker(idx)
    assert int(tiny_sae.dead_mask.sum()) > 0, "some latents should be dead after idle steps"
    aux = tiny_sae(x).aux_loss.detach()
    assert torch.isfinite(aux) and float(aux) > 0.0


def test_normalize_decoder_unit_norm(tiny_sae: TopKSAE) -> None:
    with torch.no_grad():
        tiny_sae.decoder.weight.mul_(7.0)  # break unit norm
    tiny_sae.normalize_decoder()
    col_norms = tiny_sae.decoder.weight.norm(dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


def test_dead_tracker_resets_fired_latents(tiny_sae: TopKSAE) -> None:
    x = torch.randn(4, D_IN)
    _, idx = tiny_sae.encode(x)
    tiny_sae.update_dead_tracker(idx)
    fired = torch.unique(idx)
    assert torch.all(tiny_sae.steps_since_fired[fired] == 0)


def test_resample_dead_revives(tiny_sae: TopKSAE) -> None:
    x = torch.randn(32, D_IN)
    for _ in range(5):
        _, idx = tiny_sae.encode(x)
        tiny_sae.update_dead_tracker(idx)
    n_dead = int(tiny_sae.dead_mask.sum())
    assert n_dead > 0
    revived = tiny_sae.resample_dead(x)
    assert revived > 0
    assert int(tiny_sae.dead_mask.sum()) < n_dead


# ---------------------------------------------------------------------------
# Probe math
# ---------------------------------------------------------------------------
def test_feature_biomarker_correlation_in_range() -> None:
    from scripts.sae_probe import feature_biomarker_correlation

    torch.manual_seed(0)
    acts = torch.randn(40, D_HIDDEN)
    labels = torch.randn(40)
    r = feature_biomarker_correlation(acts, labels)
    assert r.shape == (D_HIDDEN,)
    assert torch.isfinite(r).all()
    assert float(r.min()) >= -1.0 and float(r.max()) <= 1.0


def test_feature_biomarker_correlation_recovers_signal() -> None:
    """A feature constructed to track the label must score high |r|."""
    from scripts.sae_probe import feature_biomarker_correlation

    torch.manual_seed(0)
    labels = torch.randn(60)
    acts = torch.randn(60, D_HIDDEN)
    acts[:, 3] = labels * 2.0 + 0.01 * torch.randn(60)  # feature 3 ~ label
    r = feature_biomarker_correlation(acts, labels)
    assert abs(float(r[3])) > 0.9


def test_correlation_drops_nan_labels() -> None:
    from scripts.sae_probe import feature_biomarker_correlation

    acts = torch.randn(10, D_HIDDEN)
    labels = torch.full((10,), float("nan"))
    labels[:3] = torch.tensor([1.0, 2.0, 3.0])  # only 3 finite -> still computable
    r = feature_biomarker_correlation(acts, labels)
    assert torch.isfinite(r).all()


def test_top_correlated_features_ranks_by_abs_r() -> None:
    from scripts.sae_probe import top_correlated_features

    torch.manual_seed(0)
    labels = torch.randn(50)
    acts = torch.randn(50, D_HIDDEN)
    acts[:, 7] = -labels * 3.0  # strong NEGATIVE correlation
    top = top_correlated_features(acts, labels, top=5)
    assert len(top) == 5
    assert top[0][0] == 7  # strongest |r| first
    assert top[0][1] < 0  # sign preserved


def test_proxy_biomarkers_shapes() -> None:
    from scripts.sae_probe import proxy_biomarkers

    z = torch.randn(6, D_IN, 4, 4, 4)
    proxies = proxy_biomarkers(z)
    assert set(proxies) == {
        "proxy_latent_energy",
        "proxy_active_fraction",
        "proxy_spatial_spread",
    }
    for v in proxies.values():
        assert v.shape == (6,)
        assert torch.isfinite(v).all()


def test_aging_direction_shape_and_sign() -> None:
    from scripts.sae_probe import aging_direction

    young = torch.zeros(5, D_IN, 4, 4, 4)
    old = torch.ones(5, D_IN, 4, 4, 4)
    latents = torch.cat([young, old], dim=0)
    ages = torch.tensor([20.0] * 5 + [70.0] * 5)
    vec = aging_direction(latents, ages, threshold=60.0)
    assert vec.shape == (D_IN, 4, 4, 4)
    # old (ones) - young (zeros) == ones
    assert torch.allclose(vec, torch.ones_like(vec))


def test_aging_direction_empty_population_is_zero() -> None:
    from scripts.sae_probe import aging_direction

    latents = torch.randn(4, D_IN, 4, 4, 4)
    ages = torch.tensor([20.0, 25.0, 30.0, 35.0])  # all young
    vec = aging_direction(latents, ages, threshold=60.0)
    assert vec.shape == (D_IN, 4, 4, 4)
    assert float(vec.abs().sum()) == 0.0


def test_apply_steering_moves_by_scale_times_vector() -> None:
    from scripts.sae_probe import apply_steering

    z = torch.randn(3, D_IN, 4, 4, 4)
    vec = torch.randn(D_IN, 4, 4, 4)
    out = apply_steering(z, vec, scale=2.5)
    assert torch.allclose(out - z, 2.5 * vec.unsqueeze(0).expand_as(z))
    assert not torch.equal(out, z)


# ---------------------------------------------------------------------------
# LitModule fast_dev_run over a synthetic latent loader
# ---------------------------------------------------------------------------
class _SyntheticLatents(Dataset):
    """Minimal latent batch matching neurodrift.data.latents.LatentDataModule."""

    def __init__(self, n: int = 4, c: int = D_IN, d: int = 4) -> None:
        self.n, self.c, self.d = n, c, d

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "z": torch.randn(self.c, self.d, self.d, self.d),
            "age": torch.tensor(40.0 + idx),
            "sex": torch.tensor(-1),
            "dx": torch.tensor(-1),
            "apoe": torch.tensor(-1),
            "treatment": torch.tensor(0),
            "cohort": "synthetic",
            "subject": f"sub-{idx}",
            "session": "",
        }


def _collate(samples: list[dict]) -> dict:
    return {
        "z": torch.stack([s["z"] for s in samples]),
        "age": torch.stack([s["age"] for s in samples]),
        "sex": torch.stack([s["sex"] for s in samples]),
        "cohort": [s["cohort"] for s in samples],
        "subject": [s["subject"] for s in samples],
    }


def _make_lit() -> SAELitModule:
    model = TopKSAE(d_in=D_IN, d_hidden=D_HIDDEN, k=K, aux_k=AUX_K, dead_steps_threshold=2)

    def opt_partial(params):  # type: ignore[no-untyped-def]
        return torch.optim.AdamW(params, lr=1e-3)

    def sched_partial(opt):  # type: ignore[no-untyped-def]
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    return SAELitModule(model=model, optimizer_partial=opt_partial, scheduler_partial=sched_partial)


def test_tokens_reshape_is_per_voxel() -> None:
    z = torch.randn(2, D_IN, 4, 4, 4)
    tokens = _tokens(z)
    assert tokens.shape == (2 * 4 * 4 * 4, D_IN)
    # token 0 must equal z[0, :, 0, 0, 0] (channel vector at the first voxel)
    assert torch.allclose(tokens[0], z[0, :, 0, 0, 0])


def test_fast_dev_run_trains() -> None:
    """Full Lightning step through configure_optimizers + training/validation_step.

    Asserts the run completes, the loss stays finite, and L0 == k (exactly k
    nonzeros per token) — the exact path that would run on the B200s."""
    lit = _make_lit()
    loader = DataLoader(_SyntheticLatents(), batch_size=2, collate_fn=_collate)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=loader, val_dataloaders=loader)

    batch = next(iter(loader))
    out = lit.model(_tokens(batch["z"]))
    loss = (out.x_hat - _tokens(batch["z"])).pow(2).mean() + out.aux_loss
    assert torch.isfinite(loss)
    l0 = (out.acts != 0).sum(dim=1)
    assert torch.equal(l0, torch.full_like(l0, K))
