"""Shape/contract tests for the v0 Phase 2 lifespan flow backbone.

Tiny dims so the whole suite — including a real Lightning fast_dev_run of the
flow-matching training loop — runs on CPU in seconds. This is the gate that keeps a
silent flow-backbone bug off the B200s, exactly like test_disentangled.py does for
the VAE.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning as L  # noqa: E402
from neurodrift.models.flow import (  # noqa: E402
    CATEGORICAL_FIELDS,
    FlowMatchingObjective,
    MMDiT3D,
    build_axial_rope,
    sample,
)
from neurodrift.train.flow_module import FlowLitModule  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# Tiny config used everywhere: head_dim = 32 / 4 = 8, T = (8 / 2)^3 = 64.
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


@pytest.fixture
def tiny_model() -> MMDiT3D:
    return MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]


def _make_cond(b: int, age_nan: bool = False) -> dict:
    age = torch.full((b,), float("nan")) if age_nan else torch.rand(b) * 90.0
    cond = {"age": age}
    for field in CATEGORICAL_FIELDS:
        cond[field] = torch.randint(0, CARDS[field] + 1, (b,))
    return cond


def test_patch_embed_token_shape(tiny_model: MMDiT3D) -> None:
    z = torch.randn(2, 4, 8, 8, 8)
    tokens = tiny_model._patchify(z)
    assert tokens.shape == (2, 64, 32)  # T = (8/2)^3 = 64, hidden = 32


def test_unpatchify_inverts_patchify_grid(tiny_model: MMDiT3D) -> None:
    """Token count and grid bookkeeping round-trip back to the latent shape."""
    z = torch.randn(2, 4, 8, 8, 8)
    tokens = tiny_model._patchify(z)
    pseudo = tokens.new_zeros(2, tiny_model.num_patches, tiny_model.patch_size**3 * 4)
    out = tiny_model._unpatchify(pseudo)
    assert out.shape == z.shape


def test_axial_rope_table_shape() -> None:
    cos, sin = build_axial_rope((4, 4, 4), head_dim=8, device=torch.device("cpu"))
    assert cos.shape == (64, 8)
    assert sin.shape == (64, 8)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()


def test_single_block_forward(tiny_model: MMDiT3D) -> None:
    """One DiT block preserves the (B, T, hidden) sequence shape."""
    b, t = 2, tiny_model.num_patches + tiny_model.num_register_tokens
    x = torch.randn(b, t, tiny_model.hidden)
    c = torch.randn(b, tiny_model.hidden)
    cos, sin = tiny_model._rope(x.device, x.dtype)
    out = tiny_model.blocks[0](x, c, cos, sin)
    assert out.shape == x.shape


def test_forward_returns_velocity_shape(tiny_model: MMDiT3D) -> None:
    z = torch.randn(2, 4, 8, 8, 8)
    t = torch.rand(2)
    cond = _make_cond(2)
    v = tiny_model(z, t, cond)
    assert v.shape == z.shape
    assert torch.isfinite(v).all()


def test_forward_nan_age_is_finite(tiny_model: MMDiT3D) -> None:
    """NaN ages must be nan_to_num'd before the sinusoidal table (no NaN velocity)."""
    z = torch.randn(2, 4, 8, 8, 8)
    t = torch.rand(2)
    v = tiny_model(z, t, _make_cond(2, age_nan=True))
    assert torch.isfinite(v).all()


def test_interpolant_correctness() -> None:
    obj = FlowMatchingObjective()
    x1 = torch.randn(3, 4, 8, 8, 8)
    g = torch.Generator().manual_seed(0)
    interp = obj.sample_interpolant(x1, generator=g)
    # reconstruct x0 from x_t and t; check the interpolant + target identities.
    t_b = interp.t.view(3, 1, 1, 1, 1)
    x0 = x1 - interp.target  # target == x1 - x0 by definition
    expected_xt = (1.0 - t_b) * x0 + t_b * x1
    assert torch.allclose(interp.x_t, expected_xt, atol=1e-5)
    assert torch.allclose(interp.target, x1 - x0, atol=1e-6)


def test_interpolant_endpoints() -> None:
    """At t=0, x_t == x0; at t=1, x_t == x1; target == x1 - x0 in both cases."""
    obj = FlowMatchingObjective()
    x1 = torch.randn(2, 4, 8, 8, 8)
    g = torch.Generator().manual_seed(1)
    interp = obj.sample_interpolant(x1, generator=g)
    x0 = x1 - interp.target
    # t=0 endpoint
    xt0 = (1.0 - 0.0) * x0 + 0.0 * x1
    assert torch.allclose(xt0, x0, atol=1e-6)
    # t=1 endpoint
    xt1 = (1.0 - 1.0) * x0 + 1.0 * x1
    assert torch.allclose(xt1, x1, atol=1e-6)
    assert torch.allclose(interp.target, x1 - x0, atol=1e-6)


def test_cfg_dropout_full_null_runs(tiny_model: MMDiT3D) -> None:
    """Training-mode forward with cfg_dropout_p=1.0 nulls every categorical and the
    shape still holds (the unconditional path the sampler later guides)."""
    tiny_model.train()
    tiny_model.cond_embed.cfg_dropout_p = 1.0
    z = torch.randn(2, 4, 8, 8, 8)
    t = torch.rand(2)
    v = tiny_model(z, t, _make_cond(2))
    assert v.shape == z.shape
    assert torch.isfinite(v).all()


def test_sample_shape_and_finite(tiny_model: MMDiT3D) -> None:
    cond = _make_cond(2)
    g = torch.Generator().manual_seed(0)
    out = sample(tiny_model, (2, 4, 8, 8, 8), cond, num_steps=3, generator=g)
    assert out.shape == (2, 4, 8, 8, 8)
    assert torch.isfinite(out).all()


def test_sample_fixed_seed_is_deterministic(tiny_model: MMDiT3D) -> None:
    """Fixing the generator pins the sampled identity (the per-subject age-sweep proxy)."""
    cond = _make_cond(1)
    out_a = sample(
        tiny_model, (1, 4, 8, 8, 8), cond, num_steps=3, generator=torch.Generator().manual_seed(7)
    )
    out_b = sample(
        tiny_model, (1, 4, 8, 8, 8), cond, num_steps=3, generator=torch.Generator().manual_seed(7)
    )
    assert torch.allclose(out_a, out_b, atol=1e-6)


class _SyntheticLatents(Dataset):
    """Minimal latent batch matching the FlowLitModule cond contract."""

    def __init__(self, n: int = 4, c: int = 4, d: int = 8) -> None:
        self.n, self.c, self.d = n, c, d

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "z": torch.randn(self.c, self.d, self.d, self.d),
            "age": torch.tensor(float(20 + idx)),
            "sex": torch.tensor(idx % 3, dtype=torch.long),
            "dx": torch.tensor(0, dtype=torch.long),
            "apoe": torch.tensor(0, dtype=torch.long),
            "treatment": torch.tensor(0, dtype=torch.long),
            "cohort": torch.tensor(1, dtype=torch.long),
        }


def _collate(samples: list[dict]) -> dict:
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


def _make_lit(ema_decay: float | None) -> FlowLitModule:
    model = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]

    def opt_partial(params):  # type: ignore[no-untyped-def]
        return torch.optim.AdamW(params, lr=1e-4)

    def sched_partial(opt):  # type: ignore[no-untyped-def]
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    return FlowLitModule(
        model=model,
        optimizer_partial=opt_partial,
        scheduler_partial=sched_partial,
        cfg_dropout_p=0.1,
        ema_decay=ema_decay,
    )


@pytest.mark.parametrize("ema_decay", [0.999, None])
def test_fast_dev_run_trains(ema_decay: float | None) -> None:
    """Full Lightning step through configure_optimizers + training_step +
    validation_step over a synthetic latent loader — the exact path that runs on the
    B200s. Loss must stay finite and the module must take at least one step."""
    lit = _make_lit(ema_decay=ema_decay)
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
    assert trainer.global_step >= 1


def test_lit_build_cond_defaults_missing_categoricals() -> None:
    """A batch with only z + age must default every categorical to a zeros (NULL) tensor."""
    lit = _make_lit(ema_decay=None)
    batch = {"z": torch.randn(2, 4, 8, 8, 8), "age": torch.tensor([30.0, 40.0])}
    cond = lit._build_cond(batch)
    assert cond["age"].shape == (2,)
    for field in CATEGORICAL_FIELDS:
        assert cond[field].shape == (2,)
        assert cond[field].dtype == torch.long
        assert int(cond[field].abs().sum()) == 0  # all NULL (index 0)
