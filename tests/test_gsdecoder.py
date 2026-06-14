"""Shape/contract tests for the Phase 4 feed-forward Gaussian-splat decoder.

Tiny dims so the whole thing (including a real Lightning fast_dev_run over a
synthetic {z, slices, mask} loader) runs on CPU in seconds with NO gsplat installed.
gsplat is CUDA-only and must never be imported at collection time; the reference
rasteriser is what these tests and the CPU loss path use.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning as L  # noqa: E402
from neurodrift.models.gsdecoder import (  # noqa: E402
    LEVEL_COARSE,
    LEVEL_FINE,
    GaussianParams,
    GSDecoder3D,
    to_covariance,
)
from neurodrift.train.gsdecoder_module import (  # noqa: E402
    GSDecoderLitModule,
    dssim,
    total_variation,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# Tiny config shared across tests.
MODEL_DIM = 16
N_COARSE = 16
N_FINE = 16
DEPTH = 1
LATENT_C = 4
LATENT_D = 8
IMAGE_SIZE = 16


def _tiny_model(**overrides: object) -> GSDecoder3D:
    kwargs: dict[str, object] = dict(
        latent_channels=LATENT_C,
        model_dim=MODEL_DIM,
        num_coarse=N_COARSE,
        num_fine=N_FINE,
        num_heads=2,
        depth=DEPTH,
        image_size=IMAGE_SIZE,
    )
    kwargs.update(overrides)
    return GSDecoder3D(**kwargs)  # type: ignore[arg-type]


def _tiny_z(b: int = 2) -> torch.Tensor:
    return torch.randn(b, LATENT_C, LATENT_D, LATENT_D, LATENT_D)


def test_forward_gaussian_param_shapes() -> None:
    model = _tiny_model()
    b = 2
    params = model(_tiny_z(b))
    n = N_COARSE + N_FINE
    assert isinstance(params, GaussianParams)
    assert params.mu.shape == (b, n, 3)
    assert params.scale.shape == (b, n, 3)
    assert params.quat.shape == (b, n, 4)
    assert params.alpha.shape == (b, n, 1)
    assert params.intensity.shape == (b, n, 1)
    assert params.level.shape == (b, n)


def test_forward_param_ranges() -> None:
    model = _tiny_model()
    params = model(_tiny_z())
    assert torch.isfinite(params.mu).all()
    assert (params.mu.abs() <= 1.0 + 1e-5).all(), "mu must be in [-1, 1] (tanh)"
    assert (params.scale > 0).all(), "scale must be strictly positive (softplus)"
    quat_norm = params.quat.norm(dim=-1)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-4), (
        "quat must be unit-norm"
    )
    assert (params.alpha >= 0).all() and (params.alpha <= 1).all(), "alpha in [0, 1]"
    assert torch.isfinite(params.intensity).all()
    # level tags are exactly the two known constants
    levels = set(params.level.unique().tolist())
    assert levels <= {LEVEL_COARSE, LEVEL_FINE}


def test_to_covariance_symmetric_psd_finite() -> None:
    model = _tiny_model()
    params = model(_tiny_z())
    cov = to_covariance(params.scale, params.quat)
    b, n = params.scale.shape[0], params.scale.shape[1]
    assert cov.shape == (b, n, 3, 3)
    assert torch.isfinite(cov).all()
    # symmetric
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-5)
    # PSD: all eigenvalues >= 0 (symmetric -> eigvalsh)
    eig = torch.linalg.eigvalsh(cov)
    assert (eig >= -1e-5).all(), "covariance must be positive semi-definite"


def test_render_reference_shape_and_finite() -> None:
    model = _tiny_model()
    params = model(_tiny_z(2))
    img = model.render_slices_reference(params, image_size=IMAGE_SIZE)
    assert img.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.isfinite(img).all()


def test_render_reference_is_differentiable() -> None:
    model = _tiny_model()
    z = _tiny_z(1)
    img = model.render_slices_reference(model(z), image_size=IMAGE_SIZE)
    loss = img.mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "render path must propagate gradient into model parameters"
    assert all(torch.isfinite(g).all() for g in grads)


def test_hierarchy_mask_gating() -> None:
    """An active mask keeps fine Gaussians live; an all-zero mask gates them off.

    The fine tier's opacity is multiplied by the gate, so an all-zero parcellation
    mask must zero every fine alpha while leaving the coarse alphas untouched, and an
    all-ones mask must keep them. Effect on the rendered slices must differ too.
    """
    model = _tiny_model()
    model.eval()
    z = _tiny_z(2)
    full = torch.ones(2, LATENT_D, LATENT_D, LATENT_D)
    empty = torch.zeros(2, LATENT_D, LATENT_D, LATENT_D)

    with torch.no_grad():
        p_full = model(z, full)
        p_empty = model(z, empty)

    fine = slice(N_COARSE, N_COARSE + N_FINE)
    # all-zero mask -> every fine Gaussian's opacity is zeroed
    assert float(p_empty.alpha[:, fine].abs().sum()) == 0.0
    # all-ones mask -> fine opacities are NOT all zero
    assert float(p_full.alpha[:, fine].abs().sum()) > 0.0
    # coarse opacities are mask-independent (gate only touches the fine tier)
    coarse = slice(0, N_COARSE)
    assert torch.allclose(p_full.alpha[:, coarse], p_empty.alpha[:, coarse])
    # rendered output differs between the two gates
    img_full = model.render_slices_reference(p_full, IMAGE_SIZE)
    img_empty = model.render_slices_reference(p_empty, IMAGE_SIZE)
    assert not torch.allclose(img_full, img_empty)


def test_no_mask_keeps_full_fine_tier() -> None:
    model = _tiny_model()
    model.eval()
    z = _tiny_z(1)
    with torch.no_grad():
        p_none = model(z, None)
        p_full = model(z, torch.ones(1, LATENT_D, LATENT_D, LATENT_D))
    fine = slice(N_COARSE, N_COARSE + N_FINE)
    # mask=None == fully-active mask for the fine tier
    assert torch.allclose(p_none.alpha[:, fine], p_full.alpha[:, fine])


def test_loss_components_finite() -> None:
    b, s, h = 2, 3, IMAGE_SIZE
    pred = torch.rand(b, s, h, h)
    target = torch.rand(b, s, h, h)
    ds = dssim(pred, target)
    tv = total_variation(pred)
    l1 = torch.nn.functional.l1_loss(pred, target)
    assert torch.isfinite(ds) and 0.0 <= float(ds) <= 1.0
    assert torch.isfinite(tv) and float(tv) >= 0.0
    assert torch.isfinite(l1)
    # DSSIM of identical inputs is ~0
    assert float(dssim(pred, pred)) < 1e-4


class _SyntheticLatents(Dataset[dict[str, torch.Tensor]]):
    """Frozen-latent curriculum batch: {z, slices, mask}."""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z": torch.randn(LATENT_C, LATENT_D, LATENT_D, LATENT_D),
            "slices": torch.rand(3, IMAGE_SIZE, IMAGE_SIZE),
            "mask": (torch.rand(LATENT_D, LATENT_D, LATENT_D) > 0.5).float(),
        }


def _make_lit() -> GSDecoderLitModule:
    model = _tiny_model()

    def opt_partial(params):  # type: ignore[no-untyped-def]
        return torch.optim.AdamW(params, lr=1e-4)

    def sched_partial(opt):  # type: ignore[no-untyped-def]
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    return GSDecoderLitModule(
        model=model,
        optimizer_partial=opt_partial,
        scheduler_partial=sched_partial,
        dssim_weight=0.5,
        tv_weight=0.01,
        use_gsplat=False,
    )


def test_fast_dev_run_trains() -> None:
    """Full Lightning step through configure_optimizers + training_step + validation_step
    over a synthetic {z, slices, mask} loader, on CPU, with the reference rasteriser."""
    lit = _make_lit()
    loader = DataLoader(_SyntheticLatents(), batch_size=2)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=loader, val_dataloaders=loader)
    assert int(trainer.global_step) >= 1


def test_lit_step_loss_finite() -> None:
    lit = _make_lit()
    batch = {
        "z": _tiny_z(2),
        "slices": torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE),
        "mask": (torch.rand(2, LATENT_D, LATENT_D, LATENT_D) > 0.5).float(),
    }
    loss = lit._step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()


def test_reference_renderer_needs_no_gsplat() -> None:
    """The reference rasteriser must work with gsplat absent — the whole CPU contract.

    Importing gsplat is NOT required to render: building the model, forwarding, and
    rendering all succeed here. (Collection of this whole module already proves no
    top-level gsplat import exists.)
    """
    import importlib.util

    assert importlib.util.find_spec("gsplat") is None, (
        "this CPU test environment must NOT have gsplat installed"
    )
    model = _tiny_model()
    img = model.render_slices_reference(model(_tiny_z(1)), IMAGE_SIZE)
    assert torch.isfinite(img).all()


def test_gsplat_path_guarded() -> None:
    """gsplat path is exercised only when CUDA gsplat is present; skipped otherwise.

    On the CPU CI box gsplat is absent, so this skips. It documents that the real
    CUDA call lives behind pytest.importorskip and never runs at collection time.
    """
    pytest.importorskip("gsplat")
    model = _tiny_model()
    with pytest.raises(RuntimeError):
        model.render_slices_gsplat(model(_tiny_z(1)), IMAGE_SIZE)
