"""Tests for the single-process eval portfolio."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurodrift.eval.metrics import psnr, ssim3d  # noqa: E402
from neurodrift.eval.runner import evaluate  # noqa: E402
from neurodrift.models.vae3d import VAE3D  # noqa: E402


def test_psnr_identical_is_capped() -> None:
    v = torch.randn(8, 8, 8)
    assert psnr(v, v) == 99.0


def test_psnr_monotonic_in_noise() -> None:
    t = torch.randn(8, 8, 8)
    near = t + 0.01 * torch.randn_like(t)
    far = t + 0.5 * torch.randn_like(t)
    assert psnr(near, t) > psnr(far, t)


def test_ssim_identical_is_one() -> None:
    t = torch.rand(16, 16, 16)
    assert ssim3d(t, t) == pytest.approx(1.0, abs=1e-4)


def test_ssim_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError, match="D, H, W"):
        ssim3d(torch.rand(2, 8, 8, 8), torch.rand(2, 8, 8, 8))


def test_evaluate_returns_per_cohort_and_xmodal() -> None:
    mods = ("T1w", "T2w", "FLAIR")
    model = VAE3D(
        modalities=mods,
        latent_channels=4,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
    )
    b, m, d = 2, 3, 32
    target = torch.randn(b, m, d, d, d)
    present = torch.ones(b, m)
    batch = {
        "image": target.clone(),
        "target": target,
        "present_mask": present,
        "modality_mask": present,
        "cohort": ["abide", "openneuro"],
    }
    out = evaluate(model, [batch], modalities=mods, device="cpu")
    assert out["n_subjects"] == 2
    # per (cohort, modality) recon entries exist
    assert "abide/T1w" in out["recon_psnr"]
    assert "openneuro/FLAIR" in out["recon_ssim"]
    # cross-modal matrix has ordered pairs (src/dst)
    assert "T1w/T2w" in out["xmodal_psnr"]
    assert "T2w/T1w" in out["xmodal_psnr"]
    assert isinstance(out["recon_psnr_pooled"], float)
