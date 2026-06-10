"""Shape and contract tests for the v0 multimodal VAE."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neurodrift.models.vae3d import VAE3D  # noqa: E402


@pytest.fixture
def tiny_vae() -> VAE3D:
    """Same wiring as the v0 config, but small enough to run on CPU in seconds."""
    return VAE3D(
        modalities=("T1w", "T2w", "PDw", "dwi"),
        latent_channels=4,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        beta_kl=1.0e-4,
    )


def test_forward_returns_full_modality_stack(tiny_vae: VAE3D) -> None:
    b, m, d = 1, 4, 32
    x = torch.randn(b, m, d, d, d)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    tiny_vae.eval()
    out = tiny_vae(x, mask)
    assert out.recon.shape == (b, m, d, d, d)
    assert out.mu.shape == (b, 4, d // 4, d // 4, d // 4)
    assert out.logvar.shape == out.mu.shape


def test_encode_handles_default_mask(tiny_vae: VAE3D) -> None:
    x = torch.randn(1, 4, 32, 32, 32)
    tiny_vae.eval()
    mu, logvar = tiny_vae.encode(x)
    assert mu.shape == (1, 4, 8, 8, 8)
    assert logvar.shape == mu.shape


def test_encode_rejects_wrong_modality_count(tiny_vae: VAE3D) -> None:
    x = torch.randn(1, 3, 32, 32, 32)
    with pytest.raises(ValueError, match="modality"):
        tiny_vae(x)


def test_multi_resolution_preserves_4x_downsample(tiny_vae: VAE3D) -> None:
    """Latent spatial extent should always be input // 4."""
    for size in (32, 48, 64):
        x = torch.randn(1, 4, size, size, size)
        tiny_vae.eval()
        out = tiny_vae(x)
        assert out.recon.shape[-3:] == (size, size, size)
        assert out.mu.shape[-3:] == (size // 4, size // 4, size // 4)


def test_eval_mode_is_deterministic(tiny_vae: VAE3D) -> None:
    """Eval-mode reparameterize must return mu directly (no noise)."""
    tiny_vae.eval()
    x = torch.randn(1, 4, 32, 32, 32)
    out1 = tiny_vae(x)
    out2 = tiny_vae(x)
    assert torch.allclose(out1.recon, out2.recon)
