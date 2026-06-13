"""Shape/contract tests for the content/style disentangled VAE + GAN training step.

Tiny dims so the whole thing (including a real Lightning fast_dev_run that exercises
the manual-optimization GAN loop) runs on CPU in seconds. This is the gate that keeps
the next silent-bug cook off the B200s.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning as L  # noqa: E402
from neurodrift.models.vae3d import (  # noqa: E402
    DisentangledVAE3D,
    PatchDiscriminator3D,
    VAEOutput,
)
from neurodrift.train.data_module import _collate  # noqa: E402
from neurodrift.train.lightning_module import DisentangledVAELitModule  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

MODS = ("T1w", "T2w", "FLAIR")


@pytest.fixture
def tiny_model() -> DisentangledVAE3D:
    return DisentangledVAE3D(
        modalities=MODS,
        latent_channels=4,
        style_dim=8,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
    )


def test_forward_returns_full_modality_stack(tiny_model: DisentangledVAE3D) -> None:
    b, m, d = 2, 3, 16
    x = torch.randn(b, m, d, d, d)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    tiny_model.eval()
    out = tiny_model(x, mask)
    assert isinstance(out, VAEOutput)
    assert out.recon.shape == (b, m, d, d, d)
    assert out.mu.shape == (b, 4, d // 4, d // 4, d // 4)
    assert out.logvar.shape == out.mu.shape


def test_encode_decode_contract(tiny_model: DisentangledVAE3D) -> None:
    """eval.py uses encode()->(mu,logvar) and decode(z)->all slots."""
    x = torch.randn(1, 3, 16, 16, 16)
    tiny_model.eval()
    mu, _ = tiny_model.encode(x)
    assert mu.shape == (1, 4, 4, 4, 4)
    recon = tiny_model.decode(mu)
    assert recon.shape == (1, 3, 16, 16, 16)


def test_translate_single_source(tiny_model: DisentangledVAE3D) -> None:
    """Headline path: synthesize dst from one source volume."""
    src = torch.randn(2, 1, 16, 16, 16)
    tiny_model.eval()
    out = tiny_model.translate(src, src_idx=0, dst_idx=1)
    assert out.shape == (2, 1, 16, 16, 16)


def test_encode_all_shapes(tiny_model: DisentangledVAE3D) -> None:
    x = torch.randn(2, 3, 16, 16, 16)
    present = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    enc = tiny_model.encode_all(x, present)
    assert enc.content_mu.shape == (2, 3, 4, 4, 4, 4)
    assert enc.content_logvar.shape == enc.content_mu.shape
    assert enc.style.shape == (2, 3, 8)


def test_absent_slot_uses_prototype_style(tiny_model: DisentangledVAE3D) -> None:
    """A slot dropped from the input must still be synthesized (prototype style),
    i.e. forward never emits zeros for an absent-but-decoded modality."""
    x = torch.zeros(1, 3, 16, 16, 16)
    x[:, 0] = torch.randn(1, 16, 16, 16)  # only T1 present
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    tiny_model.eval()
    out = tiny_model(x, mask)
    assert out.recon[:, 1].abs().sum() > 0  # T2 slot is synthesized, not zero


def test_discriminator_patch_output() -> None:
    d = PatchDiscriminator3D(num_modalities=3, base_channels=8)
    x = torch.randn(2, 1, 32, 32, 32)
    logits = d(x, modality_idx=1)
    assert logits.dim() == 5 and logits.shape[0] == 2 and logits.shape[1] == 1
    assert logits.shape[-1] > 1  # a patch grid, not a single scalar


class _SyntheticScans(Dataset):
    """Minimal multimodal batch matching what the DataModule emits."""

    def __init__(self, n: int = 4, m: int = 3, d: int = 16) -> None:
        self.n, self.m, self.d = n, m, d

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        vol = torch.randn(self.m, self.d, self.d, self.d)
        present = torch.ones(self.m)
        return {
            "image": vol.clone(),
            "target": vol,
            "modality_mask": present.clone(),
            "present_mask": present,
            "age": torch.tensor(float("nan")),
            "cohort": "synthetic",
            "subject": f"sub-{idx}",
            "session": "",
        }


def _make_lit(use_adversarial: bool, use_perceptual: bool) -> DisentangledVAELitModule:
    model = DisentangledVAE3D(
        modalities=MODS, latent_channels=4, style_dim=8, base_channels=8, num_res_blocks=1
    )

    def opt_partial(params):  # type: ignore[no-untyped-def]
        return torch.optim.AdamW(params, lr=1e-4)

    def sched_partial(opt):  # type: ignore[no-untyped-def]
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    return DisentangledVAELitModule(
        model=model,
        optimizer_partial=opt_partial,
        scheduler_partial=sched_partial,
        kl_warmup_steps=2,
        adv_start_step=0,  # adversarial active immediately so the D step is exercised
        adv_warmup_steps=0,
        disc_base_channels=8,
        use_perceptual=use_perceptual,
        use_adversarial=use_adversarial,
        perceptual_weights=None,  # random VGG: keep the CPU test offline (no download)
    )


@pytest.mark.parametrize("use_adversarial", [True, False])
def test_fast_dev_run_trains(use_adversarial: bool) -> None:
    """Full Lightning step through configure_optimizers + manual-opt training_step +
    validation_step. With adversarial on, both G and D optimizers must step and the
    loss must stay finite — the exact path that runs on the B200s."""
    lit = _make_lit(use_adversarial=use_adversarial, use_perceptual=True)
    loader = DataLoader(_SyntheticScans(), batch_size=2, collate_fn=_collate)
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=loader, val_dataloaders=loader)
    assert int(lit._step_count.item()) >= 1


def test_nonfinite_grad_skip_zeros_grads() -> None:
    """A non-finite gradient must zero ALL grads so the optimizer step is a no-op.

    Regression for the NaN guard that only suppressed a log line: the poisoned loss
    still ran backward. Under DDP a single NaN grad is all-reduced into every rank,
    so the skip decision is made here (post-reduce, identical on all ranks) by
    zeroing grads — never by returning None (which would desync ranks).
    """
    from neurodrift.train.lightning_module import _skip_step_if_nonfinite

    lit = _make_lit(use_adversarial=False, use_perceptual=False)
    opt = torch.optim.SGD(lit.parameters(), lr=0.1)
    p = next(param for param in lit.parameters() if param.requires_grad)

    p.grad = torch.ones_like(p)
    assert _skip_step_if_nonfinite(lit, opt) is False, "finite grads must not be skipped"
    assert p.grad is not None and torch.isfinite(p.grad).all()

    p.grad = torch.full_like(p, float("nan"))
    assert _skip_step_if_nonfinite(lit, opt) is True, "NaN grads must trigger a skip"
    assert p.grad is None or float(p.grad.abs().sum()) == 0.0, "grads must be cleared on skip"


def test_adversarial_ramp_schedule() -> None:
    lit = _make_lit(use_adversarial=True, use_perceptual=False)
    lit._step_count.fill_(0)
    lit.adv_start_step = 100
    lit.adv_warmup_steps = 100
    assert lit._adv_weight() == 0.0  # before start
    lit._step_count.fill_(150)
    assert 0.0 < lit._adv_weight() < lit.adversarial_weight  # mid-ramp
    lit._step_count.fill_(500)
    assert lit._adv_weight() == pytest.approx(lit.adversarial_weight)  # saturated
