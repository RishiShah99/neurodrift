"""CPU tests for the v0 Phase 3 distillation trainer (neurodrift.train.distill_module).

Tiny teacher + half-depth student so the whole distillation loop — teacher endpoint
rollout, student one-step prediction, EMA, and a real Lightning fast_dev_run — runs on
CPU in seconds. This is the gate that keeps a silent distillation bug off the B200s, the
same discipline test_flow.py applies to the flow backbone.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import lightning as L  # noqa: E402
from neurodrift.models.flow import MMDiT3D  # noqa: E402
from neurodrift.train.distill_module import FlowDistillLitModule, load_flow_weights  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

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


def _save_teacher_ckpt(path, ema_offset: float = 0.0) -> MMDiT3D:
    """Save a Lightning-style flow ckpt (model.* + ema_model.*); return the source model."""
    teacher = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    ema = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    if ema_offset:
        with torch.no_grad():
            for p in ema.parameters():
                p.add_(ema_offset)
    state = {f"model.{k}": v for k, v in teacher.state_dict().items()}
    state.update({f"ema_model.{k}": v for k, v in ema.state_dict().items()})
    torch.save({"state_dict": state}, path)
    return ema if ema_offset else teacher


def _make_distill_lit(tmp_path, ema_decay: float | None) -> FlowDistillLitModule:
    ckpt = tmp_path / "teacher.ckpt"
    _save_teacher_ckpt(ckpt)
    student = MMDiT3D(**{**TINY, "depth": 1}, cardinalities=CARDS)  # type: ignore[arg-type]
    teacher_model = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]

    def opt_partial(params):  # type: ignore[no-untyped-def]
        return torch.optim.AdamW(params, lr=2e-4)

    def sched_partial(opt):  # type: ignore[no-untyped-def]
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    return FlowDistillLitModule(
        model=student,
        optimizer_partial=opt_partial,
        scheduler_partial=sched_partial,
        teacher_model=teacher_model,
        teacher_ckpt=str(ckpt),
        num_teacher_steps=2,
        ema_decay=ema_decay,
    )


class _SyntheticLatents(Dataset):
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


# ---------------------------------------------------------------------------
# Teacher weight loading
# ---------------------------------------------------------------------------
def test_load_flow_weights_prefers_ema(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ckpt = tmp_path / "t.ckpt"
    ema = _save_teacher_ckpt(ckpt, ema_offset=5.0)  # EMA copy is distinctively offset
    target = MMDiT3D(**TINY, cardinalities=CARDS)  # type: ignore[arg-type]
    assert load_flow_weights(target, ckpt, prefer_ema=True) == "ema"
    for k, v in target.state_dict().items():
        assert torch.allclose(v, ema.state_dict()[k])
    assert load_flow_weights(target, ckpt, prefer_ema=False) == "model"


# ---------------------------------------------------------------------------
# Distillation mechanics
# ---------------------------------------------------------------------------
def test_teacher_is_frozen_student_trains(tmp_path) -> None:  # type: ignore[no-untyped-def]
    lit = _make_distill_lit(tmp_path, ema_decay=None)
    assert all(not p.requires_grad for p in lit.teacher.parameters())
    batch = _collate([_SyntheticLatents()[i] for i in range(2)])
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    # Student receives gradient; the frozen teacher never does.
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in lit.model.parameters())
    assert all(p.grad is None for p in lit.teacher.parameters())


def test_student_onestep_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    lit = _make_distill_lit(tmp_path, ema_decay=None)
    x0 = torch.randn(2, 4, 8, 8, 8)
    cond = {"age": torch.tensor([30.0, 40.0])}
    from neurodrift.train.flow_module import build_cond_from_batch

    cond = build_cond_from_batch(lit.model, {"z": x0, "age": cond["age"]})
    x1 = lit._student_onestep(x0, cond)
    assert x1.shape == x0.shape
    assert torch.isfinite(x1).all()


def test_on_save_checkpoint_strips_teacher(tmp_path) -> None:  # type: ignore[no-untyped-def]
    lit = _make_distill_lit(tmp_path, ema_decay=0.999)
    ckpt = {
        "state_dict": {
            "model.a": torch.zeros(1),
            "teacher.b": torch.zeros(1),
            "ema_model.c": torch.zeros(1),
        }
    }
    lit.on_save_checkpoint(ckpt)
    keys = set(ckpt["state_dict"])
    assert not any(k.startswith("teacher.") for k in keys)
    assert "model.a" in keys and "ema_model.c" in keys  # student + EMA preserved


@pytest.mark.parametrize("ema_decay", [0.999, None])
def test_fast_dev_run_trains(tmp_path, ema_decay: float | None) -> None:  # type: ignore[no-untyped-def]
    """Full Lightning path (configure_optimizers + training/validation step + EMA) over a
    synthetic latent loader — the exact loop that runs on the B200s."""
    lit = _make_distill_lit(tmp_path, ema_decay=ema_decay)
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
