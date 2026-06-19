"""LightningModule for v0 Phase 3 — 1-step distillation of the lifespan flow.

The trained flow (`flow_v0`, a rectified-flow velocity model) needs ~50 Euler steps to
sample a latent, which is too slow for the live demo's age slider. This distills it into
a **1-step student**: a smaller MMDiT3D (same dims, half the depth per PLAN §5) that maps
noise `x0` directly to the teacher's ODE endpoint `x1` in a SINGLE forward, conditioned on
the same age/categorical contract.

Method — rectified-flow endpoint distillation (InstaFlow / reflow family). Per batch:

  1. Draw x0 ~ N(0, I) shaped like a real latent; take the conditioning (age + NULL
     categoricals) from the real batch so the student is distilled over the true age
     distribution, not a synthetic one.
  2. Teacher: Euler-integrate dz/dt = v_teacher(z, t, cond) from x0 (t=0) to x1 (t=1) in
     `num_teacher_steps` steps, no grad. This is the target the student must hit in one shot.
  3. Student one-step: v_s = student(x0, t=0, cond); x1_student = x0 + v_s (a single Euler
     step with dt=1). Loss = MSE(x1_student, x1_teacher).

Acceptance (PLAN §6 v0): 1-step-vs-teacher PSNR gap < 1 dB — measured separately by the
eval, not here.

The teacher is frozen, loaded from its own checkpoint, and STRIPPED from saved checkpoints
(it is reproducible from `teacher_ckpt` and would otherwise double the file); on a spot
resume `__init__` reloads it. Optimizer/scheduler/EMA/non-finite-guard wiring mirror
`FlowLitModule` so the cosine half-period is pinned to the real planned step count and a
NaN step is skipped identically on every rank.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from neurodrift.models.flow import MMDiT3D
from neurodrift.train.flow_module import build_cond_from_batch
from neurodrift.train.lightning_module import _skip_step_if_nonfinite

log = logging.getLogger("neurodrift.distill")


def load_flow_weights(
    model: torch.nn.Module, ckpt_path: str | Path, prefer_ema: bool = True
) -> str:
    """Load FlowLitModule weights into a bare MMDiT3D; prefer the EMA copy.

    The LitModule saves both the live model (`model.*`) and its EMA (`ema_model.*`); the
    EMA is the canonical sampling target for flow/diffusion models, so it is the right
    teacher. Falls back to `model.*`, then to a bare state-dict. Returns the source used.
    Mirrors `scripts/flow_eval._load_flow_weights` (kept local so the library has no
    dependency on the scripts/ dir).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    src = "bare"
    for prefix, tag in (("ema_model.", "ema"), ("model.", "model")):
        if tag == "ema" and not prefer_ema:
            continue
        sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            state, src = sub, tag
            break
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("teacher: %d missing keys (e.g. %s)", len(missing), list(missing)[:3])
    if unexpected:
        log.warning("teacher: %d unexpected keys (e.g. %s)", len(unexpected), list(unexpected)[:3])
    log.info("loaded teacher weights from %s (%s)", ckpt_path, src)
    return src


class FlowDistillLitModule(L.LightningModule):
    """1-step distillation trainer: student matches the frozen flow teacher's endpoint."""

    def __init__(
        self,
        model: MMDiT3D,
        optimizer_partial: Any,
        scheduler_partial: Any | None = None,
        *,
        teacher_model: Any,
        teacher_ckpt: str,
        num_teacher_steps: int = 50,
        prefer_ema_teacher: bool = True,
        ema_decay: float | None = 0.999,
    ) -> None:
        super().__init__()
        self.automatic_optimization = True
        # Teacher keys are stripped from saved checkpoints (see on_save_checkpoint), so a
        # resume must tolerate their absence — the teacher is reloaded fresh in __init__.
        self.strict_loading = False
        self.model = model  # the STUDENT
        self.optimizer_partial = optimizer_partial
        self.scheduler_partial = scheduler_partial
        self.num_teacher_steps = int(num_teacher_steps)

        # Teacher: instantiate the architecture (a Hydra config dict / DictConfig) and
        # load the trained weights. instantiate handles both a live DictConfig and the
        # plain dict that train.py passes via OmegaConf.to_container.
        teacher = (
            instantiate(teacher_model) if not isinstance(teacher_model, MMDiT3D) else teacher_model
        )
        load_flow_weights(teacher, teacher_ckpt, prefer_ema=prefer_ema_teacher)
        teacher.requires_grad_(False)
        teacher.eval()
        self.teacher = teacher

        self.ema_decay = ema_decay
        self.ema_model: MMDiT3D | None = None
        if ema_decay is not None:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

    # -- teacher / student rollouts ----------------------------------------
    @torch.no_grad()
    def _teacher_endpoint(self, x0: torch.Tensor, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        """Euler-integrate the teacher from a GIVEN x0 (t=0) to its latent endpoint (t=1)."""
        z = x0
        b = x0.shape[0]
        dt = 1.0 / self.num_teacher_steps
        for i in range(self.num_teacher_steps):
            t = torch.full((b,), i * dt, device=z.device, dtype=z.dtype)
            z = z + dt * self.teacher(z, t, cond)
        return z

    def _student_onestep(self, x0: torch.Tensor, cond: dict[str, torch.Tensor]) -> torch.Tensor:
        """One Euler step (dt=1) from noise: x1 = x0 + v_student(x0, t=0, cond)."""
        b = x0.shape[0]
        t0 = torch.zeros(b, device=x0.device, dtype=x0.dtype)
        v: torch.Tensor = self.model(x0, t0, cond)
        return x0 + v

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        x1_real = batch["z"]
        cond = build_cond_from_batch(self.model, batch)
        x0 = torch.randn_like(x1_real)
        x1_teacher = self._teacher_endpoint(x0, cond)
        x1_student = self._student_onestep(x0, cond)
        loss = F.mse_loss(x1_student, x1_teacher)

        bs = x1_real.shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/distill_mse", loss, on_step=False, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    @torch.no_grad()
    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        if self.ema_model is None:
            return
        decay = self.ema_decay if self.ema_decay is not None else 0.0
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters(), strict=True):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            ema_b.copy_(b)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        _skip_step_if_nonfinite(self, optimizer)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Drop the frozen teacher from the checkpoint — it is reloaded from teacher_ckpt.

        Keeps the student checkpoint lean (no point re-saving 2 GB of frozen teacher every
        save) and avoids a resume mismatch if the teacher arch ever changes.
        """
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if k.startswith("teacher.")]:
            del sd[k]

    def configure_optimizers(self) -> Any:
        # Optimize the STUDENT only (the teacher is frozen / excluded).
        optimizer = self.optimizer_partial(self.model.parameters())
        if self.scheduler_partial is None:
            return optimizer
        scheduler = self.scheduler_partial(optimizer)
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps and hasattr(scheduler, "T_max"):
            scheduler.T_max = int(total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
