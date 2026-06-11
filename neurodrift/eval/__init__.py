"""Validation portfolio: PSNR/SSIM, per-cohort + cross-modal synthesis matrix.

Single-process eval is the canonical source of v0 numbers; live DDP training
metrics are per-rank-local and not trustworthy. See `runner.evaluate`.
"""

from __future__ import annotations

from neurodrift.eval.metrics import psnr, ssim3d
from neurodrift.eval.runner import evaluate

__all__ = ["evaluate", "psnr", "ssim3d"]
