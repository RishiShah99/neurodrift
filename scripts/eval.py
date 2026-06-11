"""Single-process validation pass — the canonical v0 metrics.

Loads a trained VAE checkpoint, rebuilds the val split deterministically from the
same Hydra config used for training, and writes per-cohort + cross-modal PSNR/SSIM
to JSON. Run this on one GPU (or CPU) AFTER a cook; do not trust the per-rank
epoch metrics logged during DDP training.

Usage:
    uv run python scripts/eval.py experiment=vae_v0 \
        eval.ckpt=out/vae_v0/ckpt/last.ckpt
    # smaller/faster smoke:
    uv run python scripts/eval.py experiment=vae_v0 eval.ckpt=... eval.max_batches=10
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from neurodrift.eval import evaluate
from omegaconf import DictConfig

log = logging.getLogger("neurodrift.eval")


def _load_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load a Lightning or bare state-dict checkpoint into the raw VAE3D module.

    Lightning saves under `state_dict` with a `model.` prefix (the LitModule wraps
    the VAE as `self.model`); strip it so the weights land on the bare module.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        key = k[len("model.") :] if k.startswith("model.") else k
        cleaned[key] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning("missing keys when loading ckpt: %d (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        log.warning("unexpected keys: %d (e.g. %s)", len(unexpected), unexpected[:3])


def _limited(loader, max_batches: int | None):  # type: ignore[no-untyped-def]
    if not max_batches or max_batches <= 0:
        yield from loader
        return
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        yield b


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.get("eval", {})
    ckpt = eval_cfg.get("ckpt")
    if not ckpt:
        raise SystemExit("pass eval.ckpt=<path to .ckpt> (Lightning or bare state-dict)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("eval device=%s ckpt=%s", device, ckpt)

    model = instantiate(cfg.model)
    _load_weights(model, Path(ckpt))

    data = instantiate(cfg.data)
    # single process: no dataloader workers, deterministic val split.
    data.num_workers = 0
    data.setup("validate")
    loader = data.val_dataloader()

    metrics = evaluate(
        model,
        _limited(loader, eval_cfg.get("max_batches")),
        modalities=cfg.model.modalities,
        device=device,
        cross_modal=eval_cfg.get("cross_modal", True),
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out_path)
    log.info("recon PSNR (pooled): %.2f dB", metrics["recon_psnr_pooled"])
    log.info("cross-modal PSNR (pooled): %.2f dB", metrics["xmodal_psnr_pooled"])
    for k, v in metrics["recon_psnr"].items():
        log.info("  recon  %-24s %.2f dB", k, v)
    for k, v in metrics["xmodal_psnr"].items():
        log.info("  xmodal %-24s %.2f dB", k, v)


if __name__ == "__main__":
    main()
