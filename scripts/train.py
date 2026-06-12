"""Hydra-composed Lightning training entrypoint.

Usage:
    uv run python scripts/train.py experiment=vae_v0
    uv run python scripts/train.py experiment=vae_v0 trainer.fast_dev_run=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import call, get_class, instantiate
from neurodrift.train.lightning_module import VAELitModule
from neurodrift.utils.seed import seed_everything
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("neurodrift.train")


def _resolve_resume(ckpt_path: str | None, ckpt_dir: Path) -> str | None:
    """Map the `ckpt_path` config to an actual path for `Trainer.fit`.

    "last"/"auto" -> ckpt_dir/last.ckpt if it exists, else None (fresh start, so
    the very first launch of a spot cook just begins). null/"none"/"" -> None.
    Anything else is treated as an explicit checkpoint path.
    """
    if ckpt_path is None:
        return None
    val = str(ckpt_path).strip().lower()
    if val in ("", "none", "null", "fresh"):
        return None
    if val in ("last", "auto"):
        last = ckpt_dir / "last.ckpt"
        return str(last) if last.exists() else None
    return str(ckpt_path)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stable checkpoint dir (survives relaunch); falls back to the run dir if unset.
    ckpt_dir = Path(cfg.get("ckpt_dir") or (output_dir / "ckpt"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = instantiate(cfg.model)
    data = instantiate(cfg.data)

    def optimizer_partial(params):  # type: ignore[no-untyped-def]
        return call(cfg.optimizer, params=params)

    def scheduler_partial(optimizer):  # type: ignore[no-untyped-def]
        return call(cfg.scheduler, optimizer=optimizer)

    lit_kwargs: dict = OmegaConf.to_container(cfg.get("litmodule", {}), resolve=True) or {}
    # Pick the LightningModule class from litmodule._target_ so experiments can swap
    # in DisentangledVAELitModule (content/style + GAN) without editing this script.
    # Falls back to the v0 VAELitModule for configs that predate _target_.
    lit_target = lit_kwargs.pop("_target_", None)
    lit_cls = get_class(lit_target) if lit_target else VAELitModule
    lit = lit_cls(
        model=model,
        optimizer_partial=optimizer_partial,
        scheduler_partial=scheduler_partial,
        **lit_kwargs,
    )

    logger: L.pytorch.loggers.Logger | bool
    if cfg.log_to_tensorboard and not cfg.trainer.get("fast_dev_run", False):
        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=str(output_dir),
            name="tb",
            default_hp_metric=False,
        )
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    else:
        logger = False

    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]
    # LearningRateMonitor needs a logger to write to; skip it on fast_dev_run.
    if logger is not False:
        callbacks.insert(0, L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"))

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    fast_dev_run = bool(cfg.trainer.get("fast_dev_run", False))
    resume = None if fast_dev_run else _resolve_resume(cfg.get("ckpt_path"), ckpt_dir)
    if resume:
        log.info("resuming from checkpoint: %s", resume)
    trainer.fit(lit, datamodule=data, ckpt_path=resume)


if __name__ == "__main__":
    main()
