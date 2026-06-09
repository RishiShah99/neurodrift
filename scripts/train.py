"""Hydra-composed Lightning training entrypoint.

Usage:
    uv run python scripts/train.py experiment=vae_phase1
    uv run python scripts/train.py experiment=vae_phase1 trainer.fast_dev_run=true
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import call, instantiate
from neurodrift.train.lightning_module import VAELitModule
from neurodrift.utils.seed import seed_everything
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("neurodrift.train")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = instantiate(cfg.model)
    data = instantiate(cfg.data)

    def optimizer_partial(params):  # type: ignore[no-untyped-def]
        return call(cfg.optimizer, params=params)

    def scheduler_partial(optimizer):  # type: ignore[no-untyped-def]
        return call(cfg.scheduler, optimizer=optimizer)

    lit = VAELitModule(
        model=model,
        optimizer_partial=optimizer_partial,
        scheduler_partial=scheduler_partial,
    )

    callbacks: list[L.Callback] = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_dir / "ckpt",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]

    logger: L.pytorch.loggers.Logger | bool
    if cfg.log_to_wandb and not cfg.trainer.get("fast_dev_run", False):
        logger = L.pytorch.loggers.WandbLogger(
            project="neurodrift",
            name=cfg.experiment_name,
            save_dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        logger = False

    trainer: L.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(lit, datamodule=data)


if __name__ == "__main__":
    main()
