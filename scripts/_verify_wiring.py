"""Local CPU wiring check: compose the experiment config and instantiate exactly
the way scripts/train.py does, stopping before any data I/O. Catches Hydra typos,
_target_ resolution, and litmodule kwarg mismatches without needing the zarr corpus
or a GPU. Not a test fixture — a one-off pre-flight before launching a cook."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import call, get_class, instantiate
from omegaconf import OmegaConf

CONFIGS = str(Path(__file__).resolve().parent.parent / "configs")


def check(experiment: str) -> None:
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

    model = instantiate(cfg.model)
    data = instantiate(cfg.data)  # constructs DataModule; no setup() -> no zarr access

    def optimizer_partial(params):  # type: ignore[no-untyped-def]
        return call(cfg.optimizer, params=params)

    def scheduler_partial(optimizer):  # type: ignore[no-untyped-def]
        return call(cfg.scheduler, optimizer=optimizer)

    lit_kwargs = OmegaConf.to_container(cfg.get("litmodule", {}), resolve=True) or {}
    lit_target = lit_kwargs.pop("_target_", None)
    lit_cls = get_class(lit_target)
    lit = lit_cls(
        model=model,
        optimizer_partial=optimizer_partial,
        scheduler_partial=scheduler_partial,
        **lit_kwargs,
    )

    # configure_optimizers needs a trainer ref for estimated_stepping_batches; guard it.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{experiment}]")
    print(f"  model         = {type(model).__name__}  ({n_params / 1e6:.1f}M params)")
    print(f"  litmodule     = {type(lit).__name__}")
    print(f"  data          = {type(data).__name__}  cohorts={list(cfg.data.cohorts)}")
    print(f"  modalities    = {list(cfg.model.modalities)}")
    print(
        f"  trainer       = devices={cfg.trainer.devices} strategy={cfg.trainer.get('strategy')} "
        f"precision={cfg.trainer.precision} grad_clip={cfg.trainer.get('gradient_clip_val', 'unset')}"
    )
    print(
        f"  use_adv={getattr(lit, 'use_adversarial', None)} "
        f"use_perc={getattr(lit, 'use_perceptual', None)} "
        f"disc={'yes' if getattr(lit, 'discriminator', None) is not None else 'no'}"
    )
    # Sanity: grad-clip placement must match the optimization mode, and always 8 GPUs.
    if "DisentangledVAE" in type(lit).__name__:
        grad_clip = cfg.trainer.get("gradient_clip_val")
        if lit.automatic_optimization:
            assert grad_clip not in (None, 0, 0.0), (
                "automatic optimization needs Trainer gradient_clip_val set"
            )
        else:
            assert grad_clip in (None, 0, 0.0), (
                "manual optimization forbids Trainer gradient_clip_val; clip inside the module"
            )
        print(f"  opt_mode      = {'automatic' if lit.automatic_optimization else 'manual (GAN)'}")
        assert int(cfg.trainer.devices) == 8, "flagship must default to all 8 GPUs"


if __name__ == "__main__":
    for exp in sys.argv[1:] or ["vae_v0_disentangled", "vae_v0_disentangled_noadv", "vae_v0"]:
        check(exp)
    print("OK: all configs compose + instantiate")
