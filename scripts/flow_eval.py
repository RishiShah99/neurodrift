"""Single-process evaluation of the lifespan flow backbone — the v0 verdict.

`flow_v0.ckpt` trained and early-stopped fast (~val/loss 0.25) but was never
evaluated, so whether it learned real lifespan structure (vs. just the latent
marginal) is unknown. This script answers that, mirroring `scripts/eval.py` for the
VAE: load the checkpoint, rebuild the same deterministic data, run the canonical
checks once on one device, write JSON. Run AFTER a cook; do not trust live DDP loss.

Checks (PLAN.md §6 v0, Phase 2):
  1. Age-trajectory — fix the sampling noise (one identity), sweep age, and confirm
     the latent (and, with a VAE, the decoded brain) moves smoothly + monotonically
     with age in the right direction (ventricles up, tissue down).
  2. Population-mean MAE — per age bin, sample a population at the bin age and compare
     its mean to the held-out REAL mean. Target < 2.5%.
  3. Envelope coverage — the 90% sampled envelope should contain ~90% of held-out
     real elements at an age. Target within 5% of nominal.

Usage:
    uv run python scripts/flow_eval.py experiment=flow_v0 \
        eval.ckpt=out/flow_v0/ckpt/last.ckpt
    # richer (decoded-voxel proxies) — supply the coreg VAE used for encoding:
    uv run python scripts/flow_eval.py experiment=flow_v0 \
        eval.ckpt=flow_v0.ckpt eval.vae_ckpt=vae_v0_disentangled_noadv_coreg.ckpt
    # quick smoke:
    uv run python scripts/flow_eval.py experiment=flow_v0 eval.ckpt=... \
        eval.max_real=32 eval.num_steps=10
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from neurodrift.eval import flow_eval as fe
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger("neurodrift.flow_eval")

# Default geometry for the coreg VAE used to encode the latents (handoff: matches
# configs/model/vae3d_disentangled.yaml defaults, loads clean with strict=False).
_DEFAULT_VAE_CONFIG = Path(__file__).resolve().parents[1] / "configs/model/vae3d_disentangled.yaml"


def _load_flow_weights(model: torch.nn.Module, ckpt_path: Path, prefer_ema: bool = True) -> str:
    """Load FlowLitModule weights into the bare MMDiT3D; prefer the EMA copy.

    The LitModule saves both the live model (`model.*`) and its EMA (`ema_model.*`);
    the EMA is the canonical sampling target for flow/diffusion models. Falls back to
    `model.*`, then to a bare state-dict. Returns which source was used.
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
        log.warning("flow: %d missing keys (e.g. %s)", len(missing), list(missing)[:3])
    if unexpected:
        log.warning("flow: %d unexpected keys (e.g. %s)", len(unexpected), list(unexpected)[:3])
    log.info("loaded flow weights from %s (%s)", ckpt_path, src)
    return src


def _load_vae_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load a (Lightning or bare) VAE checkpoint into the raw module (mirrors eval.py)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning("vae: %d missing keys (e.g. %s)", len(missing), list(missing)[:3])
    if unexpected:
        log.warning("vae: %d unexpected keys (e.g. %s)", len(unexpected), list(unexpected)[:3])


def _collect_reals(dataset: Any, max_real: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Read up to `max_real` real latents with FINITE ages from a LatentZarrDataset.

    Iterates item-by-item (no collate) so a corrupt store's degraded 1-D `z` is
    skipped rather than crashing a stack, and so per-item age is available. Returns
    (latents (M,C,d,d,d), ages (M,), cohorts).
    """
    zs: list[torch.Tensor] = []
    ages: list[float] = []
    cohorts: list[str] = []
    for i in range(len(dataset)):
        item = dataset[i]
        z = item["z"]
        age = float(item["age"])
        if z.dim() != 4 or not torch.isfinite(torch.tensor(age)):
            continue  # corrupt store (1-D fallback) or unknown age — unusable here
        zs.append(z)
        ages.append(age)
        cohorts.append(str(item.get("cohort", "")))
        if len(zs) >= max_real:
            break
    if not zs:
        raise RuntimeError("no usable latents with finite ages found — check the latent root")
    return torch.stack(zs), torch.tensor(ages), cohorts


def _quantile_bins(ages: torch.Tensor, k: int) -> list[tuple[float, float, torch.Tensor]]:
    """Split sample indices into `k` equal-count age bins (balanced membership).

    Returns [(age_lo, age_hi, member_indices), ...]; equal-count (not equal-width) so
    no bin is empty on a skewed lifespan distribution. Fewer than k bins if there are
    too few samples.
    """
    order = torch.argsort(ages)
    k = max(1, min(k, len(order)))
    chunks = torch.chunk(order, k)
    bins: list[tuple[float, float, torch.Tensor]] = []
    for ch in chunks:
        if ch.numel() == 0:
            continue
        a = ages[ch]
        bins.append((float(a.min()), float(a.max()), ch))
    return bins


@torch.no_grad()
def _decode_t1(vae: torch.nn.Module, z: torch.Tensor, chunk: int) -> torch.Tensor:
    """Decode latents to the T1 (slot 0) volume in chunks; returns (B, D, H, W).

    no_grad is essential: without it every chunk's decode graph is retained (the
    returned tensors keep referencing it), so a few hundred 128^3 decodes pile up
    into hundreds of GB and OOM even a B200. Eval needs no gradients anyway.
    """
    outs: list[torch.Tensor] = []
    for s in range(0, z.shape[0], chunk):
        recon = vae.decode(z[s : s + chunk])  # (b, M, D, H, W)
        outs.append(recon[:, 0].float().cpu())
    return torch.cat(outs, dim=0)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_cfg = cfg.get("eval", {})
    ckpt = eval_cfg.get("ckpt")
    if not ckpt:
        raise SystemExit("pass eval.ckpt=<path to the flow .ckpt>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = int(eval_cfg.get("num_steps", 50))
    seed = int(eval_cfg.get("seed", 1234))
    max_real = int(eval_cfg.get("max_real", 256))
    n_sweep = int(eval_cfg.get("n_sweep", 10))
    n_bins = int(eval_cfg.get("n_bins", 4))
    cov_n = int(eval_cfg.get("cov_n", 32))
    decode_chunk = int(eval_cfg.get("decode_chunk", 2))
    log.info("flow eval device=%s ckpt=%s steps=%d", device, ckpt, num_steps)

    # --- flow model ---
    flow = instantiate(cfg.model).to(device).eval()
    weight_src = _load_flow_weights(
        flow, Path(ckpt), prefer_ema=bool(eval_cfg.get("use_ema", True))
    )

    # --- real latents (held-out val by default) ---
    data = instantiate(cfg.data)
    data.num_workers = 0
    data.setup("validate")
    split = eval_cfg.get("split", "val")
    dataset = data.train_ds if split == "train" else data.val_ds
    reals, ages, _ = _collect_reals(dataset, max_real)
    latent_shape = tuple(reals.shape[1:])
    log.info(
        "collected %d real latents (split=%s), age range %.1f-%.1f",
        reals.shape[0],
        split,
        float(ages.min()),
        float(ages.max()),
    )

    # --- optional coreg VAE for decoded-voxel proxies ---
    vae: torch.nn.Module | None = None
    vae_ckpt = eval_cfg.get("vae_ckpt")
    if vae_ckpt:
        # `or` (not .get's default): the config key exists set to null, so .get
        # returns None — fall back to the bundled VAE geometry in that case.
        vae_cfg_path = Path(eval_cfg.get("vae_config") or _DEFAULT_VAE_CONFIG)
        vae = instantiate(OmegaConf.load(vae_cfg_path)).to(device).eval()
        _load_vae_weights(vae, Path(vae_ckpt))
        log.info("decoded-voxel proxies ON (vae=%s)", vae_ckpt)

    metrics: dict[str, Any] = {
        "weight_source": weight_src,
        "num_steps": num_steps,
        "n_real": int(reals.shape[0]),
        "age_range": [float(ages.min()), float(ages.max())],
        "latent_shape": list(latent_shape),
    }

    # === 1. age-trajectory (fixed identity, sweep age) ===
    lo, hi = float(torch.quantile(ages, 0.05)), float(torch.quantile(ages, 0.95))
    sweep_ages = torch.linspace(lo, hi, n_sweep)
    swept = fe.age_sweep_latents(
        flow,
        sweep_ages.tolist(),
        latent_shape=latent_shape,
        seed=seed,
        num_steps=num_steps,
        device=device,
    )
    lat_energy = swept.flatten(1).abs().mean(dim=1).cpu()
    sweep_metrics: dict[str, Any] = {
        "ages": sweep_ages.tolist(),
        "latent_energy": lat_energy.tolist(),
        "latent_energy_vs_age_r": fe.pearson_r(lat_energy, sweep_ages),
        "latent_energy_smoothness": fe.trajectory_smoothness(lat_energy),
    }
    if vae is not None:
        t1 = _decode_t1(vae, swept, decode_chunk)
        fg = fe.foreground_fraction(t1)
        dark = fe.dark_core_fraction(t1)
        sweep_metrics["tissue_fraction"] = fg.tolist()
        sweep_metrics["ventricle_proxy"] = dark.tolist()
        sweep_metrics["tissue_vs_age_r"] = fe.pearson_r(fg, sweep_ages)
        sweep_metrics["ventricle_vs_age_r"] = fe.pearson_r(dark, sweep_ages)
        sweep_metrics["ventricle_smoothness"] = fe.trajectory_smoothness(dark)
    metrics["age_sweep"] = sweep_metrics

    # === 2. population-mean MAE per age bin ===
    bins = _quantile_bins(ages, n_bins)
    per_bin: list[dict[str, Any]] = []
    pooled_nmae: list[float] = []
    for lo_b, hi_b, idx in bins:
        bin_reals = reals[idx]
        bin_age = float(ages[idx].mean())
        n = min(int(idx.numel()), int(eval_cfg.get("pop_n", 32)))
        pop = fe.sample_population(
            flow,
            bin_age,
            n=n,
            latent_shape=latent_shape,
            base_seed=seed,
            num_steps=num_steps,
            device=device,
        )
        lat = fe.population_mean_mae(pop.mean(dim=0).cpu(), bin_reals.mean(dim=0))
        entry: dict[str, Any] = {
            "age_lo": lo_b,
            "age_hi": hi_b,
            "age_mean": bin_age,
            "n_real": int(idx.numel()),
            "n_sampled": n,
            "latent": lat,
        }
        pooled_nmae.append(lat["nmae_range"])
        if vae is not None:
            pred_t1 = _decode_t1(vae, pop, decode_chunk).mean(dim=0)
            real_t1 = _decode_t1(vae, bin_reals.to(device), decode_chunk).mean(dim=0)
            entry["decoded_t1"] = fe.population_mean_mae(pred_t1, real_t1)
        per_bin.append(entry)
    metrics["population_mean_mae"] = {
        "per_bin": per_bin,
        "pooled_nmae_range": float(sum(pooled_nmae) / len(pooled_nmae))
        if pooled_nmae
        else float("nan"),
    }

    # === 3. envelope coverage on the most-populated bin ===
    if bins:
        lo_b, hi_b, idx = max(bins, key=lambda b: b[2].numel())
        bin_age = float(ages[idx].mean())
        samples = fe.sample_population(
            flow,
            bin_age,
            n=cov_n,
            latent_shape=latent_shape,
            base_seed=seed + 9999,
            num_steps=num_steps,
            device=device,
        ).cpu()
        cov = fe.envelope_coverage(samples, reals[idx], level=0.90)
        metrics["envelope_coverage"] = {
            "age_mean": bin_age,
            "n_real": int(idx.numel()),
            "n_sampled": cov_n,
            "coverage_at_90": cov,
            "abs_error_vs_nominal": abs(cov - 0.90),
        }

    # --- write + verdict ---
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "flow_eval_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out_path)

    pooled = metrics["population_mean_mae"]["pooled_nmae_range"]
    log.info("VERDICT (vs PLAN §6 v0 Phase 2):")
    log.info("  pop-mean nMAE (range-normalised, pooled): %.4f  [target < 0.025]", pooled)
    log.info(
        "  latent-energy vs age r: %+.3f  smoothness max_jump_frac %.3f",
        sweep_metrics["latent_energy_vs_age_r"],
        sweep_metrics["latent_energy_smoothness"]["max_jump_frac"],
    )
    if vae is not None:
        log.info(
            "  ventricle proxy vs age r: %+.3f (expect > 0)  tissue vs age r: %+.3f (expect < 0)",
            sweep_metrics["ventricle_vs_age_r"],
            sweep_metrics["tissue_vs_age_r"],
        )
    if "envelope_coverage" in metrics:
        ec = metrics["envelope_coverage"]
        log.info(
            "  90%% envelope coverage: %.3f  (|err| %.3f, target < 0.05)",
            ec["coverage_at_90"],
            ec["abs_error_vs_nominal"],
        )


if __name__ == "__main__":
    main()
