"""Single-process validation pass — the canonical v0 numbers for the post.

Live DDP epoch metrics are per-rank-local (each rank reduces only the cohorts it
happened to see), so the per-cohort and cross-modal figures logged during
training are not trustworthy. This runner walks the whole val set in one process
and reports:

  * reconstruction PSNR/SSIM per cohort and per modality (all modalities fed),
  * cross-modal synthesis PSNR: feed ONLY the source modality, read the decoder's
    target-modality slot, score it against the held-out true target volume.

Cross-modal is the headline Phase-1 capability (e.g. T1w -> T2w > 28 dB) and is
exactly what the modality-dropout training objective is meant to produce.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import torch

from neurodrift.eval.metrics import psnr, ssim3d


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    batches: Iterable[dict[str, Any]],
    modalities: Sequence[str],
    device: torch.device | str = "cpu",
    cross_modal: bool = True,
) -> dict[str, Any]:
    """Run a full validation pass and return a metrics dict.

    `model` must expose `forward(image, modality_mask) -> object with .recon` and
    `encode/decode` (the v0 `VAE3D` contract). `batches` yields the DataModule's
    collated dicts (`image`, `target`, `present_mask`, `cohort`).
    """
    model = model.to(device).eval()
    mods = list(modalities)
    m = len(mods)

    recon_psnr: dict[tuple[str, str], list[float]] = defaultdict(list)
    recon_ssim: dict[tuple[str, str], list[float]] = defaultdict(list)
    xmodal_psnr: dict[tuple[str, str], list[float]] = defaultdict(list)
    xmodal_ssim: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_subjects = 0

    for batch in batches:
        image = batch["image"].to(device)
        target = batch.get("target", image).to(device)
        present = batch["present_mask"].to(device)
        cohorts = batch["cohort"]
        b = image.shape[0]
        n_subjects += b

        # full-modality reconstruction (feed everything acquired)
        out = model(target * present.view(b, m, 1, 1, 1), present)
        recon = out.recon
        for i in range(b):
            for j in range(m):
                if present[i, j] != 1.0:
                    continue
                recon_psnr[(cohorts[i], mods[j])].append(psnr(recon[i, j], target[i, j]))
                recon_ssim[(cohorts[i], mods[j])].append(ssim3d(recon[i, j], target[i, j]))

        if not cross_modal or m < 2:
            continue
        # cross-modal: feed only `src`, read decoder slot `dst`, score vs true dst
        for src in range(m):
            src_mask = torch.zeros_like(present)
            src_mask[:, src] = present[:, src]
            if src_mask.sum() == 0:
                continue
            out_x = model(target * src_mask.view(b, m, 1, 1, 1), src_mask)
            for i in range(b):
                if present[i, src] != 1.0:
                    continue
                for dst in range(m):
                    if dst == src or present[i, dst] != 1.0:
                        continue
                    xmodal_psnr[(mods[src], mods[dst])].append(
                        psnr(out_x.recon[i, dst], target[i, dst])
                    )
                    xmodal_ssim[(mods[src], mods[dst])].append(
                        ssim3d(out_x.recon[i, dst], target[i, dst])
                    )

    def _summarise(d: dict[tuple[str, str], list[float]]) -> dict[str, float]:
        return {f"{a}/{b}": _mean(v) for (a, b), v in sorted(d.items())}

    recon_summary = _summarise(recon_psnr)
    return {
        "n_subjects": n_subjects,
        "recon_psnr": recon_summary,
        "recon_ssim": _summarise(recon_ssim),
        "xmodal_psnr": _summarise(xmodal_psnr),
        "xmodal_ssim": _summarise(xmodal_ssim),
        "recon_psnr_pooled": _mean([v for vs in recon_psnr.values() for v in vs]),
        "xmodal_psnr_pooled": _mean([v for vs in xmodal_psnr.values() for v in vs]),
        "xmodal_ssim_pooled": _mean([v for vs in xmodal_ssim.values() for v in vs]),
    }
