"""E7 co-registration QA: does routing each modality through the subject's own T1
actually tighten the intra-subject cross-modal alignment?

Compares the BASELINE corpus (`zarr`, every modality independently rigid-registered
to MNI) against the CO-REGISTERED corpus (`zarr_coreg`, each modality registered to
the subject's T1 then composed with T1->MNI). Two signals, paired per subject by
(cohort, subject, session) — the same key the dataloader and E7 group on:

  1. Quantitative (numpy only, always runs): histogram MUTUAL INFORMATION between the
     reference (T1w) and moving (T2w) volumes over the shared brain mask. MI is
     contrast-agnostic, so it rises with spatial alignment regardless of the T1<->T2
     intensity difference. If E7 worked, mean coreg MI > baseline MI.
  2. Visual (optional, needs matplotlib): the reference in grayscale with the moving
     modality's edges burned in red, baseline beside co-registered. The red edges
     should hug the cortical ribbon / ventricles tighter in the coreg column.

    # on the box, after `preprocess.py --coregister`, against the local caches:
    uv run python scripts/coreg_sanity.py \
        --baseline-root /home/rishi_e3ahealth_com/zarr_local \
        --coreg-root    /home/rishi_e3ahealth_com/zarr_local_coreg \
        --cohort openneuro --n 6 --out coreg_sanity.png

    # or straight against GCS (slower; needs gcsfs auth):
    uv run python scripts/coreg_sanity.py \
        --baseline-root gs://neurodrift-data/zarr \
        --coreg-root    gs://neurodrift-data/zarr_coreg

Reads zarr directly — no model, no torch grad, no ANTs. The figure degrades to a
no-op (MI still prints) if matplotlib isn't installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import zarr
from neurodrift.train.data_module import SubjectGroup, _group_by_subject, _list_zarr_stems


class Panel(NamedTuple):
    ref_slice: np.ndarray
    moving_slice: np.ndarray
    mi: float


def _groups_with_pair(
    root: str, cohort: str, ref: str, moving: str
) -> dict[tuple[str, str, str], SubjectGroup]:
    """Subject-session groups under `root/cohort` that hold BOTH ref and moving."""
    groups = _group_by_subject(_list_zarr_stems(root, cohort))
    return {
        (g.cohort, g.subject, g.session or ""): g
        for g in groups
        if ref in g.scans_by_modality and moving in g.scans_by_modality
    }


def _load(url: str) -> np.ndarray | None:
    try:
        return np.asarray(zarr.open(url, mode="r")["data"], dtype=np.float32)
    except (KeyError, ValueError, OSError, RuntimeError):
        return None


def _mid_axial(vol: np.ndarray) -> np.ndarray:
    return vol[:, :, vol.shape[-1] // 2]


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Histogram MI over voxels in the shared brain mask (both > 0)."""
    mask = (a > 0) & (b > 0)
    if int(mask.sum()) < 1000:
        return float("nan")
    hist, _, _ = np.histogram2d(a[mask].ravel(), b[mask].ravel(), bins=bins)
    pab = hist / hist.sum()
    pa = pab.sum(axis=1)
    pb = pab.sum(axis=0)
    nz = pab > 0
    denom = (pa[:, None] * pb[None, :])[nz]
    return float(np.sum(pab[nz] * np.log(pab[nz] / denom)))


def _panel(group: SubjectGroup, ref: str, moving: str) -> Panel | None:
    ref_vol = _load(group.scans_by_modality[ref])
    mov_vol = _load(group.scans_by_modality[moving])
    if ref_vol is None or mov_vol is None or ref_vol.shape != mov_vol.shape:
        return None
    return Panel(_mid_axial(ref_vol), _mid_axial(mov_vol), _mutual_information(ref_vol, mov_vol))


def _norm(sl: np.ndarray) -> np.ndarray:
    """Percentile-clip a slice to [0,1] for display."""
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    if hi <= lo:
        return np.zeros_like(sl)
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0)


def _edges(sl: np.ndarray, pct: float = 82.0) -> np.ndarray:
    """Binary edge mask from gradient magnitude, thresholded at a high percentile."""
    gx, gy = np.gradient(sl.astype(np.float64))
    mag = np.hypot(gx, gy)
    inside = mag[sl > 0]
    if inside.size == 0:
        return np.zeros_like(sl, dtype=bool)
    return mag >= float(np.percentile(inside, pct))


def _overlay(ref_sl: np.ndarray, moving_sl: np.ndarray) -> np.ndarray:
    """Grayscale reference with the moving modality's edges burned in red (RGB)."""
    base = _norm(ref_sl)
    rgb = np.stack([base, base, base], axis=-1)
    rgb[_edges(_norm(moving_sl))] = (1.0, 0.15, 0.15)
    return np.transpose(rgb, (1, 0, 2))  # display orientation, paired with origin="lower"


def _render(
    rows: list[tuple[tuple[str, str, str], Panel | None, Panel | None]],
    ref: str,
    moving: str,
    out: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")  # headless: no DISPLAY on the fleet box
    except ImportError:
        print("matplotlib not installed — skipping the figure (MI table above stands)")
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(8, 4 * len(rows)), squeeze=False)
    for r, (key, base_p, coreg_p) in enumerate(rows):
        for c, (label, panel) in enumerate(
            [("baseline (indep. MNI)", base_p), ("co-registered (via T1)", coreg_p)]
        ):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if panel is None:
                ax.text(0.5, 0.5, "load/shape error", ha="center", va="center")
            else:
                ax.imshow(_overlay(panel.ref_slice, panel.moving_slice), origin="lower")
                ax.set_ylabel(f"{key[1]}\nMI={panel.mi:.3f}", fontsize=8)
            if r == 0:
                ax.set_title(label, fontsize=11)
    fig.suptitle(
        f"{ref} (gray) + {moving} edges (red) — baseline vs E7 co-registration", fontsize=12
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--baseline-root", default="gs://neurodrift-data/zarr")
    ap.add_argument("--coreg-root", default="gs://neurodrift-data/zarr_coreg")
    ap.add_argument("--cohort", default="openneuro")
    ap.add_argument("--ref", default="T1w", help="reference modality (the anchor)")
    ap.add_argument("--moving", default="T2w", help="modality co-registered to the ref")
    ap.add_argument("--n", type=int, default=6, help="number of subjects to render")
    ap.add_argument("--out", default="coreg_sanity.png")
    args = ap.parse_args()

    base = _groups_with_pair(args.baseline_root, args.cohort, args.ref, args.moving)
    coreg = _groups_with_pair(args.coreg_root, args.cohort, args.ref, args.moving)
    common = sorted(set(base) & set(coreg))
    if not common:
        print(
            f"no subjects with both {args.ref}+{args.moving} in BOTH roots for cohort "
            f"{args.cohort!r}\n  baseline {args.baseline_root}: {len(base)} pairs\n"
            f"  coreg    {args.coreg_root}: {len(coreg)} pairs",
            file=sys.stderr,
        )
        return 1
    picked = common[: args.n]
    print(f"{len(common)} common {args.ref}+{args.moving} pairs; using {len(picked)}\n")

    rows: list[tuple[tuple[str, str, str], Panel | None, Panel | None]] = []
    base_mis: list[float] = []
    coreg_mis: list[float] = []
    print(f"{'subject/session':<32} {'baseline MI':>12} {'coreg MI':>10} {'delta':>9}")
    for key in picked:
        base_p = _panel(base[key], args.ref, args.moving)
        coreg_p = _panel(coreg[key], args.ref, args.moving)
        rows.append((key, base_p, coreg_p))
        bmi = base_p.mi if base_p else float("nan")
        cmi = coreg_p.mi if coreg_p else float("nan")
        if base_p and coreg_p and np.isfinite(bmi) and np.isfinite(cmi):
            base_mis.append(bmi)
            coreg_mis.append(cmi)
        tag = f"{key[1]}/{key[2]}" if key[2] else key[1]
        print(f"{tag:<32} {bmi:>12.4f} {cmi:>10.4f} {cmi - bmi:>+9.4f}")

    if base_mis and coreg_mis:
        bm, cm = float(np.mean(base_mis)), float(np.mean(coreg_mis))
        verdict = "TIGHTER (E7 helped)" if cm > bm else "NOT tighter — investigate"
        print(f"\nmean MI  baseline={bm:.4f}  coreg={cm:.4f}  delta={cm - bm:+.4f}  -> {verdict}")
    else:
        print("\nno valid MI pairs (load/shape errors) — check the roots")

    _render(rows, args.ref, args.moving, Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
