"""Interpretability probe for the TopK-SAE — Phase 5.

Given a trained SAE checkpoint and a set of VAE latents, this answers two
questions the foundation-model story needs:

  1. WHICH SAE features track a biomarker? `feature_biomarker_correlation`
     correlates each feature's per-subject mean activation against a scalar label
     (Pearson r). FreeSurfer volumetrics are not present in this repo, so we ship
     `proxy_biomarkers` (honest, latent-derived volumetric proxies) AND accept a
     real labels TSV via --labels-tsv when one exists. The proxies are NOT a
     substitute for real biomarkers — they are a wiring/sanity stand-in, labelled
     as such in the output.

  2. WHAT is the "aging" direction in latent space? `aging_direction` contrasts
     the mean latent of a young vs an old population (split at --age-threshold) to
     get a mean-difference steering vector; `apply_steering(z, vec, scale)` walks a
     latent along it. Pushing a subject's latent along +vector and decoding through
     the frozen VAE is the counterfactual-aging demo (Paper 3).

All the math is importable (the functions above + `top_correlated_features` and
`per_subject_feature_means`); the CLI is a thin argparse driver around them,
structured like scripts/eval.py (a `_load_weights` that strips the LitModule
`model.` prefix; a `_latents_from_root` loader). No Hydra — the probe is a
post-hoc analysis, not a training entrypoint.

Usage:
    python scripts/sae_probe.py \
        --ckpt out/sae_v0/ckpt/last.ckpt \
        --latent-root ./latent_cache --cohorts abide openneuro \
        --d-in 16 --d-hidden 8192 --k 32 \
        --age-threshold 60 --out out/sae_v0/probe.json
    # or score against an external labels TSV (participant_id + columns of scalars):
    python scripts/sae_probe.py --ckpt ... --latent-root ... --labels-tsv labels.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch
from neurodrift.models.sae import TopKSAE

log = logging.getLogger("neurodrift.sae_probe")

# Latent shape (B, C, d, d, d): the channel axis the SAE tokenises over.
_CHANNEL_AXIS = 1


# ---------------------------------------------------------------------------
# Tokenisation + SAE activation summaries
# ---------------------------------------------------------------------------
def latent_to_tokens(z: torch.Tensor) -> torch.Tensor:
    """Latent (B, C, d, d, d) -> tokens (B*d^3, C). Mirrors the LitModule reshape."""
    c = z.shape[_CHANNEL_AXIS]
    return z.permute(0, 2, 3, 4, 1).contiguous().reshape(-1, c)


@torch.no_grad()
def per_subject_feature_means(model: TopKSAE, latents: torch.Tensor) -> torch.Tensor:
    """Mean SAE activation per (subject, feature): (B, d_hidden).

    A subject's latent is many tokens; we summarise each feature by its mean
    activation over that subject's tokens, the standard pooling for relating a
    sparse feature to a subject-level label.
    """
    b = latents.shape[0]
    n_tokens = latents[0:1].permute(0, 2, 3, 4, 1).reshape(-1, latents.shape[1]).shape[0]
    means = latents.new_zeros((b, model.d_hidden))
    for i in range(b):
        acts, _ = model.encode(latent_to_tokens(latents[i : i + 1]))
        means[i] = acts.reshape(n_tokens, model.d_hidden).mean(dim=0)
    return means


# ---------------------------------------------------------------------------
# Biomarker correlation
# ---------------------------------------------------------------------------
def feature_biomarker_correlation(acts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-feature Pearson r between a subject-level feature summary and a label.

    acts:   (B, F) per-subject feature activations (e.g. per_subject_feature_means).
    labels: (B,) scalar label per subject; NaNs are dropped pairwise.
    Returns (F,) Pearson r in [-1, 1]; a feature (or the label) with zero variance
    over the kept subjects yields 0 (undefined correlation, reported as no signal).
    """
    if acts.dim() != 2:
        raise ValueError(f"acts must be (B, F); got {tuple(acts.shape)}")
    labels = labels.to(acts.dtype)
    finite = torch.isfinite(labels)
    acts = acts[finite]
    labels = labels[finite]
    if acts.shape[0] < 2:
        return acts.new_zeros(acts.shape[1] if acts.dim() == 2 else 0)
    a = acts - acts.mean(dim=0, keepdim=True)
    y = labels - labels.mean()
    cov = (a * y.unsqueeze(1)).mean(dim=0)
    denom = a.std(dim=0, unbiased=False) * y.std(unbiased=False)
    r = cov / denom.clamp_min(1e-12)
    # Zero-variance features divided by the clamp give a spurious value; force them
    # to 0 so a dead/constant feature reads as "no correlation", not noise.
    r = torch.where(denom > 1e-12, r, torch.zeros_like(r))
    return r.clamp(-1.0, 1.0)


def top_correlated_features(
    acts: torch.Tensor, labels: torch.Tensor, top: int = 20
) -> list[tuple[int, float]]:
    """Features ranked by |Pearson r| with the label. Returns [(feature_idx, r), ...]."""
    r = feature_biomarker_correlation(acts, labels)
    top = min(top, r.numel())
    order = r.abs().topk(top).indices
    return [(int(i), float(r[i])) for i in order]


# ---------------------------------------------------------------------------
# Proxy biomarkers (honest stand-in; real labels preferred)
# ---------------------------------------------------------------------------
@torch.no_grad()
def proxy_biomarkers(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Latent-derived volumetric PROXIES, one scalar per subject.

    NOT real biomarkers. FreeSurfer volumetrics are absent in this repo, so these
    are cheap, transparent summaries of the VAE content latent that behave LIKE
    coarse morphometrics for wiring/sanity:

      * latent_energy:     mean squared activation (overall "anatomical content").
      * active_fraction:   fraction of latent voxels above the latent mean (a crude
                           tissue-vs-background volume proxy).
      * spatial_spread:    std of per-voxel L2 norm across space (heterogeneity, a
                           very loose atrophy/structure-variability proxy).

    Replace with `--labels-tsv` real FreeSurfer/region volumes for any real claim.
    z: (B, C, d, d, d). Returns {name: (B,) tensor}.
    """
    per_voxel = z.pow(2).mean(dim=_CHANNEL_AXIS)  # (B, d, d, d)
    energy = per_voxel.flatten(1).mean(dim=1)
    thresh = per_voxel.flatten(1).mean(dim=1, keepdim=True)
    active = (per_voxel.flatten(1) > thresh).float().mean(dim=1)
    norm = z.norm(dim=_CHANNEL_AXIS)  # (B, d, d, d)
    spread = norm.flatten(1).std(dim=1)
    return {
        "proxy_latent_energy": energy,
        "proxy_active_fraction": active,
        "proxy_spatial_spread": spread,
    }


def load_labels_tsv(path: str | Path, subjects: list[str]) -> dict[str, torch.Tensor]:
    """Read a labels TSV into {column_name: (B,) tensor aligned to `subjects`}.

    TSV must have a `participant_id` (or `subject`) column; every other numeric
    column becomes a biomarker. A subject missing from the file, or a non-numeric
    cell, maps to NaN (dropped pairwise in the correlation). This is the REAL-label
    path; `proxy_biomarkers` is the fallback when no TSV is given.
    """
    path = Path(path)
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = reader.fieldnames or []
        id_col = next((c for c in ("participant_id", "subject") if c in fields), None)
        if id_col is None:
            raise SystemExit("labels TSV needs a 'participant_id' or 'subject' column")
        value_cols = [c for c in fields if c != id_col]
        for row in reader:
            key = (row.get(id_col) or "").strip()
            if key and not key.startswith("sub-"):
                key = f"sub-{key}"
            if key:
                rows[key] = row
    out: dict[str, torch.Tensor] = {}
    for col in value_cols:
        vals: list[float] = []
        for s in subjects:
            cell = rows.get(s, {}).get(col, "")
            try:
                vals.append(float(cell))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        out[col] = torch.tensor(vals, dtype=torch.float32)
    return out


# ---------------------------------------------------------------------------
# Aging direction + steering
# ---------------------------------------------------------------------------
def aging_direction(latents: torch.Tensor, ages: torch.Tensor, threshold: float) -> torch.Tensor:
    """Mean-difference steering vector (old - young) in latent space.

    latents: (B, C, d, d, d); ages: (B,) (NaNs excluded from both populations).
    Splits subjects at `threshold` (>= old, < young), returns the per-element mean
    latent difference, shape (C, d, d, d) — add `scale * vector` to a latent to age
    it. Falls back to a zero vector if either population is empty (caller logs it).
    """
    finite = torch.isfinite(ages)
    z = latents[finite]
    a = ages[finite]
    old = z[a >= threshold]
    young = z[a < threshold]
    if old.shape[0] == 0 or young.shape[0] == 0:
        return latents.new_zeros(latents.shape[1:])
    return old.mean(dim=0) - young.mean(dim=0)


def apply_steering(z: torch.Tensor, vector: torch.Tensor, scale: float) -> torch.Tensor:
    """Walk a latent along a steering vector: z + scale * vector.

    z: (B, C, d, d, d) or (C, d, d, d); `vector`: (C, d, d, d) (broadcasts over B).
    Returns a new tensor (input unmodified).
    """
    return z + scale * vector


# ---------------------------------------------------------------------------
# CLI plumbing (mirrors scripts/eval.py)
# ---------------------------------------------------------------------------
def _load_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load a Lightning or bare state-dict checkpoint into the raw TopKSAE.

    Lightning saves under `state_dict` with a `model.` prefix (the LitModule wraps
    the SAE as `self.model`); strip it so the weights land on the bare module.
    Identical idiom to scripts/eval.py._load_weights.
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


def _latents_from_root(
    root: Path, cohorts: list[str]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Stack every cached latent under `root/<cohort>/*.zarr` into (B,C,d,d,d).

    Uses the read side of the shared latent store (neurodrift.data.latents) so the
    probe consumes exactly what scripts/encode_latents.py wrote. Returns
    (latents, ages, subjects). Stores with mismatched latent shapes are skipped
    with a warning (a corrupt/partial store).
    """
    from neurodrift.data.latents import list_latent_refs, read_latent_store

    zs: list[torch.Tensor] = []
    ages: list[float] = []
    subjects: list[str] = []
    ref_shape: tuple[int, ...] | None = None
    for cohort in cohorts:
        for ref in list_latent_refs(str(root), cohort):
            loaded = read_latent_store(ref.url)
            if loaded is None:
                continue
            arr, attrs = loaded
            t = torch.from_numpy(arr).float()
            if ref_shape is None:
                ref_shape = tuple(t.shape)
            if tuple(t.shape) != ref_shape:
                log.warning("skipping %s: shape %s != %s", ref.url, tuple(t.shape), ref_shape)
                continue
            zs.append(t)
            age = attrs.get("age")
            try:
                ages.append(float(age))
            except (TypeError, ValueError):
                ages.append(float("nan"))
            subjects.append(str(attrs.get("subject", ref.subject)))
    if not zs:
        raise SystemExit(f"no latent stores found under {root} for cohorts {cohorts}")
    return torch.stack(zs), torch.tensor(ages, dtype=torch.float32), subjects


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopK-SAE interpretability probe")
    p.add_argument("--ckpt", required=True, type=Path, help="trained SAE ckpt (Lightning or bare)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--latent-root", type=Path, help="root of cached <cohort>/*.zarr latents")
    src.add_argument(
        "--latents", type=Path, help="a .pt tensor file of stacked (B,C,d,d,d) latents"
    )
    p.add_argument("--cohorts", nargs="+", default=["abide", "openneuro"])
    p.add_argument("--ages", type=Path, help="optional .pt (B,) ages tensor when using --latents")
    p.add_argument("--labels-tsv", type=Path, help="real biomarker TSV (else latent proxies)")
    p.add_argument("--d-in", type=int, default=16)
    p.add_argument("--d-hidden", type=int, default=8192)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--aux-k", type=int, default=256)
    p.add_argument("--age-threshold", type=float, default=60.0, help="young/old split for steering")
    p.add_argument("--steer-scale", type=float, default=1.0, help="scale for the aging-vector demo")
    p.add_argument("--top", type=int, default=20, help="top-|r| features reported per biomarker")
    p.add_argument("--out", type=Path, default=Path("sae_probe.json"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    model = TopKSAE(d_in=args.d_in, d_hidden=args.d_hidden, k=args.k, aux_k=args.aux_k)
    _load_weights(model, args.ckpt)
    model.eval()

    if args.latent_root is not None:
        latents, ages, subjects = _latents_from_root(args.latent_root, args.cohorts)
    else:
        latents = torch.load(args.latents, map_location="cpu").float()
        ages = (
            torch.load(args.ages, map_location="cpu").float()
            if args.ages is not None
            else torch.full((latents.shape[0],), float("nan"))
        )
        subjects = [f"sub-{i}" for i in range(latents.shape[0])]
    log.info("loaded %d latents %s", latents.shape[0], tuple(latents.shape[1:]))

    acts = per_subject_feature_means(model, latents)

    if args.labels_tsv is not None:
        labels = load_labels_tsv(args.labels_tsv, subjects)
        label_source = f"tsv:{args.labels_tsv.name}"
    else:
        labels = proxy_biomarkers(latents)
        labels["age"] = ages
        label_source = "latent_proxies (NOT real biomarkers)"
    log.info("biomarker source: %s", label_source)

    correlations: dict[str, list[list[float]]] = {}
    for name, lab in labels.items():
        top = top_correlated_features(acts, lab, top=args.top)
        correlations[name] = [[idx, r] for idx, r in top]
        if top:
            log.info("  %-24s top |r|=%.3f (feature %d)", name, abs(top[0][1]), top[0][0])

    vec = aging_direction(latents, ages, args.age_threshold)
    aging_norm = float(vec.norm())
    steered = apply_steering(latents[:1], vec, args.steer_scale)
    steer_delta = float((steered - latents[:1]).abs().mean())
    if aging_norm == 0.0:
        log.warning("aging direction is zero (one age population empty) — check --age-threshold")

    n_finite_age = int(torch.isfinite(ages).sum())
    report: dict[str, Any] = {
        "n_subjects": int(latents.shape[0]),
        "latent_shape": list(latents.shape[1:]),
        "d_hidden": model.d_hidden,
        "k": model.k,
        "dead_fraction": float(model.dead_mask.float().mean()),
        "biomarker_source": label_source,
        "n_finite_age": n_finite_age,
        "age_threshold": args.age_threshold,
        "aging_direction_norm": aging_norm,
        "steering_scale": args.steer_scale,
        "steering_mean_abs_delta": steer_delta,
        "top_correlated_features": correlations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
