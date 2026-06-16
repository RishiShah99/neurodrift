"""Recover per-subject AGE for the latent corpus -> consolidated participants.tsv.

Age is the lifespan flow's only real v0 conditioning signal, but the box's raw
corpus kept only imaging (no participants.tsv) and GCS has none either. This
rebuilds it from the public sources:

  * OpenNeuro — each dataset's `participants.tsv` from the public S3 mirror
    (`https://s3.amazonaws.com/openneuro.org/<ds>/participants.tsv`). The corpus
    flattened every dataset into one `openneuro/` namespace, so we use the raw
    layout (`<raw-root>/<ds>/sub-*`) as PROVENANCE: each subject's age comes from
    the dataset whose folder contains it, and a subject id that appears in two
    datasets with conflicting ages is reported as a collision (not silently merged).
  * ABIDE — the ABIDE I phenotype CSV (fcp-indi). `SUB_ID` -> `sub-{:07d}`,
    `AGE_AT_SCAN` -> age.

Output: `<out>/openneuro_participants.tsv` + `<out>/abide_participants.tsv`, each
`participant_id<TAB>age`, consumed by `encode_latents.py --participants-dir`.

OpenNeuro `participants.tsv` age columns are wildly heterogeneous (plain numeric,
`age (5-year bins)` with `55-60` ranges, units, `n/a`); `coerce_age` handles the
common shapes and the script REPORTS per-dataset coverage + anything it could not
parse, so partial recovery is visible rather than silent.

Run on the box (has the raw provenance dirs + internet):

    python scripts/build_age_map.py --raw-root ~/scratch/raw/openneuro --out ~/ages
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import urllib.request
from pathlib import Path

log = logging.getLogger("build_age_map")

_OPENNEURO_TMPL = "https://s3.amazonaws.com/openneuro.org/{ds}/participants.tsv"
_ABIDE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b.csv"

# Age column preference: an exact, unambiguous numeric-age column wins over a binned
# / derived one. Anything else containing "age" is a last-resort fallback.
_AGE_COL_PRIORITY = ("age", "age_at_scan", "ageatscan", "age_years", "scan_age")
_NA = {"", "n/a", "na", "nan", "none", "null", "-9999", "unknown"}
_RANGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$")  # noqa: RUF001  hyphen or en-dash
_LEADING_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Pure parsing (unit-tested without network)
# ---------------------------------------------------------------------------
def coerce_age(raw: str | None) -> float:
    """A heterogeneous age cell -> float years, or NaN if missing / unparseable.

    Handles plain numerics ("55.4"), binned ranges ("55-60" -> midpoint 57.5),
    values with trailing units ("23 years" -> 23), and the usual NA spellings.
    """
    if raw is None:
        return float("nan")
    s = raw.strip()
    if s.lower() in _NA:
        return float("nan")
    m = _RANGE_RE.match(s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (lo + hi) / 2.0
    try:
        return float(s)
    except ValueError:
        pass
    lead = _LEADING_NUM_RE.search(s)  # e.g. "23 years", "age: 41"
    if lead:
        try:
            return float(lead.group(0))
        except ValueError:
            return float("nan")
    return float("nan")


def pick_age_column(fieldnames: list[str]) -> str | None:
    """Choose the best age column from a participants.tsv header (case-insensitive).

    Prefers a standard numeric-age name; else the first column whose (normalised)
    name contains "age". Returns the ORIGINAL header string, or None if none.
    """
    norm = {f: re.sub(r"[^a-z0-9]", "", f.lower()) for f in fieldnames}
    for want in _AGE_COL_PRIORITY:
        for f, n in norm.items():
            if n == want:
                return f
    for f, n in norm.items():
        if "age" in n:
            return f
    return None


def _normalize_sub(raw: str) -> str | None:
    """A participant_id cell -> canonical `sub-XXX` key, or None if empty."""
    pid = raw.strip()
    if not pid:
        return None
    return pid if pid.startswith("sub-") else f"sub-{pid}"


def parse_participants_table(text: str) -> dict[str, float]:
    """A participants.tsv body -> {sub-XXX: age}. Tolerant of any age-column shape."""
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    fields = reader.fieldnames or []
    if "participant_id" not in fields:
        return {}
    age_col = pick_age_column([f for f in fields if f != "participant_id"])
    out: dict[str, float] = {}
    for row in reader:
        key = _normalize_sub(row.get("participant_id", ""))
        if key is None:
            continue
        out[key] = coerce_age(row.get(age_col) if age_col else None)
    return out


def parse_abide_phenotype(text: str) -> dict[str, float]:
    """ABIDE I phenotype CSV -> {sub-00XXXXX: age} from SUB_ID + AGE_AT_SCAN."""
    reader = csv.DictReader(io.StringIO(text))
    out: dict[str, float] = {}
    for row in reader:
        sub_id = (row.get("SUB_ID") or "").strip()
        if not sub_id:
            continue
        try:
            key = f"sub-{int(sub_id):07d}"
        except ValueError:
            continue
        out[key] = coerce_age(row.get("AGE_AT_SCAN"))
    return out


def scan_provenance(raw_root: Path) -> dict[str, set[str]]:
    """`<raw-root>/<ds>/sub-*` -> {dataset_id: {sub-XXX, ...}}.

    The dataset folder a subject lives in is its age provenance; this also reveals
    a subject id reused across datasets (a flat-namespace collision).
    """
    prov: dict[str, set[str]] = {}
    if not raw_root.exists():
        return prov
    for ds_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        subs = {p.name for p in ds_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")}
        if subs:
            prov[ds_dir.name] = subs
    return prov


def merge_openneuro_ages(
    provenance: dict[str, set[str]],
    per_ds_ages: dict[str, dict[str, float]],
) -> tuple[dict[str, float], list[str]]:
    """Assign each subject the age from ITS dataset; report conflicting collisions.

    For a subject present in >1 dataset, a collision is only flagged when the
    datasets disagree on a *real* (non-NaN) age — the genuinely ambiguous case.
    The first real age wins so the subject still gets a usable value.
    """
    import math

    ages: dict[str, float] = {}
    collisions: list[str] = []
    for ds, subs in provenance.items():
        ds_ages = per_ds_ages.get(ds, {})
        for sub in subs:
            new = ds_ages.get(sub, float("nan"))
            if sub not in ages:
                ages[sub] = new
                continue
            old = ages[sub]
            if not math.isnan(old) and not math.isnan(new) and abs(old - new) > 1e-6:
                collisions.append(f"{sub}: {old} vs {new} (ds {ds})")
            elif math.isnan(old):
                ages[sub] = new
    return ages, collisions


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def fetch_text(url: str, timeout: int = 30) -> str | None:
    """GET a small text file; None (with a warning) on any failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text: str = resp.read().decode("utf-8", errors="replace")
            return text
    except Exception as exc:  # network / 404 / decode — all non-fatal, skip the ds
        log.warning("fetch failed %s (%s)", url, exc)
        return None


def _write_tsv(path: Path, ages: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["participant_id", "age"])
        for sub in sorted(ages):
            a = ages[sub]
            w.writerow([sub, "n/a" if a != a else f"{a:g}"])  # a!=a -> NaN


def _coverage(ages: dict[str, float]) -> tuple[int, int]:
    import math

    real = sum(1 for a in ages.values() if not math.isnan(a))
    return real, len(ages)


def build_openneuro(raw_root: Path, out: Path, tmpl: str) -> dict[str, float]:
    prov = scan_provenance(raw_root)
    log.info("openneuro: %d datasets under %s", len(prov), raw_root)
    per_ds: dict[str, dict[str, float]] = {}
    for ds in sorted(prov):
        text = fetch_text(tmpl.format(ds=ds))
        ds_ages = parse_participants_table(text) if text else {}
        real, total = _coverage({s: ds_ages.get(s, float("nan")) for s in prov[ds]})
        per_ds[ds] = ds_ages
        flag = "" if real else "  <-- NO ages parsed"
        log.info("  %s: %d subjects, %d/%d with age%s", ds, len(prov[ds]), real, total, flag)
    ages, collisions = merge_openneuro_ages(prov, per_ds)
    real, total = _coverage(ages)
    log.info("openneuro TOTAL: %d/%d subjects with a real age", real, total)
    if collisions:
        log.warning(
            "openneuro: %d cross-dataset age collisions, e.g. %s", len(collisions), collisions[:5]
        )
    _write_tsv(out / "openneuro_participants.tsv", ages)
    return ages


def build_abide(out: Path, url: str) -> dict[str, float]:
    text = fetch_text(url)
    ages = parse_abide_phenotype(text) if text else {}
    real, total = _coverage(ages)
    log.info("abide: %d/%d subjects with a real age", real, total)
    _write_tsv(out / "abide_participants.tsv", ages)
    return ages


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts", default="abide,openneuro")
    p.add_argument("--raw-root", default="", help="OpenNeuro raw root with <ds>/sub-* provenance.")
    p.add_argument("--out", required=True, help="Output dir for <cohort>_participants.tsv.")
    p.add_argument(
        "--openneuro-url", default=_OPENNEURO_TMPL, help="{ds} participants.tsv template."
    )
    p.add_argument("--abide-url", default=_ABIDE_URL)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    out = Path(args.out).expanduser()
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    if "openneuro" in cohorts:
        if not args.raw_root:
            log.error("openneuro needs --raw-root (the <ds>/sub-* provenance dir)")
            return 2
        build_openneuro(Path(args.raw_root).expanduser(), out, args.openneuro_url)
    if "abide" in cohorts:
        build_abide(out, args.abide_url)
    log.info("wrote consolidated participants.tsv to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
