"""CPU tests for scripts/build_age_map.py age recovery (no network).

Covers the load-bearing pure logic: tolerant age coercion (numeric / binned /
units / NA), age-column selection across heterogeneous OpenNeuro headers, the
ABIDE SUB_ID->sub-{:07d} mapping, provenance scanning, and the collision-aware
merge. The network fetch (fetch_text) and the real run are exercised on the box.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "build_age_map", Path(__file__).resolve().parent.parent / "scripts" / "build_age_map.py"
)
assert _SPEC and _SPEC.loader
bam = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bam)


# --- coerce_age ------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("55.4", 55.4),
        ("42", 42.0),
        ("55-60", 57.5),  # binned -> midpoint
        (f"20{chr(0x2013)}30", 25.0),  # en-dash range
        ("23 years", 23.0),  # trailing units
        ("", None),
        ("n/a", None),
        ("-9999", None),  # ABIDE/BIDS missing sentinel
        ("unknown", None),
        (None, None),
    ],
)
def test_coerce_age(raw: str | None, expected: float | None) -> None:
    out = bam.coerce_age(raw)
    if expected is None:
        assert math.isnan(out)
    else:
        assert out == pytest.approx(expected)


# --- pick_age_column -------------------------------------------------------
def test_pick_age_column_prefers_standard() -> None:
    assert bam.pick_age_column(["gender", "age", "site"]) == "age"
    assert bam.pick_age_column(["AGE_AT_SCAN", "sex"]) == "AGE_AT_SCAN"


def test_pick_age_column_binned_fallback() -> None:
    # No standard column -> the only "age"-containing header wins.
    assert bam.pick_age_column(["gender", "age (5-year bins)"]) == "age (5-year bins)"


def test_pick_age_column_none() -> None:
    assert bam.pick_age_column(["gender", "site", "handedness"]) is None


# --- parse_participants_table ---------------------------------------------
def test_parse_participants_table_standard() -> None:
    text = "participant_id\tage\tsex\nsub-01\t34.0\tM\nsub-02\tn/a\tF\n004\t56\tM\n"
    ages = bam.parse_participants_table(text)
    assert ages["sub-01"] == pytest.approx(34.0)
    assert math.isnan(ages["sub-02"])
    assert ages["sub-004"] == pytest.approx(56.0)  # bare id normalised


def test_parse_participants_table_binned_age_column() -> None:
    # ds000221-style: weird column name + binned ranges.
    text = "participant_id\tgender\tage (5-year bins)\nsub-010001\tF\t55-60\nsub-010002\tF\t65-70\n"
    ages = bam.parse_participants_table(text)
    assert ages["sub-010001"] == pytest.approx(57.5)
    assert ages["sub-010002"] == pytest.approx(67.5)


def test_parse_participants_table_no_participant_id() -> None:
    assert bam.parse_participants_table("foo\tbar\n1\t2\n") == {}


# --- parse_abide_phenotype -------------------------------------------------
def test_parse_abide_phenotype() -> None:
    text = (
        "SITE_ID,SUB_ID,AGE_AT_SCAN,SEX\nCALTECH,51456,55.4,1\nCALTECH,51457,22.9,1\nX,,-9999,1\n"
    )
    ages = bam.parse_abide_phenotype(text)
    assert ages["sub-0051456"] == pytest.approx(55.4)  # SUB_ID -> sub-{:07d}
    assert ages["sub-0051457"] == pytest.approx(22.9)
    assert "sub-" not in "".join(k for k in ages if not k.endswith(("6", "7")))  # only the 2 rows


# --- scan_provenance + merge ----------------------------------------------
def test_scan_provenance(tmp_path: Path) -> None:
    for ds, subs in {"ds000221": ["sub-010001", "sub-010002"], "ds000228": ["sub-01"]}.items():
        for s in subs:
            (tmp_path / ds / s).mkdir(parents=True)
    (tmp_path / "ds000221" / "dataset_description.json").write_text("{}")  # non-sub ignored
    prov = bam.scan_provenance(tmp_path)
    assert prov == {"ds000221": {"sub-010001", "sub-010002"}, "ds000228": {"sub-01"}}


def test_merge_assigns_per_dataset_age() -> None:
    prov = {"dsA": {"sub-01", "sub-02"}, "dsB": {"sub-09"}}
    per_ds = {"dsA": {"sub-01": 30.0, "sub-02": 40.0}, "dsB": {"sub-09": 70.0}}
    ages, collisions = bam.merge_openneuro_ages(prov, per_ds)
    assert ages == {"sub-01": 30.0, "sub-02": 40.0, "sub-09": 70.0}
    assert collisions == []


def test_merge_flags_conflicting_collision() -> None:
    # Same id in two datasets with DIFFERENT real ages -> reported, first kept.
    prov = {"dsA": {"sub-01"}, "dsB": {"sub-01"}}
    per_ds = {"dsA": {"sub-01": 30.0}, "dsB": {"sub-01": 65.0}}
    ages, collisions = bam.merge_openneuro_ages(prov, per_ds)
    assert ages["sub-01"] in (30.0, 65.0)
    assert len(collisions) == 1 and "sub-01" in collisions[0]


def test_merge_nan_then_real_fills() -> None:
    # A NaN in one dataset is backfilled by a real age in another (no collision).
    prov = {"dsA": {"sub-01"}, "dsB": {"sub-01"}}
    per_ds = {"dsA": {"sub-01": float("nan")}, "dsB": {"sub-01": 50.0}}
    ages, collisions = bam.merge_openneuro_ages(prov, per_ds)
    assert ages["sub-01"] == pytest.approx(50.0)
    assert collisions == []
