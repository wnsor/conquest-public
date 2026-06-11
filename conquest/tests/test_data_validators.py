"""Unit tests for scripts/data/_validators.py.

Covers each failure mode the validators are designed to catch — the
specific bugs that have actually hit us before (silent rows=0, schema
drift, stale data, unparseable dates, 100% non-numeric numeric columns).
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Insert the scripts/data dir on the path so the absolute import works
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from _validators import SourceSpec, validate, validate_by_name, REGISTRY  # type: ignore


TODAY = date(2026, 5, 27)


def _write_csv(tmp_path: Path, df: pd.DataFrame, name: str = "test.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ── basic shape checks ────────────────────────────────────────────────────

def test_missing_file_is_empty(tmp_path):
    spec = SourceSpec(name="x", output_path=tmp_path / "no.csv")
    res = validate(spec, TODAY)
    assert res.status == "empty"
    assert "missing" in res.issues[0]


def test_empty_dataframe_is_empty(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({"date": [], "v": []}))
    spec = SourceSpec(name="x", output_path=p, min_rows=1)
    res = validate(spec, TODAY)
    assert res.status == "empty"


def test_row_count_below_min(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({"date": ["2026-05-01"], "v": [1]}))
    spec = SourceSpec(name="x", output_path=p, min_rows=10)
    res = validate(spec, TODAY)
    assert res.status == "empty"
    assert "1 below min 10" in res.issues[0]


def test_invalid_csv_is_invalid(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_bytes(b"\x00\x01\x02not csv data at all\xff\xfe")
    spec = SourceSpec(name="x", output_path=p)
    res = validate(spec, TODAY)
    # pandas may still parse garbage into a single column — but min_rows=1 + content
    # means it could pass shape. Force schema check.
    spec2 = SourceSpec(name="x", output_path=p, required_columns=["date", "value"])
    res2 = validate(spec2, TODAY)
    assert res2.status in ("invalid", "schema_mismatch", "empty")


# ── schema checks ─────────────────────────────────────────────────────────

def test_missing_required_columns_is_schema_mismatch(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({"date": ["2026-05-01"], "v": [1]}))
    spec = SourceSpec(name="x", output_path=p, required_columns=["date", "MISSING"])
    res = validate(spec, TODAY)
    assert res.status == "schema_mismatch"
    assert "MISSING" in res.issues[0]


def test_extra_columns_are_allowed(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({"date": ["2026-05-01"], "v": [1], "extra": [42]}))
    spec = SourceSpec(name="x", output_path=p, required_columns=["date"])
    res = validate(spec, TODAY)
    assert res.status == "ok"


# ── date / staleness checks ───────────────────────────────────────────────

def test_fresh_date_passes(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({
        "date": [(TODAY - timedelta(days=1)).isoformat()], "v": [1]
    }))
    spec = SourceSpec(name="x", output_path=p, date_column="date", max_lag_days=4)
    res = validate(spec, TODAY)
    assert res.status == "ok"
    assert res.max_date == (TODAY - timedelta(days=1)).isoformat()


def test_stale_date_caught(tmp_path):
    old = (TODAY - timedelta(days=30)).isoformat()
    p = _write_csv(tmp_path, pd.DataFrame({"date": [old], "v": [1]}))
    spec = SourceSpec(name="x", output_path=p, date_column="date", max_lag_days=4)
    res = validate(spec, TODAY)
    assert res.status == "stale"
    assert "30d old" in res.issues[0]


def test_unparseable_dates_caught(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({"date": ["not-a-date", "still-bad"], "v": [1, 2]}))
    spec = SourceSpec(name="x", output_path=p, date_column="date")
    res = validate(spec, TODAY)
    assert res.status == "invalid"
    assert "100% unparseable" in res.issues[0]


# ── numeric column checks ─────────────────────────────────────────────────

def test_numeric_column_all_non_numeric_is_invalid(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({"date": ["2026-05-01"], "value": ["text only"]}))
    spec = SourceSpec(
        name="x", output_path=p,
        date_column="date", numeric_columns=["value"],
    )
    res = validate(spec, TODAY)
    assert res.status == "invalid"
    assert "100% non-numeric" in res.issues[0]


def test_numeric_column_partial_nulls_under_threshold_ok(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({
        "date": ["2026-05-01"] * 10,
        "value": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # 10% null
    }))
    spec = SourceSpec(
        name="x", output_path=p,
        date_column="date",
        numeric_columns=["value"],
        null_rate_threshold={"value": 0.20},   # threshold 20%, actual 10%
    )
    res = validate(spec, TODAY)
    assert res.status == "ok"


def test_numeric_column_over_threshold_flagged(tmp_path):
    p = _write_csv(tmp_path, pd.DataFrame({
        "date": ["2026-05-01"] * 5,
        "value": [1.0, None, None, None, None],   # 80% null
    }))
    spec = SourceSpec(
        name="x", output_path=p,
        date_column="date",
        numeric_columns=["value"],
        null_rate_threshold={"value": 0.30},   # threshold 30%, actual 80%
    )
    res = validate(spec, TODAY)
    assert res.status == "stale"
    assert "null rate" in res.issues[-1]


# ── registry sanity ──────────────────────────────────────────────────────

def test_registry_has_all_known_sources():
    """Every handler in refresh_all.HANDLERS should have a SourceSpec.

    If this test fails, a new ingester was added without a corresponding
    SourceSpec — add one to enable schema validation for that source.
    """
    sys.path.insert(0, str(ROOT / "scripts" / "data"))
    # Import refresh_all to get the canonical handler list
    try:
        import refresh_all  # type: ignore
    except Exception:
        pytest.skip("refresh_all.py not importable (missing deps in test env)")
        return
    handlers = set(refresh_all.HANDLERS.keys())
    specs = set(REGISTRY.keys())
    missing = handlers - specs
    assert not missing, (
        f"refresh_all.HANDLERS sources without a SourceSpec: {sorted(missing)}. "
        f"Add each to scripts/data/_validators.py REGISTRY."
    )


def test_validate_by_name_returns_missing_spec_for_unknown():
    res = validate_by_name("totally_made_up", TODAY)
    assert res.status == "missing_spec"
    assert "no SourceSpec" in res.issues[0]
