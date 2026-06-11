"""Integration test for refresh_all._result() — would have caught PR #9 regression.

PR #9 broke production by passing RefreshResult(name=...) where the
dataclass field is `source`. Unit tests for the validators module passed
in isolation but didn't exercise the integration path. This module fixes
that gap by testing the full _result() return path against a temporary
CSV + validator.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))


@pytest.fixture(autouse=True)
def _patch_validators_storage(monkeypatch, tmp_path):
    """Point validators.STORAGE at tmp_path so we don't need real CSVs."""
    import _validators as v   # type: ignore
    monkeypatch.setattr(v, "STORAGE", tmp_path / "storage" / "conquest")
    (tmp_path / "storage" / "conquest").mkdir(parents=True)
    # Also patch all spec paths to live under tmp_path
    for name, spec in v.REGISTRY.items():
        rel = spec.output_path.relative_to(v.WORKSPACE) if v.WORKSPACE in spec.output_path.parents else spec.output_path
        spec.output_path = tmp_path / rel
        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    yield


def test_result_returns_valid_refresh_result_on_subprocess_failure(tmp_path):
    """rc != 0 path should produce a valid RefreshResult with source= set."""
    import refresh_all as ra   # type: ignore
    out_path = tmp_path / "no-such.csv"
    result = ra._result(
        name="form4",
        out_path=out_path,
        t0=0.0,
        date_cols=["filing_date"],
        max_lag_days=10,
        today=date(2026, 5, 28),
        rc=1,
        error="subprocess timed out",
    )
    assert result.source == "form4"   # would have caught PR #9: source vs name
    assert result.status == "failed"
    assert "timed out" in (result.error or "")


def test_result_returns_valid_refresh_result_on_subprocess_success(tmp_path):
    """rc == 0 path must also return a valid RefreshResult.

    This is the EXACT path PR #9 broke — silently returned None or raised
    TypeError because of kwarg mismatch.
    """
    import refresh_all as ra   # type: ignore
    # Write a valid CSV the validator will accept
    csv_path = tmp_path / "storage" / "conquest" / "vix" / "daily.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": ["2026-05-27"],
        "vix": [18.5],
    }).to_csv(csv_path, index=False)
    # Patch the VIX spec to point at our temp CSV
    import _validators as v   # type: ignore
    v.REGISTRY["vix"].output_path = csv_path

    result = ra._result(
        name="vix",
        out_path=csv_path,
        t0=0.0,
        date_cols=["date"],
        max_lag_days=4,
        today=date(2026, 5, 28),
        rc=0,
    )
    # Critical assertions: result is well-formed AND uses source= not name=
    assert isinstance(result, ra.RefreshResult)
    assert result.source == "vix"
    # Status should not be "failed" since rc=0 and the file is well-formed
    assert result.status != "failed"
    assert result.rows == 1


def test_result_handles_missing_spec_gracefully(tmp_path):
    """If a source has no SourceSpec, _result should still produce a valid
    RefreshResult (validator returns missing_spec advisory)."""
    import refresh_all as ra   # type: ignore
    csv_path = tmp_path / "fake.csv"
    csv_path.write_text("col1,col2\n1,2\n")
    result = ra._result(
        name="totally_invented_source",
        out_path=csv_path,
        t0=0.0,
        date_cols=["col1"],
        max_lag_days=4,
        today=date(2026, 5, 28),
        rc=0,
    )
    assert isinstance(result, ra.RefreshResult)
    assert result.source == "totally_invented_source"


def test_result_combines_validator_status_with_legacy_stale(tmp_path):
    """Worst-of(legacy_status, validator.status) is the final status."""
    import refresh_all as ra   # type: ignore
    import _validators as v   # type: ignore

    # Write a CSV with very stale data — validator should flag stale
    csv_path = tmp_path / "stale.csv"
    csv_path.write_text("date,value\n2020-01-01,1.0\n")
    # Override vix spec to point here for the test
    v.REGISTRY["vix"].output_path = csv_path

    result = ra._result(
        name="vix",
        out_path=csv_path,
        t0=0.0,
        date_cols=["date"],
        max_lag_days=4,
        today=date(2026, 5, 28),
        rc=0,
    )
    # Stale date should be detected
    assert result.status in ("stale", "empty", "invalid", "failed")
    assert result.source == "vix"
