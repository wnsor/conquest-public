"""Tier1 Signal 3 — InsiderForm4Calendar.cluster_score() unit tests.

Cluster score = sum of role weights over DISTINCT insider_ciks with at least
one qualifying buy (>= min_dollar) within the lookback window.
  - Officer  → 2.0
  - Director → 1.5
  - 10pct    → 1.0

A CIK that appears multiple times counts once at its highest role weight.

Sparsity note: the production CSV currently holds ~4 rows
(`storage/conquest/insider/form4_opportunistic_buys_daily.csv`); the full
2018-2024 SEC Form 4 backfill is a separate deferred task. These tests
validate the math, not the data — production cluster_score will return
0.0 for most tickers until the backfill runs.
"""
from __future__ import annotations

import csv
import sys
from datetime import date
from io import StringIO
from pathlib import Path

# Bare-sibling import path per memory feedback_lean_bare_imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))

from edge_signals.insider_lookup import InsiderForm4Calendar  # noqa: E402


def _calendar_from_rows(rows: list[dict]) -> InsiderForm4Calendar:
    """Build a calendar from in-memory rows by routing through from_csv_text.

    Uses csv.DictWriter with default QUOTE_MINIMAL so comma-containing role
    strings (e.g. "Officer,Director" from scripts/ingest_insider_form4.py)
    survive the round-trip.
    """
    if not rows:
        return InsiderForm4Calendar.from_csv_text("")
    cols = list(rows[0].keys())
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)
    return InsiderForm4Calendar.from_csv_text(buf.getvalue())


def test_cluster_score_distinct_roles_sum():
    """1 Officer + 1 Director + 1 10pct, all distinct CIKs in window → 2.0 + 1.5 + 1.0 = 4.5."""
    cal = _calendar_from_rows([
        {"ticker": "ACOG", "transaction_date": "2024-06-03", "role": "Officer",
         "dollar_value": 100_000, "insider_cik": "CIK_A"},
        {"ticker": "ACOG", "transaction_date": "2024-06-04", "role": "Director",
         "dollar_value": 100_000, "insider_cik": "CIK_B"},
        {"ticker": "ACOG", "transaction_date": "2024-06-05", "role": "10pct",
         "dollar_value": 100_000, "insider_cik": "CIK_C"},
    ])
    assert cal.cluster_score("ACOG", date(2024, 6, 6), n_days=5) == 4.5


def test_cluster_score_dedupes_same_cik():
    """Same CIK buys twice → counted ONCE at max role weight."""
    cal = _calendar_from_rows([
        {"ticker": "MU", "transaction_date": "2024-06-01", "role": "Director",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
        {"ticker": "MU", "transaction_date": "2024-06-03", "role": "Director",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
    ])
    # Two filings, same insider — score is 1.5 (Director), not 3.0
    assert cal.cluster_score("MU", date(2024, 6, 5), n_days=5) == 1.5


def test_cluster_score_picks_max_role_for_same_cik():
    """Same CIK files as Director then later as Officer — counts as Officer (2.0)."""
    cal = _calendar_from_rows([
        {"ticker": "PLTR", "transaction_date": "2024-06-01", "role": "Director",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
        {"ticker": "PLTR", "transaction_date": "2024-06-03", "role": "Officer",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
    ])
    assert cal.cluster_score("PLTR", date(2024, 6, 5), n_days=5) == 2.0


def test_cluster_score_combined_role_string_uses_max():
    """role='Officer,Director' (insider holds both titles) → counts as Officer (2.0)."""
    cal = _calendar_from_rows([
        {"ticker": "AVGO", "transaction_date": "2024-06-03", "role": "Officer,Director",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
    ])
    assert cal.cluster_score("AVGO", date(2024, 6, 4), n_days=5) == 2.0


def test_cluster_score_window_excludes_old_buys():
    """A buy 10 days ago is outside the 5-day window → not counted."""
    cal = _calendar_from_rows([
        {"ticker": "COIN", "transaction_date": "2024-05-25", "role": "Officer",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
        {"ticker": "COIN", "transaction_date": "2024-06-03", "role": "Director",
         "dollar_value": 100_000, "insider_cik": "CIK_Y"},
    ])
    # Officer buy on 5/25 is outside 5-day window; only the Director counts
    assert cal.cluster_score("COIN", date(2024, 6, 5), n_days=5) == 1.5


def test_cluster_score_min_dollar_filter():
    """Buys below min_dollar=25_000 are filtered out."""
    cal = _calendar_from_rows([
        {"ticker": "RKLB", "transaction_date": "2024-06-03", "role": "Officer",
         "dollar_value": 10_000, "insider_cik": "CIK_X"},
        {"ticker": "RKLB", "transaction_date": "2024-06-04", "role": "Director",
         "dollar_value": 50_000, "insider_cik": "CIK_Y"},
    ])
    # Officer's $10k is below min; only Director's $50k counts
    assert cal.cluster_score("RKLB", date(2024, 6, 5), n_days=5) == 1.5


def test_cluster_score_missing_cik_is_dropped():
    """Row with empty insider_cik can't be deduped — dropped, not collapsed."""
    cal = _calendar_from_rows([
        {"ticker": "MX", "transaction_date": "2024-06-03", "role": "Director",
         "dollar_value": 100_000, "insider_cik": ""},
    ])
    assert cal.cluster_score("MX", date(2024, 6, 4), n_days=5) == 0.0


def test_cluster_score_no_buys_returns_zero():
    """Ticker absent from calendar returns 0.0 (not None, not exception)."""
    cal = _calendar_from_rows([])
    assert cal.cluster_score("UNKNOWN", date(2024, 6, 5), n_days=5) == 0.0


def test_cluster_score_future_buys_ignored():
    """Walk-forward safety: today=6/3, buy date 6/10 → not counted."""
    cal = _calendar_from_rows([
        {"ticker": "IONQ", "transaction_date": "2024-06-10", "role": "Officer",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
    ])
    assert cal.cluster_score("IONQ", date(2024, 6, 3), n_days=5) == 0.0


def test_buys_within_n_days_returns_4tuple():
    """Schema regression: buys_within_n_days returns 4-tuples after Phase 2 extension.
    Existing index-2 (dollar) consumer in main.py still works."""
    cal = _calendar_from_rows([
        {"ticker": "MU", "transaction_date": "2024-06-03", "role": "Officer",
         "dollar_value": 100_000, "insider_cik": "CIK_X"},
    ])
    buys = cal.buys_within_n_days("MU", date(2024, 6, 4), n=5)
    assert len(buys) == 1
    assert len(buys[0]) == 4
    td, role, dollar, cik = buys[0]
    assert td == date(2024, 6, 3)
    assert role == "Officer"
    assert dollar == 100_000.0
    assert cik == "CIK_X"
    # Existing consumer pattern from main.py:516 — `max(b[2] for b in buys)` — still works
    assert max(b[2] for b in buys) == 100_000.0


def test_strategy_context_has_insider_cluster_score_field():
    """Schema regression: StrategyContext must have insider_cluster_score as a dict field."""
    from datetime import datetime

    from strategies.base import StrategyContext  # noqa: E402

    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        insider_cluster_score={"PLTR": 3.5},
    )
    assert ctx.insider_cluster_score == {"PLTR": 3.5}
    ctx2 = StrategyContext(timestamp=datetime(2024, 6, 3, 15, 0))
    assert ctx2.insider_cluster_score == {}
