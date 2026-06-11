"""Tests for the point-in-time universe gate (survivorship-bias kill).

Self-contained: synthesizes a small ``as_of,ticker`` monthly-snapshot CSV (the
schema ``scripts/train_xgb_m1.py`` consumes) so no QC / network / gitignored
storage data is required. The headline test is the *no-survivorship invariant*:
a name with the strongest (forward-known) momentum that was NOT a member on the
trade date can never appear in the PIT-gated selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# bare-sibling import per memory feedback_lean_bare_imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "conquest_options"))

from edge_signals.pit_universe import (  # noqa: E402
    PitUniverse,
    pit_filter,
    select_pit_top_n,
)
from edge_signals.stock_picker import StockPicker, TickerInputs  # noqa: E402


# A 3-snapshot universe. AAA is always a member; BBB joins in Feb; FUTURE only in Mar.
PIT_CSV = (
    "as_of,ticker\n"
    "2020-01-31,AAA\n"
    "2020-01-31,CCC\n"
    "2020-02-29,AAA\n"
    "2020-02-29,BBB\n"
    "2020-02-29,CCC\n"
    "2020-03-31,AAA\n"
    "2020-03-31,BBB\n"
    "2020-03-31,CCC\n"
    "2020-03-31,FUTURE\n"
)


def _make_inputs(mom_pct: float, *, liquid: bool = True) -> TickerInputs:
    """252d price ramp giving momentum_180d ≈ 1 + mom_pct; liquidity pass/fail toggle."""
    end_price = 100.0 * (1 + mom_pct)
    prices = [100.0] * (252 - 180)
    for i in range(180):
        prices.append(100.0 + (end_price - 100.0) * (i / 179.0))
    # small alternating vol so vol_60d > 0
    prices = [p * (1 + 0.01 * ((i % 2) * 2 - 1)) for i, p in enumerate(prices)]
    return TickerInputs(
        prices_252d=prices,
        news_count_10d=[10] * 5 + [20] * 5,
        adv_dollars=100_000_000 if liquid else 1_000_000,
        option_chain_size=200,
        iv_rank=50.0,
    )


# ── parsing ──────────────────────────────────────────────────────────────────

def test_from_csv_text_parses_as_of_schema():
    u = PitUniverse.from_csv_text(PIT_CSV)
    assert len(u) == 3
    assert u.snapshot_dates[0].isoformat() == "2020-01-31"
    assert u.snapshot_dates[-1].isoformat() == "2020-03-31"


def test_date_column_alias_accepted():
    u = PitUniverse.from_csv_text("date,ticker\n2021-06-30,XYZ\n")
    assert u.members_asof("2021-07-01") == {"XYZ"}


def test_bad_schema_raises():
    with pytest.raises(ValueError):
        PitUniverse.from_csv_text("foo,bar\n1,2\n")


# ── members_asof: at-or-before / carry-forward semantics ─────────────────────

def test_members_asof_exact_snapshot_date():
    u = PitUniverse.from_csv_text(PIT_CSV)
    assert u.members_asof("2020-01-31") == {"AAA", "CCC"}
    assert u.members_asof("2020-02-29") == {"AAA", "BBB", "CCC"}


def test_members_asof_carry_forward_between_and_after():
    u = PitUniverse.from_csv_text(PIT_CSV)
    # mid-February → carries the Jan snapshot (BBB not yet a member)
    assert u.members_asof("2020-02-15") == {"AAA", "CCC"}
    # after the last snapshot → carries the Mar snapshot
    assert u.members_asof("2025-01-01") == {"AAA", "BBB", "CCC", "FUTURE"}


def test_members_asof_before_first_snapshot_empty():
    u = PitUniverse.from_csv_text(PIT_CSV)
    assert u.members_asof("2019-12-01") == set()


def test_member_added_later_excluded_earlier():
    """A ticker present only in a LATER snapshot must not appear at an earlier date."""
    u = PitUniverse.from_csv_text(PIT_CSV)
    assert "FUTURE" not in u.members_asof("2020-02-29")
    assert "FUTURE" in u.members_asof("2020-03-31")
    assert "BBB" not in u.members_asof("2020-01-31")


# ── the no-survivorship invariant ────────────────────────────────────────────

def test_no_survivorship_invariant_high_momentum_nonmember_excluded():
    u = PitUniverse.from_csv_text(PIT_CSV)
    picker = StockPicker()
    # FUTURE has the strongest momentum BUT was not a member on 2020-02-15.
    inputs = {
        "AAA": _make_inputs(0.20),
        "CCC": _make_inputs(0.10),
        "FUTURE": _make_inputs(2.00),  # would dominate any unrestricted ranking
        "BBB": _make_inputs(1.50),     # not yet a member in mid-Feb either
    }
    picked = select_pit_top_n("2020-02-15", inputs, picker, u, n=10)
    assert "FUTURE" not in picked, "survivorship leak: forward-known winner selected pre-membership"
    assert "BBB" not in picked
    assert set(picked) <= {"AAA", "CCC"}
    assert "AAA" in picked  # the legitimate member is selectable

    # Same inputs, but a date where FUTURE IS a member → now eligible.
    picked_later = select_pit_top_n("2020-03-31", inputs, picker, u, n=10)
    assert "FUTURE" in picked_later


def test_select_pit_output_always_subset_of_members():
    u = PitUniverse.from_csv_text(PIT_CSV)
    picker = StockPicker()
    inputs = {tk: _make_inputs(0.3) for tk in ("AAA", "BBB", "CCC", "FUTURE", "NOPE")}
    for d in ("2020-01-31", "2020-02-29", "2020-03-31"):
        picked = select_pit_top_n(d, inputs, picker, u, n=10)
        assert set(picked) <= u.members_asof(d)


def test_select_pit_empty_when_no_eligible_or_no_inputs():
    u = PitUniverse.from_csv_text(PIT_CSV)
    picker = StockPicker()
    inputs = {"AAA": _make_inputs(0.3)}
    assert select_pit_top_n("2019-01-01", inputs, picker, u) == []   # before first snapshot
    assert select_pit_top_n("2020-03-31", {}, picker, u) == []        # no inputs


# ── pit_filter (lightweight entry-path gate) ─────────────────────────────────

def test_pit_filter_preserves_order_and_drops_nonmembers():
    u = PitUniverse.from_csv_text(PIT_CSV)
    out = pit_filter("2020-02-29", ["FUTURE", "BBB", "AAA", "NOPE", "CCC"], u)
    assert out == ["BBB", "AAA", "CCC"]  # order preserved, non-members dropped


def test_pit_filter_case_insensitive():
    u = PitUniverse.from_csv_text(PIT_CSV)
    assert pit_filter("2020-03-31", ["future", "aaa"], u) == ["future", "aaa"]
