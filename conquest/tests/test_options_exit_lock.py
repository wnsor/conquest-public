"""Regression tests for ExitManager v22 Fix 2 (lock-once-triggered).

The bug: in fast-moving markets, premium can spike to TP for one tick then
retrace. Before this fix, positions_to_close re-evaluated price every tick
and DROPPED the close intent if the spike retraced. The strategy held past
the target, often exiting at SL or max_hold instead.

After this fix: once TP/SL fires, the (symbol, reason) is LOCKED in
_triggered_exits — re-evaluated every tick as "yes, still trying to exit"
until the caller unregisters the position (exit fills).
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))

from exit_manager import ExitManager  # noqa: E402
from strategies.base import StrategySignal  # noqa: E402


def _make_signal(tp: float | None = 1.0, sl: float | None = -0.5,
                  max_hold: int | None = None) -> StrategySignal:
    return StrategySignal(
        strategy_id="TEST",
        underlying="SPY",
        side="call",
        target_dte=35,
        edge_score=0.5,
        target_otm_pct=0.05,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        max_hold_days=max_hold,
    )


def _register(em: ExitManager, sym: str, entry_premium: float = 1.00) -> None:
    em.register(
        sym, _make_signal(),
        entry_time=datetime(2024, 6, 3, 10, 0),
        expiry=date(2024, 7, 19),
        entry_premium_per_share=entry_premium,
        contracts=1,
    )


def test_tp_lock_persists_through_price_reversal():
    """TP fires at premium=2.00 (+100%). Next tick premium retraces to 1.80
    (+80%, below TP threshold). Without lock, position would un-trigger.
    With lock, exit intent persists."""
    em = ExitManager()
    _register(em, "SPY_X", entry_premium=1.00)
    today = date(2024, 6, 5)

    # Tick 1: premium spike to 2.00 = +100% PNL = TP+100% trigger
    closes = em.positions_to_close({"SPY_X": 2.00}, today)
    assert closes == [("SPY_X", "take_profit")]

    # Tick 2: premium retraces to 1.80 = +80% (below TP)
    closes = em.positions_to_close({"SPY_X": 1.80}, today)
    assert closes == [("SPY_X", "take_profit")], (
        "TP should remain locked even after retrace below TP threshold"
    )

    # Tick 3: premium drops further to 1.30 = +30%
    closes = em.positions_to_close({"SPY_X": 1.30}, today)
    assert closes == [("SPY_X", "take_profit")], (
        "TP lock must persist regardless of intervening prices"
    )


def test_sl_lock_persists_through_price_reversal():
    """SL fires at premium=0.50 (-50%). Next tick premium recovers to 0.60.
    Without lock, position would un-trigger. With lock, exit intent persists."""
    em = ExitManager()
    _register(em, "SPY_Y", entry_premium=1.00)
    today = date(2024, 6, 5)

    # Tick 1: premium drops to 0.50 = -50% PNL = SL trigger
    closes = em.positions_to_close({"SPY_Y": 0.50}, today)
    assert closes == [("SPY_Y", "stop_loss")]

    # Tick 2: premium recovers to 0.60 = -40% (above SL)
    closes = em.positions_to_close({"SPY_Y": 0.60}, today)
    assert closes == [("SPY_Y", "stop_loss")], (
        "SL should remain locked even after price recovers above SL threshold"
    )


def test_unregister_clears_triggered_exits():
    """After exit order fills, caller calls unregister — locked exit clears."""
    em = ExitManager()
    _register(em, "SPY_Z", entry_premium=1.00)
    today = date(2024, 6, 5)

    em.positions_to_close({"SPY_Z": 2.00}, today)  # TP fires + locks
    em.unregister("SPY_Z")
    closes = em.positions_to_close({"SPY_Z": 0.50}, today)
    assert closes == [], "unregister should clear triggered_exits"
    assert not em.is_tracked("SPY_Z")


def test_multiple_positions_lock_independently():
    """Two positions can be in different lock states simultaneously."""
    em = ExitManager()
    _register(em, "A", entry_premium=1.00)
    _register(em, "B", entry_premium=1.00)
    today = date(2024, 6, 5)

    # A hits TP, B doesn't
    closes = em.positions_to_close({"A": 2.00, "B": 1.20}, today)
    assert ("A", "take_profit") in closes
    assert ("B", "take_profit") not in closes
    assert ("B", "stop_loss") not in closes

    # Next tick: A retraces (lock holds), B drops to SL
    closes = em.positions_to_close({"A": 1.50, "B": 0.50}, today)
    assert ("A", "take_profit") in closes  # lock holds
    assert ("B", "stop_loss") in closes


def test_time_stop_also_locks():
    """v22: time_stop / max_hold also locks (any exit-trigger should persist)."""
    em = ExitManager()
    em.register(
        "SPY_T", _make_signal(tp=1.0, sl=-0.5, max_hold=5),
        entry_time=datetime(2024, 6, 1, 10, 0),
        expiry=date(2024, 8, 1),
        entry_premium_per_share=1.00,
        contracts=1,
    )
    # Day 6 — max_hold (5d) exceeded
    closes = em.positions_to_close({"SPY_T": 1.10}, date(2024, 6, 7))
    assert ("SPY_T", "time_stop") in closes

    # Next call — even if max_hold logic re-evaluates the same way, still locked
    closes = em.positions_to_close({"SPY_T": 1.10}, date(2024, 6, 7))
    assert closes == [("SPY_T", "time_stop")]
