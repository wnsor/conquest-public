"""Phase 3 tests — C1 straddle, C2 strangle, max_hold_days in exit_manager."""
from __future__ import annotations

from datetime import datetime, date

from strategies.base import StrategyContext, StrategySignal
from strategies.earnings_straddle import EarningsStraddle, STRADDLE_UNIVERSE
from strategies.earnings_strangle import EarningsStrangle
from exit_manager import ExitManager


def _ctx(**overrides) -> StrategyContext:
    base = dict(
        timestamp=datetime(2026, 1, 15, 10, 0),
        underlying_prices={},
        vix=20.0,
        cstability_vote_count=0,
        iv_rank={},
        earnings_today=set(),
        earnings_within_5d=set(),
        last_earnings_surprise_pct={},
        days_since_last_earnings={},
        days_until_next_earnings={},
        underlying_momentum_30d={},
        underlying_momentum_60d={},
        uoa_active=set(),
    )
    base.update(overrides)
    return StrategyContext(**base)


class TestC1Straddle:
    def test_fires_two_legs_per_event(self):
        s = EarningsStraddle()
        ctx = _ctx(
            days_until_next_earnings={"AAPL": 2},
            iv_rank={"AAPL": 40},
            vix=20.0,
            term_regime="contango",
        )
        sigs = s.on_data(ctx)
        # 2 legs (call + put) for 1 ticker
        assert len(sigs) == 2
        sides = sorted([sig.side for sig in sigs])
        assert sides == ["call", "put"]
        assert all(sig.strategy_id == "earnings_straddle" for sig in sigs)
        assert all(sig.max_hold_days == 1 for sig in sigs)
        # ATM = delta ±0.50
        assert any(sig.target_delta == 0.5 for sig in sigs)
        assert any(sig.target_delta == -0.5 for sig in sigs)

    def test_dedup_per_event(self):
        s = EarningsStraddle()
        good = dict(days_until_next_earnings={"AAPL": 2})
        # Same event on consecutive ticks → only first fires
        sigs1 = s.on_data(_ctx(timestamp=datetime(2026, 1, 1), **good))
        sigs2 = s.on_data(_ctx(timestamp=datetime(2026, 1, 1, 11), **good))
        assert len(sigs1) == 2
        assert sigs2 == []

    def test_outside_window_blocks(self):
        s = EarningsStraddle()
        # 5 days out, 1 day out, day-of all outside [2, 3]
        for d in [5, 1, 0, 7]:
            assert s.on_data(_ctx(days_until_next_earnings={"AAPL": d})) == []

    def test_no_data_no_signal(self):
        assert EarningsStraddle().on_data(_ctx()) == []


class TestC2Strangle:
    def test_fires_two_otm_legs(self):
        s = EarningsStrangle()
        ctx = _ctx(days_until_next_earnings={"AAPL": 2})
        sigs = s.on_data(ctx)
        assert len(sigs) == 2
        # OTM = target_otm_pct=0.05, no target_delta
        assert all(sig.target_otm_pct == 0.05 for sig in sigs)
        assert all(sig.target_delta is None for sig in sigs)

    def test_different_tp_than_c1(self):
        s = EarningsStrangle()
        sigs = s.on_data(_ctx(days_until_next_earnings={"AAPL": 3}))
        assert all(sig.take_profit_pct == 1.5 for sig in sigs)
        assert all(sig.stop_loss_pct == -0.6 for sig in sigs)


class TestMaxHoldDaysExit:
    def _make_signal(self, max_hold_days=1):
        return StrategySignal(
            strategy_id="t", underlying="AAPL", side="call",
            target_dte=21, edge_score=0.5, target_delta=0.5,
            max_hold_days=max_hold_days, take_profit_pct=1.0, stop_loss_pct=-0.5,
        )

    def test_max_hold_triggers_after_n_days(self):
        em = ExitManager()
        sig = self._make_signal(max_hold_days=1)
        em.register(
            "AAPL_C_TEST", sig,
            entry_time=datetime(2026, 1, 10, 10, 0),
            expiry=date(2026, 1, 31),
            entry_premium_per_share=2.0,
            contracts=1,
        )
        # Same day → no exit
        assert em.positions_to_close({"AAPL_C_TEST": 2.0}, date(2026, 1, 10)) == []
        # 1 day later → max_hold triggers
        closes = em.positions_to_close({"AAPL_C_TEST": 2.0}, date(2026, 1, 11))
        assert ("AAPL_C_TEST", "time_stop") in closes

    def test_max_hold_takes_precedence_over_tp(self):
        em = ExitManager()
        sig = self._make_signal(max_hold_days=1)
        em.register(
            "AAPL_C_TEST", sig,
            entry_time=datetime(2026, 1, 10, 10, 0),
            expiry=date(2026, 1, 31),
            entry_premium_per_share=2.0,
            contracts=1,
        )
        # Both TP triggered (+200%) AND max_hold exceeded → max_hold fires (checked first)
        closes = em.positions_to_close({"AAPL_C_TEST": 6.0}, date(2026, 1, 11))
        assert closes[0][1] == "time_stop"

    def test_no_max_hold_means_no_calendar_exit(self):
        em = ExitManager()
        sig = self._make_signal(max_hold_days=None)
        em.register(
            "AAPL_C_TEST", sig,
            entry_time=datetime(2026, 1, 10, 10, 0),
            expiry=date(2026, 1, 31),
            entry_premium_per_share=2.0,
            contracts=1,
        )
        # 5 days later — no calendar exit, but TP hits at +200%
        closes = em.positions_to_close({"AAPL_C_TEST": 6.0}, date(2026, 1, 15))
        assert closes[0][1] == "take_profit"
