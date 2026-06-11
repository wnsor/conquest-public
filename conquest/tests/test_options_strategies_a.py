"""Unit tests for Phase 2 Category A strategies (A1, A2, A5, A8).

Tests exercise each strategy's on_data signal-emission logic with synthetic
StrategyContext objects. No Lean dependency — pure Python.
"""
from __future__ import annotations

from datetime import datetime, date

from strategies.base import StrategyContext
from strategies.momentum_otm_calls import MomentumOtmCalls, WSB_UNIVERSE
from strategies.spy_atm_calls import SpyAtmCalls
from strategies.pead_megacap import PeadMegacap, MEGACAP_UNIVERSE as PEAD_UNIVERSE
from strategies.uoa_following_calls import UoaFollowingCalls, UOA_UNIVERSE


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
        underlying_momentum_30d={},
        underlying_momentum_60d={},
        uoa_active=set(),
    )
    base.update(overrides)
    return StrategyContext(**base)


# ---------------------------------------------------------------------------
# A2 — SPY ATM call
# ---------------------------------------------------------------------------

class TestA2SpyAtmCall:
    def test_all_gates_pass_emits_signal(self):
        s = SpyAtmCalls()
        ctx = _ctx(
            vix=20.0,
            iv_rank={"SPY": 25.0},
            underlying_momentum_60d={"SPY": 1.10},
            underlying_momentum_30d={"SPY": 1.05},
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == 1
        assert sigs[0].underlying == "SPY"
        assert sigs[0].side == "call"
        assert sigs[0].target_dte == 30
        assert sigs[0].target_delta == 0.50
        assert sigs[0].edge_score > 0.5  # all confluences hit

    def test_vix_too_high_blocks(self):
        s = SpyAtmCalls()
        ctx = _ctx(vix=30.0, iv_rank={"SPY": 25}, underlying_momentum_60d={"SPY": 1.1})
        assert s.on_data(ctx) == []

    def test_momentum_negative_blocks(self):
        s = SpyAtmCalls()
        ctx = _ctx(vix=20.0, iv_rank={"SPY": 25}, underlying_momentum_60d={"SPY": 0.95})
        assert s.on_data(ctx) == []

    def test_iv_rank_too_high_blocks(self):
        s = SpyAtmCalls()
        ctx = _ctx(vix=20.0, iv_rank={"SPY": 45}, underlying_momentum_60d={"SPY": 1.1})
        assert s.on_data(ctx) == []

    def test_missing_context_data_blocks(self):
        s = SpyAtmCalls()
        assert s.on_data(_ctx(vix=None)) == []
        assert s.on_data(_ctx(underlying_momentum_60d={})) == []
        assert s.on_data(_ctx(iv_rank={})) == []

    def test_monthly_cadence_dedup(self):
        s = SpyAtmCalls()
        good = dict(vix=20.0, iv_rank={"SPY": 25}, underlying_momentum_60d={"SPY": 1.1})
        # Two ticks same month → only first emits
        assert len(s.on_data(_ctx(timestamp=datetime(2026, 1, 1), **good))) == 1
        assert s.on_data(_ctx(timestamp=datetime(2026, 1, 15), **good)) == []
        # New month → re-emits
        assert len(s.on_data(_ctx(timestamp=datetime(2026, 2, 1), **good))) == 1


# ---------------------------------------------------------------------------
# A1 — WSB OTM call
# ---------------------------------------------------------------------------

class TestA1WsbOtmCall:
    def test_emits_per_qualified_ticker(self):
        s = MomentumOtmCalls()
        ctx = _ctx(
            vix=20.0,
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: 1.08 for t in WSB_UNIVERSE},
            underlying_momentum_60d={t: 1.05 for t in WSB_UNIVERSE},
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == len(WSB_UNIVERSE)
        # v4: 12% OTM, +250% TP (was 10% / +200%)
        assert all(sig.target_otm_pct == 0.12 for sig in sigs)
        assert all(sig.take_profit_pct == 2.5 for sig in sigs)

    def test_per_ticker_cooldown(self):
        s = MomentumOtmCalls()
        good = dict(
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: 1.08 for t in WSB_UNIVERSE},
        )
        # Day 1: all fire
        sigs1 = s.on_data(_ctx(timestamp=datetime(2026, 1, 1), **good))
        assert len(sigs1) == len(WSB_UNIVERSE)
        # Day 5: cooldown still active
        assert s.on_data(_ctx(timestamp=datetime(2026, 1, 5), **good)) == []
        # Day 32: cooldown expired
        sigs2 = s.on_data(_ctx(timestamp=datetime(2026, 2, 5), **good))
        assert len(sigs2) == len(WSB_UNIVERSE)

    def test_vix_gate_blocks_universe(self):
        s = MomentumOtmCalls()
        ctx = _ctx(
            vix=28.0,
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: 1.08 for t in WSB_UNIVERSE},
        )
        assert s.on_data(ctx) == []

    def test_momentum_filter_per_ticker(self):
        s = MomentumOtmCalls()
        ctx = _ctx(
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: (1.10 if t in ("CRDO", "MU") else 0.99)
                                     for t in WSB_UNIVERSE},
        )
        sigs = s.on_data(ctx)
        ids = {sig.underlying for sig in sigs}
        assert ids == {"CRDO", "MU"}


# ---------------------------------------------------------------------------
# A5 — PEAD call
# ---------------------------------------------------------------------------

class TestA5PeadCall:
    def test_emits_within_window(self):
        s = PeadMegacap()
        ctx = _ctx(
            days_since_last_earnings={"AAPL": 2},
            last_earnings_surprise_pct={"AAPL": 10.0},
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == 1
        assert sigs[0].underlying == "AAPL"
        assert sigs[0].target_dte == 21
        assert sigs[0].target_otm_pct == 0.05

    def test_outside_window_blocks(self):
        s = PeadMegacap()
        # day 0 (= today) and day 4 both outside [1, 3]
        ctx = _ctx(
            days_since_last_earnings={"AAPL": 0, "MSFT": 4},
            last_earnings_surprise_pct={"AAPL": 10.0, "MSFT": 5.0},
        )
        assert s.on_data(ctx) == []

    def test_negative_surprise_blocks(self):
        s = PeadMegacap()
        ctx = _ctx(
            days_since_last_earnings={"AAPL": 1},
            last_earnings_surprise_pct={"AAPL": -3.0},
        )
        assert s.on_data(ctx) == []

    def test_dedup_same_event(self):
        s = PeadMegacap()
        # Same event on consecutive ticks → only fires once per (ticker, ds, surprise) key
        ctx_args = dict(
            days_since_last_earnings={"AAPL": 2},
            last_earnings_surprise_pct={"AAPL": 10.0},
        )
        sigs1 = s.on_data(_ctx(timestamp=datetime(2026, 1, 5), **ctx_args))
        sigs2 = s.on_data(_ctx(timestamp=datetime(2026, 1, 5, 11), **ctx_args))
        assert len(sigs1) == 1
        assert sigs2 == []

    def test_edge_score_scales_with_surprise(self):
        s = PeadMegacap()
        # Surprise 5% → edge ~0.25 → clamped to floor 0.3
        ctx_low = _ctx(
            days_since_last_earnings={"AAPL": 1},
            last_earnings_surprise_pct={"AAPL": 5.0},
        )
        sigs_low = s.on_data(ctx_low)
        # Fresh strategy for high-surprise test
        s2 = PeadMegacap()
        ctx_hi = _ctx(
            days_since_last_earnings={"AAPL": 1},
            last_earnings_surprise_pct={"AAPL": 25.0},
        )
        sigs_hi = s2.on_data(ctx_hi)
        assert sigs_low[0].edge_score < sigs_hi[0].edge_score
        assert sigs_hi[0].edge_score == 1.0


# ---------------------------------------------------------------------------
# A8 — UOA following call
# ---------------------------------------------------------------------------

class TestA8UoaCall:
    def test_emits_only_for_uoa_active(self):
        s = UoaFollowingCalls()
        ctx = _ctx(
            uoa_active={"AAPL", "MSFT"},
            vix=20.0,
            underlying_momentum_30d={"AAPL": 1.05, "MSFT": 1.10},
            iv_rank={"AAPL": 35.0, "MSFT": 25.0},
        )
        sigs = s.on_data(ctx)
        underlyings = {sig.underlying for sig in sigs}
        assert underlyings == {"AAPL", "MSFT"}
        for sig in sigs:
            assert sig.target_dte == 21
            assert sig.target_otm_pct == 0.05

    def test_no_uoa_no_signal(self):
        s = UoaFollowingCalls()
        ctx = _ctx(uoa_active=set(), vix=20.0)
        assert s.on_data(ctx) == []

    def test_per_ticker_cooldown_10d(self):
        s = UoaFollowingCalls()
        ctx_args = dict(uoa_active={"NVDA"})
        d1 = _ctx(timestamp=datetime(2026, 1, 1), **ctx_args)
        d5 = _ctx(timestamp=datetime(2026, 1, 5), **ctx_args)
        d15 = _ctx(timestamp=datetime(2026, 1, 15), **ctx_args)
        assert len(s.on_data(d1)) == 1
        assert s.on_data(d5) == []
        assert len(s.on_data(d15)) == 1

    def test_confluence_lifts_edge_score(self):
        s = UoaFollowingCalls()
        # 2026-05-25: denominator narrowed 7→6 after dropping lagging mom30
        # confluence (UOA is a leading-indicator strategy).
        # Only UOA → edge=1/6
        bare = _ctx(uoa_active={"NVDA"}, vix=None,
                    underlying_momentum_30d={}, iv_rank={})
        sigs = s.on_data(bare)
        assert abs(sigs[0].edge_score - 1.0 / 6.0) < 0.01
        # All 6 confluences hit → edge=1.0
        s2 = UoaFollowingCalls()
        full = _ctx(
            uoa_active={"NVDA"},
            vix=18.0,
            term_regime="contango",
            iv_rank={"NVDA": 35.0},
            skew_z={"NVDA": 1.5},
            gex_regime="long_gamma",
        )
        sigs_full = s2.on_data(full)
        assert sigs_full[0].edge_score == 1.0
