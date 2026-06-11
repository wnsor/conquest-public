"""Phase 4 tests — A_GEX_SPY, D1 LEAPS, D2 Tepper V-bottom."""
from __future__ import annotations

from datetime import datetime

from strategies.base import StrategyContext, StrategySignal
from strategies.gex_spy_selective import GexSpySelective
from strategies.cgrowth_leaps import CgrowthLeaps, CGROWTH_TOP5
from strategies.tepper_vbottom_leaps import TepperVbottomLeaps


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
        underlying_5ma_above_20ma={},
        underlying_drawdown_from_252d_high={},
        uoa_active=set(),
    )
    base.update(overrides)
    return StrategyContext(**base)


class TestA_GEX_SPY:
    def test_fires_on_full_setup(self):
        s = GexSpySelective()
        # v12: need 5+ of 8 confluences to fire (edge_score >= 0.6)
        sigs = s.on_data(_ctx(
            gex_regime="short_gamma",
            underlying_5ma_above_20ma={"SPY": True},
            vix=20.0,
            term_regime="contango",
            underlying_momentum_30d={"SPY": 1.05},   # +confluence
            cstability_vote_count=0,                  # +confluence
            iv_rank={"SPY": 30},                       # +confluence
        ))
        assert len(sigs) == 1
        assert sigs[0].underlying == "SPY"
        # v12: 7% OTM, TP+300%, 20d hold
        assert sigs[0].target_otm_pct == 0.07
        assert sigs[0].max_hold_days == 20
        assert sigs[0].take_profit_pct == 3.0

    def test_blocks_on_long_gamma(self):
        s = GexSpySelective()
        sigs = s.on_data(_ctx(
            gex_regime="long_gamma",
            underlying_5ma_above_20ma={"SPY": True},
            vix=20.0,
        ))
        assert sigs == []

    def test_blocks_no_ma_cross(self):
        s = GexSpySelective()
        sigs = s.on_data(_ctx(
            gex_regime="short_gamma",
            underlying_5ma_above_20ma={"SPY": False},
            vix=20.0,
        ))
        assert sigs == []

    def test_blocks_high_vix(self):
        s = GexSpySelective()
        sigs = s.on_data(_ctx(
            gex_regime="short_gamma",
            underlying_5ma_above_20ma={"SPY": True},
            vix=27.0,
        ))
        assert sigs == []

    def test_cooldown(self):
        s = GexSpySelective()
        good = dict(
            gex_regime="short_gamma",
            underlying_5ma_above_20ma={"SPY": True},
            vix=20.0,
            term_regime="contango",
            # enough confluences to clear v12's edge >= 0.6 gate
            underlying_momentum_30d={"SPY": 1.05},
            cstability_vote_count=0,
            iv_rank={"SPY": 30},
        )
        sigs1 = s.on_data(_ctx(timestamp=datetime(2026, 1, 1), **good))
        sigs2 = s.on_data(_ctx(timestamp=datetime(2026, 1, 15), **good))   # within 20d (v12 cooldown)
        sigs3 = s.on_data(_ctx(timestamp=datetime(2026, 1, 25), **good))   # past 20d
        assert len(sigs1) == 1
        assert sigs2 == []
        assert len(sigs3) == 1


class TestD1_CgrowthLeaps:
    def test_fires_on_qualified_names(self):
        s = CgrowthLeaps()
        ctx = _ctx(
            underlying_momentum_60d={t: 1.10 for t in CGROWTH_TOP5},
            underlying_5ma_above_20ma={t: True for t in CGROWTH_TOP5},
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == len(CGROWTH_TOP5)
        for sig in sigs:
            assert sig.side == "leaps_call"
            assert sig.target_dte == 365
            assert sig.target_delta == 0.80

    def test_per_ticker_cooldown_180d(self):
        s = CgrowthLeaps()
        good = dict(
            underlying_momentum_60d={t: 1.10 for t in CGROWTH_TOP5},
            underlying_5ma_above_20ma={t: True for t in CGROWTH_TOP5},
        )
        sigs1 = s.on_data(_ctx(timestamp=datetime(2026, 1, 1), **good))
        sigs2 = s.on_data(_ctx(timestamp=datetime(2026, 3, 1), **good))
        sigs3 = s.on_data(_ctx(timestamp=datetime(2026, 7, 5), **good))
        assert len(sigs1) == 5
        assert sigs2 == []
        assert len(sigs3) == 5

    def test_blocks_on_high_vix(self):
        s = CgrowthLeaps()
        ctx = _ctx(vix=28.0,
                   underlying_momentum_60d={t: 1.10 for t in CGROWTH_TOP5},
                   underlying_5ma_above_20ma={t: True for t in CGROWTH_TOP5})
        assert s.on_data(ctx) == []


class TestD2_TepperVbottom:
    def test_fires_on_vbottom(self):
        s = TepperVbottomLeaps()
        ctx = _ctx(
            underlying_drawdown_from_252d_high={"SPY": 0.18},   # 18% drawdown
            underlying_5ma_above_20ma={"SPY": True},
            vix=28.0,
            term_regime="backwardation",
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == 1
        assert sigs[0].underlying == "SPY"
        assert sigs[0].side == "leaps_call"
        assert sigs[0].target_dte == 180
        assert sigs[0].target_delta == 0.50
        assert sigs[0].edge_score == 1.0   # all 5 confluences hit

    def test_no_drawdown_blocks(self):
        s = TepperVbottomLeaps()
        ctx = _ctx(
            underlying_drawdown_from_252d_high={"SPY": 0.05},   # only 5% — too shallow
            underlying_5ma_above_20ma={"SPY": True},
        )
        assert s.on_data(ctx) == []

    def test_no_recovery_blocks(self):
        s = TepperVbottomLeaps()
        ctx = _ctx(
            underlying_drawdown_from_252d_high={"SPY": 0.20},
            underlying_5ma_above_20ma={"SPY": False},   # still down-trending
        )
        assert s.on_data(ctx) == []

    def test_quarterly_cooldown(self):
        s = TepperVbottomLeaps()
        good = dict(
            underlying_drawdown_from_252d_high={"SPY": 0.18},
            underlying_5ma_above_20ma={"SPY": True},
        )
        s1 = s.on_data(_ctx(timestamp=datetime(2026, 1, 1), **good))
        s2 = s.on_data(_ctx(timestamp=datetime(2026, 2, 15), **good))   # within 90d
        s3 = s.on_data(_ctx(timestamp=datetime(2026, 4, 5), **good))    # past 90d
        assert len(s1) == 1
        assert s2 == []
        assert len(s3) == 1
