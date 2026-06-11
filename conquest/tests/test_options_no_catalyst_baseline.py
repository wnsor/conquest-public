"""Unit tests for momentum_no_catalyst_baseline (TEST 1 random-entry baseline)."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from strategies.momentum_no_catalyst_baseline import MomentumNoCatalystBaseline
from strategies.momentum_otm_calls import WSB_UNIVERSE
from strategies.base import StrategyContext


def _ctx(timestamp=None, **kw):
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 16, 0)
    return StrategyContext(timestamp=timestamp, **kw)


class TestNoCatalystBaseline:
    def test_fires_on_every_universe_ticker_when_clean_regime(self):
        s = MomentumNoCatalystBaseline()
        ctx = _ctx(vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95)
        sigs = s.on_data(ctx)
        # Should fire on all universe tickers (16 names)
        underlyings = {sig.underlying for sig in sigs}
        assert underlyings == set(WSB_UNIVERSE)

    def test_no_fire_when_vix_above_25(self):
        s = MomentumNoCatalystBaseline()
        ctx = _ctx(vix=27.0, term_regime="contango")
        assert s.on_data(ctx) == []

    def test_no_fire_when_backwardation(self):
        s = MomentumNoCatalystBaseline()
        ctx = _ctx(vix=18.0, term_regime="backwardation")
        assert s.on_data(ctx) == []

    def test_no_fire_when_vix9d_acute_stress(self):
        s = MomentumNoCatalystBaseline()
        ctx = _ctx(vix=18.0, vix9d_vix_ratio=1.05, term_regime="contango")
        assert s.on_data(ctx) == []

    def test_cooldown_blocks_repeated_fires(self):
        s = MomentumNoCatalystBaseline()
        d1 = _ctx(timestamp=datetime(2026, 1, 1), vix=18.0, term_regime="contango")
        d10 = _ctx(timestamp=datetime(2026, 1, 10), vix=18.0, term_regime="contango")
        d22 = _ctx(timestamp=datetime(2026, 1, 22), vix=18.0, term_regime="contango")
        assert len(s.on_data(d1)) == len(WSB_UNIVERSE)   # all fire
        assert s.on_data(d10) == []                       # within 21d cooldown
        assert len(s.on_data(d22)) == len(WSB_UNIVERSE)   # past cooldown

    def test_signal_params_match_v16(self):
        """All signal params should mirror momentum_otm_calls v16 for clean
        comparison."""
        s = MomentumNoCatalystBaseline()
        ctx = _ctx(vix=18.0, term_regime="contango")
        sig = s.on_data(ctx)[0]
        assert sig.target_dte == 28
        assert sig.target_otm_pct == 0.15
        assert sig.take_profit_pct is None
        assert sig.stop_loss_pct == -0.4
        assert sig.max_hold_days == 21
        assert sig.side == "call"
        assert sig.max_per_trade_pct_nav == 0.08

    def test_no_signal_when_timestamp_missing(self):
        s = MomentumNoCatalystBaseline()
        assert s.on_data(_ctx(timestamp=None)) == []
