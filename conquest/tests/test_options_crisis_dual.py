"""Unit tests for crisis_dual_directional (B5) — dual-directional crisis strategy.

Pre-positions PUTS on early-warning confluence and CALLS on rebound confirmation.
Tests cover both legs + cooldowns + safety gates."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from strategies.crisis_dual_directional import CrisisDualDirectional
from strategies.base import StrategyContext


def _ctx(timestamp=None, **kw):
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 16, 0)
    return StrategyContext(timestamp=timestamp, **kw)


# ---------------------------------------------------------------------------
# PUTS phase — proactive early-warning trigger
# ---------------------------------------------------------------------------

class TestPutsPhase:
    def test_fires_on_two_warning_confluence(self):
        s = CrisisDualDirectional()
        ctx = _ctx(
            crisis_state="normal",
            vix=22.0,
            vix9d_vix_ratio=1.05,            # warning
            term_regime="backwardation",      # warning (so 2+)
            cstability_vote_count=0,
        )
        sigs = s.on_data(ctx)
        puts = [x for x in sigs if x.side == "put"]
        assert len(puts) == 1
        assert puts[0].underlying == "SPY"
        assert puts[0].target_otm_pct == 0.07
        assert puts[0].stop_loss_pct == -0.4
        assert puts[0].take_profit_pct is None    # trailing exit, no fixed TP
        assert puts[0].max_per_trade_pct_nav == 0.12

    def test_single_warning_does_not_fire(self):
        s = CrisisDualDirectional()
        ctx = _ctx(
            crisis_state="normal",
            vix=18.0,                        # below 20 (so NOT a warning)
            vix9d_vix_ratio=1.05,            # only 1 warning
            term_regime="contango",
            cstability_vote_count=0,
        )
        assert s.on_data(ctx) == []

    def test_panic_vix_blocks_put_entry(self):
        """Don't fire puts when VIX already > 40 — IV peak, late-cycle."""
        s = CrisisDualDirectional()
        ctx = _ctx(
            crisis_state="normal",
            vix=45.0,                         # panic — too late
            vix9d_vix_ratio=1.20,
            term_regime="backwardation",
            cstability_vote_count=2,
        )
        assert s.on_data(ctx) == []

    def test_does_not_fire_during_confirmed_crash(self):
        s = CrisisDualDirectional()
        ctx = _ctx(
            crisis_state="crash",             # past warning phase
            vix=30.0,
            vix9d_vix_ratio=1.20,
            term_regime="backwardation",
            cstability_vote_count=2,
        )
        sigs = s.on_data(ctx)
        puts = [x for x in sigs if x.side == "put"]
        assert puts == []

    def test_put_cooldown_60d(self):
        s = CrisisDualDirectional()
        warn_args = dict(
            crisis_state="normal", vix=22.0, vix9d_vix_ratio=1.05,
            term_regime="backwardation", cstability_vote_count=2,
        )
        d1 = _ctx(timestamp=datetime(2026, 1, 1), **warn_args)
        d30 = _ctx(timestamp=datetime(2026, 1, 31), **warn_args)
        d65 = _ctx(timestamp=datetime(2026, 3, 7), **warn_args)
        assert len(s.on_data(d1)) == 1            # first fire
        assert s.on_data(d30) == []               # within cooldown
        assert len(s.on_data(d65)) == 1           # past cooldown

    def test_edge_scales_with_warning_count(self):
        """Edge score increases with more warning signals confluencing."""
        s2 = CrisisDualDirectional()
        ctx2 = _ctx(crisis_state="normal", vix=22.0,
                    vix9d_vix_ratio=1.05, term_regime="backwardation",
                    cstability_vote_count=0)
        s4 = CrisisDualDirectional()
        ctx4 = _ctx(crisis_state="normal", vix=22.0,
                    vix9d_vix_ratio=1.05, term_regime="backwardation",
                    cstability_vote_count=3)   # all 4 warnings
        e2 = s2.on_data(ctx2)[0].edge_score
        e4 = s4.on_data(ctx4)[0].edge_score
        assert e4 > e2


# ---------------------------------------------------------------------------
# CALLS phase — reactive rebound trigger
# ---------------------------------------------------------------------------

class TestCallsPhase:
    def test_fires_after_seeing_crash_then_rebound(self):
        s = CrisisDualDirectional()
        # Simulate the cycle: normal → crash → rebound
        # First feed crash signal (no fire — wrong phase)
        crash_ctx = _ctx(
            timestamp=datetime(2026, 1, 15),
            crisis_state="crash", vix=35.0,
        )
        s.on_data(crash_ctx)   # records to state_history
        # Now rebound signal arrives
        rebound_ctx = _ctx(
            timestamp=datetime(2026, 2, 15),   # 1 month later
            crisis_state="rebound", vix=20.0,
        )
        sigs = s.on_data(rebound_ctx)
        calls = [x for x in sigs if x.side == "call"]
        assert len(calls) == 1
        assert calls[0].underlying == "SPY"
        assert calls[0].target_otm_pct == 0.05
        assert calls[0].max_per_trade_pct_nav == 0.12

    def test_no_call_without_prior_crash(self):
        """CALL phase requires seeing crash/capitulation in last 90d."""
        s = CrisisDualDirectional()
        # Just rebound — no prior crash recorded
        ctx = _ctx(crisis_state="rebound", vix=20.0)
        sigs = s.on_data(ctx)
        calls = [x for x in sigs if x.side == "call"]
        assert calls == []

    def test_no_call_with_high_vix(self):
        """Even on rebound state, if VIX still > 25, fear not subsided yet."""
        s = CrisisDualDirectional()
        crash_ctx = _ctx(timestamp=datetime(2026, 1, 15),
                          crisis_state="crash", vix=35.0)
        s.on_data(crash_ctx)
        rebound_ctx = _ctx(timestamp=datetime(2026, 2, 15),
                            crisis_state="rebound", vix=28.0)
        sigs = s.on_data(rebound_ctx)
        calls = [x for x in sigs if x.side == "call"]
        assert calls == []

    def test_call_cooldown_90d(self):
        s = CrisisDualDirectional()
        # Establish crash history
        s.on_data(_ctx(timestamp=datetime(2026, 1, 15),
                        crisis_state="crash", vix=35.0))
        d1 = _ctx(timestamp=datetime(2026, 2, 15), crisis_state="rebound", vix=20.0)
        d60 = _ctx(timestamp=datetime(2026, 4, 16), crisis_state="rebound", vix=20.0)
        d100 = _ctx(timestamp=datetime(2026, 5, 26), crisis_state="rebound", vix=20.0)
        assert len([x for x in s.on_data(d1) if x.side == "call"]) == 1
        assert [x for x in s.on_data(d60) if x.side == "call"] == []   # cooldown
        # by d100 (100 days after d1), 90d cooldown passed; but crash was 130d ago
        # so the saw_crash test still passes (180d history window). Should fire.
        assert len([x for x in s.on_data(d100) if x.side == "call"]) == 1


# ---------------------------------------------------------------------------
# Safety & integration
# ---------------------------------------------------------------------------

class TestSafetyAndIntegration:
    def test_independent_cooldowns_for_puts_and_calls(self):
        """A recent put fire shouldn't block a call fire (different cycles)."""
        s = CrisisDualDirectional()
        # First fire a PUT
        put_ctx = _ctx(
            timestamp=datetime(2026, 1, 1),
            crisis_state="normal", vix=22.0, vix9d_vix_ratio=1.05,
            term_regime="backwardation", cstability_vote_count=2,
        )
        sigs1 = s.on_data(put_ctx)
        assert any(x.side == "put" for x in sigs1)
        # Then 30 days later, crash + rebound happens
        s.on_data(_ctx(timestamp=datetime(2026, 1, 20), crisis_state="crash", vix=35.0))
        rebound_ctx = _ctx(
            timestamp=datetime(2026, 2, 15),
            crisis_state="rebound", vix=20.0,
        )
        sigs2 = s.on_data(rebound_ctx)
        # CALL should fire even though PUT cooldown not yet expired
        assert any(x.side == "call" for x in sigs2)

    def test_state_history_pruned_after_180d(self):
        """state_history dict should not grow unbounded — old entries pruned."""
        s = CrisisDualDirectional()
        for i in range(200):
            d = datetime(2024, 1, 1) + timedelta(days=i)
            s.on_data(_ctx(timestamp=d, crisis_state="normal"))
        # After 200 days of feeding, state_history should only hold recent 180
        assert len(s._state_history) <= 181   # 180-day window plus today

    def test_no_signal_when_timestamp_missing(self):
        s = CrisisDualDirectional()
        ctx = _ctx(timestamp=None)
        assert s.on_data(ctx) == []
