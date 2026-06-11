"""Unit tests for the dynamic PIT-momentum OTM-calls path (survivorship #1 kill).

Two halves, both fully offline (no QC / network / Object Store):
  1. DynamicPitMomentumCalls — ctx → signals: regime gating, cooldown, params,
     and that it ranges over ctx.active_universe (NOT a hard-coded universe).
  2. plan_rotation — the monthly rotation diff, especially the money-losing QC
     gotcha: a name we HOLD is drained (chain kept) never removed (force-liquidated).
"""
from __future__ import annotations

from datetime import date, datetime

from strategies.base import StrategyContext
from strategies.dynamic_pit_momentum_calls import DynamicPitMomentumCalls
from dyn_rotation import plan_rotation


def _ctx(timestamp=None, active_universe=None, **kw):
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 16, 0)
    if active_universe is None:
        active_universe = ["AAPL", "MSFT"]
    return StrategyContext(timestamp=timestamp, active_universe=active_universe, **kw)


# ── DynamicPitMomentumCalls: ctx → signals ───────────────────────────────────

class TestDynamicPitMomentumCalls:
    def test_fires_on_each_active_universe_name_when_clean_regime(self):
        s = DynamicPitMomentumCalls()
        ctx = _ctx(vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
                   active_universe=["AAPL", "MSFT", "NVDA"])
        sigs = s.on_data(ctx)
        assert {sig.underlying for sig in sigs} == {"AAPL", "MSFT", "NVDA"}

    def test_ranges_over_active_universe_not_static_list(self):
        """Different active_universe → different fires (proves dynamic, not WSB)."""
        s = DynamicPitMomentumCalls()
        ctx = _ctx(vix=18.0, term_regime="contango", active_universe=["XYZ"])
        sigs = s.on_data(ctx)
        assert [sig.underlying for sig in sigs] == ["XYZ"]

    def test_no_fire_when_active_universe_empty(self):
        s = DynamicPitMomentumCalls()
        ctx = _ctx(vix=18.0, term_regime="contango", active_universe=[])
        assert s.on_data(ctx) == []

    def test_no_fire_when_vix_above_25(self):
        s = DynamicPitMomentumCalls()
        assert s.on_data(_ctx(vix=27.0, term_regime="contango")) == []

    def test_no_fire_when_vix_missing(self):
        s = DynamicPitMomentumCalls()
        assert s.on_data(_ctx(vix=None, term_regime="contango")) == []

    def test_no_fire_when_backwardation(self):
        s = DynamicPitMomentumCalls()
        assert s.on_data(_ctx(vix=18.0, term_regime="backwardation")) == []

    def test_no_fire_when_vix9d_acute_stress(self):
        s = DynamicPitMomentumCalls()
        assert s.on_data(_ctx(vix=18.0, vix9d_vix_ratio=1.2, term_regime="contango")) == []

    def test_no_signal_when_timestamp_missing(self):
        s = DynamicPitMomentumCalls()
        assert s.on_data(_ctx(timestamp=None)) == []

    def test_cooldown_keys_on_ctx_last_entry_not_emit(self):
        """Cooldown reads ctx.last_entry_date (entry-based), so a failed emit never
        burns the cycle. Entry 5d ago → suppressed; 22d ago → eligible again."""
        s = DynamicPitMomentumCalls()
        no_entry = _ctx(timestamp=datetime(2026, 1, 22), vix=18.0, term_regime="contango",
                        active_universe=["AAPL"], last_entry_date={})
        recent = _ctx(timestamp=datetime(2026, 1, 22), vix=18.0, term_regime="contango",
                      active_universe=["AAPL"], last_entry_date={"AAPL": date(2026, 1, 17)})
        stale = _ctx(timestamp=datetime(2026, 1, 22), vix=18.0, term_regime="contango",
                     active_universe=["AAPL"], last_entry_date={"AAPL": date(2025, 12, 30)})
        assert len(s.on_data(no_entry)) == 1   # never entered → fires
        assert s.on_data(recent) == []          # entered 5d ago → within 21d
        assert len(s.on_data(stale)) == 1       # entered 23d ago → past cooldown

    def test_no_internal_self_throttle(self):
        """The strategy keeps no emit memory — repeated calls with no recorded entry
        BOTH fire (entry tracking is external, in main.py). This is the throttle fix."""
        s = DynamicPitMomentumCalls()
        c = _ctx(vix=18.0, term_regime="contango", active_universe=["AAPL"], last_entry_date={})
        assert len(s.on_data(c)) == 1
        assert len(s.on_data(c)) == 1

    def test_signal_params_mirror_momentum_otm_v16(self):
        s = DynamicPitMomentumCalls()
        sig = s.on_data(_ctx(vix=18.0, term_regime="contango", active_universe=["AAPL"]))[0]
        assert sig.strategy_id == "dynamic_pit_momentum_calls"
        assert sig.side == "call"
        assert sig.target_dte == 28
        assert sig.target_otm_pct == 0.15
        assert sig.take_profit_pct is None
        assert sig.stop_loss_pct == -0.4
        assert sig.max_hold_days == 21
        assert sig.max_per_trade_pct_nav == 0.08
        assert sig.edge_score == 0.6

    def test_declares_empty_static_universe(self):
        """No hand-picked names baked in — the whole point of the refactor."""
        assert DynamicPitMomentumCalls().universe == []


# ── plan_rotation: monthly diff (the QC drain-not-liquidate invariant) ────────

class TestPlanRotation:
    def test_entrant_gets_chain_added(self):
        plan = plan_rotation(
            current_active={"AAPL"},
            draining=set(),
            new_set={"AAPL", "MSFT"},
            subscribed={"AAPL"},
            has_position=lambda t: False,
        )
        assert plan.entrants == ["MSFT"]
        assert plan.add_chain == ["MSFT"]
        assert plan.new_active == {"AAPL", "MSFT"}
        assert plan.remove == []

    def test_held_leaver_drains_never_removed(self):
        """A leaver we hold MUST drain (chain kept for ExitManager), not remove."""
        plan = plan_rotation(
            current_active={"AAPL", "TSLA"},
            draining=set(),
            new_set={"AAPL"},
            subscribed={"AAPL", "TSLA"},
            has_position=lambda t: t == "TSLA",
        )
        assert plan.drain == ["TSLA"]
        assert "TSLA" in plan.new_draining
        assert "TSLA" not in plan.remove          # the invariant
        assert "TSLA" not in plan.new_active

    def test_flat_leaver_removed(self):
        plan = plan_rotation(
            current_active={"AAPL", "TSLA"},
            draining=set(),
            new_set={"AAPL"},
            subscribed={"AAPL", "TSLA"},
            has_position=lambda t: False,
        )
        assert plan.remove == ["TSLA"]
        assert "TSLA" not in plan.new_draining
        assert plan.drain == []

    def test_drainer_now_flat_is_removed(self):
        plan = plan_rotation(
            current_active={"AAPL"},
            draining={"TSLA"},
            new_set={"AAPL"},
            subscribed={"AAPL", "TSLA"},
            has_position=lambda t: False,
        )
        assert plan.remove == ["TSLA"]
        assert "TSLA" not in plan.new_draining

    def test_drainer_still_held_stays_draining(self):
        plan = plan_rotation(
            current_active={"AAPL"},
            draining={"TSLA"},
            new_set={"AAPL"},
            subscribed={"AAPL", "TSLA"},
            has_position=lambda t: t == "TSLA",
        )
        assert plan.remove == []
        assert plan.new_draining == {"TSLA"}

    def test_draining_name_reentering_topn_reactivates_without_readd(self):
        """Held name that left then momentum-returns: back to active, off draining,
        and NOT re-added (its chain is still subscribed)."""
        plan = plan_rotation(
            current_active={"AAPL"},
            draining={"TSLA"},
            new_set={"AAPL", "TSLA"},
            subscribed={"AAPL", "TSLA"},
            has_position=lambda t: t == "TSLA",
        )
        assert "TSLA" in plan.new_active
        assert "TSLA" not in plan.new_draining
        assert plan.add_chain == []     # already subscribed → no re-add

    def test_noop_when_topn_unchanged(self):
        plan = plan_rotation(
            current_active={"AAPL", "MSFT"},
            draining=set(),
            new_set={"AAPL", "MSFT"},
            subscribed={"AAPL", "MSFT"},
            has_position=lambda t: False,
        )
        assert plan.entrants == []
        assert plan.add_chain == []
        assert plan.drain == []
        assert plan.remove == []
        assert plan.new_active == {"AAPL", "MSFT"}
