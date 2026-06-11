"""Unit tests for v_REFLEX_v2 — leading-only reflexivity detector."""
from __future__ import annotations

from datetime import datetime, timedelta

from strategies.reflex_ignition_v2 import ReflexIgnitionV2
from strategies.momentum_otm_calls import WSB_UNIVERSE
from strategies.base import StrategyContext


def _ctx_full_trigger(ticker="MSTR", timestamp=None):
    """A context where ALL 3 leading + both confirmations fire."""
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 16, 0)
    return StrategyContext(
        timestamp=timestamp,
        vix=18.0,
        term_regime="contango",
        vix9d_vix_ratio=0.95,
        short_interest_velocity={ticker: 0.30},       # L1: +30% WoW
        insider_count_5d={ticker: 4},                  # L2: 4 distinct insiders
        news_propagation_5d={ticker: 2.00},            # L3: 2× attention
        volume_spike={ticker: 5.0},                    # C1: 5× volume
        implied_move_vs_realized={ticker: 3.0},        # C2: IM 3× realized
        skew_z={ticker: -0.3},                         # regime OK
    )


class TestReflexIgnitionV2:
    def test_fires_on_3_leading_plus_2_confirm(self):
        s = ReflexIgnitionV2()
        sigs = s.on_data(_ctx_full_trigger("MSTR"))
        mstr = [sig for sig in sigs if sig.underlying == "MSTR"]
        assert len(mstr) == 1
        assert mstr[0].edge_score == 1.0   # max edge
        assert mstr[0].side == "call"
        assert mstr[0].target_dte == 45
        assert mstr[0].target_otm_pct == 0.20
        assert mstr[0].stop_loss_pct == -0.50
        assert "REFLEX_v2 L=3/3 (si=True,ins=True,news=True)" in mstr[0].notes

    def test_fires_on_2_leading_plus_1_confirm_min(self):
        """Minimum threshold: 2 of 3 leading + 1 of 2 confirm."""
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            short_interest_velocity={"MSTR": 0.25},   # L1: yes
            insider_count_5d={"MSTR": 3},              # L2: yes
            news_propagation_5d={"MSTR": 1.0},         # L3: no (1.0 < 1.5)
            volume_spike={"MSTR": 5.0},                # C1: yes
            implied_move_vs_realized={"MSTR": 1.0},    # C2: no
            skew_z={"MSTR": -0.3},
        )
        sigs = s.on_data(ctx)
        mstr = [s for s in sigs if s.underlying == "MSTR"]
        assert len(mstr) == 1
        # 2/3 lead * (1 + 1/2 confirm) = 0.667 * 1.5 = 1.0
        assert mstr[0].edge_score == 1.0

    def test_no_fire_on_1_leading(self):
        """1 of 3 leading is not enough — must be 2+."""
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            short_interest_velocity={"MSTR": 0.25},   # L1: yes
            insider_count_5d={"MSTR": 1},              # L2: no
            news_propagation_5d={"MSTR": 1.0},         # L3: no
            volume_spike={"MSTR": 5.0},
            implied_move_vs_realized={"MSTR": 5.0},
            skew_z={"MSTR": -0.3},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_on_2_leading_zero_confirm(self):
        """Need at least 1 confirmation alongside leading triggers."""
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            short_interest_velocity={"MSTR": 0.25},   # L1: yes
            insider_count_5d={"MSTR": 3},              # L2: yes
            news_propagation_5d={"MSTR": 1.0},         # L3: no
            volume_spike={"MSTR": 1.0},                # C1: no
            implied_move_vs_realized={"MSTR": 1.0},    # C2: no
            skew_z={"MSTR": -0.3},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_data_missing(self):
        """Empty context (data not yet backfilled) should not fire."""
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            # All leading-signal dicts empty — simulates fresh deploy before
            # FINRA/Form4/GDELT have been backfilled
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_vix_too_high(self):
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=32.0,   # > 30
            term_regime="contango",
            short_interest_velocity={"MSTR": 0.30},
            insider_count_5d={"MSTR": 4},
            news_propagation_5d={"MSTR": 2.0},
            volume_spike={"MSTR": 5.0},
            implied_move_vs_realized={"MSTR": 3.0},
            skew_z={"MSTR": -0.3},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_on_backwardation(self):
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            term_regime="backwardation",
            short_interest_velocity={"MSTR": 0.30},
            insider_count_5d={"MSTR": 4},
            news_propagation_5d={"MSTR": 2.0},
            volume_spike={"MSTR": 5.0},
            implied_move_vs_realized={"MSTR": 3.0},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_on_acute_stress(self):
        s = ReflexIgnitionV2()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            vix9d_vix_ratio=1.10,   # > 1.05
            short_interest_velocity={"MSTR": 0.30},
            insider_count_5d={"MSTR": 4},
            news_propagation_5d={"MSTR": 2.0},
            volume_spike={"MSTR": 5.0},
            implied_move_vs_realized={"MSTR": 3.0},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_skew_panicking(self):
        """skew_z > 0.5 means puts-side panic — kills signal."""
        s = ReflexIgnitionV2()
        ctx = _ctx_full_trigger("MSTR")
        ctx_panic = StrategyContext(
            timestamp=ctx.timestamp,
            vix=ctx.vix, term_regime=ctx.term_regime,
            vix9d_vix_ratio=ctx.vix9d_vix_ratio,
            short_interest_velocity=ctx.short_interest_velocity,
            insider_count_5d=ctx.insider_count_5d,
            news_propagation_5d=ctx.news_propagation_5d,
            volume_spike=ctx.volume_spike,
            implied_move_vs_realized=ctx.implied_move_vs_realized,
            skew_z={"MSTR": 1.5},   # ← panic
        )
        assert s.on_data(ctx_panic) == []

    def test_cooldown_30d(self):
        s = ReflexIgnitionV2()
        ctx1 = _ctx_full_trigger("MSTR", timestamp=datetime(2026, 1, 1))
        ctx15 = _ctx_full_trigger("MSTR", timestamp=datetime(2026, 1, 15))
        ctx40 = _ctx_full_trigger("MSTR", timestamp=datetime(2026, 2, 5))
        assert len(s.on_data(ctx1)) >= 1
        assert not any(sig.underlying == "MSTR" for sig in s.on_data(ctx15))
        assert any(sig.underlying == "MSTR" for sig in s.on_data(ctx40))

    def test_strategy_does_not_use_any_lagging_signal(self):
        """The implementation must not reference momentum_30d/60d/MAs/RSI/etc.

        Lagging-indicator reference would be a code smell. Spot-check by
        running with all leading+confirming OK but all-zero momentum-like
        fields — should still fire because lagging fields are never checked.
        """
        s = ReflexIgnitionV2()
        ctx = _ctx_full_trigger("MSTR")
        # Note: ctx doesn't even SET underlying_momentum_30d etc. — they're
        # empty dicts by default. The strategy must work without them.
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)

    def test_universe_matches_wsb(self):
        s = ReflexIgnitionV2()
        assert s.universe == WSB_UNIVERSE
