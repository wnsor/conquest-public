"""Unit tests for v_REFLEX (reflex_ignition) — reflexivity ignition detector."""
from __future__ import annotations

from datetime import datetime

from strategies.reflex_ignition import ReflexIgnition
from strategies.momentum_otm_calls import WSB_UNIVERSE
from strategies.base import StrategyContext


def _ctx_full_ignition(ticker="MSTR", timestamp=None):
    """A context where ALL 5 accelerators fire for `ticker`."""
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 16, 0)
    return StrategyContext(
        timestamp=timestamp,
        vix=18.0,
        term_regime="contango",
        vix9d_vix_ratio=0.95,
        news_volume_spike={ticker: 5.0},        # (1) attention spike
        news_sentiment_24h={ticker: 0.40},      # (2) sentiment positive
        uoa_active={ticker},                    # (2) UOA reinforces
        volume_spike={ticker: 4.0},             # (3) $-volume spike
        underlying_momentum_30d={ticker: 1.25}, # (3) + (5) momentum
        underlying_momentum_60d={ticker: 1.10}, # (5) 30d > 60d = accelerating
        skew_z={ticker: -0.3},                  # (4) skew not panicking
    )


def _ctx_minimal(timestamp=None, **kw):
    if timestamp is None:
        timestamp = datetime(2026, 1, 15, 16, 0)
    return StrategyContext(timestamp=timestamp, **kw)


class TestReflexIgnition:
    def test_fires_when_5of5_accelerators_active(self):
        s = ReflexIgnition()
        ctx = _ctx_full_ignition("MSTR")
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)
        # Confluence 5/5 → edge=1.0
        mstr_sig = next(sig for sig in sigs if sig.underlying == "MSTR")
        assert mstr_sig.edge_score == 1.0
        assert mstr_sig.side == "call"
        assert mstr_sig.target_dte == 45
        assert mstr_sig.target_otm_pct == 0.20
        assert mstr_sig.stop_loss_pct == -0.50
        assert mstr_sig.max_hold_days == 45
        assert "REFLEX_n=5/5" in mstr_sig.notes

    def test_fires_at_4of5_confluence(self):
        s = ReflexIgnition()
        # Drop accelerator (3): no volume spike
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            news_volume_spike={"MSTR": 5.0},
            news_sentiment_24h={"MSTR": 0.40},
            uoa_active={"MSTR"},
            volume_spike={"MSTR": 1.5},   # ← BELOW trigger
            underlying_momentum_30d={"MSTR": 1.25},
            underlying_momentum_60d={"MSTR": 1.10},
            skew_z={"MSTR": -0.3},
        )
        sigs = s.on_data(ctx)
        mstr = [s for s in sigs if s.underlying == "MSTR"]
        assert len(mstr) == 1
        assert mstr[0].edge_score == 0.8   # 4/5

    def test_no_fire_at_3of5(self):
        s = ReflexIgnition()
        # Drop accelerators (3) and (5)
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            news_volume_spike={"MSTR": 5.0},
            news_sentiment_24h={"MSTR": 0.40},
            uoa_active={"MSTR"},
            volume_spike={"MSTR": 1.5},      # no
            underlying_momentum_30d={"MSTR": 1.05},  # below 1.10 trigger
            underlying_momentum_60d={"MSTR": 1.10},
            skew_z={"MSTR": -0.3},
        )
        sigs = s.on_data(ctx)
        assert not any(s.underlying == "MSTR" for s in sigs)

    def test_no_fire_on_high_vix(self):
        s = ReflexIgnition()
        ctx = _ctx_full_ignition("MSTR")
        ctx = StrategyContext(
            timestamp=ctx.timestamp,
            vix=32.0,                   # over 30 = risk-off, reflex dead
            term_regime=ctx.term_regime,
            vix9d_vix_ratio=ctx.vix9d_vix_ratio,
            news_volume_spike=ctx.news_volume_spike,
            news_sentiment_24h=ctx.news_sentiment_24h,
            uoa_active=ctx.uoa_active,
            volume_spike=ctx.volume_spike,
            underlying_momentum_30d=ctx.underlying_momentum_30d,
            underlying_momentum_60d=ctx.underlying_momentum_60d,
            skew_z=ctx.skew_z,
        )
        assert s.on_data(ctx) == []

    def test_no_fire_on_backwardation(self):
        s = ReflexIgnition()
        # All-accel context but in backwardation
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            term_regime="backwardation",      # forced liquidation regime
            news_volume_spike={"MSTR": 5.0},
            news_sentiment_24h={"MSTR": 0.40},
            uoa_active={"MSTR"},
            volume_spike={"MSTR": 4.0},
            underlying_momentum_30d={"MSTR": 1.25},
            underlying_momentum_60d={"MSTR": 1.10},
            skew_z={"MSTR": -0.3},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_on_acute_stress(self):
        s = ReflexIgnition()
        # All-accel but VIX9D/VIX > 1.10
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            term_regime="contango",
            vix9d_vix_ratio=1.15,
            news_volume_spike={"MSTR": 5.0},
            news_sentiment_24h={"MSTR": 0.40},
            uoa_active={"MSTR"},
            volume_spike={"MSTR": 4.0},
            underlying_momentum_30d={"MSTR": 1.25},
            underlying_momentum_60d={"MSTR": 1.10},
            skew_z={"MSTR": -0.3},
        )
        assert s.on_data(ctx) == []

    def test_momentum_must_be_accelerating(self):
        """30d > 60d required (rate-of-change rising, not just stable up)."""
        s = ReflexIgnition()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            news_volume_spike={"MSTR": 5.0},
            news_sentiment_24h={"MSTR": 0.40},
            uoa_active={"MSTR"},
            volume_spike={"MSTR": 4.0},
            underlying_momentum_30d={"MSTR": 1.15},  # up, but
            underlying_momentum_60d={"MSTR": 1.30},  # 60d > 30d = decelerating
            skew_z={"MSTR": -0.3},
        )
        sigs = s.on_data(ctx)
        # Should still fire at 4/5 (drops accel 5)
        mstr = [s for s in sigs if s.underlying == "MSTR"]
        assert len(mstr) == 1
        assert mstr[0].edge_score == 0.8

    def test_skew_panic_kills_signal(self):
        """High skew_z = puts crowded = doubt = no ignition."""
        s = ReflexIgnition()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            news_volume_spike={"MSTR": 5.0},
            news_sentiment_24h={"MSTR": 0.40},
            uoa_active={"MSTR"},
            volume_spike={"MSTR": 4.0},
            underlying_momentum_30d={"MSTR": 1.25},
            underlying_momentum_60d={"MSTR": 1.10},
            skew_z={"MSTR": 1.5},  # ← skew elevated = doubt rising
        )
        sigs = s.on_data(ctx)
        # 4/5 fires (drops accel 4)
        mstr = [s for s in sigs if s.underlying == "MSTR"]
        assert len(mstr) == 1

    def test_cooldown_30d_blocks_re_entry(self):
        s = ReflexIgnition()
        ctx1 = _ctx_full_ignition("MSTR", timestamp=datetime(2026, 1, 1))
        ctx20 = _ctx_full_ignition("MSTR", timestamp=datetime(2026, 1, 20))
        ctx40 = _ctx_full_ignition("MSTR", timestamp=datetime(2026, 2, 5))
        assert len(s.on_data(ctx1)) >= 1
        # Within 30d cooldown — no fire
        sigs20 = s.on_data(ctx20)
        assert not any(sig.underlying == "MSTR" for sig in sigs20)
        # Past 30d cooldown — fires again
        sigs40 = s.on_data(ctx40)
        assert any(sig.underlying == "MSTR" for sig in sigs40)

    def test_universe_matches_momentum_otm_calls(self):
        s = ReflexIgnition()
        assert s.universe == WSB_UNIVERSE

    def test_no_signal_when_timestamp_missing(self):
        s = ReflexIgnition()
        ctx = _ctx_minimal(timestamp=None)
        assert s.on_data(ctx) == []
