"""Unit tests for 6 standby mix-and-match strategies (2026-05-26)."""
from __future__ import annotations

from datetime import datetime

from strategies.tail_hedge_regime import TailHedgeRegime
from strategies.vvix_divergence import VvixDivergence
from strategies.retail_attention_cascade import RetailAttentionCascade
from strategies.earnings_revision_momentum import EarningsRevisionMomentum
from strategies.quad_confluence import QuadConfluence
from strategies.activist_drift import ActivistDrift
from strategies.eightk_burst import EightKBurst
from strategies.base import StrategyContext


# ── v_TAIL_HEDGE_REGIME ────────────────────────────────────────────────

class TestTailHedgeRegime:
    def test_fires_on_high_skew_low_vix_pct(self):
        s = TailHedgeRegime()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            cboe_skew=135.0, vix_percentile_1y=0.15, vix9d_vix_ratio=0.95,
            vix=15.0,
        )
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "SPY" and sig.side == "put" for sig in sigs)

    def test_no_fire_when_skew_below_threshold(self):
        s = TailHedgeRegime()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            cboe_skew=125.0, vix_percentile_1y=0.15, vix9d_vix_ratio=0.95)
        assert s.on_data(ctx) == []

    def test_no_fire_when_already_stressed(self):
        s = TailHedgeRegime()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            cboe_skew=135.0, vix_percentile_1y=0.15, vix9d_vix_ratio=1.10)
        assert s.on_data(ctx) == []

    def test_no_fire_when_data_missing(self):
        s = TailHedgeRegime()
        ctx = StrategyContext(timestamp=datetime(2026, 1, 15))
        assert s.on_data(ctx) == []


# ── v_VVIX_DIVERGENCE ──────────────────────────────────────────────────

class TestVvixDivergence:
    def test_fires_on_elevated_vvix_calm_vix(self):
        s = VvixDivergence()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vvix=135.0, vix=18.0, vix_percentile_1y=0.30)
        sigs = s.on_data(ctx)
        assert any(sig.side == "put" for sig in sigs)

    def test_no_fire_when_vix_already_high(self):
        s = VvixDivergence()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vvix=135.0, vix=27.0, vix_percentile_1y=0.30)
        assert s.on_data(ctx) == []


# ── v_RETAIL_ATTENTION_CASCADE ─────────────────────────────────────────

class TestRetailAttentionCascade:
    def test_fires_on_wsb_plus_gtrends_plus_volume(self):
        s = RetailAttentionCascade()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            wsb_mention_velocity={"MSTR": 4.0},
            google_trends_velocity={"MSTR": 2.5},
            volume_spike={"MSTR": 4.0},
            skew_z={"MSTR": -0.2})
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)

    def test_no_fire_when_only_wsb(self):
        s = RetailAttentionCascade()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            wsb_mention_velocity={"MSTR": 5.0},   # only this
            volume_spike={"MSTR": 4.0})
        assert s.on_data(ctx) == []


# ── v_EARNINGS_REVISION_MOMENTUM ───────────────────────────────────────

class TestEarningsRevisionMomentum:
    def test_fires_on_positive_revision_in_window(self):
        s = EarningsRevisionMomentum()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            earnings_revision_velocity={"MSTR": 0.08},
            days_until_next_earnings={"MSTR": 20},
            volume_spike={"MSTR": 3.0})
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)

    def test_no_fire_too_close_to_earnings(self):
        s = EarningsRevisionMomentum()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            earnings_revision_velocity={"MSTR": 0.08},
            days_until_next_earnings={"MSTR": 2},   # too close, IV crush risk
            volume_spike={"MSTR": 3.0})
        assert s.on_data(ctx) == []


# ── v_QUAD_CONFLUENCE ──────────────────────────────────────────────────

class TestQuadConfluence:
    def test_fires_on_all_4_leading_plus_confirm(self):
        s = QuadConfluence()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            short_interest_velocity={"MSTR": 0.30},
            insider_count_5d={"MSTR": 4},
            news_propagation_5d={"MSTR": 2.0},
            wsb_mention_velocity={"MSTR": 3.0},
            volume_spike={"MSTR": 5.0},
            implied_move_vs_realized={"MSTR": 2.5},
            skew_z={"MSTR": -0.3})
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)
        sig = next(s for s in sigs if s.underlying == "MSTR")
        assert sig.max_per_trade_pct_nav == 0.15
        assert sig.edge_score == 1.0

    def test_no_fire_if_any_leading_misses(self):
        s = QuadConfluence()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            short_interest_velocity={"MSTR": 0.10},   # below threshold
            insider_count_5d={"MSTR": 4},
            news_propagation_5d={"MSTR": 2.0},
            wsb_mention_velocity={"MSTR": 3.0},
            volume_spike={"MSTR": 5.0})
        assert s.on_data(ctx) == []


# ── v_ACTIVIST_DRIFT ───────────────────────────────────────────────────

class TestActivistDrift:
    def test_fires_on_13d_with_insider_confirm(self):
        s = ActivistDrift()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            recent_13d_filing={"MSTR"},
            insider_cluster_score={"MSTR": 2.0})
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)

    def test_no_fire_without_13d(self):
        s = ActivistDrift()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            recent_13d_filing=set(),
            insider_cluster_score={"MSTR": 2.0})
        assert s.on_data(ctx) == []


# ── v_8K_BURST ─────────────────────────────────────────────────────────

class TestEightKBurst:
    def test_fires_on_burst_with_insider_present(self):
        s = EightKBurst()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            recent_8k_count={"MSTR": 4},
            insider_count_5d={"MSTR": 2},
            volume_spike={"MSTR": 3.0},
            skew_z={"MSTR": -0.2})
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)

    def test_no_fire_when_below_burst_threshold(self):
        s = EightKBurst()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0,
            recent_8k_count={"MSTR": 2},
            insider_count_5d={"MSTR": 2},
            volume_spike={"MSTR": 3.0})
        assert s.on_data(ctx) == []
