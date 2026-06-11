"""Unit tests for the 4 new leading-indicator strategies drafted 2026-05-26:
  - v_SHORT_SQUEEZE_PURE
  - v_IMPLIED_MOVE_DIVERGENCE
  - v_TRIPLE_CONFLUENCE
  - v_VIX_TERM_RECOVERY
  - v_NETWORK_PROPAGATION

Each strategy must:
  1. Fire when leading conditions + confirms align
  2. NOT fire when data is missing (fail-closed)
  3. NOT fire when regime is hostile
  4. Respect cooldown
"""
from __future__ import annotations

from datetime import datetime, timedelta

from strategies.short_squeeze_pure import ShortSqueezePure
from strategies.implied_move_divergence import ImpliedMoveDivergence
from strategies.triple_confluence import TripleConfluence
from strategies.vix_term_recovery import VixTermRecovery
from strategies.network_propagation import NetworkPropagation
from strategies.dealer_opex_squeeze import DealerOpexSqueeze
from strategies.base import StrategyContext


# ── v_SHORT_SQUEEZE_PURE ───────────────────────────────────────────────

class TestShortSqueezePure:
    def _ctx(self, ticker="MSTR", timestamp=None, **kw):
        if timestamp is None:
            timestamp = datetime(2026, 1, 15)
        base = dict(
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            short_interest_velocity={ticker: 0.40},   # +40% WoW
            insider_cluster_score={ticker: 2.0},       # insiders accumulating
            volume_spike={ticker: 5.0},
            skew_z={ticker: -0.3},
        )
        base.update(kw)
        return StrategyContext(timestamp=timestamp, **base)

    def test_fires_when_all_gates_pass(self):
        s = ShortSqueezePure()
        sigs = s.on_data(self._ctx("MSTR"))
        assert any(sig.underlying == "MSTR" for sig in sigs)
        sig = next(s for s in sigs if s.underlying == "MSTR")
        assert sig.side == "call"
        assert sig.target_dte == 45
        assert sig.max_hold_days == 60
        assert "SQUEEZE" in sig.notes

    def test_no_fire_below_si_velocity_threshold(self):
        s = ShortSqueezePure()
        ctx = self._ctx("MSTR", short_interest_velocity={"MSTR": 0.15})
        assert s.on_data(ctx) == []

    def test_no_fire_when_insiders_selling(self):
        s = ShortSqueezePure()
        ctx = self._ctx("MSTR", insider_cluster_score={"MSTR": -0.5})
        assert s.on_data(ctx) == []

    def test_no_fire_when_volume_thin(self):
        s = ShortSqueezePure()
        ctx = self._ctx("MSTR", volume_spike={"MSTR": 1.5})
        assert s.on_data(ctx) == []

    def test_no_fire_in_high_vix(self):
        s = ShortSqueezePure()
        ctx = self._ctx("MSTR", vix=28.0)
        assert s.on_data(ctx) == []

    def test_cooldown(self):
        s = ShortSqueezePure()
        d1 = self._ctx("MSTR", timestamp=datetime(2026, 1, 1))
        d20 = self._ctx("MSTR", timestamp=datetime(2026, 1, 20))
        d60 = self._ctx("MSTR", timestamp=datetime(2026, 3, 1))
        assert len(s.on_data(d1)) >= 1
        assert not any(sig.underlying == "MSTR" for sig in s.on_data(d20))
        assert any(sig.underlying == "MSTR" for sig in s.on_data(d60))


# ── v_IMPLIED_MOVE_DIVERGENCE ──────────────────────────────────────────

class TestImpliedMoveDivergence:
    def _ctx(self, ticker="MSTR", **kw):
        base = dict(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            implied_move_vs_realized={ticker: 3.0},
            days_until_next_earnings={ticker: 4},
            earnings_within_5d={ticker},
        )
        base.update(kw)
        return StrategyContext(**base)

    def test_fires_on_high_im_with_earnings_confirm(self):
        s = ImpliedMoveDivergence()
        sigs = s.on_data(self._ctx("MSTR"))
        assert any(sig.underlying == "MSTR" for sig in sigs)
        sig = next(s for s in sigs if s.underlying == "MSTR")
        assert sig.target_otm_pct == 0.05
        assert sig.target_dte == 30
        assert sig.max_hold_days <= 14   # tightened by earnings proximity

    def test_no_fire_without_confirmation(self):
        s = ImpliedMoveDivergence()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            implied_move_vs_realized={"MSTR": 3.0},
            # No earnings_within_5d, no UOA
            uoa_active=set(),
            days_until_next_earnings={"MSTR": 60},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_im_below_threshold(self):
        # 2026-05-27 retune: threshold dropped 2.5 → 1.4; below = 1.3.
        s = ImpliedMoveDivergence()
        ctx = self._ctx("MSTR", implied_move_vs_realized={"MSTR": 1.3})
        assert s.on_data(ctx) == []

    def test_uoa_alone_as_confirmation_works(self):
        s = ImpliedMoveDivergence()
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            implied_move_vs_realized={"MSTR": 3.0},
            uoa_active={"MSTR"},
            days_until_next_earnings={"MSTR": 60},   # no earnings soon
        )
        assert any(sig.underlying == "MSTR" for sig in s.on_data(ctx))


# ── v_TRIPLE_CONFLUENCE ────────────────────────────────────────────────

class TestTripleConfluence:
    def _ctx(self, ticker="MSTR", **kw):
        base = dict(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango", vix9d_vix_ratio=0.95,
            short_interest_velocity={ticker: 0.30},
            insider_count_5d={ticker: 4},
            news_propagation_5d={ticker: 2.0},
            volume_spike={ticker: 5.0},
            implied_move_vs_realized={ticker: 3.0},
            skew_z={ticker: -0.3},
        )
        base.update(kw)
        return StrategyContext(**base)

    def test_fires_on_full_confluence(self):
        s = TripleConfluence()
        sigs = s.on_data(self._ctx("MSTR"))
        assert any(sig.underlying == "MSTR" for sig in sigs)
        sig = next(s for s in sigs if s.underlying == "MSTR")
        assert sig.max_per_trade_pct_nav == 0.12
        assert sig.target_otm_pct == 0.25
        assert sig.target_dte == 60
        assert "TRIPLE" in sig.notes

    def test_no_fire_if_any_leading_misses(self):
        s = TripleConfluence()
        # Drop L1 (SI velocity)
        ctx = self._ctx("MSTR", short_interest_velocity={"MSTR": 0.15})
        assert s.on_data(ctx) == []

    def test_no_fire_if_any_confirm_misses(self):
        s = TripleConfluence()
        # Drop C1 (volume_spike)
        ctx = self._ctx("MSTR", volume_spike={"MSTR": 1.5})
        assert s.on_data(ctx) == []

    def test_skew_zero_required(self):
        """Stricter than v_REFLEX_v2's 0.5 — must be ≤ 0."""
        s = TripleConfluence()
        ctx = self._ctx("MSTR", skew_z={"MSTR": 0.3})  # > 0
        assert s.on_data(ctx) == []


# ── v_VIX_TERM_RECOVERY ────────────────────────────────────────────────

class TestVixTermRecovery:
    def test_fires_when_ratio_inverts_from_elevated(self):
        s = VixTermRecovery()
        # Day 1-5: elevated ratio (stress regime)
        for d in range(1, 6):
            ctx = StrategyContext(
                timestamp=datetime(2026, 1, d),
                vix=22.0, term_regime="flat", vix9d_vix_ratio=1.15,
            )
            s.on_data(ctx)
        # Day 6: ratio drops below 0.95 — RECOVERY trigger
        ctx_recov = StrategyContext(
            timestamp=datetime(2026, 1, 6),
            vix=20.0, term_regime="contango", vix9d_vix_ratio=0.90,
        )
        sigs = s.on_data(ctx_recov)
        assert any(sig.underlying == "SPY" for sig in sigs)
        sig = next(s for s in sigs if s.underlying == "SPY")
        assert sig.target_otm_pct == 0.0   # ATM
        assert sig.max_per_trade_pct_nav == 0.15

    def test_no_fire_without_prior_stress(self):
        s = VixTermRecovery()
        # All days have moderate ratio (no prior stress)
        for d in range(1, 7):
            ctx = StrategyContext(
                timestamp=datetime(2026, 1, d),
                vix=18.0, term_regime="contango", vix9d_vix_ratio=0.92,
            )
            assert s.on_data(ctx) == []

    def test_no_fire_in_active_panic(self):
        s = VixTermRecovery()
        for d in range(1, 6):
            s.on_data(StrategyContext(
                timestamp=datetime(2026, 1, d),
                vix=40.0, term_regime="flat", vix9d_vix_ratio=1.20,
            ))
        # VIX still > 35 = panic, even if ratio inverts → no fire
        ctx = StrategyContext(
            timestamp=datetime(2026, 1, 6),
            vix=38.0, term_regime="flat", vix9d_vix_ratio=0.90,
        )
        assert s.on_data(ctx) == []


# ── v_NETWORK_PROPAGATION ──────────────────────────────────────────────

class TestNetworkPropagation:
    def _ctx(self, ticker="MSTR", **kw):
        base = dict(
            timestamp=datetime(2026, 1, 15),
            vix=18.0, term_regime="contango",
            news_propagation_5d={ticker: 2.5},
            volume_spike={ticker: 3.0},
        )
        base.update(kw)
        return StrategyContext(**base)

    def test_fires_on_strong_propagation_with_volume(self):
        s = NetworkPropagation()
        sigs = s.on_data(self._ctx("MSTR"))
        assert any(sig.underlying == "MSTR" for sig in sigs)
        sig = next(s for s in sigs if s.underlying == "MSTR")
        assert sig.target_dte == 30
        assert sig.max_hold_days == 30
        assert "NETPROP" in sig.notes

    def test_no_fire_below_propagation_threshold(self):
        s = NetworkPropagation()
        ctx = self._ctx("MSTR", news_propagation_5d={"MSTR": 1.5})
        assert s.on_data(ctx) == []

    def test_no_fire_without_volume_confirm(self):
        s = NetworkPropagation()
        ctx = self._ctx("MSTR", volume_spike={"MSTR": 1.0})
        assert s.on_data(ctx) == []

    def test_no_fire_in_high_vix(self):
        s = NetworkPropagation()
        ctx = self._ctx("MSTR", vix=27.0)
        assert s.on_data(ctx) == []


# ── v_DEALER_OPEX_SQUEEZE ──────────────────────────────────────────────

class TestDealerOpexSqueeze:
    """Mid-month OPEX window + dealer-hedging-stress proxies.

    Calendar: 3rd Friday of Jan 2026 = 2026-01-16. We test on 2026-01-13
    (3 trading days before) which is squarely in the 7-day OPEX window.
    """
    def _ctx(self, ticker="MSTR", timestamp=None, gex_regime="flip_zone",
             uoa=True, mom30=1.05, vix=18.0, **kw):
        if timestamp is None:
            timestamp = datetime(2026, 1, 13)  # 3 days before 3rd Friday
        base = dict(
            vix=vix, term_regime="contango",
            gex_regime=gex_regime,
            uoa_active={ticker} if uoa else set(),
            underlying_momentum_30d={ticker: mom30},
        )
        base.update(kw)
        return StrategyContext(timestamp=timestamp, **base)

    def test_fires_in_opex_window_with_uoa(self):
        s = DealerOpexSqueeze()
        sigs = s.on_data(self._ctx("MSTR", uoa=True, mom30=1.05))
        assert any(sig.underlying == "MSTR" for sig in sigs), (
            "Expected MSTR fire on opex_d-3 with UOA active + mom30=1.05"
        )

    def test_fires_via_or_logic_strong_mom_without_uoa(self):
        """2026-05-27 v2 OR-logic — strong-mom alone is sufficient."""
        s = DealerOpexSqueeze()
        sigs = s.on_data(self._ctx("MSTR", uoa=False, mom30=1.10))
        assert any(sig.underlying == "MSTR" for sig in sigs), (
            "Expected fire via strong-mom OR branch even without UOA"
        )

    def test_no_fire_outside_opex_window(self):
        s = DealerOpexSqueeze()
        # 2026-01-30 is 14 days AFTER the 3rd Friday — well outside window
        ctx = self._ctx("MSTR", timestamp=datetime(2026, 1, 30))
        assert s.on_data(ctx) == []

    def test_no_fire_when_vix_too_high(self):
        s = DealerOpexSqueeze()
        ctx = self._ctx("MSTR", vix=26.0)
        assert s.on_data(ctx) == []

    def test_no_fire_when_neither_uoa_nor_strong_mom(self):
        s = DealerOpexSqueeze()
        ctx = self._ctx("MSTR", uoa=False, mom30=1.03)  # mom not strong enough
        assert s.on_data(ctx) == []

    def test_no_fire_when_mom_below_floor(self):
        """mom30 < 1.02 floor blocks even if UOA active."""
        s = DealerOpexSqueeze()
        ctx = self._ctx("MSTR", uoa=True, mom30=0.99)
        assert s.on_data(ctx) == []

    def test_fires_when_gex_is_none(self):
        """2026-05-27 v2 fix — gex None default permissive (was 0-fire bug)."""
        s = DealerOpexSqueeze()
        ctx = self._ctx("MSTR", gex_regime=None)
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs), (
            "v2 should default gex None → flip_zone (permissive). "
            "Reproduces the bug that caused 0 fires in 8 BTs."
        )
