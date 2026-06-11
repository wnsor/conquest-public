"""End-to-end strategy-fires tests.

The Conquest Options BTs have produced 0 trades in many runs. Two
hypotheses:
  1. The strategy gate logic is wrong → tested HERE with mock contexts
     simulating COVID 2020 / GFC 2009 V-bottoms.
  2. The QC runtime isn't populating the context fields (price_history
     empty, ctx.crisis_state stuck on 'normal', etc.) → that needs cloud
     BT logs to verify.

This file tests (1). If a strategy doesn't fire here despite hand-rolled
COVID-perfect inputs, the strategy is broken regardless of the runtime.

Each test:
  - Constructs a StrategyContext that mirrors COVID-like or GFC-like
    real data on the day the strategy SHOULD fire.
  - Calls strategy.on_data(ctx).
  - Asserts the strategy emitted ≥1 signal.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

# bare sibling imports per memory feedback_lean_bare_imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))

from strategies.base import StrategyContext  # noqa: E402
from strategies.tepper_vbottom_leaps import TepperVbottomLeaps  # noqa: E402
from strategies.crisis_rebound_basket import CrisisReboundBasket  # noqa: E402
from strategies.spy_crisis_put import SpyCrisisPut  # noqa: E402
from strategies.gex_spy_selective import GexSpySelective  # noqa: E402
from strategies.gex_spy_baseline import GexSpyBaseline  # noqa: E402
from strategies.momentum_otm_calls import MomentumOtmCalls  # noqa: E402
from strategies.cgrowth_leaps import CgrowthLeaps  # noqa: E402
from edge_signals.crisis_detector import CrisisDetector  # noqa: E402


# ============================================================
# D2 Tepper V-bottom LEAPS
# ============================================================

def _covid_v_bottom_ctx() -> StrategyContext:
    """A context approximating SPY on April 7, 2020 — the V-bottom recovery
    point. SPY was ~$260, peak Feb 19 was $339 → dd=23%. 5MA was rising
    above 20MA. VIX still elevated ~45."""
    return StrategyContext(
        timestamp=datetime(2020, 4, 7, 15, 0),
        underlying_prices={"SPY": 260.0},
        vix=45.0,
        vix3m=38.0,
        vix_term_ratio=1.18,
        term_regime="backwardation",
        underlying_drawdown_from_252d_high={"SPY": 0.23},   # 23% dd
        underlying_5ma_above_20ma={"SPY": True},
        underlying_momentum_30d={"SPY": 0.85},
        underlying_momentum_60d={"SPY": 0.83},
        gex_total=-30.0,
        gex_regime="short_gamma",
        crisis_state="rebound",       # CrisisDetector should be in rebound by now
        crisis_vix_peak=82.0,
        cstability_vote_count=3,      # all defensive votes firing
        iv_rank={"SPY": 95.0},
        historical_vol_30d={"SPY": 0.55},
    )


def test_d2_tepper_fires_on_covid_v_bottom():
    """D2 should emit exactly 1 signal on the COVID V-bottom date."""
    d2 = TepperVbottomLeaps()
    ctx = _covid_v_bottom_ctx()
    signals = d2.on_data(ctx)
    assert len(signals) == 1, (
        f"D2 should fire on COVID V-bottom (dd=23%, 5MA>20MA), got {len(signals)}"
    )
    sig = signals[0]
    assert sig.underlying == "SPY"
    assert sig.side == "leaps_call"
    assert sig.target_dte == 180
    assert sig.target_delta == 0.50


def test_d2_tepper_does_not_fire_in_normal_market():
    """D2 must NOT fire when drawdown < 10%."""
    d2 = TepperVbottomLeaps()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 1, 15, 0),
        underlying_prices={"SPY": 530.0},
        vix=14.0,
        underlying_drawdown_from_252d_high={"SPY": 0.02},
        underlying_5ma_above_20ma={"SPY": True},
    )
    signals = d2.on_data(ctx)
    assert len(signals) == 0


def test_d2_tepper_does_not_fire_if_5ma_below_20ma():
    """D2 needs 5MA up-cross (recovery confirmation)."""
    d2 = TepperVbottomLeaps()
    ctx = _covid_v_bottom_ctx()
    # Override: trend still down
    ctx = StrategyContext(
        timestamp=ctx.timestamp,
        underlying_drawdown_from_252d_high=ctx.underlying_drawdown_from_252d_high,
        underlying_5ma_above_20ma={"SPY": False},
    )
    signals = d2.on_data(ctx)
    assert len(signals) == 0


def test_d2_tepper_respects_cooldown():
    """D2 fires once then waits 90 days."""
    d2 = TepperVbottomLeaps()
    ctx = _covid_v_bottom_ctx()
    first = d2.on_data(ctx)
    assert len(first) == 1
    # Same day, same gates — should NOT re-fire
    second = d2.on_data(ctx)
    assert len(second) == 0


# ============================================================
# CrisisReboundBasket
# ============================================================

def test_crisis_rebound_basket_fires_when_state_rebound():
    """CrisisReboundBasket emits 7 signals (SPY+QQQ+5 names) when state=rebound."""
    basket = CrisisReboundBasket()
    ctx = _covid_v_bottom_ctx()
    signals = basket.on_data(ctx)
    assert len(signals) == 7, (
        f"basket should fire all 7 contracts when state=rebound, got {len(signals)}"
    )
    underlyings = {s.underlying for s in signals}
    assert underlyings == {"SPY", "QQQ", "NVDA", "AMD", "META", "GOOGL", "MSFT"}


def test_crisis_rebound_basket_does_not_fire_in_normal():
    """CrisisReboundBasket waits for state=rebound; other states don't fire."""
    basket = CrisisReboundBasket()
    for state in (None, "normal", "warning", "crash", "capitulation", "recovery"):
        ctx = StrategyContext(timestamp=datetime(2020, 4, 7, 15, 0), crisis_state=state)
        signals = basket.on_data(ctx)
        assert len(signals) == 0, f"basket fired in state={state!r}"


def test_crisis_rebound_basket_180d_lock():
    """Once fired, basket is locked for 180 days."""
    basket = CrisisReboundBasket()
    ctx = _covid_v_bottom_ctx()
    first = basket.on_data(ctx)
    assert len(first) == 7
    # Same context, immediately — no re-fire
    second = basket.on_data(ctx)
    assert len(second) == 0


# ============================================================
# B1 SPY crisis put
# ============================================================

def test_b1_spy_crisis_put_fires_in_warning():
    """B1 fires when CrisisDetector says 'warning' AND VIX in 22-50 range."""
    b1 = SpyCrisisPut()
    ctx = StrategyContext(
        timestamp=datetime(2020, 3, 5, 15, 0),
        underlying_prices={"SPY": 305.0},
        vix=30.0,
        term_regime="backwardation",
        crisis_state="warning",
        crisis_vix_peak=35.0,
        cstability_vote_count=2,
    )
    signals = b1.on_data(ctx)
    assert len(signals) == 1, "B1 should fire in 'warning' state with vix=30"


def test_b1_spy_crisis_put_skips_capitulation():
    """B1 deliberately skips capitulation (too late, IV peaked)."""
    b1 = SpyCrisisPut()
    ctx = StrategyContext(
        timestamp=datetime(2020, 3, 23, 15, 0),
        vix=82.0,
        crisis_state="capitulation",
        cstability_vote_count=3,
    )
    signals = b1.on_data(ctx)
    assert len(signals) == 0, "B1 should NOT fire in capitulation (too late)"


# ============================================================
# A_GEX SPY call (selective + baseline)
# ============================================================

def test_a_gex_selective_fires_with_short_gamma():
    """A_GEX selective fires when ALL hard gates pass."""
    strat = GexSpySelective()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),  # Monday
        underlying_prices={"SPY": 530.0},
        vix=14.0,
        term_regime="contango",
        gex_total=-25.0,
        gex_regime="short_gamma",        # required
        underlying_5ma_above_20ma={"SPY": True},  # required
        underlying_drawdown_from_252d_high={"SPY": 0.02},
        underlying_momentum_30d={"SPY": 1.04},
        underlying_momentum_60d={"SPY": 1.08},
        cstability_vote_count=0,
        iv_rank={"SPY": 25.0},           # cheap IV (helps edge)
        iv_hv_ratio={"SPY": 0.9},        # IV cheap relative to HV
    )
    signals = strat.on_data(ctx)
    assert len(signals) >= 1, "A_GEX selective should fire when all hard gates pass"


def test_a_gex_selective_blocked_by_long_gamma():
    """A_GEX requires gex_regime == 'short_gamma'."""
    strat = GexSpySelective()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        vix=14.0,
        gex_regime="long_gamma",
        underlying_5ma_above_20ma={"SPY": True},
    )
    signals = strat.on_data(ctx)
    assert len(signals) == 0


def test_a_gex_baseline_fires_when_gates_pass():
    """A_GEX baseline fires unconditionally when hard gates pass (no edge threshold)."""
    strat = GexSpyBaseline()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        underlying_prices={"SPY": 530.0},
        vix=14.0,
        gex_regime="short_gamma",
        underlying_5ma_above_20ma={"SPY": True},
        term_regime="contango",
        cstability_vote_count=0,
        underlying_momentum_30d={"SPY": 1.04},
    )
    signals = strat.on_data(ctx)
    assert len(signals) == 1


# ============================================================
# A1 WSB OTM call
# ============================================================

def _a1_baseline_ctx(ticker: str = "MU") -> StrategyContext:
    """A1 baseline context — passes all v6 catalyst-first gates: iv_rank<40 + at least one catalyst."""
    return StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        underlying_prices={ticker: 150.0},
        vix=14.0,
        iv_rank={ticker: 25.0},  # v6 requires <40 (cheap options pre-IV expansion)
        uoa_active={ticker},
        insider_recent_buys={ticker: 250_000},
        underlying_momentum_30d={ticker: 1.05},  # v6 has no momentum gate
        iv_hv_ratio={ticker: 0.9},
        cstability_vote_count=0,
        # v6 catalyst gate: at least one required
        volume_spike={ticker: 5.0},          # >4 = institutional flow
        insider_cluster_score={ticker: 2.0},  # >=1.5 = insiders buying
    )


def test_a1_wsb_fires_with_iv_uoa_insider_confluence():
    """A1 WSB call needs IV-rank<60 + mom30>1.0 (hard gates) + IV/HV gate."""
    strat = MomentumOtmCalls()
    signals = strat.on_data(_a1_baseline_ctx("MU"))
    assert any(s.underlying == "MU" for s in signals), (
        "A1 should fire on MU with cheap IV + momentum"
    )


def test_a1_tier1_volume_spike_boosts_edge():
    """Tier1 Signal 2: high volume_spike adds 1-2 confluences → higher edge_score."""
    strat_low = MomentumOtmCalls()
    sigs_low = strat_low.on_data(_a1_baseline_ctx("MU"))
    edge_low = next(s.edge_score for s in sigs_low if s.underlying == "MU")

    # Same ctx but with a 6x volume spike (>5.0 threshold → +2 confluences)
    strat_high = MomentumOtmCalls()
    ctx_high = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        underlying_prices={"MU": 150.0},
        vix=14.0,
        iv_rank={"MU": 25.0},
        uoa_active={"MU"},
        insider_recent_buys={"MU": 250_000},
        underlying_momentum_30d={"MU": 1.10},
        iv_hv_ratio={"MU": 0.9},
        cstability_vote_count=0,
        volume_spike={"MU": 6.0},
    )
    sigs_high = strat_high.on_data(ctx_high)
    edge_high = next(s.edge_score for s in sigs_high if s.underlying == "MU")
    assert edge_high > edge_low, (
        f"6x volume_spike should raise edge_score above baseline; "
        f"baseline={edge_low}, with_spike={edge_high}"
    )


def test_a1_tier1_insider_cluster_boosts_edge():
    """Tier1 Signal 3: cluster_score ≥3.0 adds +2 confluences."""
    strat_low = MomentumOtmCalls()
    edge_low = next(s.edge_score for s in strat_low.on_data(_a1_baseline_ctx("MU"))
                    if s.underlying == "MU")

    strat_high = MomentumOtmCalls()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        underlying_prices={"MU": 150.0}, vix=14.0,
        iv_rank={"MU": 25.0}, uoa_active={"MU"},
        insider_recent_buys={"MU": 250_000},
        underlying_momentum_30d={"MU": 1.10}, iv_hv_ratio={"MU": 0.9},
        cstability_vote_count=0,
        insider_cluster_score={"MU": 4.5},  # Officer + Director + 10pct (max strong)
    )
    edge_high = next(s.edge_score for s in strat_high.on_data(ctx) if s.underlying == "MU")
    assert edge_high > edge_low


def test_a1_tier1_news_sentiment_boosts_edge():
    """Tier1 Signal 1: positive news_sentiment_24h > 0.3 adds +1 confluence."""
    strat_low = MomentumOtmCalls()
    edge_low = next(s.edge_score for s in strat_low.on_data(_a1_baseline_ctx("MU"))
                    if s.underlying == "MU")

    strat_high = MomentumOtmCalls()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        underlying_prices={"MU": 150.0}, vix=14.0,
        iv_rank={"MU": 25.0}, uoa_active={"MU"},
        insider_recent_buys={"MU": 250_000},
        underlying_momentum_30d={"MU": 1.10}, iv_hv_ratio={"MU": 0.9},
        cstability_vote_count=0,
        news_sentiment_24h={"MU": 0.55},   # strongly positive (>0.3)
        news_volume_spike={"MU": 4.5},     # 4.5x average (>3.0)
    )
    edge_high = next(s.edge_score for s in strat_high.on_data(ctx) if s.underlying == "MU")
    assert edge_high > edge_low


def test_a1_tier1_signals_do_not_bypass_hard_gates():
    """Strong Tier1 confluences must NOT override hard gates (mom30 < 1.0)."""
    strat = MomentumOtmCalls()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        underlying_prices={"MU": 150.0}, vix=14.0,
        iv_rank={"MU": 25.0},
        underlying_momentum_30d={"MU": 0.85},   # FAIL: < 1.0
        iv_hv_ratio={"MU": 0.9},
        # Massively positive Tier1 confluences:
        volume_spike={"MU": 10.0},
        insider_cluster_score={"MU": 6.0},
        news_sentiment_24h={"MU": 0.9},
        news_volume_spike={"MU": 8.0},
    )
    signals = strat.on_data(ctx)
    assert not any(s.underlying == "MU" for s in signals), (
        "Hard gate mom30<1.0 must reject despite max Tier1 confluences"
    )


# ============================================================
# CrisisDetector state machine
# ============================================================

def test_crisis_detector_reaches_rebound_on_covid():
    """Walk the detector through synthetic COVID-like daily data and assert
    it reaches the 'rebound' state."""
    detector = CrisisDetector()
    base = date(2020, 1, 1)

    # 30 days normal market (VIX 15-18)
    for d in range(30):
        detector.update(
            today=base.replace(day=1) if False else base,
            vix=15.0 + (d % 4),
            vix_term_ratio=0.9,
            term_regime="contango",
            spy_drawdown_from_252d_high=0.01,
            spy_5ma_above_20ma=True,
        )

    from datetime import timedelta
    # Crash phase: 40 days of escalating VIX, dd, backwardation (Feb-Mar 2020)
    for d in range(40):
        detector.update(
            today=base + timedelta(days=30 + d),
            vix=20 + d * 1.5,                # ramps to 80
            vix_term_ratio=1.0 + d * 0.02,
            term_regime="backwardation",
            spy_drawdown_from_252d_high=0.05 + d * 0.005,  # ramps to 25%
            spy_5ma_above_20ma=False,
        )
    # At this point VIX peaked ~80, dd ~25%, state should be 'capitulation' or 'crash'
    assert detector.state in ("crash", "capitulation"), (
        f"after 40 days of escalating stress, state should be crash/capit, "
        f"got {detector.state!r}"
    )
    assert detector.vix_peak >= 70

    # Rebound phase: 10 days of VIX dropping, dd still > 10%, 5MA crossing up
    for d in range(10):
        detector.update(
            today=base + timedelta(days=70 + d),
            vix=80 - d * 5,                  # 80 → 30 (62% drop from peak)
            vix_term_ratio=0.95,
            term_regime="flat",
            spy_drawdown_from_252d_high=0.20,  # still 20% dd
            spy_5ma_above_20ma=True,           # 5MA up-cross
        )
    # Now state should be 'rebound'
    assert detector.state == "rebound", (
        f"after V-bottom signals, state should be 'rebound', got {detector.state!r}"
    )


# ============================================================
# Smoke: every registered strategy has a working on_data
# ============================================================

def test_every_registered_strategy_can_handle_empty_context():
    """Every strategy in ENABLED_STRATEGIES must NOT crash given an empty
    StrategyContext. Should return []."""
    from strategies import ENABLED_STRATEGIES
    empty_ctx = StrategyContext(timestamp=datetime(2024, 1, 2, 15, 0))
    for strat in ENABLED_STRATEGIES:
        try:
            signals = strat.on_data(empty_ctx)
            assert isinstance(signals, list), (
                f"{strat.id} returned {type(signals)} instead of list"
            )
        except Exception as e:
            raise AssertionError(f"{strat.id} crashed on empty ctx: {e}") from e
