"""Tests for the v22 max_per_trade_pct_nav override on StrategySignal.

When set, the override REPLACES config.base_pct_nav as the base allocation
AND lifts the portfolio cap. Lets crisis-window strategies (D2 Tepper,
CrisisRebound, D1 LEAPS) take 25% positions while per-trade-gate
strategies stay at the global 10%.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "conquest_options"))

from position_sizer import SizerConfig, size_position  # noqa: E402
from strategies.base import StrategySignal  # noqa: E402


def _signal(otm_pct=0.05, edge=1.0, max_pct_nav: float | None = None) -> StrategySignal:
    return StrategySignal(
        strategy_id="TEST",
        underlying="SPY",
        side="call",
        target_dte=35,
        edge_score=edge,
        target_otm_pct=otm_pct,
        take_profit_pct=2.0,
        stop_loss_pct=-0.5,
        max_per_trade_pct_nav=max_pct_nav,
    )


def test_default_cap_still_10pct():
    """No override → config.base_pct_nav (10%) applies as before."""
    cfg = SizerConfig(mode="flat", base_pct_nav=0.10, portfolio_cap_pct_nav=0.30)
    sig = _signal(max_pct_nav=None, edge=1.0)
    res = size_position(sig, contract_mid_price=2.00, nav=10_000, config=cfg)
    # 10% × $10k = $1000 cap, $200/contract → 5 contracts
    assert res.contracts == 5
    assert res.capital_committed_dollars == 1000.0


def test_override_25pct_lifts_allocation():
    """Override to 25% → 5x more capital allocated than default."""
    cfg = SizerConfig(mode="flat", base_pct_nav=0.10, portfolio_cap_pct_nav=0.30)
    sig = _signal(max_pct_nav=0.25, edge=1.0)
    res = size_position(sig, contract_mid_price=2.00, nav=10_000, config=cfg)
    # 25% × $10k = $2500 cap, $200/contract → 12 contracts
    assert res.contracts == 12, f"expected 12 with 25% cap, got {res.contracts}"
    assert res.capital_committed_dollars == 2400.0


def test_override_allows_expensive_contract_otherwise_unaffordable():
    """The D2 Tepper case: 180-DTE ATM SPY call costs ~$2500/contract.
    Default 10% cap = $1000 → 0 contracts (unaffordable).
    Override 25% cap = $2500 → 1 contract (affordable)."""
    cfg = SizerConfig(mode="flat", base_pct_nav=0.10, portfolio_cap_pct_nav=0.30,
                       min_premium_dollars=50.0)
    nav = 10_000

    # Without override: blocked
    sig_default = _signal(max_pct_nav=None, edge=1.0)
    res_def = size_position(sig_default, contract_mid_price=25.00, nav=nav, config=cfg)
    assert res_def.contracts == 0, (
        f"expected 0 contracts at 10% cap with $2500 premium, got {res_def.contracts}"
    )

    # With override: allowed
    sig_override = _signal(max_pct_nav=0.25, edge=1.0)
    res_or = size_position(sig_override, contract_mid_price=25.00, nav=nav, config=cfg)
    assert res_or.contracts == 1, (
        f"expected 1 contract at 25% cap with $2500 premium, got {res_or.contracts}"
    )


def test_override_lifts_portfolio_cap():
    """Override can exceed config.portfolio_cap_pct_nav (was a hard ceiling)."""
    cfg = SizerConfig(mode="flat", base_pct_nav=0.10, portfolio_cap_pct_nav=0.10)
    # default portfolio cap = 10%; signal wants 25%
    sig = _signal(max_pct_nav=0.25, edge=1.0)
    res = size_position(sig, contract_mid_price=20.00, nav=10_000, config=cfg)
    # Without v22 fix: cap would clamp to 10% = $1000 → 0 contracts at $2000/contract
    # With v22: portfolio_cap effectively becomes max(10%, 25%) = 25% → $2500 cap
    # → 1 contract
    assert res.contracts == 1


def test_edge_scaling_still_applies_to_override():
    """The edge_score scaling factor (0.5 + 0.5*edge) still applies on top
    of the override base."""
    cfg = SizerConfig(mode="flat", base_pct_nav=0.10, portfolio_cap_pct_nav=0.30)
    # edge=0.0 → scaled_pct = 0.25 * 0.5 = 12.5%
    sig_low = _signal(max_pct_nav=0.25, edge=0.0)
    res_low = size_position(sig_low, contract_mid_price=2.00, nav=10_000, config=cfg)
    # 12.5% × $10k = $1250, $200/contract → 6 contracts
    assert res_low.contracts == 6

    # edge=1.0 → scaled_pct = 0.25 * 1.0 = 25%
    sig_high = _signal(max_pct_nav=0.25, edge=1.0)
    res_high = size_position(sig_high, contract_mid_price=2.00, nav=10_000, config=cfg)
    assert res_high.contracts == 12


def test_d2_tepper_has_25pct_override():
    """Verify D2 Tepper actually sets max_per_trade_pct_nav=0.25."""
    from strategies.tepper_vbottom_leaps import TepperVbottomLeaps
    from strategies.base import StrategyContext
    from datetime import datetime
    d2 = TepperVbottomLeaps()
    ctx = StrategyContext(
        timestamp=datetime(2020, 4, 7, 15, 0),
        underlying_drawdown_from_252d_high={"SPY": 0.23},
        underlying_5ma_above_20ma={"SPY": True},
        vix=45.0,
        term_regime="backwardation",
    )
    signals = d2.on_data(ctx)
    assert len(signals) == 1
    assert signals[0].max_per_trade_pct_nav == 0.25, (
        f"D2 should set 25% cap, got {signals[0].max_per_trade_pct_nav}"
    )


def test_crisis_rebound_basket_per_leg_caps():
    """CrisisRebound basket: SPY/QQQ legs at 10%, name legs at 6%."""
    from strategies.crisis_rebound_basket import CrisisReboundBasket
    from strategies.base import StrategyContext
    from datetime import datetime
    basket = CrisisReboundBasket()
    ctx = StrategyContext(
        timestamp=datetime(2020, 4, 7, 15, 0),
        crisis_state="rebound",
        crisis_vix_peak=82.0,
    )
    signals = basket.on_data(ctx)
    assert len(signals) == 7
    for s in signals:
        if s.underlying in ("SPY", "QQQ"):
            assert s.max_per_trade_pct_nav == 0.10, (
                f"{s.underlying} should be 10%, got {s.max_per_trade_pct_nav}"
            )
        else:
            assert s.max_per_trade_pct_nav == 0.06, (
                f"{s.underlying} should be 6%, got {s.max_per_trade_pct_nav}"
            )


def test_cgrowth_leaps_has_25pct_override():
    """D1 cgrowth_leaps sets 25% cap for DITM LEAPS."""
    from strategies.cgrowth_leaps import CgrowthLeaps
    from strategies.base import StrategyContext
    from datetime import datetime
    d1 = CgrowthLeaps()
    ctx = StrategyContext(
        timestamp=datetime(2024, 6, 3, 15, 0),
        vix=14.0,
        underlying_momentum_60d={"AAPL": 1.08, "MSFT": 1.10},
        underlying_5ma_above_20ma={"AAPL": True, "MSFT": True},
        underlying_drawdown_from_252d_high={"AAPL": 0.05, "MSFT": 0.06},
        cstability_vote_count=0,
    )
    signals = d1.on_data(ctx)
    assert len(signals) >= 1, "D1 should fire on at least one cgrowth name with mom60>1.05 + 5MA>20MA"
    for s in signals:
        assert s.max_per_trade_pct_nav == 0.25, (
            f"{s.underlying} D1 LEAPS should be 25%, got {s.max_per_trade_pct_nav}"
        )
