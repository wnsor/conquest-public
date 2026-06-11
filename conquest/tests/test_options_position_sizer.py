"""Unit tests for conquest_options.position_sizer."""
from __future__ import annotations

from position_sizer import SizerConfig, size_position
from strategies.base import StrategySignal


def _signal(edge: float = 1.0) -> StrategySignal:
    return StrategySignal(
        strategy_id="t",
        underlying="SPY",
        side="call",
        target_dte=30,
        edge_score=edge,
        target_delta=0.5,
    )


class TestFlatSizer:
    def test_basic_sizing(self):
        # NAV $10k, base 1.5%, edge=1.0 → $150 target. Premium $5/share → $500 cost.
        # $150 / $500 = 0 contracts (floor). Should be skipped.
        cfg = SizerConfig(mode="flat", base_pct_nav=0.015)
        r = size_position(_signal(1.0), contract_mid_price=5.0, nav=10_000, config=cfg)
        assert r.contracts == 0
        assert r.skipped_reason is not None

    def test_sized_when_premium_fits(self):
        # NAV $10k, base 5%, edge=1.0 → $500 target. Premium $1/share → $100 cost.
        # $500 / $100 = 5 contracts.
        cfg = SizerConfig(mode="flat", base_pct_nav=0.05)
        r = size_position(_signal(1.0), contract_mid_price=1.0, nav=10_000, config=cfg)
        assert r.contracts == 5
        assert r.capital_committed_dollars == 500.0
        assert r.skipped_reason is None

    def test_edge_score_scales_sizing(self):
        cfg = SizerConfig(mode="flat", base_pct_nav=0.10, min_premium_dollars=0)
        # edge=0.0 → 0.5×base = 5% NAV = $500 → 5 contracts at $100
        r0 = size_position(_signal(0.0), contract_mid_price=1.0, nav=10_000, config=cfg)
        # edge=1.0 → 1.0×base = 10% NAV = $1000 → 10 contracts
        r1 = size_position(_signal(1.0), contract_mid_price=1.0, nav=10_000, config=cfg)
        assert r0.contracts == 5
        assert r1.contracts == 10

    def test_portfolio_cap_caps_sizing(self):
        # base 20% NAV, cap 10% → cap binds
        cfg = SizerConfig(mode="flat", base_pct_nav=0.20, portfolio_cap_pct_nav=0.10,
                          min_premium_dollars=0)
        r = size_position(_signal(1.0), contract_mid_price=1.0, nav=10_000, config=cfg)
        assert r.contracts == 10  # $1000 cap / $100 per contract

    def test_min_premium_floor_skips(self):
        # NAV $100, target $1.50 < min_premium_dollars $50 → skip
        cfg = SizerConfig(mode="flat", base_pct_nav=0.015, min_premium_dollars=50)
        r = size_position(_signal(1.0), contract_mid_price=1.0, nav=100, config=cfg)
        assert r.contracts == 0
        assert "floor" in (r.skipped_reason or "").lower()

    def test_non_positive_premium_skips(self):
        cfg = SizerConfig(mode="flat")
        r = size_position(_signal(1.0), contract_mid_price=0, nav=10_000, config=cfg)
        assert r.contracts == 0


class TestKellySizer:
    def test_kelly_needs_history(self):
        cfg = SizerConfig(mode="kelly")
        r = size_position(_signal(1.0), contract_mid_price=1.0, nav=10_000, config=cfg)
        assert r.contracts == 0
        assert "kelly" in (r.skipped_reason or "").lower()

    def test_kelly_positive_edge_sizes(self):
        # p=0.5, b=3 → full Kelly = (3*0.5 - 0.5)/3 = 1/3 = 0.333
        # cap_fraction=0.25 → 0.0833; * (0.5+0.5*1) = 0.0833 of NAV
        # NAV $10k → $833 target; premium $100/contract → 8 contracts
        cfg = SizerConfig(mode="kelly", kelly_cap_fraction=0.25, portfolio_cap_pct_nav=0.50,
                          min_premium_dollars=0)
        hist = {"win_rate": 0.5, "win_loss_ratio": 3.0}
        r = size_position(_signal(1.0), contract_mid_price=1.0, nav=10_000,
                          config=cfg, strategy_history=hist)
        assert r.contracts == 8

    def test_kelly_negative_edge_zero_sized(self):
        # p=0.3, b=1 → full Kelly = (0.3 - 0.7)/1 = -0.4 → clamp to 0
        cfg = SizerConfig(mode="kelly", min_premium_dollars=0)
        hist = {"win_rate": 0.3, "win_loss_ratio": 1.0}
        r = size_position(_signal(1.0), contract_mid_price=1.0, nav=10_000,
                          config=cfg, strategy_history=hist)
        assert r.contracts == 0
