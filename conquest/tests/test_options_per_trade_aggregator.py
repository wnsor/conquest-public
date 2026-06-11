"""Unit tests for scripts/aggregate_per_trade_metrics.compute_strategy_metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make scripts/ importable for the test
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from aggregate_per_trade_metrics import (
    compute_strategy_metrics,
    passes_gate,
    _max_consecutive_true,
    _sortino,
)


def _journal_df(pnls):
    """Build a synthetic trade journal from a list of pnl_pct values."""
    return pd.DataFrame({
        "strategy_id": ["t"] * len(pnls),
        "pnl_pct": pnls,
        "r_multiple": pnls,
        "dte_at_open": [30] * len(pnls),
        "dte_at_close": [10] * len(pnls),  # 20-day hold each
    })


class TestComputeMetrics:
    def test_basic_metrics(self):
        # 5 wins of +50%, 3 losses of -30% → mean = 0.5*5 + (-0.3)*3 = 2.5 - 0.9 = 1.6 / 8 = 0.2 → 20%
        # PF = 2.5 / 0.9 ≈ 2.78
        # WR = 5/8 = 62.5%
        m = compute_strategy_metrics(_journal_df([0.5]*5 + [-0.3]*3), total_backtest_days=365)
        assert m["n_trades"] == 8
        assert abs(m["expectancy_pct"] - 20.0) < 0.01
        assert abs(m["profit_factor"] - 2.778) < 0.01
        assert abs(m["win_rate_pct"] - 62.5) < 0.01

    def test_max_loss_streak(self):
        # Pattern: win, loss, loss, win, loss, loss, loss, win
        pnls = [0.5, -0.3, -0.2, 0.1, -0.4, -0.5, -0.2, 0.3]
        m = compute_strategy_metrics(_journal_df(pnls), total_backtest_days=365)
        assert m["max_loss_streak"] == 3

    def test_empty_journal(self):
        m = compute_strategy_metrics(pd.DataFrame(), total_backtest_days=365)
        assert m == {"n_trades": 0}

    def test_all_wins_profit_factor_inf(self):
        m = compute_strategy_metrics(_journal_df([0.5, 0.3, 0.4]), total_backtest_days=365)
        # All wins → gross_losses = 0 → PF = inf
        import math
        assert math.isinf(m["profit_factor"])

    def test_time_in_market_pct(self):
        # 10 trades, 20-day hold each = 200 trade-days; 365 total → 54.8%
        m = compute_strategy_metrics(_journal_df([0.1]*10), total_backtest_days=365)
        assert abs(m["time_in_market_pct"] - 54.79) < 0.01


class TestPromotionGate:
    def test_pass_when_all_satisfied(self):
        m = {
            "n_trades": 100, "expectancy_pct": 20.0, "profit_factor": 2.5,
            "win_rate_pct": 45.0, "r_mean": 0.7, "sortino_per_trade": 2.5,
            "max_loss_streak": 5,
        }
        passed, fails = passes_gate(m)
        assert passed
        assert fails == []

    def test_fail_on_one_metric(self):
        m = {
            "n_trades": 100, "expectancy_pct": 10.0,  # below 15
            "profit_factor": 2.5, "win_rate_pct": 45.0, "r_mean": 0.7,
            "sortino_per_trade": 2.5, "max_loss_streak": 5,
        }
        passed, fails = passes_gate(m)
        assert not passed
        assert any("expectancy_pct" in f for f in fails)

    def test_fail_on_sample_size(self):
        m = {
            "n_trades": 30, "expectancy_pct": 30.0, "profit_factor": 3.0,
            "win_rate_pct": 50.0, "r_mean": 1.0, "sortino_per_trade": 3.0,
            "max_loss_streak": 3,
        }
        passed, fails = passes_gate(m)
        assert not passed
        assert any("n_trades" in f for f in fails)


class TestHelpers:
    def test_max_consecutive(self):
        assert _max_consecutive_true([False, True, True, False, True, True, True, False]) == 3
        assert _max_consecutive_true([]) == 0
        assert _max_consecutive_true([False, False]) == 0
        assert _max_consecutive_true([True, True, True]) == 3

    def test_sortino_handles_no_downside(self):
        assert _sortino(pd.Series([0.1, 0.2, 0.3])) is None

    def test_sortino_basic(self):
        s = pd.Series([0.1, -0.05, 0.2, -0.1, 0.15])
        v = _sortino(s)
        assert v is not None
        assert v > 0
