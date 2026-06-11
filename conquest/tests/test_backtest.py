"""Sanity tests for conquest.backtest — engine + costs + metrics + ranker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.backtest import IBCostModel, backtest, metrics_summary, rank_models
from conquest.models import EqualWeight, all_models


@pytest.fixture
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    cols = ["A", "B", "C"]
    data = {c: (1 + 0.0003 + rng.normal(0, 0.01, len(dates))).cumprod() * 100 for c in cols}
    return pd.DataFrame(data, index=dates)


def test_zero_signal_equity_flat(prices):
    """All-zero weights ⇒ NAV stays at initial capital, no costs incurred."""
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    res = backtest(prices, signals, IBCostModel())
    assert res.equity.iloc[0] == pytest.approx(100_000)
    assert res.equity.iloc[-1] == pytest.approx(100_000)
    assert (res.returns.abs() < 1e-12).all()


def test_single_asset_full_weight_tracks_asset(prices):
    """100% weight in A every day ⇒ portfolio return ≈ A's return (minus a tiny first-day cost)."""
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    signals["A"] = 1.0
    res = backtest(prices, signals, cost_model=IBCostModel(bps_per_turnover=0.0))
    a_total = prices["A"].iloc[-1] / prices["A"].iloc[0] - 1
    portfolio_total = res.equity.iloc[-1] / res.equity.iloc[0] - 1
    assert portfolio_total == pytest.approx(a_total, rel=1e-6)


def test_costs_reduce_returns(prices):
    """A non-zero cost model should reduce net returns vs gross when there's turnover."""
    signals = EqualWeight().signal(prices)
    # Force monthly rebalance to generate turnover
    res = backtest(prices, signals, cost_model=IBCostModel(bps_per_turnover=10.0), rebalance_freq="ME")
    assert (res.returns <= res.gross_returns + 1e-12).all()
    # At least one bar must have a real cost charge
    assert (res.gross_returns - res.returns).max() > 0


def test_equity_starts_at_initial_capital(prices):
    signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    res = backtest(prices, signals, initial_capital=250_000)
    assert res.equity.iloc[0] == pytest.approx(250_000)


def test_metrics_summary_keys(prices):
    signals = EqualWeight().signal(prices)
    res = backtest(prices, signals)
    m = metrics_summary(res.returns, res.equity, res.turnover)
    # Check the v0 base keys are always present (subset check; v4+ may add more)
    base_keys = {
        "annual_return", "annual_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "hit_rate", "profit_factor", "turnover_annual",
    }
    assert base_keys.issubset(m.keys())


def test_max_drawdown_non_positive():
    """Equity series with a dip ⇒ drawdown is negative; equity series only rising ⇒ 0."""
    from conquest.backtest.metrics import max_drawdown
    rising = pd.Series(np.linspace(100, 200, 100))
    assert max_drawdown(rising) == pytest.approx(0.0)
    dipping = pd.Series([100, 110, 90, 100, 110])
    assert max_drawdown(dipping) < 0


def test_rank_models_returns_sorted(prices):
    df, results = rank_models(all_models(), prices, sort_by="sharpe", rebalance_freq="ME")
    # Must be sorted descending by sharpe
    sharpes = df["sharpe"].dropna()
    assert (sharpes.diff().dropna() <= 1e-12).all()
    # Each model represented in `results`
    assert set(results.keys()) == {m.name for m in all_models()}
