"""Vectorized daily backtest engine.

Convention
----------
Weights are set at the close of day t-1 and held through day t. Daily portfolio
return = sum(weight_{t-1} * symbol_return_t). Costs are charged on weight changes
(turnover) on day t-1 and amortized into that day's net return.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from conquest.backtest.costs import IBCostModel


@dataclass
class BacktestResult:
    equity: pd.Series           # NAV time series
    returns: pd.Series          # daily NET returns
    gross_returns: pd.Series    # daily GROSS returns (no costs)
    turnover: pd.Series         # daily turnover (sum |Δw|)
    weights: pd.DataFrame       # forward-filled weights actually used
    initial_capital: float


def _resample_to_freq(weights: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Keep only the LAST weight in each period (per `freq` offset alias); ffill in between."""
    rebal_mask = pd.Series(False, index=weights.index)
    for _, group in weights.groupby(pd.Grouper(freq=freq)):
        if len(group) > 0:
            rebal_mask.loc[group.index[-1]] = True
    out = weights.where(rebal_mask)
    return out.ffill().fillna(0)


def backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    cost_model: IBCostModel | None = None,
    initial_capital: float = 100_000,
    rebalance_freq: str | None = None,
) -> BacktestResult:
    """Run a vectorized backtest.

    Args:
        prices: date x symbol close prices.
        signals: date x symbol target weights. NaN treated as 0.
        cost_model: defaults to IBCostModel(2 bps).
        initial_capital: starting NAV.
        rebalance_freq: pandas offset alias, e.g. "M" (monthly), "W" (weekly), "Q".
            If set, only the last `signals` value in each period is applied (ffill between).
            If None, signals are used as-is (caller responsible for cadence).
    """
    cost_model = cost_model or IBCostModel()

    if not prices.index.equals(signals.index):
        signals = signals.reindex(prices.index)
    weights = signals.fillna(0)

    if rebalance_freq:
        weights = _resample_to_freq(weights, rebalance_freq)

    rets = prices.pct_change().fillna(0)
    lagged_w = weights.shift(1).fillna(0)
    gross_returns = (lagged_w * rets).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    cost_fraction = cost_model.cost_fraction(turnover)
    net_returns = gross_returns - cost_fraction
    equity = (1 + net_returns).cumprod() * initial_capital

    return BacktestResult(
        equity=equity,
        returns=net_returns,
        gross_returns=gross_returns,
        turnover=turnover,
        weights=weights,
        initial_capital=initial_capital,
    )
