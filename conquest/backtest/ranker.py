"""Run multiple models on the same prices and return a ranked DataFrame of metrics."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from conquest.models.base import Model
from conquest.backtest.costs import IBCostModel
from conquest.backtest.engine import backtest, BacktestResult
from conquest.backtest.metrics import metrics_summary


def rank_models(
    models: Iterable[Model],
    prices: pd.DataFrame,
    cost_model: IBCostModel | None = None,
    initial_capital: float = 100_000,
    rebalance_freq: str | None = "ME",
    sort_by: str = "sharpe",
    benchmark: str | None = "SPY",
    bootstrap_n_iter: int = 500,
    bootstrap_block_size: int = 1,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Bake-off: evaluate every model on `prices`, return (ranked metrics, per-model results).

    Default `rebalance_freq="ME"` (month-end) matches the cstability cadence; override as needed.
    `benchmark`: ticker present in `prices` to use for v5 relative metrics
    (beta / up_capture / down_capture). Pass None to skip those columns.
    `bootstrap_n_iter`: iterations for the Sharpe CI bootstrap (~500 is fast).
    `bootstrap_block_size`: 1 = iid bootstrap (back-compat); >1 (e.g., 21) enables
    moving-block bootstrap to capture daily-return autocorrelation. v8+ work uses 21.
    """
    bench_returns: pd.Series | None = None
    if benchmark and benchmark in prices.columns:
        bench_returns = prices[benchmark].pct_change().fillna(0)

    rows: list[dict] = []
    results: dict[str, BacktestResult] = {}
    for model in models:
        signals = model.signal(prices)
        result = backtest(prices, signals, cost_model, initial_capital, rebalance_freq)
        results[model.name] = result
        m = metrics_summary(
            result.returns,
            result.equity,
            result.turnover,
            benchmark_returns=bench_returns,
            bootstrap_n_iter=bootstrap_n_iter,
            weights=result.weights,
            bootstrap_block_size=bootstrap_block_size,
        )
        rows.append({"model": model.name, **m})
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values(sort_by, ascending=False), results
