"""Hedged-backtest wrapper: equity sleeve (via `engine.backtest`) + put-roll overlay.

Composition rule:
    combined_ret_t = equity_ret_t + hedge_ret_t

`hedge_ret` already nets premium paid/received, MTM moves, and trading costs
(see conquest.options.hedge.HedgePnL). The equity backtest still applies its
own equity-side cost model.

Lookahead discipline:
- Contract count at roll-date *t* is set from NAV-through-*t-1*. The simulator
  receives `equity_nav` aligned daily; on each roll date it queries `nav_t`,
  but in practice the *change* at t (today's equity return) is already
  reflected — we use t-1 NAV via shift to keep this clean.
- IV at *t* uses VIX/VIX3M closes at *t* (same-day; this is fine because the
  Lean version reads the daily series after the close).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from conquest.backtest.engine import backtest, BacktestResult
from conquest.backtest.costs import IBCostModel
from conquest.options.roll import PutRollSimulator, RollSchedule
from conquest.options.sizing import Sizer, NotionalSizer
from conquest.options.costs import OptionsCostModel
from conquest.options.hedge import HedgePnL


@dataclass
class HedgedBacktestResult:
    equity: pd.Series           # combined NAV (equity + hedge)
    returns: pd.Series          # combined daily returns
    equity_returns_only: pd.Series   # equity-sleeve daily returns (no hedge)
    hedge_returns: pd.Series         # hedge daily return contribution
    gross_returns: pd.Series         # equity gross + hedge (no equity-side costs subtracted, hedge is already net)
    turnover: pd.Series              # equity-side turnover
    weights: pd.DataFrame            # equity weights actually used
    initial_capital: float
    roll_log: pd.DataFrame           # raw simulator output for diagnostics


def _rolling_beta(equity_ret: pd.Series, spy_ret: pd.Series, window: int = 60) -> pd.Series:
    """Rolling β of equity sleeve vs SPY, computed with `window` trading days.

    Returns a Series aligned to equity_ret.index. Uses 1.0 as the warmup
    fallback (conservative; matches the v8.5 top-5-momentum sleeve's
    long-run β).
    """
    aligned = pd.concat([equity_ret.rename("eq"), spy_ret.rename("sp")], axis=1).dropna()
    cov = aligned["eq"].rolling(window).cov(aligned["sp"])
    var = aligned["sp"].rolling(window).var()
    beta = (cov / var).reindex(equity_ret.index)
    return beta.fillna(1.0)


def hedged_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    spx: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
    sizer: Sizer,
    roll_schedule: RollSchedule,
    options_cost_model: Optional[OptionsCostModel] = None,
    equity_cost_model: Optional[IBCostModel] = None,
    initial_capital: float = 100_000,
    rebalance_freq: Optional[str] = "ME",
    beta_window: int = 60,
    strike_offset: float = -0.05,
    skew_per_5pct_otm: float = 2.0,
) -> HedgedBacktestResult:
    """Run an equity backtest, overlay a put-roll, return combined result.

    Args:
        prices: date × symbol equity close prices.
        signals: date × symbol target weights.
        spx: SPY (or ^GSPC) daily close — the put underlying.
        vix, vix3m: VIX term structure for IV.
        sizer: Sizer instance.
        roll_schedule: RollSchedule (ConstantRoll or VIXConditionalRoll).
        options_cost_model: defaults to OptionsCostModel().
        equity_cost_model: defaults to IBCostModel(2 bps).
        initial_capital: starting NAV (applied to combined curve).
        rebalance_freq: passed through to equity engine.
        beta_window: rolling-β window for sizing (default 60 trading days).
        strike_offset: (K-S)/S; default -0.05 (5% OTM put). -0.10 = 10% OTM (cheaper).
        skew_per_5pct_otm: vol points added per 5%-OTM step (linear in |offset|).
    """
    options_cost_model = options_cost_model or OptionsCostModel()

    # 1) Run the equity sleeve as-is.
    eq = backtest(
        prices=prices,
        signals=signals,
        cost_model=equity_cost_model,
        initial_capital=initial_capital,
        rebalance_freq=rebalance_freq,
    )

    # 2) Compute rolling beta of equity returns vs SPY.
    spy_ret = spx.pct_change().reindex(eq.returns.index).fillna(0.0)
    beta_series = _rolling_beta(eq.returns, spy_ret, window=beta_window)

    # 3) Run put-roll simulation, sized off equity NAV.
    sim = PutRollSimulator(
        tenor_days=roll_schedule.tenor_days,
        strike_offset=strike_offset,
        skew_per_5pct_otm=skew_per_5pct_otm,
    )
    roll_log = sim.simulate(
        spx=spx.reindex(eq.returns.index).ffill(),
        vix=vix.reindex(eq.returns.index).ffill(),
        vix3m=vix3m.reindex(eq.returns.index).ffill(),
        equity_nav=eq.equity,
        equity_beta=beta_series,
        sizer=sizer,
        schedule=roll_schedule,
        cost_model=options_cost_model,
    )

    # 4) Convert roll log → daily return contribution.
    hedge_ret = HedgePnL.from_roll(roll_log, eq.equity).reindex(eq.returns.index).fillna(0.0)

    # 5) Combine and re-cumulate.
    combined_ret = eq.returns.add(hedge_ret, fill_value=0.0)
    combined_equity = (1 + combined_ret).cumprod() * initial_capital

    return HedgedBacktestResult(
        equity=combined_equity,
        returns=combined_ret,
        equity_returns_only=eq.returns,
        hedge_returns=hedge_ret,
        gross_returns=eq.gross_returns + hedge_ret,
        turnover=eq.turnover,
        weights=eq.weights,
        initial_capital=initial_capital,
        roll_log=roll_log,
    )
