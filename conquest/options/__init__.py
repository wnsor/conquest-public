"""Options layer for protective-puts overlays on equity sleeves.

Pandas-vectorized Black-Scholes pricer + put-roll simulator + hedge PnL
aggregator + sizing modes + per-leg cost model. Designed to wrap an existing
equity backtest result (`conquest.backtest.engine.BacktestResult`) into a
`HedgedBacktestResult` via `conquest.backtest.hedged.hedged_backtest`.

Hypothesis being tested in v1: protective puts on SPY (proxy for cgrowth's
top-5 momentum sleeve) cap MaxDD around -15% and lift Calmar above v8.5
LIVE's 0.94, *after* paying the vol risk premium — Israelov 2014 critique
addressed by the VIX-conditional roll variant.
"""
from __future__ import annotations

from conquest.options.pricing import (
    BlackScholes,
    bs_put_price,
    bs_put_delta,
    bs_put_gamma,
    bs_put_vega,
)
from conquest.options.implied_vol import vix_to_spx_iv
from conquest.options.sizing import (
    Sizer,
    NotionalSizer,
    DeltaTargetSizer,
)
from conquest.options.costs import OptionsCostModel
from conquest.options.roll import (
    RollSchedule,
    ConstantRoll,
    VIXConditionalRoll,
    RegimeTriggeredRoll,
    PutRollSimulator,
)
from conquest.options.hedge import HedgePnL
from conquest.options.uoa import uoa_flag, uoa_flag_series

__all__ = [
    "BlackScholes",
    "bs_put_price",
    "bs_put_delta",
    "bs_put_gamma",
    "bs_put_vega",
    "vix_to_spx_iv",
    "Sizer",
    "NotionalSizer",
    "DeltaTargetSizer",
    "OptionsCostModel",
    "RollSchedule",
    "ConstantRoll",
    "VIXConditionalRoll",
    "RegimeTriggeredRoll",
    "PutRollSimulator",
    "HedgePnL",
    "uoa_flag",
    "uoa_flag_series",
]
