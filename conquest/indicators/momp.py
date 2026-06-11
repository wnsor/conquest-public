"""MOMP — momentum percent.

Parity target: Lean's `self.MOMP(symbol, period)`.
Returns (price_t - price_{t-period}) / price_{t-period} as a decimal.

Scale convention: Lean's `self.MOMP(symbol, period)` returns the same value
multiplied by 100 (i.e., percent form). Multiply this output by 100 when
comparing to Lean. The pandas API stays decimal so the bake-off and research
callers don't have to compensate.
"""
from __future__ import annotations
import pandas as pd


def momp(prices: pd.Series, period: int) -> pd.Series:
    """Percent momentum over `period` bars (NaN for the first `period` bars)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return prices.pct_change(periods=period).rename("momp")
