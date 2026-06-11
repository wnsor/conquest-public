"""TRIX — triple-smoothed exponential moving average rate-of-change.

Parity target: Lean's `self.TRIX(symbol, period)`.

Definition: TRIX_t = (EMA3_t - EMA3_{t-1}) / EMA3_{t-1}, where EMA3 is the EMA(period)
of the EMA(period) of the EMA(period) of the price series. Returned as a decimal
(e.g., 0.0012 for +12 bps).

Scale convention: Lean's `self.TRIX(symbol, period)` returns the same value
multiplied by 100 (i.e., percent form: 0.12 for +12 bps). Multiply this output
by 100 when comparing to Lean. The pandas API stays decimal so the bake-off
and research callers don't have to compensate.
"""
from __future__ import annotations
import pandas as pd


def trix(prices: pd.Series, period: int = 15) -> pd.Series:
    """Compute TRIX on a price series."""
    if period <= 0:
        raise ValueError("period must be > 0")
    e1 = prices.ewm(span=period, adjust=False).mean()
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return e3.pct_change().rename("trix")
