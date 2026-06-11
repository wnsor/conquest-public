"""SMA — simple moving average.

Parity target: Lean's `self.SMA(symbol, period)`.
"""
from __future__ import annotations
import pandas as pd


def sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple moving average over `period` bars (NaN for the first period-1 bars)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return prices.rolling(window=period, min_periods=period).mean().rename("sma")
