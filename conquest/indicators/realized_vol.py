"""Realized volatility — annualized standard deviation of log returns.

Returns sigma_n = sqrt(252) * std(log_returns over `period` bars).
NaN for the first `period` bars. Use in research / training pipelines;
inside Lean Algorithms use `self.STD(symbol, period)` (raw stddev of prices,
non-annualized — different scale).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def realized_vol(prices: pd.Series, period: int, annualize: bool = True) -> pd.Series:
    """Annualized realized vol of log returns over `period` bars."""
    if period <= 1:
        raise ValueError("period must be > 1")
    log_returns = np.log(prices / prices.shift(1))
    sigma = log_returns.rolling(window=period, min_periods=period).std()
    if annualize:
        sigma = sigma * np.sqrt(252)
    return sigma.rename(f"realized_vol_{period}d")
