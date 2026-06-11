"""Realized volatility from price returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(
    prices: pd.DataFrame,
    lookback: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """Rolling realized volatility per symbol from log returns.

    Args:
        prices: date x symbol close prices.
        lookback: rolling window in trading days.
        annualize: scale to annual via sqrt(252).

    Returns:
        DataFrame same shape as `prices`; NaN before the window is full.
    """
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    log_rets = np.log(prices / prices.shift(1))
    vol = log_rets.rolling(lookback).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
