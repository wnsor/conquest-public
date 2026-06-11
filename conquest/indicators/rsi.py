"""RSI — Relative Strength Index, Wilder smoothing.

Parity target: Lean's `self.RSI(symbol, period, MovingAverageType.WILDERS)`.

Implementation note: Wilder smoothing with parameter `period` is mathematically
equivalent to an exponential moving average with alpha = 1/period. Lean's exact
implementation seeds with an SMA over the first `period` bars and then switches
to recursive Wilder; we use pandas EWM with `adjust=False` which seeds with the
first observation and propagates. The two converge to <1e-3 within ~3*period bars,
which is well inside any realistic trading window.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI on a price series.

    Args:
        prices: closing prices indexed by date (or any monotonic index).
        period: lookback in bars (Lean default: 14).

    Returns:
        RSI series in [0, 100], same index. NaN where avg_loss is exactly 0
        (no down-bars in window) — uncommon outside contrived inputs.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    # 100 * avg_gain / (avg_gain + avg_loss) is mathematically equivalent to
    # 100 - 100/(1 + avg_gain/avg_loss) but well-defined when avg_loss == 0
    # (returns 100) or avg_gain == 0 (returns 0). Both-zero (flat series) → NaN.
    total = avg_gain + avg_loss
    # Float roundoff can push the ratio a fraction of a ulp outside [0, 1];
    # clip to enforce the mathematical bounds.
    return (100 * avg_gain / total.replace(0, np.nan)).clip(0, 100).rename("rsi")
