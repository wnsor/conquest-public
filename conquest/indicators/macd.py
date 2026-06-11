"""MACD — Moving Average Convergence Divergence with EMA smoothing.

Parity target: Lean's `self.MACD(symbol, fast, slow, signal, MovingAverageType.EXPONENTIAL)`.
Pandas EWM `span=N, adjust=False` matches Lean's standard EMA exactly: alpha = 2/(N+1),
recursive, seeded with the first observation.
"""
from __future__ import annotations
import pandas as pd


def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD on a price series.

    Args:
        prices: closing prices indexed by date.
        fast:   period of the fast EMA (default 12).
        slow:   period of the slow EMA (default 26).
        signal: period of the signal-line EMA over the MACD line (default 9).

    Returns:
        DataFrame with columns ["line", "signal", "histogram"], same index as `prices`.
    """
    if not (0 < fast < slow):
        raise ValueError("require 0 < fast < slow")
    if signal <= 0:
        raise ValueError("signal must be > 0")
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"line": line, "signal": sig, "histogram": line - sig})
