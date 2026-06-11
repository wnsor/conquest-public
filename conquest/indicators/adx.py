"""Average Directional Index (ADX), pandas-vectorized.

ADX measures trend *strength* (regardless of direction). The standard reads:
    ADX > 25  → trending market (use trend strategies)
    ADX < 20  → chopping / mean-reverting market (avoid trend strategies)

Implementation matches Lean's built-in ADX (Wilder smoothing). Inputs are
OHLC bars; we use High, Low, Close (Open isn't part of the formula).

Components computed and exposed:
    +DI, -DI    — directional indicators (price action up vs down)
    DX          — |+DI - -DI| / (+DI + -DI), normalized 0-100
    ADX         — Wilder-smoothed DX over `period`
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adx(
    ohlc: pd.DataFrame,
    period: int = 14,
    return_components: bool = False,
) -> pd.Series | pd.DataFrame:
    """Compute ADX from OHLC bars.

    Args:
        ohlc: DataFrame with columns including 'High', 'Low', 'Close' (case-sensitive).
        period: Wilder smoothing period (default 14, the textbook value).
        return_components: if True, return a DataFrame with adx, plus_di, minus_di, dx.
                           Otherwise just the ADX Series.

    Returns:
        Series of ADX values clipped to [0, 100], or a 4-column DataFrame if
        ``return_components=True``.
    """
    if period <= 1:
        raise ValueError("period must be > 1")
    needed = {"High", "Low", "Close"}
    missing = needed - set(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc missing columns: {missing}")

    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range = max(H-L, |H-prev_close|, |L-prev_close|)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=ohlc.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=ohlc.index,
    )

    # Wilder's smoothing == EMA with alpha = 1/period (adjust=False)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean().clip(0, 100).rename("adx")

    if return_components:
        return pd.DataFrame(
            {
                "adx": adx_val,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "dx": dx,
            }
        )
    return adx_val
