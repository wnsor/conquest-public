"""Volume-spike signal — today's dollar-volume vs trailing-N average.

Used in cspec's directional/activity filter (`vol_spike > 1.5x` to admit a
name into ranking) and as a composite-score axis (z-scored across the
universe at each rebalance).

We use **dollar volume** (close × volume) rather than share volume so the
signal isn't biased toward cheap stocks with naturally high share counts,
and so the spike threshold is comparable across price regimes.

Parity target: cspec/main.py runs the same arithmetic at rebalance time on
Lean's history slice. The pandas form is for research / backtest.
"""
from __future__ import annotations

import pandas as pd


def dollar_volume_spike(close: pd.Series, volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Today's $-volume divided by the prior `lookback`-day mean $-volume.

    Args:
        close: daily close prices (pd.Series indexed by date).
        volume: daily share volume.
        lookback: trailing window for the average (default 20).

    Returns:
        pd.Series with one entry per date. NaN until the rolling window has
        `lookback+1` valid points (`+1` because we exclude the current bar
        from the average).

    A value of 3.0 means today's $-volume is 3× the prior 20d average.
    """
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    dv = (close * volume).astype(float)
    avg = dv.rolling(window=lookback, min_periods=lookback).mean().shift(1)
    return (dv / avg).rename("dollar_volume_spike")


def volume_spike(volume: pd.Series, lookback: int = 20) -> pd.Series:
    """Share-volume version of the spike. Provided for completeness — cspec
    uses `dollar_volume_spike` by default."""
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    v = volume.astype(float)
    avg = v.rolling(window=lookback, min_periods=lookback).mean().shift(1)
    return (v / avg).rename("volume_spike")
