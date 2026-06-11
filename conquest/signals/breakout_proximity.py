"""52-week-high breakout-proximity signal.

`breakout_proximity(close, lookback)` returns `close / rolling_max(close,
lookback).shift(1)` — i.e. how close today's close is to the highest close
over the prior `lookback` bars.

- 1.0 means today's close equals the prior `lookback`-day max.
- > 1.0 means today's close is above the prior max — a fresh N-day high.
- < 1.0 means we're below the high (e.g. 0.85 = 15% below).

Used as a smooth analog to the binary "broke 52-week high" flag — captures
both freshly-breaking-out names (~1.0+) and consolidation-near-highs names
(~0.95-1.0). Rank-friendlier than a binary threshold.

The shift(1) excludes the current bar from the max so today's close can
exceed the lookback max. cspec uses `lookback=252` (~52 weeks).
"""
from __future__ import annotations

import pandas as pd


def breakout_proximity(close: pd.Series, lookback: int = 252) -> pd.Series:
    """Today's close / (prior `lookback`-day max). NaN until the rolling
    window has `lookback+1` valid points.
    """
    if lookback <= 1:
        raise ValueError("lookback must be > 1")
    rolling_max = close.rolling(window=lookback, min_periods=lookback).max().shift(1)
    return (close / rolling_max).rename("breakout_proximity")
