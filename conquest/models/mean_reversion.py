"""RSI Mean Reversion — long when RSI crosses below `oversold`, exit when above `exit_level`.

Stateful per-symbol: enter at oversold, hold until exit_level. Across active longs:
equal weight, capped at `max_positions` (most-oversold first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.indicators import rsi
from conquest.models.base import Model


class MeanReversion(Model):
    name = "mean_reversion"

    def __init__(
        self,
        oversold: float = 30.0,
        exit_level: float = 50.0,
        max_positions: int = 5,
        rsi_period: int = 14,
    ):
        self.oversold = oversold
        self.exit_level = exit_level
        self.max_positions = max_positions
        self.rsi_period = rsi_period

    def _entry_state(self, r: pd.Series) -> pd.Series:
        """Return a Bool series: True on dates we're currently long this symbol."""
        held = pd.Series(False, index=r.index)
        flag = False
        for i, v in enumerate(r.values):
            if pd.isna(v):
                held.iloc[i] = flag
                continue
            if not flag and v < self.oversold:
                flag = True
            elif flag and v > self.exit_level:
                flag = False
            held.iloc[i] = flag
        return held

    def signal(self, prices, regime=None, vol=None):
        rsi_df = prices.apply(lambda c: rsi(c, self.rsi_period))
        in_pos = pd.DataFrame(
            {sym: self._entry_state(rsi_df[sym]) for sym in prices.columns},
            index=prices.index,
        )

        # Among active longs, prefer most-oversold (lowest RSI) → take up to max_positions
        # Vectorized: mask out inactive, rank ascending by RSI, keep top max_positions per row
        rsi_active = rsi_df.where(in_pos)
        ranks = rsi_active.rank(axis=1, ascending=True, method="first")
        mask = in_pos & (ranks <= self.max_positions)

        n_per_row = mask.sum(axis=1)
        weights = mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)
        return weights
