"""Dual Momentum — rank by 12-month return, hold top-N, reject names with negative momentum.

The "dual" gate: relative momentum (rank vs peers) AND absolute momentum (return > 0).
Classic Antonacci dual-momentum framing, single asset class.

Allocation across the top-N selected names supports two modes:
- 'equal' (default): 1/N each — the v0-v8.5 LIVE behaviour.
- 'rank':  linear-decreasing weight by momentum rank (top-1 gets the largest
  share, top-N the smallest). With top_n=5 the weights are (5,4,3,2,1)/15.
  Concentrates exposure in the strongest-signal names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.indicators import momp
from conquest.models.base import Model


class DualMomentum(Model):
    name = "dual_momentum"

    def __init__(self, top_n: int = 5, lookback: int = 252, weighting: str = "equal"):
        if weighting not in ("equal", "rank"):
            raise ValueError("weighting must be 'equal' or 'rank'")
        self.top_n = top_n
        self.lookback = lookback
        self.weighting = weighting
        if weighting == "rank":
            self.name = f"dual_momentum_rank"

    def signal(self, prices, regime=None, vol=None):
        ret = prices.apply(lambda c: momp(c, self.lookback))
        positive = ret > 0
        ranks = ret.where(positive).rank(axis=1, ascending=False, method="first")
        mask = positive & (ranks <= self.top_n)

        if self.weighting == "equal":
            n_per_row = mask.sum(axis=1)
            return mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)

        # Rank-weighted: w_i ∝ (top_n + 1 - rank_i) for selected names; renormalise per row
        rank_weight = (self.top_n + 1 - ranks).where(mask, 0).fillna(0)
        row_sum = rank_weight.sum(axis=1)
        return rank_weight.div(row_sum.replace(0, np.nan), axis=0).fillna(0)
