"""Trend Follow — MACD bull + TRIX>0 gate, then rank by `momp_period`-day MOMP, top-N equal-weight.

Pandas-vectorized counterpart of the cgrowth Lean Algorithm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.indicators import macd, trix, momp
from conquest.models.base import Model


class TrendFollow(Model):
    name = "trend_follow"

    def __init__(
        self,
        top_n: int = 3,
        momp_period: int = 90,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        trix_period: int = 15,
    ):
        self.top_n = top_n
        self.momp_period = momp_period
        self.macd_params = (macd_fast, macd_slow, macd_signal)
        self.trix_period = trix_period

    def signal(self, prices, regime=None, vol=None):
        macd_hist = prices.apply(lambda c: macd(c, *self.macd_params)["histogram"])
        trix_df = prices.apply(lambda c: trix(c, self.trix_period))
        momp_df = prices.apply(lambda c: momp(c, self.momp_period))

        passes = (macd_hist > 0) & (trix_df > 0)
        ranks = momp_df.where(passes).rank(axis=1, ascending=False, method="first")
        mask = passes & (ranks <= self.top_n)

        n_per_row = mask.sum(axis=1)
        weights = mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)
        return weights
