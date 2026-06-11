"""Momentum Consensus — RSI>50 + MACD bull + TRIX>0 vote, top-N equal-weight.

This is the pandas-vectorized counterpart of the cstability Lean Algorithm. The
Algorithm uses Lean's event-driven indicators; this module uses conquest.indicators
(which target Lean parity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.indicators import rsi, macd, trix
from conquest.models.base import Model


class MomentumConsensus(Model):
    name = "momentum_consensus"

    def __init__(
        self,
        top_n: int = 5,
        min_score: int = 2,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        trix_period: int = 15,
    ):
        self.top_n = top_n
        self.min_score = min_score
        self.rsi_period = rsi_period
        self.macd_params = (macd_fast, macd_slow, macd_signal)
        self.trix_period = trix_period

    def signal(self, prices, regime=None, vol=None):
        rsi_df = prices.apply(lambda c: rsi(c, self.rsi_period))
        macd_hist = prices.apply(
            lambda c: macd(c, *self.macd_params)["histogram"]
        )
        trix_df = prices.apply(lambda c: trix(c, self.trix_period))

        score = (
            (rsi_df > 50).astype(int)
            + (macd_hist > 0).astype(int)
            + (trix_df > 0).astype(int)
        ).astype(float)

        # Tiebreak with RSI level (small additive perturbation, won't change integer score order)
        composite = score + rsi_df.fillna(0) / 1000.0
        ranks = composite.rank(axis=1, ascending=False, method="first")
        mask = (score >= self.min_score) & (ranks <= self.top_n)

        n_per_row = mask.sum(axis=1)
        # Need at least 2 active to diversify (matches cstability Algorithm)
        n_per_row = n_per_row.where(n_per_row >= 2, np.nan)
        weights = mask.div(n_per_row, axis=0).fillna(0)
        return weights
