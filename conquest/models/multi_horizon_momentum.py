"""MultiHorizonMomentum - dual_momentum but require multiple lookbacks to agree.

Standard dual_momentum ranks names by 252d return and keeps top-N where
return > 0. This variant additionally requires the 60d AND 180d returns to
also be positive, before considering a name eligible for the top-N. Names
that pass all three momentum gates are then ranked by the longest horizon
(252d) for selection.

Why: requiring confirmation across short, medium, and long horizons filters
out names that are mean-reverting on shorter timescales (false positives
during regime transitions). Tighter entry criterion -> fewer trades, better
trade quality. Pinned at (60, 180, 252) - no hyperparameter search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.indicators import momp
from conquest.models.base import Model


class MultiHorizonMomentum(Model):
    name = "multi_horizon_momentum"

    def __init__(
        self,
        top_n: int = 20,
        horizons: tuple[int, ...] = (60, 180, 252),
    ):
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if len(horizons) < 2:
            raise ValueError("horizons must be at least 2 lookbacks")
        if any(h < 5 for h in horizons):
            raise ValueError("horizons must each be >= 5 trading days")
        self.top_n = top_n
        self.horizons = tuple(sorted(horizons))
        self.rank_horizon = self.horizons[-1]

    def signal(self, prices, regime=None, vol=None):
        positive_all = pd.DataFrame(True, index=prices.index, columns=prices.columns)
        for h in self.horizons:
            r = prices.apply(lambda c: momp(c, h))
            positive_all &= r > 0

        rank_ret = prices.apply(lambda c: momp(c, self.rank_horizon))
        ranks = rank_ret.where(positive_all).rank(axis=1, ascending=False, method="first")
        mask = positive_all & (ranks <= self.top_n)

        n_per_row = mask.sum(axis=1)
        weights = mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)
        return weights
