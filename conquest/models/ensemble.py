"""Ensemble — average weights across constituent models.

Default constituents: MomentumConsensus + TrendFollow + DualMomentum. Each
sub-model's weights row already sums to <=1, so the average sums to <=1 as well.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from conquest.models.base import Model
from conquest.models.momentum_consensus import MomentumConsensus
from conquest.models.trend_follow import TrendFollow
from conquest.models.dual_momentum import DualMomentum


class Ensemble(Model):
    name = "ensemble"

    def __init__(self, top_n: int = 5, sub_models: Sequence[Model] | None = None):
        self.subs = list(sub_models) if sub_models is not None else [
            MomentumConsensus(top_n=top_n),
            TrendFollow(top_n=top_n),
            DualMomentum(top_n=top_n),
        ]

    def signal(self, prices, regime=None, vol=None):
        sub_weights = [m.signal(prices, regime, vol) for m in self.subs]
        avg = sum(sub_weights) / len(sub_weights)
        return avg
