"""VolTargeted - portfolio-level volatility targeting wrapper.

Scales the base model's weights so that the ex-ante annualized portfolio
volatility matches a target. Cap is hardcoded at 1.0x gross because the
project does not take on margin.

Ex-ante vol is estimated from the past `lookback` days of returns on the
SELECTED names (those with nonzero weight at each row), via the standard
sqrt(w' Sigma w) decomposition. Computing covariance only over the active
subset keeps this fast even on a 500-name universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model


class VolTargeted(Model):
    def __init__(
        self,
        base_model: Model,
        target: float = 0.10,
        lookback: int = 60,
        cap: float = 1.0,
    ):
        if not (0 < target < 1):
            raise ValueError("target must be in (0, 1) — annualized vol fraction")
        if lookback < 20:
            raise ValueError("lookback must be >= 20 trading days for stable cov")
        if cap > 1.0:
            raise ValueError("cap must be <= 1.0 — project rule, no margin")
        self.base = base_model
        self.target = target
        self.lookback = lookback
        self.cap = cap
        self.name = f"{base_model.name}_vol_targeted"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        rets = prices.pct_change()
        scale = pd.Series(1.0, index=weights.index)
        for i in range(self.lookback, len(weights)):
            w_row = weights.iloc[i]
            active = w_row[w_row.ne(0)]
            if active.empty:
                continue
            cols = active.index
            past = rets.iloc[i - self.lookback:i][cols]
            past = past.dropna(how="all")
            if len(past) < self.lookback // 2:
                continue
            cov_ann = past.cov().values * 252.0
            cov_ann = np.nan_to_num(cov_ann, nan=0.0)
            wv = active.values
            ex_ante_var = float(wv @ cov_ann @ wv)
            if ex_ante_var <= 0:
                continue
            ex_ante_vol = np.sqrt(ex_ante_var)
            scale.iloc[i] = min(self.target / ex_ante_vol, self.cap)
        return weights.mul(scale, axis=0)
