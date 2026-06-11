"""LowVolFiltered - pre-filter universe to low-realized-vol names before base ranking.

Rationale: stock-momentum on a 500-name universe routinely concentrates in
high-vol single-stocks at trend tops (e.g., AI/semis 2024). Low-vol names
that ALSO have positive momentum tend to be steadier compounders (utilities
and staples in early-cycle, healthcare in mid-cycle, etc.). Filtering to
the lowest `keep_fraction` of names by trailing realized vol BEFORE
momentum ranking biases the strategy toward those names.

Mechanism: at each row, compute trailing 60d realized vol per ticker; keep
the bottom `keep_fraction` (e.g. 0.5 = bottom half = lowest-vol 50%); zero
out everyone else's weight after the base model has assigned weights.
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


class LowVolFiltered(Model):
    def __init__(
        self,
        base_model: Model,
        keep_fraction: float = 0.5,
        lookback: int = 60,
    ):
        if not (0 < keep_fraction <= 1):
            raise ValueError("keep_fraction must be in (0, 1]")
        if lookback < 20:
            raise ValueError("lookback must be >= 20 trading days for stable vol estimate")
        self.base = base_model
        self.keep_fraction = keep_fraction
        self.lookback = lookback
        self.name = f"{base_model.name}_low_vol_filtered"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        if self.keep_fraction >= 1.0:
            return weights

        rets = prices.pct_change()
        rolling_vol = rets.rolling(self.lookback, min_periods=self.lookback // 2).std()

        # Per row: rank tickers by vol ascending; keep the lowest `keep_fraction`.
        ranks = rolling_vol.rank(axis=1, method="first", ascending=True)
        n_valid = rolling_vol.notna().sum(axis=1)
        cutoff = (n_valid * self.keep_fraction).round().clip(lower=1).astype(int)
        # broadcast cutoff so each row's mask is "rank <= cutoff[row]"
        keep_mask = ranks.le(cutoff, axis=0)

        return weights.where(keep_mask, 0.0)
