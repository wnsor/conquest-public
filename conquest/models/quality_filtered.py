"""QualityFiltered - pre-filter universe to high-quality names before base ranking.

"Quality" here is a price-derived proxy: trailing realized Sharpe of each name's
own daily returns over `lookback` days. High-Sharpe stocks have steady positive
drift with low volatility - the empirical signature of "quality" companies in
the Quality Minus Junk literature (Frazzini, Pedersen).

Mechanism: at each row, rank tickers by trailing Sharpe descending. Keep the
TOP `keep_fraction` (e.g., 0.3 = top 30%). Then mask the base model's weights
to zero on names that didn't survive the quality filter, AND renormalize the
surviving weights so gross stays at 1.0x (in contrast to LowVolFiltered which
intentionally lets gross drop). This gives "concentrate in quality momentum
names" rather than "go to cash when momentum picks low-quality names."

Why a price-derived proxy:
- No fundamentals API needed (yfinance Ticker.info is current-snapshot only,
  introduces survivorship bias when applied historically).
- No data dependency for a backtest spanning 11 years.
- Quality factor literature shows trailing Sharpe is a reasonable proxy for
  the multi-factor quality composite (ROE / low-debt / earnings stability).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model


class QualityFiltered(Model):
    def __init__(
        self,
        base_model: Model,
        keep_fraction: float = 0.3,
        lookback: int = 252,
        renormalize: bool = True,
    ):
        if not (0 < keep_fraction <= 1):
            raise ValueError("keep_fraction must be in (0, 1]")
        if lookback < 60:
            raise ValueError("lookback must be >= 60 trading days for stable Sharpe estimate")
        self.base = base_model
        self.keep_fraction = keep_fraction
        self.lookback = lookback
        self.renormalize = renormalize
        self.name = f"{base_model.name}_quality_filtered"

    def signal(self, prices, regime=None, vol=None):
        weights = self.base.signal(prices, regime, vol)
        if self.keep_fraction >= 1.0:
            return weights

        rets = prices.pct_change()
        rolling_mean = rets.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        rolling_std = rets.rolling(self.lookback, min_periods=self.lookback // 2).std()
        # Annualized Sharpe per stock per day
        rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

        # Per row: keep top `keep_fraction` by trailing Sharpe (descending: 1 = best)
        ranks = rolling_sharpe.rank(axis=1, method="first", ascending=False)
        n_valid = rolling_sharpe.notna().sum(axis=1)
        cutoff = (n_valid * self.keep_fraction).round().clip(lower=1).astype(int)
        keep_mask = ranks.le(cutoff, axis=0)

        filtered = weights.where(keep_mask, 0.0)

        if self.renormalize:
            orig_gross = weights.abs().sum(axis=1)
            new_gross = filtered.abs().sum(axis=1).replace(0, np.nan)
            scale = (orig_gross / new_gross).fillna(0)
            filtered = filtered.mul(scale, axis=0)
        return filtered
