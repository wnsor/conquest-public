"""AdxGated — wrap any base model; only hold names whose ADX exceeds `min_adx`.

Use case
--------
Trend-following strategies (`trend_follow`, `dual_momentum`) lose money in
chopping/sideways markets where indicator signals fire spuriously. ADX
measures *trend strength* independent of direction:
    ADX > 25  → trending
    ADX < 20  → chopping

Filtering the base model's selections through "ADX must be > min_adx" should
suppress whipsaws in flat regimes — a textbook gate for trend strategies.

Per-name ADX requires OHLC bars (not just close prices). The wrapper takes
``ohlc`` at construction time as a ``dict[ticker, DataFrame]`` (consistent
with how ``VixGated`` takes a vix series and ``RegimeGated`` takes a regime
series — gates inject their context via constructor, not via the signal call).
"""
from __future__ import annotations

import pandas as pd

from conquest.indicators.adx import adx
from conquest.models.base import Model


class AdxGated(Model):
    def __init__(
        self,
        base_model: Model,
        ohlc: dict[str, pd.DataFrame] | None = None,
        adx_period: int = 14,
        min_adx: float = 25.0,
    ):
        self.base = base_model
        self.ohlc = ohlc or {}
        self.adx_period = adx_period
        self.min_adx = min_adx
        self.name = f"{base_model.name}_adx_gated"

    def signal(self, prices, regime=None, vol=None):
        base_w = self.base.signal(prices, regime, vol)
        if not self.ohlc:
            return base_w  # no OHLC → pass through

        # Compute ADX per ticker, aligned to prices.index
        adx_cols: dict[str, pd.Series] = {}
        for ticker in prices.columns:
            if ticker not in self.ohlc:
                continue
            df = self.ohlc[ticker]
            try:
                a = adx(df, self.adx_period)
            except ValueError:
                # OHLC missing required columns for this ticker; skip the gate for it
                continue
            adx_cols[ticker] = a.reindex(prices.index).ffill()

        if not adx_cols:
            return base_w

        adx_df = pd.DataFrame(adx_cols).reindex(columns=prices.columns).fillna(0)
        trending = adx_df > self.min_adx
        return base_w.where(trending, 0)
