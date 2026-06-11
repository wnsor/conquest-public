"""Faber GTAA-5 — Mebane Faber, "A Quantitative Approach to Tactical Asset Allocation".

Reference: Faber 2007, SSRN 962461. Classic 10-month SMA cross strategy on
5 broad asset classes. Each asset gets equal weight (1/N) when its price is
above its 10-month SMA, else allocated to cash (or kept at 0 if no cash asset).

Universe (5 broad asset classes):
    - SPY  (US large-cap)
    - EFA  (Developed international)
    - VNQ  (REIT)
    - PDBC (Commodities; or DBC if PDBC unavailable)
    - IEF  (Intermediate Treasuries; bond proxy)

Why the SMA cross
-----------------
A 10-month SMA on monthly closes is the textbook trend filter. Originally
proposed in 1973 (Faber points to even earlier sources), it captures broad
trend regimes without curve-fitting the lookback. The "go to cash if below
SMA" rule provides crash protection — Faber's original 2007 backtest showed
significantly lower drawdowns than buy-and-hold for the same return.

This is *not* a forecasting model — it's purely reactive. Strength: simple,
robust, easy to verify. Weakness: lags regime shifts by ~half the SMA window.

Anti-overfit notes
------------------
- 10-month SMA is the canonical Faber number; do not tune.
- Equal weight (1/N) on top-of-SMA assets, no scoring, no top-N selection.
- Rebalance monthly (Faber spec).
- "Cash" defaults to BIL/SHV (or just zeroing out the position).
"""
from __future__ import annotations

import pandas as pd

from conquest.models.base import Model


GTAA_5_UNIVERSE = ["SPY", "EFA", "VNQ", "PDBC", "IEF"]


class FaberGTAA(Model):
    """Faber GTAA: hold 1/N each asset when price > N-month SMA, else cash/zero.

    Args:
        universe: ETF tickers (default: 5 broad asset classes).
        sma_months: SMA window in calendar months (default: 10, Faber spec).
        cash_ticker: ticker to allocate to when an asset is below SMA. If None
            or not present in price columns, the slice goes to zero (gross
            exposure shrinks).
    """
    name = "faber_gtaa"

    def __init__(
        self,
        universe: list[str] | None = None,
        sma_months: int = 10,
        cash_ticker: str | None = None,
    ):
        self.universe = universe or list(GTAA_5_UNIVERSE)
        self.sma_months = sma_months
        self.cash_ticker = cash_ticker
        # Convert months to trading days (~21 bd/mo)
        self.sma_window_days = sma_months * 21

    def signal(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        vol: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        missing = set(self.universe) - set(prices.columns)
        if missing:
            raise ValueError(f"FaberGTAA: missing required tickers: {missing}")

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        slice_w = 1.0 / len(self.universe)

        # SMA per asset
        for t in self.universe:
            p = prices[t]
            sma = p.rolling(self.sma_window_days, min_periods=self.sma_window_days).mean()
            above = (p > sma).reindex(weights.index, fill_value=False)
            # Where above SMA: this asset gets its slice
            weights.loc[above, t] = slice_w
            # Where below SMA: slice goes to cash if available
            below = ~above
            if self.cash_ticker and self.cash_ticker in prices.columns:
                weights.loc[below, self.cash_ticker] = (
                    weights.loc[below, self.cash_ticker] + slice_w
                )
            # else: leave slice at 0 (gross exposure shrinks below 1.0)

        return weights
