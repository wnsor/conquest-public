"""Residual Momentum — rank stocks by IDIOSYNCRATIC return, not total return.

Reference: Blitz, Huij, Martens (2011) "Residual Momentum", J. Empirical Finance.
Reported result: ~2x risk-adjusted profits vs total-return momentum, with more
consistent performance across regimes and less concentration in extreme deciles.

Methodology (CAPM-residual v1; FF3 extension is straightforward later)
----------------------------------------------------------------------
For each stock and each date t:

  1. Rolling regression over the last `beta_window` business days (default 60):
        stock_return_τ = α + β · market_return_τ + ε_τ
  2. Residual return: r_resid_τ = stock_return_τ - β · market_return_τ
     (we drop α to avoid in-sample fitting; Blitz uses pure residuals)
  3. Cumulative residual over `lookback`: ∏(1 + r_resid_τ) - 1, last `lookback` days.
  4. Rank stocks by cumulative residual return; take top-N where residual > 0.

Why this beats total-return momentum
------------------------------------
Top-5 by 180d total return concentrates in high-β names that just rallied.
In 2020, that was concentrated growth tech (NVDA, AAPL, etc.) — they ran
because the MARKET ran, not because of stock-specific edge. When the market
flipped (March 2020), high-β positions got hammered.

Residual momentum strips the market component. Stocks that ranked high on
residuals went up MORE than their β predicted — genuine alpha. They're more
likely to keep outperforming idiosyncratically.

Anti-overfit notes
------------------
- Single new parameter: `beta_window` (default 60d, standard practitioner value).
- Reuses existing `lookback` (180d for v8.5 parity).
- No extra data dependencies for v1 (CAPM uses SPY which is already in panel).
- FF3 v2 would add SMB + HML factors (Fama-French daily file from Ken French).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model


class ResidualMomentum(Model):
    """Rank by residual (CAPM-stripped) cumulative return; hold top-N if positive.

    Args:
        market_ticker: column name in `prices` to use as the market proxy.
            Default "SPY". If not present, falls back to equal-weighted average
            of all columns (mathematically the simplest "market" but noisier
            than SPY for an S&P 500 universe).
        top_n: number of stocks to hold (matches DualMomentum default).
        lookback: residual-cumulation window in trading days (default 180 for
            v8.5 parity).
        beta_window: rolling regression window in trading days (default 60 —
            standard practitioner value; 30 too noisy, 120 too slow).
    """
    name = "residual_momentum"

    def __init__(
        self,
        market_ticker: str = "SPY",
        top_n: int = 5,
        lookback: int = 180,
        beta_window: int = 60,
    ):
        self.market_ticker = market_ticker
        self.top_n = top_n
        self.lookback = lookback
        self.beta_window = beta_window

    def signal(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        vol: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Identify market series. Prefer the named ticker; else equal-weight average.
        if self.market_ticker in prices.columns:
            market_prices = prices[self.market_ticker]
            stock_cols = [c for c in prices.columns if c != self.market_ticker]
        else:
            market_prices = prices.mean(axis=1)
            stock_cols = list(prices.columns)

        # Daily returns
        stock_returns = prices[stock_cols].pct_change()
        market_returns = market_prices.pct_change()

        # Rolling β per stock vs market (pandas rolling cov/var is vectorized)
        # cov(stock, market) / var(market), window = beta_window
        var_m = market_returns.rolling(self.beta_window).var()
        # broadcast: covariance of each stock column with market
        cov = stock_returns.rolling(self.beta_window).cov(market_returns)
        beta = cov.div(var_m, axis=0).fillna(1.0).clip(-3.0, 3.0)

        # Residual return per day per stock
        # r_resid = r_stock - β * r_market
        market_broadcast = pd.DataFrame(
            np.broadcast_to(market_returns.values[:, None], stock_returns.shape),
            index=stock_returns.index, columns=stock_returns.columns,
        )
        residual = stock_returns - beta * market_broadcast

        # Cumulative residual return over `lookback` window:
        #   (1 + r_resid_τ) compounded, minus 1
        # Use log-sum trick for numerical stability.
        # Don't fillna before log1p — let NaN from beta-warmup propagate so the
        # rolling sum requires `lookback` valid values (full warmup = beta_window + lookback).
        log1p_resid = np.log1p(residual)
        cum_log_resid = log1p_resid.rolling(self.lookback, min_periods=self.lookback).sum()
        cum_resid = np.expm1(cum_log_resid)

        # Rank: top-N positive cumulative residual
        positive = cum_resid > 0
        ranks = cum_resid.where(positive).rank(axis=1, ascending=False, method="first")
        mask = positive & (ranks <= self.top_n)
        n_per_row = mask.sum(axis=1)
        weights = mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)

        # Reindex to original prices columns (add back market with weight 0 if it was dropped)
        full_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in weights.columns:
            full_weights[c] = weights[c]
        return full_weights
