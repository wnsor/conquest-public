"""MultiBetaResidualMomentum — average residual returns across multiple β-windows.

Motivation: ResidualMomentum with β-60 fails Train Sharpe; with β-252 fails
Holdout Sharpe. The β-window picks which regime gate to fail — classic bias-
variance tradeoff in the β estimate.

This wrapper computes residuals at multiple β-window lengths (e.g., 60/120/252)
and averages them before ranking. Smooths out the regime-specific failures of
single-window β estimates.

Reference: similar to "shrinkage estimator" approach in Bayesian regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model
from conquest.models.residual_momentum import ResidualMomentum


class MultiBetaResidualMomentum(Model):
    """Average residual cumulative returns across multiple β-window estimates.

    Args:
        market_ticker: column name for market proxy (default "SPY")
        top_n: top-N to hold (default 5)
        lookback: cumulative residual window (default 180)
        beta_windows: list of β-window sizes to average over.
            Default [60, 120, 252] — short/medium/long captures different regimes.
    """
    name = "multi_beta_residual_momentum"

    def __init__(
        self,
        market_ticker: str = "SPY",
        top_n: int = 5,
        lookback: int = 180,
        beta_windows: list[int] | None = None,
    ):
        self.market_ticker = market_ticker
        self.top_n = top_n
        self.lookback = lookback
        self.beta_windows = beta_windows or [60, 120, 252]

    def signal(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        vol: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # Run a ResidualMomentum at each β-window, accumulate cumulative-residual scores
        # (we re-implement here rather than calling the existing model, so we can average
        # the SCORES not the binary picks).
        if self.market_ticker in prices.columns:
            market_prices = prices[self.market_ticker]
            stock_cols = [c for c in prices.columns if c != self.market_ticker]
        else:
            market_prices = prices.mean(axis=1)
            stock_cols = list(prices.columns)

        stock_returns = prices[stock_cols].pct_change()
        market_returns = market_prices.pct_change()

        cum_resid_per_window = []
        for bw in self.beta_windows:
            var_m = market_returns.rolling(bw).var()
            cov = stock_returns.rolling(bw).cov(market_returns)
            beta = cov.div(var_m, axis=0).fillna(1.0).clip(-3.0, 3.0)
            market_broadcast = pd.DataFrame(
                np.broadcast_to(market_returns.values[:, None], stock_returns.shape),
                index=stock_returns.index, columns=stock_returns.columns,
            )
            residual = stock_returns - beta * market_broadcast
            log1p_resid = np.log1p(residual)
            cum_log_resid = log1p_resid.rolling(self.lookback, min_periods=self.lookback).sum()
            cum_resid_per_window.append(np.expm1(cum_log_resid))

        # Average cumulative residuals across β-windows
        avg_cum_resid = sum(cum_resid_per_window) / len(cum_resid_per_window)

        # Top-N positive
        positive = avg_cum_resid > 0
        ranks = avg_cum_resid.where(positive).rank(axis=1, ascending=False, method="first")
        mask = positive & (ranks <= self.top_n)
        n_per_row = mask.sum(axis=1)
        weights = mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)

        # Reindex to original prices columns
        full_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in weights.columns:
            full_weights[c] = weights[c]
        return full_weights
