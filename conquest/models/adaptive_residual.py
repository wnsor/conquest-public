"""Adaptive Residual Momentum — three regime-conditioning approaches.

Motivation: single-β residual momentum has a regime-dependent train/holdout
tradeoff. β=60 fails train, β=252 fails holdout. Hypothesis: residual signal
is MORE valuable in high-vol regimes (where β shifts rapidly) and LESS valuable
in low-vol regimes (where total-return momentum is cleaner).

Three approaches tested:

A) `VolAdaptiveResidualMomentum`
   - residual_weight = sigmoid((realized_vol - threshold) / scale)
   - Low realized vol → low residual weight (lean on total-return momentum)
   - High realized vol → high residual weight (strip beta during stress)

B) `TwoStageMomentum`
   - Stage 1: top_n_stage1 by total-return momentum
   - Stage 2: among those, top_k_stage2 by residual momentum
   - Picks names that went up AND went up idiosyncratically

C) `RegimeAdaptiveResidualMomentum`
   - residual_weight depends on macro regime probability (P(stress))
   - Stress regime → heavier residual; calm regime → heavier total
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from conquest.models.base import Model


def _compute_residuals(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    beta_window: int,
) -> pd.DataFrame:
    """Per-stock residual returns from rolling CAPM regression."""
    var_m = market_returns.rolling(beta_window).var()
    cov = stock_returns.rolling(beta_window).cov(market_returns)
    beta = cov.div(var_m, axis=0).fillna(1.0).clip(-3.0, 3.0)
    market_broadcast = pd.DataFrame(
        np.broadcast_to(market_returns.values[:, None], stock_returns.shape),
        index=stock_returns.index, columns=stock_returns.columns,
    )
    return stock_returns - beta * market_broadcast


def _cumulative(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    log1p = np.log1p(returns)
    return np.expm1(log1p.rolling(lookback, min_periods=lookback).sum())


def _split_market(prices: pd.DataFrame, market_ticker: str):
    if market_ticker in prices.columns:
        market = prices[market_ticker]
        stock_cols = [c for c in prices.columns if c != market_ticker]
    else:
        market = prices.mean(axis=1)
        stock_cols = list(prices.columns)
    return prices[stock_cols], market


def _select_top_n(scores: pd.DataFrame, n: int, require_positive: pd.DataFrame | None = None) -> pd.DataFrame:
    if require_positive is None:
        positive = scores > -np.inf
    else:
        positive = require_positive
    ranks = scores.where(positive).rank(axis=1, ascending=False, method="first")
    mask = positive & (ranks <= n)
    n_per_row = mask.sum(axis=1)
    return mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)


class VolAdaptiveResidualMomentum(Model):
    """Residual weight scales with realized portfolio vol (proxy via SPY vol).

    residual_weight_t = sigmoid((vol_t - vol_pivot) / vol_scale)
    blended_score_t = (1 - rw_t) * total_rank_t + rw_t * residual_rank_t
    """
    name = "vol_adaptive_residual"

    def __init__(
        self,
        market_ticker: str = "SPY",
        top_n: int = 5,
        lookback: int = 180,
        beta_window: int = 120,
        vol_window: int = 21,
        vol_pivot: float = 0.012,   # 1.2%/day = ~19% annualized (mid-VIX)
        vol_scale: float = 0.005,   # sigmoid steepness
    ):
        self.market_ticker = market_ticker
        self.top_n = top_n
        self.lookback = lookback
        self.beta_window = beta_window
        self.vol_window = vol_window
        self.vol_pivot = vol_pivot
        self.vol_scale = vol_scale

    def signal(self, prices, regime=None, vol=None):
        stock_prices, market_prices = _split_market(prices, self.market_ticker)
        stock_returns = stock_prices.pct_change()
        market_returns = market_prices.pct_change()

        # Total return cumulative
        total_cum = _cumulative(stock_returns, self.lookback)
        # Residual return cumulative
        residual_returns = _compute_residuals(stock_returns, market_returns, self.beta_window)
        resid_cum = _cumulative(residual_returns, self.lookback)

        # Realized market vol (rolling std of daily SPY returns)
        realized_vol = market_returns.rolling(self.vol_window).std()

        # Sigmoid residual weight, indexed daily
        rw = 1 / (1 + np.exp(-(realized_vol - self.vol_pivot) / self.vol_scale))
        rw = rw.fillna(0.5)  # default 50/50 during warmup

        # Cross-sectional ranks (percentile, 0 to 1)
        total_rank = total_cum.rank(axis=1, ascending=True, pct=True)
        resid_rank = resid_cum.rank(axis=1, ascending=True, pct=True)

        # Daily blended score
        rw_broadcast = pd.DataFrame(
            np.broadcast_to(rw.values[:, None], total_rank.shape),
            index=total_rank.index, columns=total_rank.columns,
        )
        blend = (1 - rw_broadcast) * total_rank + rw_broadcast * resid_rank

        positive = (total_cum > 0) & (resid_cum > 0)
        weights = _select_top_n(blend, self.top_n, require_positive=positive)
        full = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in weights.columns:
            full[c] = weights[c]
        return full


class TwoStageMomentum(Model):
    """Two-stage filter: top_n_stage1 by total return, then top_k_stage2 by residual.

    Picks stocks that went up AND went up idiosyncratically. Avoids the rank-blend
    confusion of hybrid models — the two filters operate sequentially.
    """
    name = "two_stage_momentum"

    def __init__(
        self,
        market_ticker: str = "SPY",
        top_n_stage1: int = 20,
        top_k_stage2: int = 5,
        lookback: int = 180,
        beta_window: int = 120,
    ):
        self.market_ticker = market_ticker
        self.top_n_stage1 = top_n_stage1
        self.top_k_stage2 = top_k_stage2
        self.lookback = lookback
        self.beta_window = beta_window

    def signal(self, prices, regime=None, vol=None):
        stock_prices, market_prices = _split_market(prices, self.market_ticker)
        stock_returns = stock_prices.pct_change()
        market_returns = market_prices.pct_change()

        total_cum = _cumulative(stock_returns, self.lookback)
        residual_returns = _compute_residuals(stock_returns, market_returns, self.beta_window)
        resid_cum = _cumulative(residual_returns, self.lookback)

        # Stage 1: top_n by total return (positive only)
        positive_total = total_cum > 0
        stage1_ranks = total_cum.where(positive_total).rank(axis=1, ascending=False, method="first")
        stage1_mask = positive_total & (stage1_ranks <= self.top_n_stage1)

        # Stage 2: among stage1 picks, top_k by residual return (positive only)
        # Set residual to -inf for stocks NOT in stage1 so they get pushed to bottom
        stage1_resid = resid_cum.where(stage1_mask)
        positive_resid = resid_cum > 0
        stage2_mask = stage1_mask & positive_resid
        stage2_ranks = stage1_resid.where(stage2_mask).rank(axis=1, ascending=False, method="first")
        final_mask = stage2_mask & (stage2_ranks <= self.top_k_stage2)

        n_per_row = final_mask.sum(axis=1)
        weights = final_mask.div(n_per_row.replace(0, np.nan), axis=0).fillna(0)
        full = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in weights.columns:
            full[c] = weights[c]
        return full


class RegimeAdaptiveResidualMomentum(Model):
    """Residual weight depends on macro stress probability.

    Stress regime (high P(stagflation) or P(deflation)) → heavier residual weight.
    Calm regime → heavier total return weight.

    Pass `regime` arg to .signal() as a Series of stress probability in [0,1].
    Falls back to 0.5/0.5 weighting if no regime series provided.
    """
    name = "regime_adaptive_residual"

    def __init__(
        self,
        market_ticker: str = "SPY",
        top_n: int = 5,
        lookback: int = 180,
        beta_window: int = 120,
        stress_pivot: float = 0.40,
        stress_scale: float = 0.10,
    ):
        self.market_ticker = market_ticker
        self.top_n = top_n
        self.lookback = lookback
        self.beta_window = beta_window
        self.stress_pivot = stress_pivot
        self.stress_scale = stress_scale

    def signal(self, prices, regime=None, vol=None):
        stock_prices, market_prices = _split_market(prices, self.market_ticker)
        stock_returns = stock_prices.pct_change()
        market_returns = market_prices.pct_change()

        total_cum = _cumulative(stock_returns, self.lookback)
        residual_returns = _compute_residuals(stock_returns, market_returns, self.beta_window)
        resid_cum = _cumulative(residual_returns, self.lookback)

        # Residual weight from regime stress
        if regime is None:
            stress = pd.Series(0.5, index=prices.index)
        else:
            stress = regime.reindex(prices.index, method="ffill").fillna(0.5)
        rw = 1 / (1 + np.exp(-(stress - self.stress_pivot) / self.stress_scale))

        total_rank = total_cum.rank(axis=1, ascending=True, pct=True)
        resid_rank = resid_cum.rank(axis=1, ascending=True, pct=True)
        rw_broadcast = pd.DataFrame(
            np.broadcast_to(rw.values[:, None], total_rank.shape),
            index=total_rank.index, columns=total_rank.columns,
        )
        blend = (1 - rw_broadcast) * total_rank + rw_broadcast * resid_rank

        positive = (total_cum > 0) & (resid_cum > 0)
        weights = _select_top_n(blend, self.top_n, require_positive=positive)
        full = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for c in weights.columns:
            full[c] = weights[c]
        return full
