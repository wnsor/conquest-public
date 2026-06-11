"""Fractional Kelly sizer.

For continuous returns the optimal-growth (full-Kelly) fraction is::

    f* = μ_excess / σ²

In a multi-asset, zero-correlation simplification each name's weight is f*ᵢ
independently, then the row is rescaled to a leverage cap.

Why fractional (default ½)?
    Full Kelly maximises expected log-wealth in the limit but is famously
    volatile in practice. μ̂ is noisily estimated (over any reasonable window),
    and full Kelly amplifies that noise into wild swings — drawdowns get worse,
    not better, despite higher expected log-growth. Half-Kelly captures most
    of the asymptotic growth advantage with much lower variance and far less
    sensitivity to μ̂ noise. Quarter-Kelly is even safer; some practitioners
    tune fraction in [0.25, 0.5] empirically per strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_weights(
    expected_returns: pd.DataFrame,
    vol: pd.DataFrame,
    fraction: float = 0.5,
    leverage_cap: float = 1.5,
    long_only: bool = True,
) -> pd.DataFrame:
    """Per-name fractional-Kelly target weights.

    weights[i] = fraction × expected_returns[i] / vol[i]²

    Args:
        expected_returns: date x symbol annualized μ̂.
        vol:              date x symbol annualized σ̂ (must be > 0).
        fraction:         0.5 = half-Kelly (default; safer than full).
        leverage_cap:     max sum of |weights| per row; rows over cap get scaled down.
        long_only:        clip negative weights to 0 (no shorting).
    """
    raw = fraction * expected_returns / (vol ** 2)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    if long_only:
        raw = raw.clip(lower=0)
    abs_sum = raw.abs().sum(axis=1)
    over_cap = abs_sum > leverage_cap
    scale = pd.Series(1.0, index=raw.index)
    scale.loc[over_cap] = leverage_cap / abs_sum.loc[over_cap]
    return raw.mul(scale, axis=0)
