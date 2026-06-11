"""Portfolio concentration metrics — Herfindahl-Hirschman Index and friends.

These operate on the time-varying weights DataFrame produced by the backtest
engine. Concentration is the *flip side* of diversification: a strategy with
"5 holdings" might actually be 80% in one name (HHI ≈ 0.65, effective N ≈ 1.5).

Background
----------
HHI was originally an antitrust market-concentration metric. Adapted to
portfolios:
    HHI = Σᵢ wᵢ²        (sum of squared portfolio weights)

Range: [1/N, 1]. Lower = more diversified.
- 100% in one name: HHI = 1.0, effective N = 1
- Equal-weight 5 names: HHI = 0.20, effective N = 5
- Equal-weight 19 names: HHI ≈ 0.053, effective N ≈ 19

The squared weights make HHI sensitive to a small number of large positions —
adding a 30% position increases HHI more than adding ten 3% positions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hhi_series(weights: pd.DataFrame) -> pd.Series:
    """Per-bar HHI = Σᵢ wᵢ². Bars where the row sums to 0 (all cash) get NaN."""
    sq = (weights.abs() ** 2).sum(axis=1)
    row_invested = weights.abs().sum(axis=1) > 1e-12
    return sq.where(row_invested, np.nan)


def avg_hhi(weights: pd.DataFrame) -> float:
    """Mean HHI across all *invested* bars (bars where any weight is non-zero)."""
    s = hhi_series(weights).dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())


def max_hhi(weights: pd.DataFrame) -> float:
    """Worst (highest) HHI observed — peak concentration."""
    s = hhi_series(weights).dropna()
    if s.empty:
        return float("nan")
    return float(s.max())


def effective_n_series(weights: pd.DataFrame) -> pd.Series:
    """Per-bar 1/HHI = 'effective number of equally-weighted holdings'."""
    h = hhi_series(weights)
    return (1.0 / h.replace(0, np.nan)).rename("effective_n")


def avg_effective_n(weights: pd.DataFrame) -> float:
    """Average effective number of holdings across invested bars."""
    s = effective_n_series(weights).dropna()
    if s.empty:
        return float("nan")
    return float(s.mean())


def max_single_name_weight(weights: pd.DataFrame) -> float:
    """Largest single-name weight observed at any time. A complementary check
    to HHI — flags the worst single concentration even if HHI is moderate."""
    if weights.empty:
        return float("nan")
    return float(weights.abs().max().max())
