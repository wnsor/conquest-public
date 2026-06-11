"""Implied-volatility proxy for SPX/SPY options from VIX term structure.

The plan
--------
We need an IV input to price SPY puts at arbitrary tenor & strike, daily,
across 2010-2024. We don't have a full SPX vol surface — but we do have
VIX (~30d ATM) and VIX3M (~93d ATM) from `conquest.data.vix_term`. So:

    σ(T)        = sqrt-T interp between VIX and VIX3M
    σ(T, strike) = σ(T) + skew_premium(strike)   # fixed +2 vol pts at -5% OTM

Limitations (documented for the SC1 sanity check):
- VIX is SPX index vol, not SPY ETF vol — small basis (SPY ≈ 0.99 × SPX).
- Skew is hard-coded as +2 vol points at -5% OTM (defensible for "normal"
  regime; widens to +5-8 in stress). One knob — calibrate via SC1.
- Beyond 93d we clamp to VIX3M (no extrapolation). Fine for our 21d / 63d
  tenors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Standard "anchor" tenors for VIX / VIX3M in trading days.
VIX_TENOR_DAYS = 21    # VIX targets 30 calendar days ≈ 21 trading days
VIX3M_TENOR_DAYS = 63  # VIX3M targets 93 calendar days ≈ 63 trading days


def vix_to_spx_iv(
    vix: pd.Series,
    vix3m: pd.Series,
    tenor_days: int,
    strike_offset: float = -0.05,
    skew_per_5pct_otm: float = 2.0,
) -> pd.Series:
    """Daily IV estimate for an SPX put of given tenor and strike offset.

    Args:
        vix: ^VIX close, in vol points (e.g. 18.0 means 18% annualized).
        vix3m: ^VIX3M close, same scale.
        tenor_days: trading-day tenor of the option (21 = 1mo, 63 = 3mo).
        strike_offset: (K-S)/S; -0.05 = 5% OTM put.
        skew_per_5pct_otm: vol points added per 5%-OTM step. Default 2.0 vol pts
            per 5% OTM. Linear in |offset|.

    Returns:
        Series indexed like the intersection of vix/vix3m, in vol points (e.g.
        18.0 not 0.18). Caller divides by 100 before passing to BS pricer.
    """
    common = vix.index.intersection(vix3m.index)
    v = vix.reindex(common).astype(float)
    v3 = vix3m.reindex(common).astype(float)

    # sqrt-T interpolation in variance (more correct than linear in vol)
    if tenor_days <= VIX_TENOR_DAYS:
        sigma_atm = v
    elif tenor_days >= VIX3M_TENOR_DAYS:
        sigma_atm = v3
    else:
        # weight on VIX3M ∈ [0, 1] from sqrt-T position
        t = np.sqrt(tenor_days)
        t1 = np.sqrt(VIX_TENOR_DAYS)
        t2 = np.sqrt(VIX3M_TENOR_DAYS)
        w = (t - t1) / (t2 - t1)
        # Variance interp: v(T)^2 = (1-w) v1^2 + w v2^2
        var = (1 - w) * v ** 2 + w * v3 ** 2
        sigma_atm = np.sqrt(var)

    # Skew premium: linear in |strike_offset|, normalized per 5% OTM
    skew = skew_per_5pct_otm * abs(strike_offset) / 0.05
    sigma = sigma_atm + skew
    return sigma.rename(f"iv_{tenor_days}d_{int(strike_offset * 100)}pct")
