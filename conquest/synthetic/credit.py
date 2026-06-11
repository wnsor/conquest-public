"""Synthesize HYG/LQD daily prices from credit-spread inputs.

Model
-----
A corporate-bond ETF's total return decomposes into:

  r_bond[t] = duration * (-Δy_t) + carry_t + credit_drift_t

For high-yield (HYG-like):
  Δy = Δ(5Y_Treasury) + Δ(HY-Treasury_spread)
  carry = (5Y_yield + HY_spread) / 252
  duration ≈ 4.0 years (HYG empirical)
  credit_drift ≈ -0.5 * default_rate * loss_given_default / 252  (default cushion)

For investment-grade (LQD-like):
  Δy = Δ(7Y_Treasury) + Δ(BBB-AAA_spread)
  carry = (7Y_yield + IG_spread) / 252
  duration ≈ 8.5 years (LQD empirical)
  credit_drift ≈ 0  (negligible default rate)

Inputs from FRED (all back to 1953+ for Aaa/Baa; T-yields back to 1962):
- AAA  = Moody's seasoned Aaa corporate yield (monthly)
- BAA  = Moody's seasoned Baa corporate yield (monthly)
- DGS5 = 5-year Treasury constant maturity (daily, 1962+)
- DGS7 = 7-year Treasury (daily, 1969+)

HY spread proxy: use BAA-AAA spread scaled. Real HY spread averages ~2.5x
the IG spread (Baa-Aaa). Calibration on 2007-2026 where real HYG exists:
  HY_spread_synth = (BAA - AAA) * scale + intercept
  → fit scale ~2.7, intercept ~3.5 (in pct points)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def synthesize_credit_etf(
    treasury_yield: pd.Series,            # annualized, in pct
    credit_spread: pd.Series,             # annualized, in pct points
    duration_years: float = 4.0,
    default_rate_annual: float = 0.04,
    lgd: float = 0.65,
    initial_nav: float = 100.0,
) -> pd.Series:
    """Synthesize daily credit-bond ETF total-return series.

    Parameters
    ----------
    treasury_yield
        Daily forward-filled risk-free yield (pct annualized).
    credit_spread
        Daily forward-filled spread (pct points annualized).
    duration_years
        Effective duration (HYG ≈ 4.0; LQD ≈ 8.5).
    default_rate_annual
        Annual default rate (HY ~4%; IG ~0.5%).
    lgd
        Loss given default (0.65 for HY senior unsecured; 0.40 IG).
    initial_nav
        Starting NAV (cosmetic).

    Returns
    -------
    Daily NAV series (price + reinvested coupon, like an ETF).
    """
    df = pd.DataFrame({"y": treasury_yield, "s": credit_spread}).dropna()
    df = df.sort_index()

    # Total yield = risk-free + spread (annualized %)
    df["total_yield"] = df["y"] + df["s"]

    # Daily price change from yield change: ΔP/P ≈ -duration * Δy
    df["dy"] = df["y"].diff().fillna(0.0)
    df["ds"] = df["s"].diff().fillna(0.0)
    df["price_change"] = -duration_years * (df["dy"] + df["ds"]) / 100  # /100 to convert pct → fraction

    # Daily carry: total_yield / 252
    df["carry"] = df["total_yield"] / 100 / 252

    # Default drag: amortized over the year
    df["default_drag"] = -default_rate_annual * lgd / 252

    # Daily total return
    df["r"] = df["price_change"] + df["carry"] + df["default_drag"]

    nav = initial_nav * (1 + df["r"]).cumprod()
    return nav


def calibrate_hy_spread_from_baa_aaa(
    real_hy_spread: pd.Series,
    baa_yield: pd.Series,
    aaa_yield: pd.Series,
) -> dict:
    """OLS: real_HY_spread = intercept + scale * (BAA - AAA).

    Real HY spread can come from BAMLH0A0HYM2 (ICE BoA High Yield OAS,
    available from 1996-12 in FRED). BAA and AAA from Moody's go back to
    1919 in FRED.
    """
    s_baa_aaa = (baa_yield - aaa_yield).resample("D").ffill()
    df = pd.DataFrame({"hy": real_hy_spread, "baa_aaa": s_baa_aaa}).dropna()
    if len(df) < 100:
        return {"error": f"insufficient overlap: {len(df)} days"}

    y = df["hy"].values
    X = np.column_stack([np.ones(len(df)), df["baa_aaa"].values])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "intercept": float(beta[0]),
        "scale": float(beta[1]),
        "r_squared": float(r2),
        "n_days": int(len(df)),
    }


def synthesize_hy_spread(
    baa_yield: pd.Series,
    aaa_yield: pd.Series,
    intercept: float = 3.5,
    scale: float = 2.7,
) -> pd.Series:
    """Generate HY spread proxy from BAA-AAA. Resampled to daily."""
    s = (baa_yield - aaa_yield).resample("D").ffill()
    return (intercept + scale * s).clip(lower=1.0, upper=20.0)  # sanity clip


def validate_hyg_synthesizer(
    real_hyg_prices: pd.Series,
    treasury_5y: pd.Series,
    hy_spread: pd.Series,
    duration_years: float = 4.0,
) -> dict:
    """Validate against real HYG total-return prices."""
    synth_nav = synthesize_credit_etf(treasury_5y, hy_spread, duration_years=duration_years,
                                        default_rate_annual=0.04, lgd=0.65)
    real = real_hyg_prices.copy()
    real.index = pd.to_datetime(real.index)

    aligned = pd.DataFrame({"real": real, "synth": synth_nav}).dropna()
    if len(aligned) < 100:
        return {"error": f"insufficient overlap: {len(aligned)} days"}

    # Convert both to daily returns for correlation
    r_real = aligned["real"].pct_change().dropna()
    r_synth = aligned["synth"].pct_change().reindex(r_real.index)

    # CAGR
    n_yr = len(aligned) / 252
    cagr_real = (aligned["real"].iloc[-1] / aligned["real"].iloc[0]) ** (1/n_yr) - 1
    cagr_synth = (aligned["synth"].iloc[-1] / aligned["synth"].iloc[0]) ** (1/n_yr) - 1

    # Daily return correlation
    diff = r_synth - r_real
    mr, ms = r_real.mean(), r_synth.mean()
    cov = ((r_real - mr) * (r_synth - ms)).mean()
    vr = ((r_real - mr) ** 2).mean()
    vs = ((r_synth - ms) ** 2).mean()
    corr = (cov / math.sqrt(vr * vs)) if vr > 0 and vs > 0 else 0.0

    return {
        "n_days_aligned": int(len(aligned)),
        "cagr_real": float(cagr_real),
        "cagr_synth": float(cagr_synth),
        "cagr_error_pp": float(cagr_synth - cagr_real),
        "daily_return_correlation": float(corr),
        "passes_validation": abs(cagr_synth - cagr_real) < 0.03 and corr > 0.5,
    }
