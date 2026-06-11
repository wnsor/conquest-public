"""Synthesize VIX from S&P 500 daily returns.

Model
-----
VIX is implied vol from SPX option prices. Pre-1990 we have no options data.
But empirically:

  VIX ≈ a + b * realized_vol_30d + c * realized_vol_5d_max + d * skew_indicator

where realized_vol terms are annualized and the skew indicator captures fear
premium that doesn't show in pure realized vol.

Calibration: regress real VIX (1990-2020) on these features. Coefficients
typically:
  a ≈ 5.0  (always-on fear premium, even at zero realized vol)
  b ≈ 0.85 (most of VIX is just realized vol)
  c ≈ 0.20 (rapid spikes get amplified in implied)
  d ≈ varies (skew premium during crashes)

Validation discipline: model must reproduce VIX correlation > 0.85 on the
training set, with mean error within 3 points, before being used pre-1990.

VIX3M
-----
VIX3M (90-day) launched 2007-12. Pre-launch:
  VIX3M ≈ smoothed(VIX, 60-day window) + risk-premium term-structure adjustment

Empirical: VIX3M / VIX ≈ 1.07 average (term-structure mean), with the ratio
inverting during stress (backwardation: ratio < 1.0 → recession-correlated).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def synthesize_vix(
    sp500_returns: pd.Series,
    a: float = 5.0,
    b: float = 0.85,
    c: float = 0.20,
    skew_premium: float = 2.0,
    vol_lookback_days: int = 30,
    spike_lookback_days: int = 5,
) -> pd.Series:
    """Synthesize daily VIX from S&P 500 returns.

    Parameters
    ----------
    sp500_returns
        Daily returns of SPY / SPX / ^GSPC. Datetime index.
    a, b, c
        Linear-model coefficients (intercept, realized-vol coef, spike coef).
    skew_premium
        Extra additive when negative-skew week (down moves > up moves significantly).
    vol_lookback_days
        Window for realized vol (default 30).
    spike_lookback_days
        Window for "recent spike" detection (default 5).

    Returns
    -------
    Pandas Series of synthesized VIX values (daily). VIX is annualized vol in pct.
    """
    if not isinstance(sp500_returns.index, pd.DatetimeIndex):
        sp500_returns = sp500_returns.copy()
        sp500_returns.index = pd.to_datetime(sp500_returns.index)

    # Annualized realized vol (in pct, like VIX)
    sigma_30d = sp500_returns.rolling(vol_lookback_days, min_periods=10).std() * math.sqrt(252) * 100

    # Recent 5-day max abs return (in pct, annualized)
    max_5d = sp500_returns.rolling(spike_lookback_days, min_periods=2).apply(
        lambda x: np.abs(x).max(), raw=True
    ) * math.sqrt(252) * 100

    # Skew indicator: fraction of negative days in past 5 days
    neg_5d = (sp500_returns < 0).rolling(spike_lookback_days, min_periods=2).mean()
    skew_term = skew_premium * (neg_5d - 0.5).clip(lower=0)  # only when > 50% down days

    vix_synth = a + b * sigma_30d + c * max_5d + skew_term * 10  # scale skew to pct-points

    # Floor at 5 (VIX rarely below 9)
    return vix_synth.clip(lower=5.0)


def validate_vix(
    real_vix: pd.Series,
    sp500_returns: pd.Series,
    a: float = 5.0,
    b: float = 0.85,
    c: float = 0.20,
    skew_premium: float = 2.0,
) -> dict:
    """Validate VIX synthesizer against real VIX data.

    Returns dict with correlation, mean error, RMSE.
    """
    synth = synthesize_vix(sp500_returns, a=a, b=b, c=c, skew_premium=skew_premium)
    aligned = pd.DataFrame({"real": real_vix, "synth": synth}).dropna()
    if len(aligned) < 100:
        return {"error": f"insufficient overlap: {len(aligned)} aligned days"}

    real = aligned["real"]
    s = aligned["synth"]
    diff = s - real

    mr, ms = real.mean(), s.mean()
    cov = ((real - mr) * (s - ms)).mean()
    vr = ((real - mr) ** 2).mean()
    vs = ((s - ms) ** 2).mean()
    corr = (cov / math.sqrt(vr * vs)) if vr > 0 and vs > 0 else 0.0

    return {
        "n_days_aligned": int(len(aligned)),
        "mean_real": float(real.mean()),
        "mean_synth": float(s.mean()),
        "mean_error": float(diff.mean()),
        "mae": float(diff.abs().mean()),
        "rmse": float(math.sqrt((diff ** 2).mean())),
        "correlation": float(corr),
        "passes_validation": abs(diff.mean()) < 3.0 and corr > 0.85,
    }


def calibrate_vix_coefficients(
    real_vix: pd.Series,
    sp500_returns: pd.Series,
    vol_lookback_days: int = 30,
    spike_lookback_days: int = 5,
) -> dict:
    """OLS calibration of VIX = a + b*sigma_30d + c*max_5d using overlap data.

    Returns optimal (a, b, c, skew_premium) on the training set + diagnostics.
    """
    if not isinstance(sp500_returns.index, pd.DatetimeIndex):
        sp500_returns = sp500_returns.copy()
        sp500_returns.index = pd.to_datetime(sp500_returns.index)

    sigma_30d = sp500_returns.rolling(vol_lookback_days, min_periods=10).std() * math.sqrt(252) * 100
    max_5d = sp500_returns.rolling(spike_lookback_days, min_periods=2).apply(
        lambda x: np.abs(x).max(), raw=True
    ) * math.sqrt(252) * 100
    neg_5d = (sp500_returns < 0).rolling(spike_lookback_days, min_periods=2).mean()
    skew_term = (neg_5d - 0.5).clip(lower=0) * 10

    df = pd.DataFrame({
        "vix": real_vix,
        "sigma_30d": sigma_30d,
        "max_5d": max_5d,
        "skew_term": skew_term,
    }).dropna()
    if len(df) < 100:
        return {"error": f"insufficient data: {len(df)} aligned days"}

    # OLS: y = beta_0 + beta_1*x1 + beta_2*x2 + beta_3*x3
    y = df["vix"].values
    X = np.column_stack([
        np.ones(len(df)),
        df["sigma_30d"].values,
        df["max_5d"].values,
        df["skew_term"].values,
    ])
    beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "c": float(beta[2]),
        "skew_premium": float(beta[3]),
        "r_squared": float(r2),
        "n_days": int(len(df)),
    }
