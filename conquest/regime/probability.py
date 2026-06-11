"""Probabilistic 4-quadrant regime classifier — predicts NEXT-month regime distribution.

Where the deterministic `RegimeClassifier` returns one regime label per month based
on TODAY's z-scored GDP_YoY and CPI_YoY, this module returns the PROBABILITY
distribution over the 4 regimes for NEXT month — a leading signal that lets the
strategy de-risk PRE-EMPTIVELY.

Methodology (matches the user's GBM spreadsheet at the regime layer):
1. Z-score each YoY series with a 60-month rolling window (same as classifier).
2. Compute monthly CHANGES in those z-scores.
3. Estimate the drift mu and volatility sigma of those changes over a trailing
   24-month window.
4. Forecast next-month z-value: z_{t+1} ~ N(z_t + mu, sigma^2).
5. Use the analytical normal CDF to compute P(z_GDP_next > 0) and P(z_CPI_next > 0).
   (This is faster than Monte Carlo and identical at the limit.)
6. Joint regime probabilities under the independence assumption:
       P(Inflation)    = P(GDP>0) * P(CPI>0)
       P(Disinflation) = P(GDP>0) * (1 - P(CPI>0))
       P(Stagflation)  = (1 - P(GDP>0)) * P(CPI>0)
       P(Deflation)    = (1 - P(GDP>0)) * (1 - P(CPI>0))

Anti-overfit notes:
- The change-window length (24mo) is fixed, not optimized in-sample.
- Independence between GDP and CPI shocks is a simplifying assumption; in
  practice they correlate (~0.3 historically). Adding a correlation term is a
  v9.5 candidate if this version generalizes.
- Forward-looking by exactly one month — calibrated for monthly rebalance cadence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


REGIME_PROB_COLS: list[str] = [
    "p_inflation", "p_disinflation", "p_stagflation", "p_deflation",
]


def regime_probabilities(
    gdp_yoy: pd.Series,
    cpi_yoy: pd.Series,
    *,
    zscore_window: int = 60,
    change_window: int = 24,
) -> pd.DataFrame:
    """Return monthly DataFrame with [p_inflation, p_disinflation, p_stagflation, p_deflation].

    Args:
        gdp_yoy: monthly (or quarterly upsampled to monthly) GDP YoY growth.
        cpi_yoy: monthly CPI YoY inflation.
        zscore_window: rolling window for z-score normalization (default 60mo).
        change_window: rolling window for drift / vol of the z-score changes (default 24mo).

    Returns:
        DataFrame indexed by month-end, columns [p_inflation, p_disinflation,
        p_stagflation, p_deflation], each in [0, 1]. NaN where insufficient
        history. Probabilities sum to ~1.0 per row (modulo numerical roundoff).
    """
    if not (gdp_yoy.index.is_monotonic_increasing and cpi_yoy.index.is_monotonic_increasing):
        gdp_yoy = gdp_yoy.sort_index()
        cpi_yoy = cpi_yoy.sort_index()

    # Align to monthly frequency
    gdp_m = gdp_yoy.resample("ME").last() if hasattr(gdp_yoy.index, "freq") else gdp_yoy
    cpi_m = cpi_yoy.resample("ME").last() if hasattr(cpi_yoy.index, "freq") else cpi_yoy

    # Z-score each series (Bridgewater "above trend" = positive z)
    z_gdp = _zscore(gdp_m, zscore_window)
    z_cpi = _zscore(cpi_m, zscore_window)

    # Monthly z-score changes
    dz_gdp = z_gdp.diff()
    dz_cpi = z_cpi.diff()

    # Rolling drift and volatility of the changes
    mu_gdp = dz_gdp.rolling(change_window, min_periods=change_window // 2).mean()
    sd_gdp = dz_gdp.rolling(change_window, min_periods=change_window // 2).std()
    mu_cpi = dz_cpi.rolling(change_window, min_periods=change_window // 2).mean()
    sd_cpi = dz_cpi.rolling(change_window, min_periods=change_window // 2).std()

    # Forecast next-month z-value: z_{t+1} ~ N(z_t + mu, sd^2)
    # P(z_{t+1} > 0) = 1 - Phi((0 - z_t - mu) / sd)
    sd_gdp_safe = sd_gdp.replace(0, np.nan)
    sd_cpi_safe = sd_cpi.replace(0, np.nan)
    p_gdp_pos = pd.Series(
        1 - norm.cdf(((0 - z_gdp - mu_gdp) / sd_gdp_safe).values),
        index=z_gdp.index,
    )
    p_cpi_pos = pd.Series(
        1 - norm.cdf(((0 - z_cpi - mu_cpi) / sd_cpi_safe).values),
        index=z_cpi.index,
    )

    # Align to common monthly index
    common = p_gdp_pos.index.intersection(p_cpi_pos.index)
    p_g = p_gdp_pos.reindex(common)
    p_c = p_cpi_pos.reindex(common)

    out = pd.DataFrame({
        "p_inflation":    p_g * p_c,
        "p_disinflation": p_g * (1 - p_c),
        "p_stagflation":  (1 - p_g) * p_c,
        "p_deflation":    (1 - p_g) * (1 - p_c),
    }, index=common)
    return out


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std = series.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return (series - mean) / std


def probability_to_daily(
    prob_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Forward-fill monthly probabilities to a daily business-day index."""
    if daily_index is None:
        daily_index = pd.date_range(prob_df.index[0], prob_df.index[-1], freq="B")
    return prob_df.reindex(daily_index, method="ffill")
