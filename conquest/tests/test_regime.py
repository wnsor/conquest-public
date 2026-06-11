"""Tests for the regime classifier on synthetic GDP and CPI series."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.regime import RegimeClassifier, REGIME_LABELS


def _quarterly(years: int, drift: float, noise: float = 0.5, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=years * 4, freq="QE")
    return pd.Series(drift + rng.normal(0, noise, len(idx)), index=idx, name="gdp_yoy")


def _monthly(years: int, drift: float, noise: float = 0.3, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=years * 12, freq="ME")
    return pd.Series(drift + rng.normal(0, noise, len(idx)), index=idx, name="cpi_yoy")


def test_classify_columns_and_labels():
    gdp = _quarterly(years=20, drift=2.5)
    cpi = _monthly(years=20, drift=2.0)
    df = RegimeClassifier(min_dwell_months=1).classify(gdp, cpi)
    assert {"gdp_yoy", "cpi_yoy", "gdp_yoy_z", "cpi_yoy_z", "regime", "confidence"} == set(df.columns)
    labels = df["regime"].dropna().unique()
    assert all(lab in REGIME_LABELS for lab in labels)


def test_growth_above_inflation_above_yields_inflation():
    """Both GDP and CPI shift upward in the second half ⇒ z-scores positive ⇒ Inflation."""
    qn = 20 * 4
    mn = 20 * 12
    gdp_vals = np.concatenate([np.full(qn // 2, 1.5), np.full(qn // 2, 4.5)])
    cpi_vals = np.concatenate([np.full(mn // 2, 1.0), np.full(mn // 2, 5.0)])
    gdp = pd.Series(gdp_vals, index=pd.date_range("2000-01-01", periods=qn, freq="QE"))
    cpi = pd.Series(cpi_vals, index=pd.date_range("2000-01-01", periods=mn, freq="ME"))
    df = RegimeClassifier(lookback_months=60, min_dwell_months=1).classify(gdp, cpi)
    assert df["regime"].dropna().iloc[-1] == "Inflation"


def test_growth_below_inflation_above_yields_stagflation():
    qn = 20 * 4
    mn = 20 * 12
    gdp_vals = np.concatenate([np.full(qn // 2, 4.0), np.full(qn // 2, 0.5)])
    cpi_vals = np.concatenate([np.full(mn // 2, 1.0), np.full(mn // 2, 5.0)])
    gdp = pd.Series(gdp_vals, index=pd.date_range("2000-01-01", periods=qn, freq="QE"))
    cpi = pd.Series(cpi_vals, index=pd.date_range("2000-01-01", periods=mn, freq="ME"))
    df = RegimeClassifier(lookback_months=60, min_dwell_months=1).classify(gdp, cpi)
    assert df["regime"].dropna().iloc[-1] == "Stagflation"


def test_growth_below_inflation_below_yields_deflation():
    qn = 20 * 4
    mn = 20 * 12
    gdp_vals = np.concatenate([np.full(qn // 2, 4.0), np.full(qn // 2, 0.5)])
    cpi_vals = np.concatenate([np.full(mn // 2, 4.0), np.full(mn // 2, 0.5)])
    gdp = pd.Series(gdp_vals, index=pd.date_range("2000-01-01", periods=qn, freq="QE"))
    cpi = pd.Series(cpi_vals, index=pd.date_range("2000-01-01", periods=mn, freq="ME"))
    df = RegimeClassifier(lookback_months=60, min_dwell_months=1).classify(gdp, cpi)
    assert df["regime"].dropna().iloc[-1] == "Deflation"


def test_classify_to_daily_ffilled_continuity():
    gdp = _quarterly(years=10, drift=2.5)
    cpi = _monthly(years=10, drift=2.0)
    daily = RegimeClassifier().classify_to_daily(gdp, cpi)
    # Most successive daily rows carry the same regime forward; transitions are rare
    valid = daily["regime"].dropna()
    same_as_prev = (valid.iloc[1:].values == valid.iloc[:-1].values)
    assert same_as_prev.mean() > 0.9


def test_min_dwell_blocks_one_month_blip():
    """A 1-month spike in CPI z-score should NOT cause a regime change with min_dwell=2."""
    qn = 20 * 4
    mn = 20 * 12
    gdp = pd.Series(np.full(qn, 2.0), index=pd.date_range("2000-01-01", periods=qn, freq="QE"))
    cpi_arr = np.full(mn, 1.0)
    cpi_arr[120] = 6.0   # one-month spike
    cpi = pd.Series(cpi_arr, index=pd.date_range("2000-01-01", periods=mn, freq="ME"))
    df_dwell = RegimeClassifier(lookback_months=60, min_dwell_months=2).classify(gdp, cpi)
    df_nodwell = RegimeClassifier(lookback_months=60, min_dwell_months=1).classify(gdp, cpi)
    # The non-dwell version may flip at the spike; the dwell version should not pick it up
    spike_month = df_dwell.index[120]
    # Find the regime immediately before and at the spike under both versions
    if spike_month in df_dwell.index:
        # In the dwell version, the regime at the spike row equals its predecessor
        pos = df_dwell.index.get_loc(spike_month)
        if pos > 0:
            assert df_dwell["regime"].iloc[pos] == df_dwell["regime"].iloc[pos - 1]
