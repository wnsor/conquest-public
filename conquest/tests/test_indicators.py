"""Sanity + bound tests for conquest.indicators.

Phase 1 v0: validates mathematical properties (bounds, monotonicity in trends, edge cases).
Strict numerical parity vs Lean's built-ins requires running Lean to capture a fixture; that
upgrade lands once the Lean smoke test produces fixture CSVs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.indicators import rsi, macd, trix, sma, momp, realized_vol


@pytest.fixture
def trending_up() -> pd.Series:
    """100 daily bars, +0.5%/day drift with small noise."""
    rng = np.random.default_rng(42)
    rets = 0.005 + rng.normal(0, 0.001, 100)
    return pd.Series((1 + rets).cumprod() * 100, name="px")


@pytest.fixture
def trending_down() -> pd.Series:
    rng = np.random.default_rng(42)
    rets = -0.005 + rng.normal(0, 0.001, 100)
    return pd.Series((1 + rets).cumprod() * 100, name="px")


@pytest.fixture
def constant() -> pd.Series:
    return pd.Series([100.0] * 100, name="px")


# ---------------- RSI ----------------

def test_rsi_bounded(trending_up):
    r = rsi(trending_up, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_uptrend_above_70(trending_up):
    assert rsi(trending_up, 14).iloc[-1] > 70


def test_rsi_downtrend_below_30(trending_down):
    assert rsi(trending_down, 14).iloc[-1] < 30


def test_rsi_constant_neutral_or_nan(constant):
    # Constant prices => 0/0; pandas yields NaN. That's acceptable.
    r = rsi(constant, 14)
    assert r.iloc[20:].isna().all() or (r.iloc[20:].dropna() == pytest.approx(50.0, abs=1.0)).all()


# ---------------- MACD ---------------

def test_macd_columns(trending_up):
    m = macd(trending_up)
    assert list(m.columns) == ["line", "signal", "histogram"]


def test_macd_uptrend_line_positive(trending_up):
    # In a sustained uptrend, fast EMA > slow EMA, so MACD line > 0.
    # We avoid asserting on (line - signal) here because once the trend reaches
    # steady-state both EMAs converge to a near-constant gap and the signal line
    # catches up to the MACD line — at that point histogram sign is noise-driven.
    assert macd(trending_up)["line"].iloc[-1] > 0


def test_macd_downtrend_line_negative(trending_down):
    assert macd(trending_down)["line"].iloc[-1] < 0


def test_macd_signal_lags_during_acceleration():
    # During *active acceleration* of the trend (not steady-state), the signal lags
    # the line. Use a series that's still in the ramp-up phase.
    prices = pd.Series((1 + 0.01) ** pd.Series(range(40)) * 100, name="px")
    m = macd(prices).iloc[-1]
    # Line is rising toward a higher steady-state, so signal (EMA-of-line) trails below it.
    assert m["line"] > m["signal"]


def test_macd_validates_periods():
    p = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        macd(p, fast=26, slow=12)   # fast must be < slow
    with pytest.raises(ValueError):
        macd(p, signal=0)


# ---------------- TRIX ---------------

def test_trix_uptrend_positive(trending_up):
    assert trix(trending_up, 15).iloc[-1] > 0


def test_trix_downtrend_negative(trending_down):
    assert trix(trending_down, 15).iloc[-1] < 0


def test_trix_constant_zero_or_nan(constant):
    t = trix(constant, 15).dropna()
    if not t.empty:
        assert (t.abs() < 1e-12).all()


# ---------------- SMA ----------------

def test_sma_warmup_then_equals_mean(trending_up):
    s = sma(trending_up, 20)
    assert s.iloc[:19].isna().all()
    assert s.iloc[19] == pytest.approx(trending_up.iloc[:20].mean())


def test_sma_constant_passes_through(constant):
    s = sma(constant, 10).dropna()
    assert (s == 100.0).all()


# ---------------- MOMP ---------------

def test_momp_uptrend_positive(trending_up):
    assert momp(trending_up, 30).iloc[-1] > 0


def test_momp_downtrend_negative(trending_down):
    assert momp(trending_down, 30).iloc[-1] < 0


def test_momp_warmup_nan(trending_up):
    m = momp(trending_up, 30)
    assert m.iloc[:30].isna().all()
    assert not m.iloc[30:].isna().any()


# ---------------- realized_vol -------

def test_realized_vol_warmup_nan(trending_up):
    v = realized_vol(trending_up, 30)
    assert v.iloc[:30].isna().all()
    assert not v.iloc[30:].isna().any()


def test_realized_vol_low_for_steady_trend(trending_up):
    """A series with +0.5%/day drift + tiny noise has low realized vol."""
    v = realized_vol(trending_up, 60).iloc[-1]
    # Daily noise σ=0.001 ⇒ annualized ≈ 0.001 * sqrt(252) ≈ 0.016
    assert 0.005 < v < 0.05


def test_realized_vol_invalid_period():
    s = pd.Series([100.0, 101.0, 102.0])
    with pytest.raises(ValueError):
        realized_vol(s, 1)


def test_realized_vol_annualization_factor(trending_up):
    """sqrt(252) annualization should give ~16x the raw daily σ."""
    raw = realized_vol(trending_up, 60, annualize=False).iloc[-1]
    ann = realized_vol(trending_up, 60, annualize=True).iloc[-1]
    assert ann == pytest.approx(raw * (252 ** 0.5), rel=1e-6)
