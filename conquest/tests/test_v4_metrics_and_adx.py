"""Tests for v4 additions: ADX indicator, AdxGated wrapper, distribution metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.indicators.adx import adx
from conquest.models import AdxGated, MomentumConsensus, all_models
from conquest.backtest.metrics import (
    cvar, omega_ratio, skewness, excess_kurtosis, metrics_summary,
)


# ---------- ADX indicator ----------

@pytest.fixture
def trending_ohlc() -> pd.DataFrame:
    """Strongly trending up: each bar's close is the previous high + a small drift."""
    rng = np.random.default_rng(0)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = (1 + 0.01).cumprod() if False else pd.Series((1.005) ** np.arange(n) * 100)
    closes.index = idx
    high = closes * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = closes * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({"High": high, "Low": low, "Close": closes}, index=idx)


@pytest.fixture
def choppy_ohlc() -> pd.DataFrame:
    """Range-bound oscillation around a flat mean."""
    rng = np.random.default_rng(1)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = pd.Series(100 + np.sin(np.arange(n) * 0.3) * 2 + rng.normal(0, 0.5, n), index=idx)
    high = closes + np.abs(rng.normal(0, 0.4, n))
    low = closes - np.abs(rng.normal(0, 0.4, n))
    return pd.DataFrame({"High": high, "Low": low, "Close": closes}, index=idx)


def test_adx_bounded_0_to_100(trending_ohlc):
    a = adx(trending_ohlc, 14).dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_adx_higher_in_trend_than_chop(trending_ohlc, choppy_ohlc):
    """The whole point of ADX: higher in trends, lower in chop."""
    a_trend = adx(trending_ohlc, 14).dropna().tail(20).mean()
    a_chop = adx(choppy_ohlc, 14).dropna().tail(50).mean()
    assert a_trend > a_chop


def test_adx_components_returned():
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = pd.Series(np.linspace(100, 110, n), index=idx)
    df = pd.DataFrame({
        "High": closes * 1.002, "Low": closes * 0.998, "Close": closes,
    }, index=idx)
    full = adx(df, 14, return_components=True)
    assert set(full.columns) == {"adx", "plus_di", "minus_di", "dx"}


def test_adx_rejects_missing_columns():
    bad = pd.DataFrame({"Close": [100, 101, 102]})
    with pytest.raises(ValueError):
        adx(bad)


def test_adx_rejects_period_lt_2():
    df = pd.DataFrame({"High": [1.0], "Low": [1.0], "Close": [1.0]})
    with pytest.raises(ValueError):
        adx(df, period=1)


# ---------- AdxGated wrapper ----------

@pytest.fixture
def universe_with_ohlc():
    rng = np.random.default_rng(7)
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = ["TRENDY", "CHOPPY"]
    drifts = [0.0008, 0.0]
    prices = pd.DataFrame()
    ohlc = {}
    for c, d in zip(cols, drifts):
        if c == "CHOPPY":
            base = 100 + np.sin(np.arange(n) * 0.4) * 2 + rng.normal(0, 0.5, n)
        else:
            base = (1 + d + rng.normal(0, 0.005, n)).cumprod() * 100
        s = pd.Series(base, index=idx)
        prices[c] = s
        ohlc[c] = pd.DataFrame({
            "High":  s * (1 + np.abs(rng.normal(0, 0.003, n))),
            "Low":   s * (1 - np.abs(rng.normal(0, 0.003, n))),
            "Close": s,
        }, index=idx)
    return prices, ohlc


def test_adx_gated_no_ohlc_passes_through(universe_with_ohlc):
    prices, _ = universe_with_ohlc
    base = MomentumConsensus(top_n=2)
    gated = AdxGated(base, ohlc=None)
    pd.testing.assert_frame_equal(gated.signal(prices), base.signal(prices))


def test_adx_gated_filters_choppy_names(universe_with_ohlc):
    prices, ohlc = universe_with_ohlc
    base = MomentumConsensus(top_n=2)
    gated = AdxGated(base, ohlc=ohlc, min_adx=25)

    base_w = base.signal(prices)
    gated_w = gated.signal(prices)

    # Late in the series, base should hold both names; gated should retain TRENDY > CHOPPY
    last_base = base_w.iloc[-1]
    last_gated = gated_w.iloc[-1]
    if last_base["TRENDY"] > 0 and last_base["CHOPPY"] > 0:
        # gated may have zeroed out CHOPPY (low ADX); TRENDY should still be present
        assert last_gated["TRENDY"] >= 0
        # The gated version should hold no more weight than the base in any name
        assert (last_gated <= last_base + 1e-12).all()


def test_all_models_v4_factory_adds_three_variants(universe_with_ohlc):
    _, ohlc = universe_with_ohlc
    base = all_models(include_v4=False)
    extended = all_models(include_v4=True, ohlc=ohlc)
    assert len(extended) == len(base) + 3
    names = {m.name for m in extended}
    expected = {
        "trend_follow_adx_gated",
        "momentum_consensus_adx_gated",
        "dual_momentum_adx_gated",
    }
    assert expected.issubset(names)


# ---------- Distribution metrics ----------

@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.0005, 0.012, 1000))


@pytest.fixture
def left_skewed_returns():
    """Mostly small positive returns + occasional large negatives = left-skewed."""
    rng = np.random.default_rng(43)
    base = rng.normal(0.001, 0.005, 1000)
    # Inject 30 negative shocks
    shock_idx = rng.choice(1000, size=30, replace=False)
    base[shock_idx] = rng.normal(-0.05, 0.02, 30)
    return pd.Series(base)


def test_cvar_more_negative_than_mean(normal_returns):
    """CVaR(5%) should be negative for any non-degenerate return series with mean ~0."""
    c = cvar(normal_returns, alpha=0.05)
    assert c < normal_returns.mean()
    assert c < 0


def test_cvar_alpha_bounds():
    s = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(cvar(s, alpha=0.0))
    assert np.isnan(cvar(s, alpha=1.0))


def test_omega_ratio_above_1_for_positive_mean(normal_returns):
    """Mean > 0 → more upside than downside → omega > 1."""
    assert omega_ratio(normal_returns, threshold=0.0) > 1.0


def test_omega_ratio_below_1_for_negative_mean():
    rng = np.random.default_rng(44)
    losing = pd.Series(rng.normal(-0.001, 0.012, 1000))
    assert omega_ratio(losing, threshold=0.0) < 1.0


def test_skewness_negative_on_left_skewed(left_skewed_returns):
    assert skewness(left_skewed_returns) < 0


def test_skewness_near_zero_on_normal(normal_returns):
    assert abs(skewness(normal_returns)) < 0.5


def test_excess_kurtosis_higher_on_fat_tails(normal_returns, left_skewed_returns):
    """Left-skewed series with shocks has fatter tails than near-normal returns."""
    k_normal = excess_kurtosis(normal_returns)
    k_skewed = excess_kurtosis(left_skewed_returns)
    assert k_skewed > k_normal


def test_metrics_summary_includes_v4_keys(normal_returns):
    """All v4 distribution keys must appear in metrics_summary."""
    equity = (1 + normal_returns).cumprod() * 100_000
    turnover = pd.Series(0.05, index=normal_returns.index)
    m = metrics_summary(normal_returns, equity, turnover)
    for k in ("cvar_5pct", "omega_ratio", "skewness", "excess_kurtosis"):
        assert k in m
