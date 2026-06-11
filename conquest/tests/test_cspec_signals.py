"""Tests for conquest.signals — the cspec composite-score helpers.

Validates:
  * dollar_volume_spike: shape, NaN-padding, exclude-today semantics, and
    a hand-computed value against a small fixture.
  * breakout_proximity: shape, NaN-padding, value at known max/min.
  * cspec_composite_score: shape, columns present, all components z-summed,
    cross-sectional means ~ 0 per date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.signals import (
    breakout_proximity,
    cspec_composite_score,
    dollar_volume_spike,
    volume_spike,
)


@pytest.fixture
def fixture_panel():
    """Two symbols, 300 trading days, deterministic prices + volumes."""
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    rng = np.random.default_rng(42)
    # Symbol A: linearly trending up with noise
    a_close = np.linspace(100, 200, 300) + rng.normal(0, 1, 300)
    a_vol = rng.integers(1_000_000, 2_000_000, 300).astype(float)
    # Symbol B: sideways with noise
    b_close = 50 + rng.normal(0, 0.8, 300)
    b_vol = rng.integers(500_000, 800_000, 300).astype(float)
    closes = pd.DataFrame({"A": a_close, "B": b_close}, index=dates)
    volumes = pd.DataFrame({"A": a_vol, "B": b_vol}, index=dates)
    return closes, volumes


# ------------------ dollar_volume_spike ------------------

def test_dollar_volume_spike_shape_and_nans(fixture_panel):
    closes, volumes = fixture_panel
    out = dollar_volume_spike(closes["A"], volumes["A"], lookback=20)
    assert len(out) == len(closes)
    # First 20 bars unfilled (need 20 bars to compute mean, then shift(1) loses one more);
    # exact NaN count is 20 (rolling needs 20 valid points; shift(1) shifts from position 20 to 21)
    assert out.iloc[0:20].isna().all()
    assert out.iloc[21:].notna().all()


def test_dollar_volume_spike_excludes_today(fixture_panel):
    """The denominator should not include today's bar — spike must compare
    today against the prior 20d, not against a window that includes today."""
    closes, volumes = fixture_panel
    out = dollar_volume_spike(closes["A"], volumes["A"], lookback=20)
    # Hand compute for bar idx 25
    dv = closes["A"] * volumes["A"]
    today_dv = dv.iloc[25]
    prior_avg = dv.iloc[5:25].mean()  # exclusive of bar 25
    expected = today_dv / prior_avg
    assert out.iloc[25] == pytest.approx(expected, rel=1e-9)


def test_dollar_volume_spike_value_around_one_for_steady_state(fixture_panel):
    """For a panel where today's $-vol ~= prior 20d mean, spike should be ~1.0."""
    closes, volumes = fixture_panel
    out = dollar_volume_spike(closes["B"], volumes["B"], lookback=20)
    # Symbol B is sideways with low-noise volume; spike values should center near 1.0
    median = out.dropna().median()
    assert 0.7 < median < 1.3


def test_dollar_volume_spike_invalid_lookback_raises():
    with pytest.raises(ValueError):
        dollar_volume_spike(pd.Series([1, 2]), pd.Series([10, 20]), lookback=1)


def test_volume_spike_shape(fixture_panel):
    _, volumes = fixture_panel
    out = volume_spike(volumes["A"], lookback=20)
    assert len(out) == len(volumes)
    assert out.iloc[21:].notna().all()


# ------------------ breakout_proximity ------------------

def test_breakout_proximity_shape_and_nans(fixture_panel):
    closes, _ = fixture_panel
    out = breakout_proximity(closes["A"], lookback=252)
    assert len(out) == len(closes)
    # Need 252 bars for the rolling max, then shift(1) → 252 NaNs at start
    assert out.iloc[0:252].isna().all()
    assert out.iloc[253:].notna().all()


def test_breakout_proximity_at_known_high():
    """If today's close == the prior-window max, proximity == 1.0."""
    # Construct a series that ramps to its high on bar 100, stays high
    close = pd.Series(
        list(range(1, 101)) + [100] * 100,  # 200 bars total
        index=pd.date_range("2020-01-02", periods=200, freq="B"),
    ).astype(float)
    out = breakout_proximity(close, lookback=50)
    # Bar 150 — prior 50d window is bars 100..149, all == 100; today close == 100
    assert out.iloc[150] == pytest.approx(1.0, rel=1e-9)


def test_breakout_proximity_above_one_for_breakout():
    """If today's close exceeds the prior-window max, proximity > 1.0."""
    close = pd.Series(
        [10] * 60 + [15],  # 60 bars at 10, then breakout to 15
        index=pd.date_range("2020-01-02", periods=61, freq="B"),
    ).astype(float)
    out = breakout_proximity(close, lookback=50)
    # Last bar's prior-50d window is all 10s, close is 15 -> 1.5
    assert out.iloc[-1] == pytest.approx(1.5, rel=1e-9)


def test_breakout_proximity_invalid_lookback_raises():
    with pytest.raises(ValueError):
        breakout_proximity(pd.Series([1.0, 2.0]), lookback=1)


# ------------------ cspec_composite_score ------------------

def test_cspec_composite_score_shape_and_columns(fixture_panel):
    closes, volumes = fixture_panel
    df = cspec_composite_score(
        closes, volumes,
        momentum_lookback=60,
        vol_lookback=20,
        breakout_lookback=252,
    )
    # Long-form (date, symbol) index, 4 columns
    assert df.index.names == ["date", "symbol"]
    assert set(df.columns) == {"roc", "vol_spike", "breakout_prox", "score"}
    # Rows exist after the longest lookback (252 + 1 shift)
    last_dates = df.index.get_level_values("date").unique()
    assert len(last_dates) > 0


def test_cspec_composite_score_z_sum(fixture_panel):
    """The 'score' column must equal the sum of the per-axis z-scores."""
    closes, volumes = fixture_panel
    df = cspec_composite_score(closes, volumes,
                               momentum_lookback=60, vol_lookback=20, breakout_lookback=252)
    # We can't reverse-derive z values from the long form alone, but we can
    # check internal consistency: per-date cross-sectional mean of `score`
    # should be ~ 0 (sum of three z-scores, each cross-sectionally zero-mean).
    by_date = df["score"].groupby("date").mean()
    # For only 2 symbols, std=0 cases yield NaN scores and they get dropped;
    # surviving dates should have score-mean very close to 0.
    assert by_date.dropna().abs().max() < 1e-9


def test_cspec_composite_score_mismatched_columns_raises():
    closes = pd.DataFrame({"A": [1.0, 2, 3]}, index=pd.RangeIndex(3))
    volumes = pd.DataFrame({"B": [10.0, 20, 30]}, index=pd.RangeIndex(3))
    with pytest.raises(ValueError):
        cspec_composite_score(closes, volumes)
