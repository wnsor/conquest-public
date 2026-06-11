"""Tests for conquest.vol — realized vol, inverse-vol weighting, vol-targeting, Kelly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.vol import (
    realized_vol, inverse_vol_weights, vol_target_scale, kelly_weights,
)


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    cols = ["A", "B", "C"]
    return pd.DataFrame(
        {c: (1 + 0.0003 + rng.normal(0, 0.01, len(dates))).cumprod() * 100 for c in cols},
        index=dates,
    )


def test_realized_vol_shape_and_positivity(prices):
    v = realized_vol(prices, lookback=20)
    assert v.shape == prices.shape
    valid = v.dropna()
    assert (valid > 0).all().all()
    # Daily σ ≈ 1% → annualised ≈ 16%; allow a wide band
    assert 0.10 < float(valid.iloc[-1].mean()) < 0.30


def test_realized_vol_lookback_changes_value(prices):
    short = realized_vol(prices, lookback=10).dropna()
    long = realized_vol(prices, lookback=60).dropna()
    assert not np.allclose(short.iloc[-1].values, long.iloc[-1].values)


def test_realized_vol_rejects_lookback_lt_2():
    px = pd.DataFrame({"A": [100.0, 101, 102]})
    with pytest.raises(ValueError):
        realized_vol(px, lookback=1)


def test_inverse_vol_weights_sum_to_one(prices):
    v = realized_vol(prices, lookback=20).dropna()
    w = inverse_vol_weights(v)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
    assert (w >= 0).all().all()


def test_inverse_vol_weights_lower_vol_higher_weight():
    v = pd.DataFrame({"A": [0.10], "B": [0.40]})
    w = inverse_vol_weights(v)
    assert w["A"].iloc[0] > w["B"].iloc[0]


def test_vol_target_hits_target():
    """Equal-weight 2 iid assets each σ=20%: portfolio σ = 0.20/√2 ≈ 0.1414.
    Target 10% should scale weights by 10/14.14 ≈ 0.707."""
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]})
    vol = pd.DataFrame({"A": [0.20], "B": [0.20]})
    scaled = vol_target_scale(weights, vol, target_vol=0.10, leverage_cap=2.0)
    pv = float(np.sqrt(((scaled ** 2) * (vol ** 2)).sum(axis=1)).iloc[0])
    assert abs(pv - 0.10) < 1e-6


def test_vol_target_respects_leverage_cap():
    weights = pd.DataFrame({"A": [1.0]})
    vol = pd.DataFrame({"A": [0.05]})  # need 10x leverage to hit 50% vol → cap kicks in
    scaled = vol_target_scale(weights, vol, target_vol=0.50, leverage_cap=1.5)
    assert scaled.iloc[0, 0] == pytest.approx(1.5)


def test_kelly_zero_mu_yields_zero():
    mu = pd.DataFrame({"A": [0.0], "B": [0.0]})
    sigma = pd.DataFrame({"A": [0.10], "B": [0.20]})
    w = kelly_weights(mu, sigma, fraction=0.5, long_only=True)
    assert (w == 0).all().all()


def test_kelly_lower_vol_gets_more_weight():
    """Same μ on both names; lower σ gets more weight (vol²)."""
    mu = pd.DataFrame({"A": [0.10], "B": [0.10]})
    sigma = pd.DataFrame({"A": [0.10], "B": [0.20]})
    w = kelly_weights(mu, sigma, fraction=0.5, leverage_cap=10.0)
    # Ratio ≈ (0.20² / 0.10²) = 4
    assert abs(w["A"].iloc[0] / w["B"].iloc[0] - 4.0) < 1e-6


def test_kelly_long_only_clips_negative_mu():
    mu = pd.DataFrame({"A": [-0.05], "B": [0.10]})
    sigma = pd.DataFrame({"A": [0.10], "B": [0.10]})
    w = kelly_weights(mu, sigma, fraction=1.0, leverage_cap=10.0, long_only=True)
    assert w["A"].iloc[0] == 0
    assert w["B"].iloc[0] > 0


def test_kelly_caps_extreme_leverage():
    """μ/σ² blowup → row should be scaled to leverage cap."""
    mu = pd.DataFrame({"A": [0.20]})
    sigma = pd.DataFrame({"A": [0.05]})  # full Kelly = 0.20/0.0025 = 80x
    w = kelly_weights(mu, sigma, fraction=1.0, leverage_cap=1.5, long_only=True)
    assert w.iloc[0, 0] == pytest.approx(1.5)
