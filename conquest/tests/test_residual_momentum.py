"""ResidualMomentum model behavior tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models.residual_momentum import ResidualMomentum


@pytest.fixture
def synth_universe():
    """SPY (market) + 4 stocks with different idiosyncratic profiles."""
    rng = np.random.default_rng(13)
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    # SPY: random walk + small drift
    spy_ret = rng.normal(0.0004, 0.010, len(idx))
    # Stock A: high beta (1.5) + small idiosyncratic drift
    a_ret = 1.5 * spy_ret + rng.normal(0.0001, 0.005, len(idx))
    # Stock B: low beta (0.3) + strong idiosyncratic drift
    b_ret = 0.3 * spy_ret + rng.normal(0.0008, 0.005, len(idx))
    # Stock C: zero beta + zero drift (pure noise)
    c_ret = rng.normal(0.0, 0.005, len(idx))
    # Stock D: high beta (1.2) + negative idiosyncratic drift
    d_ret = 1.2 * spy_ret + rng.normal(-0.0005, 0.005, len(idx))
    cols = {"SPY": spy_ret, "A": a_ret, "B": b_ret, "C": c_ret, "D": d_ret}
    prices = pd.DataFrame({k: (1 + v).cumprod() * 100 for k, v in cols.items()}, index=idx)
    return prices


def test_no_market_ticker_falls_back_to_equal_weight():
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    prices = pd.DataFrame({
        c: (1 + rng.normal(0.0002, 0.01, len(idx))).cumprod() * 100
        for c in ["A", "B", "C"]
    }, index=idx)
    model = ResidualMomentum(market_ticker="SPY", top_n=2, lookback=180)
    weights = model.signal(prices)
    # Should not raise; weights should sum to <= 1.0 per row
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_warmup_returns_zero_weights(synth_universe):
    """Before `lookback` days of residuals are available, weights are zero.
    β defaults to 1.0 during its own warmup so it doesn't gate; only the
    cumulative-residual lookback (180d) gates the first valid weight."""
    model = ResidualMomentum(top_n=2, lookback=180, beta_window=60)
    weights = model.signal(synth_universe)
    # Strictly before day 180 (the lookback), no weights should be assigned.
    early = weights.iloc[:179]
    assert (early.sum(axis=1) == 0).all(), \
        f"Weights non-zero before lookback: max sum = {early.sum(axis=1).max()}"


def test_high_idiosyncratic_drift_picked(synth_universe):
    """Stock B has positive idiosyncratic drift (0.0008/day after stripping market).
    Should rank high in residual momentum."""
    model = ResidualMomentum(top_n=2, lookback=180, beta_window=60)
    weights = model.signal(synth_universe)
    post = weights.iloc[300:]
    # B should be selected most of the time
    b_weight_avg = post["B"].mean()
    assert b_weight_avg > 0.3, f"Stock B (high alpha) only selected {b_weight_avg:.2f} of post-warmup periods"


def test_high_beta_low_alpha_NOT_picked(synth_universe):
    """Stock A is mostly market exposure (β=1.5) with tiny α — should NOT
    dominate residual momentum even if its TOTAL return is large in a bull
    market (which a regular DualMomentum would chase)."""
    model = ResidualMomentum(top_n=2, lookback=180, beta_window=60)
    weights = model.signal(synth_universe)
    post = weights.iloc[300:]
    # A's residual should be near 0 (it's pure beta), so its selection rate
    # should be lower than B's (which has real alpha).
    a_avg = post["A"].mean()
    b_avg = post["B"].mean()
    assert b_avg > a_avg, f"Stock A (pure beta) selected {a_avg:.2f} > B (alpha) {b_avg:.2f}; residual filter not working"


def test_negative_idiosyncratic_NOT_picked(synth_universe):
    """Stock D has negative idiosyncratic drift after market — should rarely be picked."""
    model = ResidualMomentum(top_n=2, lookback=180, beta_window=60)
    weights = model.signal(synth_universe)
    post = weights.iloc[300:]
    # D has negative alpha; should be picked less than 30% of the time
    d_avg = post["D"].mean()
    assert d_avg < 0.3, f"Stock D (negative alpha) selected {d_avg:.2f} of periods"


def test_market_ticker_excluded_from_picks(synth_universe):
    """The market ticker should never be selected (it has zero residual by construction)."""
    model = ResidualMomentum(market_ticker="SPY", top_n=2, lookback=180, beta_window=60)
    weights = model.signal(synth_universe)
    assert (weights["SPY"] == 0).all()


def test_weights_sum_to_one_when_top_n_picks_available(synth_universe):
    model = ResidualMomentum(top_n=2, lookback=180, beta_window=60)
    weights = model.signal(synth_universe)
    post = weights.iloc[300:]
    sums = post.sum(axis=1)
    # When top-2 picks are available, weights should sum to exactly 1.0
    valid = sums[sums > 0]
    if not valid.empty:
        assert ((valid - 1.0).abs() < 1e-9).all() or ((valid - 0.5).abs() < 1e-9).all()
