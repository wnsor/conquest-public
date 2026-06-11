"""Sanity tests for conquest.models — each model must produce well-formed weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models import (
    EqualWeight, MomentumConsensus, MeanReversion,
    TrendFollow, DualMomentum, Ensemble, all_models,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """500 daily bars of 5 synthetic ETFs with mixed trends."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    drifts = [0.0006, 0.0004, 0.0002, 0.0000, -0.0002]
    cols = ["A", "B", "C", "D", "E"]
    data = {}
    for col, drift in zip(cols, drifts):
        rets = drift + rng.normal(0, 0.012, len(dates))
        data[col] = (1 + rets).cumprod() * 100
    return pd.DataFrame(data, index=dates)


@pytest.mark.parametrize("model", all_models())
def test_signal_shape(model, prices):
    w = model.signal(prices)
    assert w.shape == prices.shape
    assert (w.index == prices.index).all()
    assert list(w.columns) == list(prices.columns)


@pytest.mark.parametrize("model", all_models())
def test_signal_no_nan(model, prices):
    w = model.signal(prices)
    assert not w.isna().any().any(), f"{model.name} produced NaN weights"


@pytest.mark.parametrize("model", all_models())
def test_signal_row_sum_bounded(model, prices):
    w = model.signal(prices)
    abs_sum = w.abs().sum(axis=1)
    # Allow tiny float overshoot
    assert (abs_sum <= 1.0 + 1e-9).all(), (
        f"{model.name} row sum exceeds 1.0 — max {abs_sum.max():.6f}"
    )


def test_equal_weight_sums_to_one(prices):
    w = EqualWeight().signal(prices)
    # All five symbols active everywhere → weight 0.2 each
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(w.iloc[-1].values, 0.2, atol=1e-9)


def test_momentum_consensus_top_n_respected(prices):
    m = MomentumConsensus(top_n=2)
    w = m.signal(prices)
    # On any active row, at most top_n names with non-zero weight
    nonzero_per_row = (w > 0).sum(axis=1)
    assert (nonzero_per_row <= 2).all()


def test_dual_momentum_rejects_negative(prices):
    m = DualMomentum(top_n=5, lookback=60)
    w = m.signal(prices)
    # The deeply-negative-drift series ("E") should rarely (ideally never) be held
    e_active_pct = (w["E"] > 0).mean()
    assert e_active_pct < 0.5, f"DualMomentum held losing series too often: {e_active_pct:.2%}"
