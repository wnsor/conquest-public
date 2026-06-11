"""Tests for v2 model wrappers: RegimeGated, VixGated, KellySized."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models import (
    EqualWeight, MomentumConsensus, DualMomentum,
    RegimeGated, VixGated, KellySized, all_models,
)


@pytest.fixture
def prices():
    rng = np.random.default_rng(101)
    dates = pd.date_range("2018-01-01", periods=600, freq="B")
    cols = ["SPY", "QQQ", "IWM", "TLT", "AGG"]
    drifts = [0.0005, 0.0007, 0.0004, 0.0001, 0.00005]
    return pd.DataFrame(
        {c: (1 + d + rng.normal(0, 0.012, len(dates))).cumprod() * 100
         for c, d in zip(cols, drifts)},
        index=dates,
    )


def test_regime_gated_no_regime_passes_through(prices):
    base = MomentumConsensus(top_n=3)
    gated = RegimeGated(base, regime_series=None)
    pd.testing.assert_frame_equal(gated.signal(prices), base.signal(prices))


def test_regime_gated_scales_in_risk_off(prices):
    base = EqualWeight()  # always 1/N each
    n_half = len(prices) // 2
    regime = pd.Series(
        ["Stagflation"] * n_half + ["Inflation"] * (len(prices) - n_half),
        index=prices.index,
    )
    gated = RegimeGated(base, regime_series=regime, risk_off_factor=0.5)
    g = gated.signal(prices)
    b = base.signal(prices)
    assert np.allclose(g.iloc[:n_half].values, b.iloc[:n_half].values * 0.5)
    assert np.allclose(g.iloc[n_half:].values, b.iloc[n_half:].values)


def test_vix_gated_no_vix_passes_through(prices):
    base = MomentumConsensus(top_n=3)
    gated = VixGated(base, vix_series=None)
    pd.testing.assert_frame_equal(gated.signal(prices), base.signal(prices))


def test_vix_gated_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        VixGated(EqualWeight(), vix_series=None, vix_high=10.0, vix_low=20.0)


def test_vix_gated_asymmetric_hysteresis(prices):
    base = EqualWeight()
    vix = pd.Series(10.0, index=prices.index)
    vix.iloc[100:200] = 30.0    # spike high
    vix.iloc[200:300] = 20.0    # still > 15 — should remain risk-off (asymmetric)
    vix.iloc[300:] = 12.0       # below 15 → exit risk-off

    g = VixGated(base, vix_series=vix, vix_high=25, vix_low=15,
                 risk_off_factor=0.5).signal(prices)
    b = base.signal(prices)

    # Pre-spike: full
    assert np.allclose(g.iloc[50].values, b.iloc[50].values)
    # During spike: risk-off
    assert np.allclose(g.iloc[150].values, b.iloc[150].values * 0.5)
    # Between spike and exit: still risk-off (asymmetric — VIX above 15 but below 25)
    assert np.allclose(g.iloc[250].values, b.iloc[250].values * 0.5)
    # After exit: full
    assert np.allclose(g.iloc[350].values, b.iloc[350].values)


def test_kelly_sized_holds_only_base_selected_names(prices):
    base = MomentumConsensus(top_n=2)
    sized = KellySized(base, lookback=60, fraction=0.5)
    base_w = base.signal(prices)
    sized_w = sized.signal(prices)
    base_active = base_w > 0
    leakage = (~base_active) & (sized_w > 0)
    assert not leakage.any().any()


def test_kelly_sized_respects_leverage_cap(prices):
    base = DualMomentum(top_n=5, lookback=60)
    sized = KellySized(base, lookback=60, fraction=1.0, leverage_cap=1.0)
    w = sized.signal(prices)
    assert (w.abs().sum(axis=1) <= 1.0 + 1e-9).all()


def test_all_models_v2_factory_size_and_names():
    base = all_models(include_v2=False)
    extended = all_models(include_v2=True)
    assert len(extended) == len(base) + 6
    names = {m.name for m in extended}
    expected = {
        "momentum_consensus_regime_gated",
        "dual_momentum_regime_gated",
        "momentum_consensus_vix_gated",
        "dual_momentum_vix_gated",
        "momentum_consensus_kelly",
        "dual_momentum_kelly",
    }
    assert expected.issubset(names)
