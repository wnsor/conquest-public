"""Keller HAA model behavior tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models.keller_haa import KellerHAA, _avg_1_3_6_12_momentum


@pytest.fixture
def synthetic_universe():
    """9 assets, 600 bars, with controllable momentum profiles."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    # All 8 offensive + TIP canary
    cols = ["SPY", "IWM", "EFA", "EEM", "VNQ", "PDBC", "IEF", "TLT", "TIP"]
    drifts = {
        "SPY": 0.0006, "IWM": 0.0005, "EFA": 0.0004, "EEM": 0.0003,
        "VNQ": 0.0002, "PDBC": 0.0001, "IEF": 0.0001, "TLT": 0.0001,
        "TIP": 0.0001,
    }
    data = {
        c: (1 + drifts[c] + rng.normal(0, 0.008, len(idx))).cumprod() * 100
        for c in cols
    }
    return pd.DataFrame(data, index=idx)


def test_avg_momentum_basic():
    """1/3/6/12 momentum on a constant series → 0; on a strictly rising series → positive."""
    flat = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=300, freq="B"))
    assert (_avg_1_3_6_12_momentum(flat).dropna() == 0).all()
    rising = pd.Series(np.linspace(100, 200, 300), index=pd.date_range("2024-01-01", periods=300, freq="B"))
    assert (_avg_1_3_6_12_momentum(rising).dropna() > 0).all()


def test_haa_rejects_missing_tickers(synthetic_universe):
    df = synthetic_universe.drop(columns=["TIP"])
    haa = KellerHAA()
    with pytest.raises(ValueError, match="missing required tickers"):
        haa.signal(df)


def test_haa_warmup_returns_zero_weights(synthetic_universe):
    """Before 12mo of history is available, weights should all be zero."""
    haa = KellerHAA()
    weights = haa.signal(synthetic_universe)
    # First ~252 days have NaN momentum on at least one series
    early = weights.iloc[:200]
    assert (early.sum(axis=1) == 0).all() or (early.sum(axis=1).max() < 1e-9)


def test_haa_risk_off_when_canary_negative():
    """Force TIP to decline → canary < 0 → 100% IEF."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    data = {c: pd.Series(100.0, index=idx) for c in
            ["SPY", "IWM", "EFA", "EEM", "VNQ", "PDBC", "IEF", "TLT"]}
    # TIP declines steadily
    data["TIP"] = pd.Series(np.linspace(100, 80, 400), index=idx)
    df = pd.DataFrame(data)
    haa = KellerHAA()
    weights = haa.signal(df)
    # After warmup (~252d), allocations should be 100% IEF
    post = weights.iloc[280:]
    assert (post["IEF"] > 0.99).all()
    other_cols = [c for c in df.columns if c != "IEF"]
    assert (post[other_cols].sum(axis=1) < 0.01).all()


def test_haa_risk_on_picks_top_4():
    """When canary > 0 and offensive assets all rising at different rates,
    HAA should pick the 4 strongest at 25% each."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    # All offensive assets rising; SPY/IWM/EFA/EEM strongest, others flat
    drifts = {
        "SPY": 0.001, "IWM": 0.0009, "EFA": 0.0008, "EEM": 0.0007,
        "VNQ": 0.0001, "PDBC": 0.0001, "IEF": 0.0001, "TLT": 0.0001,
        "TIP": 0.0005,  # canary positive
    }
    data = {c: pd.Series((1 + drifts[c]).cumprod() if False else
                         np.cumprod([1.0] + [1 + drifts[c]] * (len(idx) - 1)) * 100,
                         index=idx) for c in drifts}
    df = pd.DataFrame(data)
    haa = KellerHAA()
    weights = haa.signal(df)
    post = weights.iloc[280:]
    # Top-4 should each get 0.25
    for t in ["SPY", "IWM", "EFA", "EEM"]:
        assert (abs(post[t] - 0.25) < 1e-9).all(), f"{t} not at 25%"
    # weak ones should NOT be selected (they have positive but tiny momentum, may
    # still be top-4 actually; this assertion is about the strongest 4 winning)


def test_haa_negative_momentum_slice_goes_to_defensive():
    """Risk-on but a top-4 pick has negative momentum → its slice goes to IEF."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    # Canary positive
    tip = pd.Series(np.cumprod([1.0] + [1.0005] * 399) * 100, index=idx)
    # 3 strong winners; rest flat or negative
    data = {
        "SPY": pd.Series(np.cumprod([1.0] + [1.001] * 399) * 100, index=idx),
        "IWM": pd.Series(np.cumprod([1.0] + [1.0009] * 399) * 100, index=idx),
        "EFA": pd.Series(np.cumprod([1.0] + [1.0008] * 399) * 100, index=idx),
        # Top-4 will include one of the declining ones
        "EEM": pd.Series(np.cumprod([1.0] + [0.999] * 399) * 100, index=idx),
        "VNQ": pd.Series(np.cumprod([1.0] + [0.998] * 399) * 100, index=idx),
        "PDBC": pd.Series(np.cumprod([1.0] + [0.998] * 399) * 100, index=idx),
        "IEF": pd.Series(np.cumprod([1.0] + [0.999] * 399) * 100, index=idx),
        "TLT": pd.Series(np.cumprod([1.0] + [0.998] * 399) * 100, index=idx),
        "TIP": tip,
    }
    df = pd.DataFrame(data)
    haa = KellerHAA()
    weights = haa.signal(df)
    post = weights.iloc[280:]
    # SPY, IWM, EFA each at 0.25
    for t in ["SPY", "IWM", "EFA"]:
        assert (abs(post[t] - 0.25) < 1e-9).all()
    # The 4th pick has negative momentum → its 0.25 slice goes to IEF
    assert (post["IEF"] >= 0.25 - 1e-9).all()


def test_haa_weights_sum_to_one_post_warmup(synthetic_universe):
    haa = KellerHAA()
    weights = haa.signal(synthetic_universe)
    post = weights.iloc[300:]
    sums = post.sum(axis=1)
    # Allow tiny float wiggle
    assert ((sums - 1.0).abs() < 1e-9).all()
