"""Faber GTAA model behavior tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.models.faber_gtaa import FaberGTAA, GTAA_5_UNIVERSE


@pytest.fixture
def synth_universe():
    """5 assets + IEF as cash, 600 bars."""
    rng = np.random.default_rng(11)
    idx = pd.date_range("2018-01-01", periods=600, freq="B")
    drifts = {"SPY": 0.0008, "EFA": 0.0004, "VNQ": 0.0003, "PDBC": 0.0001, "IEF": 0.00005}
    data = {c: (1 + drifts[c] + rng.normal(0, 0.008, len(idx))).cumprod() * 100 for c in drifts}
    return pd.DataFrame(data, index=idx)


def test_missing_tickers_raises(synth_universe):
    df = synth_universe.drop(columns=["SPY"])
    gtaa = FaberGTAA()
    with pytest.raises(ValueError, match="missing required tickers"):
        gtaa.signal(df)


def test_warmup_zero_weights():
    """First sma_window_days bars should have zero weight (no SMA available)."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    data = {c: pd.Series(np.linspace(100, 110, 400), index=idx) for c in GTAA_5_UNIVERSE}
    df = pd.DataFrame(data)
    gtaa = FaberGTAA(sma_months=10)
    weights = gtaa.signal(df)
    early = weights.iloc[:200]  # 10mo = 210bd; first 200 should be all zero
    assert (early.sum(axis=1) == 0).all()


def test_above_sma_gets_slice():
    """If price always above SMA, asset gets 1/N consistently."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    # Strictly rising assets → always above SMA after warmup
    data = {c: pd.Series(np.linspace(100, 200, 400), index=idx) for c in GTAA_5_UNIVERSE}
    df = pd.DataFrame(data)
    gtaa = FaberGTAA(sma_months=10)
    weights = gtaa.signal(df)
    post = weights.iloc[300:]
    for t in GTAA_5_UNIVERSE:
        assert (abs(post[t] - 0.20) < 1e-9).all()


def test_below_sma_zero_weight_no_cash():
    """If price always below SMA and no cash ticker, asset gets 0."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    data = {c: pd.Series(np.linspace(200, 100, 400), index=idx) for c in GTAA_5_UNIVERSE}
    df = pd.DataFrame(data)
    gtaa = FaberGTAA(sma_months=10, cash_ticker=None)
    weights = gtaa.signal(df)
    post = weights.iloc[300:]
    assert (post.sum(axis=1) == 0).all()


def test_cash_ticker_receives_below_sma_slices():
    """If cash_ticker is set, below-SMA slices accumulate to cash."""
    idx = pd.date_range("2018-01-01", periods=400, freq="B")
    universe = ["SPY", "EFA", "VNQ", "PDBC", "IEF"]
    # SPY rising (above SMA), others falling (below SMA), with BIL as cash
    data = {
        "SPY":  pd.Series(np.linspace(100, 200, 400), index=idx),
        "EFA":  pd.Series(np.linspace(200, 100, 400), index=idx),
        "VNQ":  pd.Series(np.linspace(200, 100, 400), index=idx),
        "PDBC": pd.Series(np.linspace(200, 100, 400), index=idx),
        "IEF":  pd.Series(np.linspace(200, 100, 400), index=idx),
        "BIL":  pd.Series(100.0, index=idx),
    }
    df = pd.DataFrame(data)
    gtaa = FaberGTAA(universe=universe, sma_months=10, cash_ticker="BIL")
    weights = gtaa.signal(df)
    post = weights.iloc[300:]
    # SPY at 0.20, others at 0, BIL at 0.80
    assert (abs(post["SPY"] - 0.20) < 1e-9).all()
    assert (abs(post["BIL"] - 0.80) < 1e-9).all()
