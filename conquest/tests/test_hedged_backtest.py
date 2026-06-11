"""hedged_backtest: equity sleeve + put-roll overlay composition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.backtest.hedged import hedged_backtest
from conquest.options.costs import OptionsCostModel
from conquest.options.roll import ConstantRoll, VIXConditionalRoll
from conquest.options.sizing import NotionalSizer


@pytest.fixture
def synthetic_world():
    """Two-asset synthetic world with an inline crash."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    # Asset A: random walk with mild positive drift
    rets_a = rng.normal(0.0006, 0.012, len(idx))
    rets_a[200:220] = -0.025  # ~50% drawdown over 20 days, then resumes
    a = pd.Series((1 + rets_a).cumprod() * 100, index=idx)
    # Asset B: another random walk
    rets_b = rng.normal(0.0004, 0.010, len(idx))
    b = pd.Series((1 + rets_b).cumprod() * 100, index=idx)
    prices = pd.DataFrame({"A": a, "B": b})

    # Equal-weighted weights, monthly rebal
    signals = pd.DataFrame(0.5, index=idx, columns=["A", "B"])

    # SPY proxy: reuse asset A as the "market" (so beta ≈ 1)
    spx = a.copy()

    # VIX: low normally, spike with the crash
    vix = pd.Series(18.0, index=idx)
    vix.iloc[200:220] = 50.0
    vix3m = pd.Series(20.0, index=idx)
    vix3m.iloc[200:220] = 35.0

    return prices, signals, spx, vix, vix3m


def test_zero_size_hedge_matches_equity_only(synthetic_world):
    """Sizer with fraction=0 should leave returns identical to equity-only."""
    prices, signals, spx, vix, vix3m = synthetic_world
    out = hedged_backtest(
        prices=prices, signals=signals,
        spx=spx, vix=vix, vix3m=vix3m,
        sizer=NotionalSizer(fraction=0.0),
        roll_schedule=ConstantRoll(tenor_days=63),
        rebalance_freq="ME",
    )
    # hedge return should be all zeros (no contracts ever opened)
    assert out.roll_log["contracts"].max() == 0
    assert (out.hedge_returns.abs() < 1e-12).all()
    # Combined ≡ equity-only
    np.testing.assert_allclose(
        out.returns.values, out.equity_returns_only.values, atol=1e-12,
    )


def test_hedge_helps_in_crash(synthetic_world):
    """100% notional hedge should reduce MaxDD vs equity-only on the crash window."""
    from conquest.backtest.metrics import max_drawdown

    prices, signals, spx, vix, vix3m = synthetic_world
    eq_only = hedged_backtest(
        prices=prices, signals=signals,
        spx=spx, vix=vix, vix3m=vix3m,
        sizer=NotionalSizer(fraction=0.0),
        roll_schedule=ConstantRoll(tenor_days=63),
        rebalance_freq="ME",
    )
    hedged = hedged_backtest(
        prices=prices, signals=signals,
        spx=spx, vix=vix, vix3m=vix3m,
        sizer=NotionalSizer(fraction=1.0),
        roll_schedule=ConstantRoll(tenor_days=63),
        rebalance_freq="ME",
    )
    dd_eq = max_drawdown(eq_only.equity)
    dd_hedge = max_drawdown(hedged.equity)
    # Hedged drawdown should be less negative (i.e. closer to 0)
    assert dd_hedge > dd_eq, f"hedged dd {dd_hedge:.3%} not better than eq-only {dd_eq:.3%}"


def test_combined_returns_decompose(synthetic_world):
    """combined_returns ≈ equity_returns_only + hedge_returns (small float diff allowed)."""
    prices, signals, spx, vix, vix3m = synthetic_world
    out = hedged_backtest(
        prices=prices, signals=signals,
        spx=spx, vix=vix, vix3m=vix3m,
        sizer=NotionalSizer(fraction=0.5),
        roll_schedule=ConstantRoll(tenor_days=63),
        rebalance_freq="ME",
    )
    expected = out.equity_returns_only + out.hedge_returns
    np.testing.assert_allclose(out.returns.values, expected.values, atol=1e-12)


def test_lookahead_check(synthetic_world):
    """Shuffling future SPX prices shouldn't change the result of an already-run backtest.

    (Validates: the backtest result depends only on prices up to time t for each t.)
    Implementation: run twice, modifying spx[400:] in the second run; the
    returns at indices [0, 400) must be unchanged.
    """
    prices, signals, spx, vix, vix3m = synthetic_world
    out_a = hedged_backtest(
        prices=prices, signals=signals,
        spx=spx, vix=vix, vix3m=vix3m,
        sizer=NotionalSizer(fraction=1.0),
        roll_schedule=ConstantRoll(tenor_days=63),
        rebalance_freq="ME",
    )
    spx_b = spx.copy()
    spx_b.iloc[400:] = spx_b.iloc[400:].sample(frac=1.0, random_state=99).values
    out_b = hedged_backtest(
        prices=prices, signals=signals,
        spx=spx_b, vix=vix, vix3m=vix3m,
        sizer=NotionalSizer(fraction=1.0),
        roll_schedule=ConstantRoll(tenor_days=63),
        rebalance_freq="ME",
    )
    # Returns up to (but not including) day 400 must match within tiny float epsilon.
    np.testing.assert_allclose(
        out_a.returns.iloc[:400].values,
        out_b.returns.iloc[:400].values,
        atol=1e-10,
    )
