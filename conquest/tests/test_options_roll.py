"""PutRollSimulator: schedule, sizing, expiry handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.options.costs import OptionsCostModel
from conquest.options.roll import (
    ConstantRoll, VIXConditionalRoll, PutRollSimulator,
)
from conquest.options.sizing import NotionalSizer, DeltaTargetSizer


@pytest.fixture
def calm_market():
    """Flat-vol calm world: SPX flat at 100, VIX = 18, VIX3M = 22, NAV $100k."""
    idx = pd.date_range("2020-01-02", periods=200, freq="B")
    spx = pd.Series(100.0, index=idx)
    vix = pd.Series(18.0, index=idx)
    vix3m = pd.Series(22.0, index=idx)
    nav = pd.Series(100_000.0, index=idx)
    beta = pd.Series(1.0, index=idx)
    return spx, vix, vix3m, nav, beta


def test_constant_roll_opens_immediately(calm_market):
    spx, vix, vix3m, nav, beta = calm_market
    sim = PutRollSimulator(tenor_days=63, strike_offset=-0.05)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    # First row should be a roll day (entry)
    assert out["is_roll_day"].iloc[0]
    assert out["contracts"].iloc[0] > 0
    assert out["premium_paid"].iloc[0] > 0


def test_constant_roll_cadence(calm_market):
    spx, vix, vix3m, nav, beta = calm_market
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    roll_days = out.index[out["is_roll_day"]]
    # We expect entries at day 0 and day 63 (expiry of first → close & re-open).
    # Note: at expiry the close & open both fall on same day, so it counts as one roll day.
    assert len(roll_days) >= 2
    # 2nd roll occurs at or after index 63
    assert (roll_days[1] - roll_days[0]).days >= 63


def test_vix_conditional_skips_high_vix():
    """When VIX is always high, no entry should ever occur."""
    idx = pd.date_range("2020-01-02", periods=100, freq="B")
    spx = pd.Series(100.0, index=idx)
    vix = pd.Series(30.0, index=idx)  # high
    vix3m = pd.Series(28.0, index=idx)
    nav = pd.Series(100_000.0, index=idx)
    beta = pd.Series(1.0, index=idx)
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=VIXConditionalRoll(threshold=15.0, tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    assert out["is_roll_day"].sum() == 0
    assert (out["contracts"] == 0).all()
    assert out["premium_paid"].sum() == 0


def test_vix_conditional_opens_when_calm():
    idx = pd.date_range("2020-01-02", periods=100, freq="B")
    spx = pd.Series(100.0, index=idx)
    vix = pd.Series(12.0, index=idx)  # low
    vix3m = pd.Series(14.0, index=idx)
    nav = pd.Series(100_000.0, index=idx)
    beta = pd.Series(1.0, index=idx)
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=VIXConditionalRoll(threshold=15.0, tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    assert out["is_roll_day"].iloc[0]
    assert out["contracts"].iloc[0] > 0


def test_notional_sizing_formula(calm_market):
    spx, vix, vix3m, nav, beta = calm_market
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    # contracts = nav * fraction * beta / (S * 100) = 100_000 * 1.0 * 1.0 / (100 * 100) = 10
    assert out["contracts"].iloc[0] == pytest.approx(10.0)


def test_half_notional_sizing(calm_market):
    spx, vix, vix3m, nav, beta = calm_market
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=0.5),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    # 5 contracts at half notional
    assert out["contracts"].iloc[0] == pytest.approx(5.0)


def test_beta_adjustment(calm_market):
    """β=1.5 should size 50% larger than β=1.0."""
    spx, vix, vix3m, nav, _ = calm_market
    beta_15 = pd.Series(1.5, index=nav.index)
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta_15,
        sizer=NotionalSizer(fraction=1.0),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    assert out["contracts"].iloc[0] == pytest.approx(15.0)


def test_expiry_clears_position():
    """At day 63 (tenor), the position should expire & a new one open."""
    idx = pd.date_range("2020-01-02", periods=130, freq="B")
    spx = pd.Series(100.0, index=idx)
    vix = pd.Series(18.0, index=idx)
    vix3m = pd.Series(22.0, index=idx)
    nav = pd.Series(100_000.0, index=idx)
    beta = pd.Series(1.0, index=idx)
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    # By day 63 the original position should have expired & a new one opened.
    # After 63 trading days (idx[63]) we expect at least 2 roll days observed.
    assert out["is_roll_day"].iloc[:64].sum() >= 2


def test_crash_payoff():
    """Big SPX drop after entry should cause a positive MTM on the put."""
    idx = pd.date_range("2020-01-02", periods=60, freq="B")
    spx = pd.Series(100.0, index=idx)
    spx.iloc[30:] = 80.0  # 20% drop after day 30
    vix = pd.Series(18.0, index=idx)
    vix.iloc[30:] = 35.0  # vol spike with the crash
    vix3m = pd.Series(22.0, index=idx)
    vix3m.iloc[30:] = 30.0
    nav = pd.Series(100_000.0, index=idx)
    beta = pd.Series(1.0, index=idx)
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta,
        sizer=NotionalSizer(fraction=1.0),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    # Strike was 95 (5% OTM at S=100). Post-crash S=80 → ITM by $15.
    # MTM at day 31 should be near intrinsic: $15/sh × 100 × 10 contracts = $15k,
    # less small discount for r > 0 (puts can trade below intrinsic in BS world).
    # Threshold $14k allows for ~7% discount headroom.
    assert out["mtm_value"].iloc[31] > 14_000


def test_delta_target_zero_when_beta_below_target():
    """If equity beta < target_net_delta, sizer should return 0 contracts."""
    idx = pd.date_range("2020-01-02", periods=10, freq="B")
    spx = pd.Series(100.0, index=idx)
    vix = pd.Series(18.0, index=idx)
    vix3m = pd.Series(22.0, index=idx)
    nav = pd.Series(100_000.0, index=idx)
    beta_low = pd.Series(0.5, index=idx)  # below target
    sim = PutRollSimulator(tenor_days=63)
    out = sim.simulate(
        spx=spx, vix=vix, vix3m=vix3m,
        equity_nav=nav, equity_beta=beta_low,
        sizer=DeltaTargetSizer(target_net_delta=0.7),
        schedule=ConstantRoll(tenor_days=63),
        cost_model=OptionsCostModel(),
    )
    assert (out["contracts"] == 0).all()
