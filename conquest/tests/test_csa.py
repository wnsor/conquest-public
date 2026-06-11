"""Sanity tests for the CSA combiner.

These run on synthetic equity curves + synthetic price parquets so they don't
depend on yfinance or QC cloud data. The goal is to prove the rebalance loop is
dollar-correct, the bridge handoff doesn't double-count, and the metric
helpers produce sensible numbers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conquest.backtest.csa import (
    CSAConfig,
    CryptoBridgeSegment,
    align_returns,
    compute_metrics,
    default_crypto_bridge,
    default_weights,
    first_trading_days_of_month,
    load_aux_returns,
    load_fund_returns,
    probabilistic_sharpe_ratio,
    simulate,
)


@pytest.fixture
def synthetic_curves(tmp_path: Path) -> dict:
    """Build two fund equity JSONs and three aux price parquets covering 2020-2024.

    Funds: cstability rises 10%/yr, cgrowth rises 25%/yr (smooth daily compounding).
    Aux:   GLD rises 5%/yr, BIL rises 2%/yr, GBTC rises 30%/yr.
    Tickers BITO + IBIT are aliased to GBTC (single proxy active during this window).
    """
    dates = pd.bdate_range("2020-01-02", "2024-12-31")  # business days
    n = len(dates)

    def daily_return_for(annual_pct: float) -> float:
        return (1.0 + annual_pct) ** (1 / 252.0) - 1.0

    def grow(start: float, daily_ret: float) -> pd.Series:
        return pd.Series(
            start * (1.0 + daily_ret) ** np.arange(n), index=dates
        )

    cstab_eq = grow(25_000.0, daily_return_for(0.10))
    cgrow_eq = grow(25_000.0, daily_return_for(0.25))

    def write_curve_json(p: Path, fund: str, eq: pd.Series) -> None:
        p.write_text(json.dumps({
            "fund": fund,
            "version_label": "v11 LIVE",
            "engine": "synthetic",
            "period": {"start": eq.index[0].strftime("%Y-%m-%d"),
                       "end":   eq.index[-1].strftime("%Y-%m-%d")},
            "start_equity": float(eq.iloc[0]),
            "values": [
                {"date": d.strftime("%Y-%m-%d"), "equity": float(v)}
                for d, v in eq.items()
            ],
        }))

    cstab_p = tmp_path / "cstability_lean_equity.json"
    cgrow_p = tmp_path / "cgrowth_lean_equity.json"
    write_curve_json(cstab_p, "cstability", cstab_eq)
    write_curve_json(cgrow_p, "cgrowth", cgrow_eq)

    def write_close_parquet(p: Path, daily_ret: float) -> None:
        prices = 100.0 * (1.0 + daily_ret) ** np.arange(n)
        pd.DataFrame({"close": prices}, index=dates).to_parquet(p)

    gld_p = tmp_path / "gld_close.parquet"
    bil_p = tmp_path / "bil_close.parquet"
    gbtc_p = tmp_path / "gbtc_close.parquet"
    write_close_parquet(gld_p, daily_return_for(0.05))
    write_close_parquet(bil_p, daily_return_for(0.02))
    write_close_parquet(gbtc_p, daily_return_for(0.30))

    return {
        "fund_curves": {"cstability": cstab_p, "cgrowth": cgrow_p},
        "aux_prices": {"GLD": gld_p, "BIL": bil_p, "GBTC": gbtc_p,
                       "BITO": gbtc_p, "IBIT": gbtc_p},
        "dates": dates,
        "fund_equity": {"cstability": cstab_eq, "cgrowth": cgrow_eq},
    }


def test_load_fund_returns_round_trips(synthetic_curves):
    rets = load_fund_returns(synthetic_curves["fund_curves"]["cgrowth"])
    expected_daily = (1.25) ** (1 / 252.0) - 1.0
    assert len(rets) == len(synthetic_curves["dates"]) - 1
    np.testing.assert_allclose(rets.values, expected_daily, rtol=1e-10)


def test_first_trading_days_of_month_picks_first_each_month():
    idx = pd.bdate_range("2024-01-02", "2024-04-30")
    firsts = first_trading_days_of_month(idx)
    assert len(firsts) == 4  # Jan, Feb, Mar, Apr
    expected = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01"),
                pd.Timestamp("2024-03-01"), pd.Timestamp("2024-04-01")]
    assert firsts == expected


def test_load_aux_returns_bridge_handoff(synthetic_curves):
    bridge = [
        CryptoBridgeSegment("2020-01-01", "2022-01-01", "BIL"),
        CryptoBridgeSegment("2022-01-01", "2099-01-01", "GBTC"),
    ]
    aux, proxy_map = load_aux_returns(synthetic_curves["aux_prices"], bridge)
    assert "crypto" in aux.columns
    assert {"GLD", "BIL"}.issubset(aux.columns)
    pre_handoff = aux.loc["2021-12-31", "crypto"]
    post_handoff = aux.loc["2022-01-03", "crypto"]
    expected_bil_daily = (1.02) ** (1 / 252.0) - 1.0
    expected_gbtc_daily = (1.30) ** (1 / 252.0) - 1.0
    assert abs(pre_handoff - expected_bil_daily) < 1e-9
    assert abs(post_handoff - expected_gbtc_daily) < 1e-9
    assert proxy_map[0]["proxy"] == "BIL"
    assert proxy_map[1]["proxy"] == "GBTC"


def test_simulate_single_sleeve_degeneracy_cgrowth(synthetic_curves):
    """With weights = {cgrowth: 1.0}, CSA equity must equal cgrowth fund
    equity * (start_capital / fund_start_capital) at every date."""
    bridge = [CryptoBridgeSegment("2020-01-01", "2099-01-01", "GBTC")]
    cfg = CSAConfig(
        fund_curves=synthetic_curves["fund_curves"],
        aux_prices=synthetic_curves["aux_prices"],
        weights={"cgrowth": 1.0, "cstability": 0.0, "crypto": 0.0,
                 "GLD": 0.0, "BIL": 0.0},
        crypto_bridge=bridge,
        start_capital=50_000.0,
        window=("2020-01-02", "2024-12-31"),
    )
    fund_rets = {
        "cstability": load_fund_returns(cfg.fund_curves["cstability"]),
        "cgrowth": load_fund_returns(cfg.fund_curves["cgrowth"]),
    }
    aux_rets, _ = load_aux_returns(cfg.aux_prices, cfg.crypto_bridge)
    aligned = align_returns(fund_rets, aux_rets, cfg.window)
    result = simulate(aligned, cfg)

    # Expected: $50k * (cgrowth_eq[t] / cgrowth_eq[t0]) on the day-1 (which is
    # the second business day, since pct_change drops the first row).
    cgrow_eq = synthetic_curves["fund_equity"]["cgrowth"]
    aligned_dates = aligned.index
    seed_eq = cgrow_eq.loc[aligned_dates[0]]
    # The first row of the simulator already applies day-0 return; check at
    # t=10 days in to avoid edge-cases.
    test_date = aligned_dates[10]
    expected = 50_000.0 * (cgrow_eq.loc[test_date] / seed_eq) * (1 + (cgrow_eq.loc[aligned_dates[0]] / cgrow_eq.iloc[aligned_dates.get_loc(aligned_dates[0]) - 1] - 1))
    # Simpler check: the daily return on cgrowth is constant; compounding 10
    # days gives a known value. Use that instead.
    daily_ret = (1.25) ** (1 / 252.0) - 1.0
    expected_simple = 50_000.0 * (1.0 + daily_ret) ** 11  # 11 returns applied through index 10
    assert abs(result.equity.iloc[10] - expected_simple) / expected_simple < 1e-6


def test_simulate_balanced_weights_compounds_correctly(synthetic_curves):
    """With weights split evenly between two perfectly-known sleeves and no
    rebalancing churn (constant daily returns), the CSA equity at any date is
    deterministic and we can check it analytically."""
    bridge = [CryptoBridgeSegment("2020-01-01", "2099-01-01", "GBTC")]
    cfg = CSAConfig(
        fund_curves=synthetic_curves["fund_curves"],
        aux_prices=synthetic_curves["aux_prices"],
        weights={"cgrowth": 0.5, "cstability": 0.5, "crypto": 0.0,
                 "GLD": 0.0, "BIL": 0.0},
        crypto_bridge=bridge,
        start_capital=50_000.0,
        window=("2020-01-02", "2024-12-31"),
    )
    fund_rets = {
        "cstability": load_fund_returns(cfg.fund_curves["cstability"]),
        "cgrowth": load_fund_returns(cfg.fund_curves["cgrowth"]),
    }
    aux_rets, _ = load_aux_returns(cfg.aux_prices, cfg.crypto_bridge)
    aligned = align_returns(fund_rets, aux_rets, cfg.window)
    result = simulate(aligned, cfg)

    # CSA should be > buy-and-hold of the lower-return sleeve and < buy-and-hold
    # of the higher-return sleeve at every date past the seed.
    daily_low = (1.10) ** (1 / 252.0) - 1.0
    daily_high = (1.25) ** (1 / 252.0) - 1.0
    n_days = len(result.equity)
    low_only = 50_000.0 * (1.0 + daily_low) ** n_days
    high_only = 50_000.0 * (1.0 + daily_high) ** n_days
    assert low_only < result.equity.iloc[-1] < high_only
    # Sanity: end equity > start, and CAGR is between 10% and 25%.
    assert result.metrics["end_equity"] > 50_000.0
    assert 0.10 < result.metrics["cagr"] < 0.25


def test_rebalance_dates_count_and_dates_are_first_trading_days(synthetic_curves):
    cfg = CSAConfig(
        fund_curves=synthetic_curves["fund_curves"],
        aux_prices=synthetic_curves["aux_prices"],
        weights=default_weights(),
        crypto_bridge=[CryptoBridgeSegment("2020-01-01", "2099-01-01", "GBTC")],
        start_capital=50_000.0,
        window=("2020-01-02", "2024-12-31"),
    )
    fund_rets = {
        "cstability": load_fund_returns(cfg.fund_curves["cstability"]),
        "cgrowth": load_fund_returns(cfg.fund_curves["cgrowth"]),
    }
    aux_rets, _ = load_aux_returns(cfg.aux_prices, cfg.crypto_bridge)
    aligned = align_returns(fund_rets, aux_rets, cfg.window)
    result = simulate(aligned, cfg)

    # 5 years * 12 months ~= 60, minus 1 because the first trading day is the
    # seed (skipped). Allow some tolerance for synthetic-calendar edge cases.
    assert 55 <= len(result.rebalance_dates) <= 60
    for d in result.rebalance_dates:
        same_month = aligned.index[(aligned.index.year == d.year)
                                   & (aligned.index.month == d.month)]
        assert d == same_month[0]


def test_compute_metrics_cgrowth_matches_known_cagr(synthetic_curves):
    eq = synthetic_curves["fund_equity"]["cgrowth"]
    metrics = compute_metrics(eq)
    assert abs(metrics["cagr"] - 0.25) < 0.01
    assert metrics["max_dd"] == pytest.approx(0.0, abs=1e-6) or metrics["max_dd"] < 0.0
    assert metrics["end_equity"] > metrics["start_equity"]
    assert 100 < metrics["n_days"]


def test_psr_returns_probability_in_unit_interval():
    rng = np.random.default_rng(42)
    # Daily mu=0.0015, sigma=0.01 -> Sharpe ~2.4 annualized
    high_rets = pd.Series(rng.normal(0.0015, 0.01, size=500))
    p_high = probabilistic_sharpe_ratio(high_rets, sr_benchmark=0.0)
    assert 0.0 <= p_high <= 1.0
    assert p_high > 0.95  # strong Sharpe, plenty of data -> high confidence

    # Negative-Sharpe series should produce p < 0.5
    bad_rets = pd.Series(rng.normal(-0.0005, 0.01, size=500))
    p_bad = probabilistic_sharpe_ratio(bad_rets, sr_benchmark=0.0)
    assert p_bad < 0.5

    # Tiny series -> NaN
    p_short = probabilistic_sharpe_ratio(pd.Series([0.01, -0.01, 0.005]))
    assert np.isnan(p_short)


def test_default_weights_sum_to_one():
    assert abs(sum(default_weights().values()) - 1.0) < 1e-9


def test_default_crypto_bridge_covers_window():
    bridge = default_crypto_bridge()
    assert bridge[0].start == "2008-01-01"
    assert pd.Timestamp(bridge[-1].end_exclusive) >= pd.Timestamp("2026-12-31")
    # Boundaries are contiguous (next.start == prev.end_exclusive)
    for prev, nxt in zip(bridge[:-1], bridge[1:]):
        assert prev.end_exclusive == nxt.start
