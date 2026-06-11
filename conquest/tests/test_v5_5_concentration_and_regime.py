"""Tests for v5.5: HHI concentration metrics + per-regime decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.backtest.concentration import (
    hhi_series, avg_hhi, max_hhi, effective_n_series, avg_effective_n,
    max_single_name_weight,
)
from conquest.backtest.regime_breakdown import per_regime_stats, regime_breakdown_table
from conquest.backtest.engine import BacktestResult


# ---------- HHI ----------

def test_hhi_one_for_single_name():
    w = pd.DataFrame({"A": [1.0, 1.0]})
    assert avg_hhi(w) == pytest.approx(1.0)
    assert max_hhi(w) == pytest.approx(1.0)


def test_hhi_minimum_for_equal_weight():
    """Equal-weight 5 names → HHI = 5 * (0.2)^2 = 0.20."""
    w = pd.DataFrame({c: [0.2, 0.2] for c in "ABCDE"})
    assert avg_hhi(w) == pytest.approx(0.20)


def test_hhi_higher_when_concentrated():
    concentrated = pd.DataFrame({"A": [0.8], "B": [0.1], "C": [0.1]})
    diversified = pd.DataFrame({"A": [0.34], "B": [0.33], "C": [0.33]})
    assert avg_hhi(concentrated) > avg_hhi(diversified)


def test_hhi_nan_for_all_cash():
    """Bars where the row sums to 0 (all cash) get NaN HHI; with no invested bars, avg is NaN."""
    w = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    assert np.isnan(avg_hhi(w))


def test_effective_n_inverse_of_hhi():
    w = pd.DataFrame({c: [0.2] * 3 for c in "ABCDE"})  # equal-weight 5
    assert avg_effective_n(w) == pytest.approx(5.0, abs=1e-9)


def test_max_single_name_weight():
    w = pd.DataFrame({"A": [0.7, 0.3], "B": [0.3, 0.7]})
    assert max_single_name_weight(w) == pytest.approx(0.7)


# ---------- per-regime stats ----------

@pytest.fixture
def returns_and_regime():
    rng = np.random.default_rng(0)
    n = 600
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Two-regime universe: returns 50% bigger in 'good' regime
    regime = pd.Series(
        ["Disinflation"] * (n // 2) + ["Stagflation"] * (n - n // 2),
        index=idx,
    )
    drift = pd.Series(
        np.where(regime == "Disinflation", 0.001, -0.0005),
        index=idx,
    )
    rets = drift + rng.normal(0, 0.01, n)
    return rets, regime


def test_per_regime_stats_returns_one_entry_per_regime(returns_and_regime):
    returns, regime = returns_and_regime
    stats = per_regime_stats(returns, regime)
    assert set(stats.keys()) == {"Disinflation", "Stagflation"}


def test_per_regime_stats_disinflation_higher_sharpe(returns_and_regime):
    returns, regime = returns_and_regime
    stats = per_regime_stats(returns, regime)
    # By construction Disinflation has positive drift, Stagflation negative
    assert stats["Disinflation"]["sharpe"] > stats["Stagflation"]["sharpe"]


def test_per_regime_stats_includes_required_keys(returns_and_regime):
    returns, regime = returns_and_regime
    stats = per_regime_stats(returns, regime)
    expected = {"n_days", "sharpe", "hit_rate", "annual_return", "max_dd"}
    for label, vals in stats.items():
        assert set(vals.keys()) == expected


def test_per_regime_stats_short_regime_returns_nan():
    """Regime with < 30 observations should yield NaN metrics."""
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, n), index=idx)
    regime = pd.Series(["Disinflation"] * (n - 5) + ["Stagflation"] * 5, index=idx)
    stats = per_regime_stats(rets, regime)
    assert np.isnan(stats["Stagflation"]["sharpe"])
    assert stats["Stagflation"]["n_days"] == 5


def test_regime_breakdown_table_shape(returns_and_regime):
    """The wide-form regime table from a fake per-model results dict."""
    returns, regime = returns_and_regime
    equity = (1 + returns).cumprod() * 100_000
    turnover = pd.Series(0.05, index=returns.index)
    weights = pd.DataFrame({"A": [0.5] * len(returns), "B": [0.5] * len(returns)},
                           index=returns.index)
    results = {
        "model_a": BacktestResult(equity=equity, returns=returns,
                                   gross_returns=returns, turnover=turnover,
                                   weights=weights, initial_capital=100_000),
        "model_b": BacktestResult(equity=equity, returns=returns,
                                   gross_returns=returns, turnover=turnover,
                                   weights=weights, initial_capital=100_000),
    }
    table = regime_breakdown_table(results, regime)
    # 2 models × 2 regimes = 4 rows
    assert len(table) == 4
    assert set(table.columns) == {
        "model", "regime", "n_days", "sharpe", "hit_rate", "annual_return", "max_dd"
    }


# ---------- metrics_summary integration ----------

def test_metrics_summary_includes_v5_5_keys_when_weights_provided():
    from conquest.backtest.metrics import metrics_summary
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    equity = (1 + rets).cumprod() * 100_000
    turnover = pd.Series(0.05, index=idx)
    weights = pd.DataFrame({c: [0.2] * n for c in "ABCDE"}, index=idx)

    m = metrics_summary(rets, equity, turnover, weights=weights, bootstrap_n_iter=200)
    for k in ("hhi_avg", "hhi_max", "effective_n_avg", "max_single_weight"):
        assert k in m
    # Equal-weight 5 ⇒ HHI ~ 0.20, effective N ~ 5
    assert m["hhi_avg"] == pytest.approx(0.20, abs=1e-9)
    assert m["effective_n_avg"] == pytest.approx(5.0, abs=1e-9)


def test_metrics_summary_omits_v5_5_keys_without_weights():
    from conquest.backtest.metrics import metrics_summary
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    equity = (1 + rets).cumprod() * 100_000
    turnover = pd.Series(0.05, index=idx)
    m = metrics_summary(rets, equity, turnover, bootstrap_n_iter=200)
    for k in ("hhi_avg", "hhi_max", "effective_n_avg", "max_single_weight"):
        assert k not in m
