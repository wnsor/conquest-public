"""Tests for v4.5 drawdown-shape metrics: Ulcer Index, Avg DD, Time in DD %, Recovery Factor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conquest.backtest.metrics import (
    ulcer_index, avg_drawdown, time_in_drawdown_pct, recovery_factor,
    metrics_summary,
)


def test_ulcer_index_zero_for_strictly_rising_equity():
    """No drawdowns ⇒ ulcer = 0."""
    equity = pd.Series(np.linspace(100, 200, 100))
    assert ulcer_index(equity) == pytest.approx(0.0)


def test_ulcer_index_positive_with_drawdowns():
    equity = pd.Series([100, 110, 105, 95, 100, 105, 110])  # dips below peak
    u = ulcer_index(equity)
    assert u > 0


def test_ulcer_index_penalises_deep_more_than_shallow():
    """Two equity series with same average drawdown but different depths:
    deep_then_recover should have HIGHER ulcer than chronic_shallow because
    the squared term inflates the penalty for big dips."""
    deep = pd.Series([100, 100, 100, 50, 100, 100, 100, 100])  # one big dip
    shallow = pd.Series([100, 95, 95, 95, 95, 95, 95, 95])     # chronic small dip
    # Same mean drawdown (~6%), but ulcer should differ
    assert avg_drawdown(deep) == pytest.approx(avg_drawdown(shallow), abs=0.02)
    assert ulcer_index(deep) > ulcer_index(shallow)


def test_avg_drawdown_non_positive():
    """avg_drawdown is ≤ 0 (drawdowns are non-positive)."""
    equity = pd.Series([100, 110, 95, 105, 90, 100])
    assert avg_drawdown(equity) <= 0


def test_avg_drawdown_zero_for_strict_uptrend():
    equity = pd.Series(np.linspace(100, 200, 50))
    assert avg_drawdown(equity) == pytest.approx(0.0)


def test_time_in_dd_pct_in_unit_interval():
    equity = pd.Series([100, 95, 100, 105, 95, 105, 110, 100])
    t = time_in_drawdown_pct(equity)
    assert 0 <= t <= 1


def test_time_in_dd_pct_one_for_strict_downtrend():
    """Strict downtrend ⇒ every bar after first is below previous all-time high."""
    equity = pd.Series(np.linspace(100, 50, 50))
    t = time_in_drawdown_pct(equity)
    # First bar can't be in DD; all others should be
    assert t > 0.95


def test_time_in_dd_pct_zero_for_strict_uptrend():
    equity = pd.Series(np.linspace(100, 200, 50))
    assert time_in_drawdown_pct(equity) == pytest.approx(0.0)


def test_recovery_factor_positive_when_profitable():
    equity = pd.Series([100, 90, 100, 120, 110, 130])
    rf = recovery_factor(equity)
    assert rf > 0
    # Net profit is 30%; max DD is -10/100 = -10%; rf ≈ 3
    assert rf == pytest.approx(3.0, rel=0.1)


def test_recovery_factor_handles_no_drawdown():
    """No drawdown ⇒ recovery factor undefined (NaN)."""
    equity = pd.Series(np.linspace(100, 200, 50))
    assert np.isnan(recovery_factor(equity))


def test_metrics_summary_includes_v4_5_keys():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 252))
    equity = (1 + rets).cumprod() * 100_000
    turnover = pd.Series(0.05, index=rets.index)
    m = metrics_summary(rets, equity, turnover)
    for k in ("ulcer_index", "avg_drawdown", "time_in_dd_pct", "recovery_factor"):
        assert k in m
