"""Unit tests for conquest_options/edge_signals/stock_picker.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# bare-sibling import per memory feedback_lean_bare_imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "conquest_options"))

from edge_signals.stock_picker import (  # noqa: E402
    DEFAULT_WEIGHTS, ScoredTicker, StockPicker, TickerInputs,
    _breakout_score, _liquidity_gate, _minmax_scale,
    _momentum_180d, _news_growth_5d, _vol_60d, _vol_z_score,
)


# ── pure function tests ─────────────────────────────────────────────────

def test_momentum_180d_basic():
    # 270 prices: index [0..89]=100, [90..179]=120, [180..269]=150
    # prices[-180] = prices[90] = 120; prices[-1] = prices[269] = 150
    # 150 / 120 = 1.25
    prices = [100.0] * 90 + [120.0] * 90 + [150.0] * 90
    assert _momentum_180d(prices) == pytest.approx(1.25, abs=0.01)


def test_momentum_180d_insufficient_history_none():
    assert _momentum_180d([100.0] * 10) is None
    assert _momentum_180d([]) is None


def test_vol_60d_constant_prices_is_zero():
    prices = [100.0] * 100
    v = _vol_60d(prices)
    assert v == 0.0 or v is None    # near-zero acceptable


def test_vol_60d_volatile_prices_positive():
    prices = [100.0]
    for i in range(99):
        prices.append(prices[-1] * (1.02 if i % 2 == 0 else 0.98))
    v = _vol_60d(prices)
    assert v is not None and v > 0


def test_vol_z_score_neg_when_below_mean():
    history = [0.02] * 100   # baseline vol = 2%
    z = _vol_z_score(0.01, history)
    # Current 0.01 < mean 0.02 → z < 0; but variance is 0 here. Use varied history.
    history_varied = [0.015, 0.02, 0.025, 0.03, 0.02, 0.015, 0.02, 0.025] * 40
    z = _vol_z_score(0.015, history_varied)
    assert z is not None and z < 0


def test_news_growth_5d_acceleration():
    # 5 days each: prior = 10/day, recent = 30/day → 3x
    counts = [10] * 5 + [30] * 5
    assert _news_growth_5d(counts) == pytest.approx(3.0, abs=0.01)


def test_news_growth_5d_no_history_none():
    assert _news_growth_5d([10] * 3) is None


def test_breakout_score_at_high():
    prices = [100.0] * 30 + [200.0] * 30   # 60 prices
    prices.append(200.0)   # at the high
    score = _breakout_score(prices)
    assert score == pytest.approx(1.0, abs=0.01)


def test_breakout_score_at_low():
    prices = [200.0] * 30 + [100.0] * 30
    prices.append(100.0)
    score = _breakout_score(prices)
    assert score == pytest.approx(0.0, abs=0.01)


def test_liquidity_gate_passes_when_both_above_min():
    assert _liquidity_gate(adv_dollars=100_000_000, option_chain_size=200) is True


def test_liquidity_gate_fails_when_adv_too_low():
    assert _liquidity_gate(adv_dollars=10_000_000, option_chain_size=200) is False


def test_liquidity_gate_fails_when_chain_too_thin():
    assert _liquidity_gate(adv_dollars=100_000_000, option_chain_size=10) is False


def test_minmax_scale_basic():
    out = _minmax_scale({"a": 1.0, "b": 2.0, "c": 3.0})
    assert out["a"] == pytest.approx(0.0)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(1.0)


def test_minmax_scale_drops_none():
    out = _minmax_scale({"a": 1.0, "b": None, "c": 3.0})
    assert "b" not in out
    assert "a" in out and "c" in out


# ── StockPicker integration ────────────────────────────────────────────

def _make_inputs(name, mom_pct, vol, news_recent, news_prior, breakout):
    """Helper: synthesize 252d price history producing target factors.

    Builds a clean linear ramp so momentum_180d = (1 + mom_pct) exactly:
      prices[0..71]    = 100.0 (flat past)
      prices[72..251]  = linear from 100 to 100*(1+mom_pct)
    Then overlays vol noise on the FULL 252d so vol_60d > 0.
    """
    n = 252
    end_price = 100.0 * (1 + mom_pct)
    # Build base ramp: prices[-180] = 100 (start of ramp), prices[-1] = end_price
    prices = [100.0] * (n - 180)
    for i in range(180):
        t = i / 179.0
        prices.append(100.0 + (end_price - 100.0) * t)
    # Apply vol noise across full series. Use deterministic alternating pattern.
    noisy = []
    for i, p in enumerate(prices):
        noisy.append(p * (1 + vol * ((i % 2) * 2 - 1)))
    if breakout:
        noisy[-1] = max(noisy)
    return TickerInputs(
        prices_252d=noisy,
        news_count_10d=[news_prior] * 5 + [news_recent] * 5,
        adv_dollars=100_000_000,
        option_chain_size=200,
        iv_rank=50.0,
    )


def test_picker_ranks_high_momentum_above_low():
    p = StockPicker()
    out = p.rank({
        "HIGH": _make_inputs("HIGH", mom_pct=0.50, vol=0.01,
                              news_recent=30, news_prior=10, breakout=True),
        "LOW":  _make_inputs("LOW",  mom_pct=-0.20, vol=0.05,
                              news_recent=5, news_prior=15, breakout=False),
    })
    assert len(out) == 2
    assert out[0].ticker == "HIGH"
    assert out[1].ticker == "LOW"


def test_picker_excludes_liquidity_failures():
    p = StockPicker()
    out = p.rank({
        "OK": _make_inputs("OK", 0.30, 0.01, 30, 10, True),
        "ILLIQUID": TickerInputs(
            prices_252d=[100.0] * 252,
            news_count_10d=[10] * 10,
            adv_dollars=1_000_000,    # below 50M min
            option_chain_size=200,
        ),
    })
    assert len(out) == 1
    assert out[0].ticker == "OK"


def test_top_n_returns_ticker_list():
    p = StockPicker()
    inputs = {
        f"T{i}": _make_inputs(f"T{i}", mom_pct=0.10 + i * 0.05, vol=0.02,
                               news_recent=20, news_prior=15, breakout=False)
        for i in range(10)
    }
    top_3 = p.top_n(inputs, n=3)
    assert len(top_3) == 3
    # Highest mom should rank first
    assert top_3[0] == "T9"


def test_empty_inputs_returns_empty():
    p = StockPicker()
    assert p.rank({}) == []
    assert p.top_n({}) == []
