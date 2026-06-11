"""Unit tests for scripts/export_breadth.py compute_breadth — the BULL options
gate signal math (% S&P members > 200DMA).

The generator is also validated end-to-end against the frozen historical series
(seam check mean|Δ|=0.0000), but these pure-function tests guard the breadth
math + the zero-denominator warmup path (a prior int ZeroDivisionError) against
regressions, with no network dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from export_breadth import compute_breadth, SMA_WINDOW  # noqa: E402


def _frame() -> pd.DataFrame:
    """210 business days, 3 tickers: 2 rising (end above 200DMA), 1 falling."""
    n = SMA_WINDOW + 10
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "RISE_A": np.linspace(100.0, 200.0, n),   # last price > trailing mean → above
            "FALL_B": np.linspace(200.0, 100.0, n),   # last price < trailing mean → below
            "RISE_C": np.linspace(50.0, 150.0, n),    # above
        },
        index=idx,
    )


def test_breadth_value_is_known_fraction():
    out = compute_breadth(_frame())
    # 2 of 3 names above their 200DMA on the last day → breadth = 2/3.
    assert abs(out["pct_above_200d_ma"].iloc[-1] - 2.0 / 3.0) < 1e-6


def test_breadth_in_unit_interval():
    out = compute_breadth(_frame())
    assert ((out["pct_above_200d_ma"] >= 0.0) & (out["pct_above_200d_ma"] <= 1.0)).all()


def test_no_zero_division_on_warmup():
    """The first SMA_WINDOW-1 rows have no valid 200DMA (den==0). compute_breadth
    must NOT raise (the int-division-by-zero bug) and must drop those dates."""
    frame = _frame()
    out = compute_breadth(frame)  # would raise ZeroDivisionError pre-fix
    # only fully-warmed dates survive: n - SMA_WINDOW + 1 of them
    assert len(out) == len(frame) - SMA_WINDOW + 1
    assert out.index.min() >= pd.Timestamp("2024-01-01") + pd.tseries.offsets.BDay(SMA_WINDOW - 1)


def test_all_above_is_one_all_below_is_zero():
    n = SMA_WINDOW + 5
    idx = pd.bdate_range("2024-01-01", periods=n)
    up = pd.DataFrame({"X": np.linspace(1.0, 5.0, n), "Y": np.linspace(2.0, 9.0, n)}, index=idx)
    down = pd.DataFrame({"X": np.linspace(5.0, 1.0, n), "Y": np.linspace(9.0, 2.0, n)}, index=idx)
    assert compute_breadth(up)["pct_above_200d_ma"].iloc[-1] == 1.0
    assert compute_breadth(down)["pct_above_200d_ma"].iloc[-1] == 0.0
