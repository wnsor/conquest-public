"""Parity test scaffold: conquest.indicators (pandas) vs Lean's built-in indicators.

Status (v7.1): active strict parity verified for RSI / MACD / TRIX / MOMP. ADX
is wired with a stub that auto-enables once high/low are captured.

Implementation conventions:
- Skip windows account for Wilder/EMA convergence: Lean enters warmed-up after
  `set_warm_up(timedelta(days=400))`; pandas cold-starts on the captured close
  series. Skipping the convergence band makes the comparison fair.
- MOMP / TRIX are scaled x100 in the assertion since Lean returns percent while
  conquest.indicators returns decimal (the bake-off + research API stays decimal).

Captured data: ``storage/conquest/parity/lean_indicators.json`` (1761 daily
records, SPY 2018-2024).

Re-running the data capture (also picks up new high/low fields for ADX):
    lean backtest lean_parity_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conquest.indicators.adx import adx
from conquest.indicators.macd import macd
from conquest.indicators.momp import momp
from conquest.indicators.rsi import rsi
from conquest.indicators.trix import trix


def _find_parity_json() -> Path | None:
    """Locate lean_indicators.json by walking up from the test file.

    Works in the main checkout (storage/ is at the workspace root) and in
    `.claude/worktrees/*` worktrees (storage/ lives in the parent checkout
    five levels above the test file).
    """
    here = Path(__file__).resolve()
    for ancestor in list(here.parents)[2:8]:
        candidate = ancestor / "storage" / "conquest" / "parity" / "lean_indicators.json"
        if candidate.exists():
            return candidate
    return None


PARITY_JSON = _find_parity_json()


def _has_parity_data() -> bool:
    return PARITY_JSON is not None


@pytest.fixture(scope="module")
def parity_df() -> pd.DataFrame:
    if not _has_parity_data():
        pytest.skip("Run `lean backtest lean_parity_check` first.")
    records = json.loads(PARITY_JSON.read_text())
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@pytest.mark.skipif(not _has_parity_data(), reason="No parity data captured.")
def test_parity_data_captured(parity_df):
    """Sanity: confirm Lean parity backtest produced sensible records."""
    assert len(parity_df) > 1000
    for col in ("close", "rsi", "macd_line", "macd_signal", "trix", "momp", "adx"):
        assert col in parity_df.columns
    assert parity_df["close"].between(200, 700).all()
    assert parity_df["rsi"].between(0, 100).all()
    assert parity_df["adx"].between(0, 100).all()


# ---------------------------------------------------------------------------
# Strict parity tests. Skip windows and tolerances per v7.1 reconciliation:
#   RSI(14)         skip 150   atol 1e-2
#   MACD(12,26,9)   skip 130   atol 1e-3   (line + signal + histogram)
#   TRIX(15)        skip 200   atol 1e-3   after x100 percent scale
#   MOMP(90)        skip 90    atol 1e-9   after x100 percent scale
#   ADX(14)         skip 140   atol 5e-3   (auto-skipped until H/L captured)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_parity_data(), reason="No parity data captured.")
def test_rsi_parity_strict(parity_df):
    pandas_rsi = rsi(parity_df["close"], period=14)
    lean_rsi = parity_df["rsi"]
    skip = 150
    np.testing.assert_allclose(
        pandas_rsi.iloc[skip:].values,
        lean_rsi.iloc[skip:].values,
        atol=1e-2,
        err_msg="RSI(14) parity failure past warmup",
    )


@pytest.mark.skipif(not _has_parity_data(), reason="No parity data captured.")
def test_macd_parity_strict(parity_df):
    pandas_macd = macd(parity_df["close"], fast=12, slow=26, signal=9)
    skip = 130
    for col in ("line", "signal", "histogram"):
        np.testing.assert_allclose(
            pandas_macd[col].iloc[skip:].values,
            parity_df[f"macd_{col}"].iloc[skip:].values,
            atol=1e-3,
            err_msg=f"MACD {col} parity failure past warmup",
        )


@pytest.mark.skipif(not _has_parity_data(), reason="No parity data captured.")
def test_trix_parity_strict(parity_df):
    pandas_trix = trix(parity_df["close"], period=15)
    lean_trix = parity_df["trix"]
    skip = 200
    np.testing.assert_allclose(
        pandas_trix.iloc[skip:].values * 100.0,
        lean_trix.iloc[skip:].values,
        atol=1e-3,
        err_msg="TRIX(15) parity failure past warmup (after x100 percent scale)",
    )


@pytest.mark.skipif(not _has_parity_data(), reason="No parity data captured.")
def test_momp_parity_strict(parity_df):
    pandas_momp = momp(parity_df["close"], period=90)
    lean_momp = parity_df["momp"]
    skip = 90
    np.testing.assert_allclose(
        pandas_momp.iloc[skip:].values * 100.0,
        lean_momp.iloc[skip:].values,
        atol=1e-9,
        err_msg="MOMP(90) parity failure (after x100 percent scale)",
    )


@pytest.mark.skipif(not _has_parity_data(), reason="No parity data captured.")
def test_adx_parity_strict(parity_df):
    """ADX(14) parity. Auto-skips until lean_parity_check captures H/L.

    To enable: rerun `lean backtest lean_parity_check` (the Algorithm now logs
    bar.high and bar.low alongside close) and the test activates automatically.
    """
    if "high" not in parity_df.columns or "low" not in parity_df.columns:
        pytest.skip("ADX parity requires high/low columns; rerun lean_parity_check.")
    ohlc = pd.DataFrame({
        "High":  parity_df["high"],
        "Low":   parity_df["low"],
        "Close": parity_df["close"],
    })
    pandas_adx = adx(ohlc, period=14)
    lean_adx = parity_df["adx"]
    skip = 140
    np.testing.assert_allclose(
        pandas_adx.iloc[skip:].values,
        lean_adx.iloc[skip:].values,
        atol=5e-3,
        err_msg="ADX(14) parity failure past warmup",
    )
