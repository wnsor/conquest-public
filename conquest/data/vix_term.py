"""VIX term structure + credit-stress proxy from yfinance.

Why not FRED HY OAS? ICE Data Indices restricted FRED's BAMLH0A0HYM2 to the
last 3 years as of April 2026 — useless for the 2014-2024 backtest window.
Instead we compute a credit-stress PROXY from ETFs we already have:

    hy_stress = log(HYG_t / IEF_t) - log(HYG_{t-window} / IEF_{t-window})

When HYG underperforms IEF, credit is stressing. This is highly correlated
with HY OAS widening but is computed from instruments we can backtest.

VIX term-structure signals:
- VIX9D / VIX (front-month vol curve): inversion (>1) = near-term stress imminent
- VIX / VIX3M: inversion = backwardation = high-stress regime

All series fetched from yfinance, cached as parquet for repeat use.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


VIX_SYMBOLS = {
    "^VIX":   "vix",
    "^VIX9D": "vix9d",
    "^VIX3M": "vix3m",
    "^VVIX":  "vvix",
    "^MOVE":  "move",
}

CREDIT_PROXY_TICKERS = {
    "HYG": "hyg",
    "IEF": "ief",
}


def fetch_vix_term(
    cache_dir: Path | None = None,
    refresh: bool = False,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Fetch VIX, VIX9D, VIX3M, VVIX, MOVE — all close prices, daily."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / \
                    "data" / "alternative" / "conquest" / "raw" / "prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "vix_term_daily.parquet"

    if cache_path.exists() and not refresh:
        df = pd.read_parquet(cache_path)
        if df.index[0] <= pd.Timestamp(start) and df.index[-1] >= pd.Timestamp(end):
            return df.loc[start:end]

    import yfinance as yf
    out = {}
    for sym, label in VIX_SYMBOLS.items():
        df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            continue
        c = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        if hasattr(c, "shape") and len(c.shape) > 1:
            c = c.iloc[:, 0]
        out[label] = c
    panel = pd.concat(out, axis=1).sort_index()
    panel.index.name = "date"
    panel.to_parquet(cache_path)
    return panel


def credit_stress_proxy(
    hyg: pd.Series,
    ief: pd.Series,
    lookback_days: int = 60,
) -> pd.Series:
    """Synthetic HY stress proxy: relative drawdown of HYG vs IEF.

    Returns a series in [-inf, +inf] where:
        > 0  = HYG outperforming IEF (risk-on, credit calm)
        < 0  = HYG underperforming IEF (credit stress)
        sharp negative spikes = credit stress events

    Computed as the difference of log-returns over `lookback_days`:
        proxy_t = log(HYG_t/HYG_{t-N}) - log(IEF_t/IEF_{t-N})

    Aligns to the intersection of the two input series.
    """
    import numpy as np
    common = hyg.index.intersection(ief.index)
    h = hyg.reindex(common).ffill()
    i = ief.reindex(common).ffill()
    log_h = np.log(h)
    log_i = np.log(i)
    proxy = (log_h - log_h.shift(lookback_days)) - (log_i - log_i.shift(lookback_days))
    return proxy.rename("hy_stress_proxy")


def vix_term_inversion(
    vix: pd.Series,
    vix3m: pd.Series,
) -> pd.Series:
    """Returns VIX/VIX3M ratio. > 1 = backwardation (stress regime). < 1 = contango (calm)."""
    common = vix.index.intersection(vix3m.index)
    return (vix.reindex(common) / vix3m.reindex(common)).rename("vix_term_ratio")
