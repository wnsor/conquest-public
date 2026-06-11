"""S&P 500 constituents and GICS sector mapping.

Fetches the current S&P 500 list from Wikipedia (which mirrors the official
Standard & Poor's index methodology). Caches the result locally as parquet.

Survivorship-bias caveat
------------------------
This is the *current* S&P 500 — names that have been added since 2018 are
included, names that have been delisted/removed are not. A backtest from
2018-01-01 using this list implicitly assumes those ~480 names that "made it"
are the universe — overstating realized returns by ~50-100bps/year vs a true
point-in-time universe. Acceptable for v6 proof-of-concept; tighten in v6.x
by pulling historical constituents (QC's US Equity Constituents dataset, or
parsing Wikipedia's revision history).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests


WORKSPACE = Path(__file__).resolve().parent.parent.parent
CACHE = WORKSPACE / "data" / "alternative" / "conquest" / "raw" / "universe" / "sp500.parquet"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500(refresh: bool = False) -> pd.DataFrame:
    """Return the current S&P 500 constituent list with GICS sectors.

    Columns: ticker, security_name, gics_sector, gics_sub_industry, hq_location.
    Cached as parquet under data/alternative/conquest/raw/universe/.

    Tickers are normalized to yfinance format (e.g. 'BRK.B' → 'BRK-B').
    """
    if CACHE.exists() and not refresh:
        return pd.read_parquet(CACHE)

    print(f"Fetching S&P 500 constituents from Wikipedia ...")
    # Wikipedia returns 403 for bare urllib requests; use requests with a UA.
    resp = requests.get(
        WIKIPEDIA_URL,
        headers={"User-Agent": "conquest-research/1.0 (https://github.com/wnsor/conquest)"},
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    raw = tables[0]
    # Wikipedia column names occasionally drift; map flexibly
    cols = {c: c.lower().strip().replace(" ", "_") for c in raw.columns}
    raw = raw.rename(columns=cols)

    # Standardize column names
    rename = {}
    if "symbol" in raw.columns:
        rename["symbol"] = "ticker"
    elif "ticker" not in raw.columns:
        raise RuntimeError(f"Unexpected Wikipedia columns: {list(raw.columns)}")
    if "security" in raw.columns:
        rename["security"] = "security_name"
    if "gics_sector" not in raw.columns:
        # Sometimes appears as just "sector"
        for c in raw.columns:
            if "sector" in c:
                rename[c] = "gics_sector"
                break
    if "gics_sub-industry" in raw.columns:
        rename["gics_sub-industry"] = "gics_sub_industry"
    if "headquarters_location" in raw.columns:
        rename["headquarters_location"] = "hq_location"

    df = raw.rename(columns=rename)
    keep = [c for c in ["ticker", "security_name", "gics_sector",
                        "gics_sub_industry", "hq_location"] if c in df.columns]
    df = df[keep].copy()

    # Normalize tickers for yfinance: dots → dashes (BRK.B → BRK-B, BF.B → BF-B)
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    print(f"Cached {len(df)} S&P 500 names to {CACHE}.")
    return df


def sector_map(refresh: bool = False) -> dict[str, str]:
    """Return ticker → GICS sector mapping. Convenience wrapper around fetch_sp500()."""
    df = fetch_sp500(refresh=refresh)
    return dict(zip(df["ticker"], df["gics_sector"]))


def tickers(refresh: bool = False) -> list[str]:
    """Return just the list of S&P 500 tickers, yfinance-normalized."""
    return fetch_sp500(refresh=refresh)["ticker"].tolist()
