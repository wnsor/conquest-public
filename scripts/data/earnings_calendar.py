"""
Build an earnings calendar from Yahoo Finance (forward + historical EPS surprise).

The DIY-mode alternative to Polygon.io: pull per-ticker earnings calendar
from yfinance for the strategies that need it (PEAD calls A5, pre-earnings
straddles C1/C2, FOMC/CPI/NFP plays C3-C5). Yahoo provides both forward
announce dates and historical actual-vs-estimate EPS surprise.

A future Phase-2 enhancement is to cross-validate against SEC EDGAR 8-K
Item 2.02 filings (canonical earnings-release date). Phase 0 ships the
yfinance pull only — it's enough to wire downstream strategy testing.

Output:
    storage/conquest/options/earnings_calendar.csv
        columns: ticker, earnings_date, time_of_day, eps_estimate,
                 eps_actual, surprise_pct

Usage:
    python scripts/data/earnings_calendar.py --smoke 90d           # next 90d forward, smoke universe
    python scripts/data/earnings_calendar.py --tickers AAPL,MSFT   # specific tickers
    python scripts/data/earnings_calendar.py --universe sp100      # all S&P 100
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_CSV = WORKSPACE / "storage" / "conquest" / "options" / "earnings_calendar.csv"

# Smoke universe — a small representative basket. Phase 2 will expand to
# the full strategy-specific universe (e.g. cgrowth top-30 for PEAD calls).
SMOKE_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "V", "WMT", "COST", "UNH", "JNJ", "PG",
]


def fetch_ticker(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        df = t.earnings_dates  # forward + historical, indexed by datetime
    except Exception as e:
        print(f"  {ticker}: fetch error: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [str(c).strip() for c in df.columns]
    # yfinance returns columns like: 'Earnings Date', 'EPS Estimate',
    # 'Reported EPS', 'Surprise(%)' — normalize.
    rename = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("(", "").replace(")", "").replace("%", "")
        if "earningsdate" in cl:
            rename[c] = "earnings_date"
        elif "epsestimate" in cl:
            rename[c] = "eps_estimate"
        elif "reportedeps" in cl or "epsactual" in cl:
            rename[c] = "eps_actual"
        elif "surprise" in cl:
            rename[c] = "surprise_pct"
    df = df.rename(columns=rename)
    df["ticker"] = ticker
    # Time of day: Yahoo distinguishes BMO (before market open) / AMC
    # (after market close) via the time-of-day portion of the datetime.
    if "earnings_date" in df.columns:
        ts = pd.to_datetime(df["earnings_date"], errors="coerce")
        df["time_of_day"] = ts.dt.strftime("%H:%M").where(ts.notna(), None)
        df["earnings_date"] = ts.dt.strftime("%Y-%m-%d")
    keep = ["ticker", "earnings_date", "time_of_day", "eps_estimate",
            "eps_actual", "surprise_pct"]
    have = [c for c in keep if c in df.columns]
    return df[have].copy()


def main():
    ap = argparse.ArgumentParser(description="Pull earnings calendar from Yahoo Finance")
    ap.add_argument("--tickers", help="Comma-separated tickers")
    ap.add_argument("--universe", choices=["smoke", "sp100", "sp500"],
                    help="Predefined universe")
    ap.add_argument("--smoke", type=str,
                    help="Smoke mode: '90d' filters to forward dates within window using SMOKE_UNIVERSE.")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="Sleep between Yahoo requests to avoid throttling")
    args = ap.parse_args()

    if args.smoke:
        unit = args.smoke[-1].lower()
        n = int(args.smoke[:-1])
        days = n if unit == "d" else n * 365
        tickers = SMOKE_UNIVERSE
        cutoff_start = datetime.now().date()
        cutoff_end = cutoff_start + timedelta(days=days)
        print(f"SMOKE: forward window {cutoff_start}..{cutoff_end}, universe={tickers}")
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
        cutoff_start = cutoff_end = None
    elif args.universe == "smoke":
        tickers = SMOKE_UNIVERSE
        cutoff_start = cutoff_end = None
    elif args.universe in ("sp100", "sp500"):
        sys.exit(f"--universe {args.universe} not wired in Phase 0; use --tickers or --smoke")
    else:
        sys.exit("Pass --tickers, --universe, or --smoke")

    parts = []
    for t in tickers:
        df = fetch_ticker(t)
        if not df.empty:
            print(f"  {t}: {len(df)} earnings rows")
            parts.append(df)
        time.sleep(args.sleep)

    if not parts:
        sys.exit("No earnings data fetched")

    out = pd.concat(parts, ignore_index=True)
    if args.smoke:
        ed = pd.to_datetime(out["earnings_date"], errors="coerce").dt.date
        out = out[(ed >= cutoff_start) & (ed <= cutoff_end)].copy()
        print(f"Filtered to {len(out)} rows in smoke window")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
