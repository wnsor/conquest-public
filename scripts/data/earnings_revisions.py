"""Fetch analyst EPS-revision velocity per ticker via yfinance.

REWRITE 2026-05-27: original used `info["currentQuarterEpsEstimate"]` which
yfinance removed from the `info` dict, causing 100% None returns. Switched
to `Ticker.eps_trend` which natively returns current + 7d/30d/60d/90d-ago
EPS estimates as a DataFrame — exactly the revision-velocity signal the
strategy wants.

revision_30d_pct = (current_est - eps_30d_ago) / abs(eps_30d_ago)
  > +5%  in 30d → analyst upgrades flowing → drift premium into next earnings
  < -5%        → downgrades → reversal / vol-expansion bet

Output: storage/conquest/options/earnings_revisions_daily.csv
Schema: date, ticker, period, current_eps, eps_30d_ago, revision_30d_pct,
        eps_90d_ago, revision_90d_pct

Universe: WSB single-names + S&P 100 megacaps (where the
earnings_revision_momentum standby strategy looks).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
OUT_CSV = WORKSPACE / "storage" / "conquest" / "options" / "earnings_revisions_daily.csv"

WSB_UNIVERSE = [
    "CRDO", "MU", "NBIS", "NOK", "MX", "PL", "DRAM", "LPTH", "SOUN", "RDDT",
    "MSTR", "COIN", "PLTR", "RKLB", "APP", "SMCI",
]
EXTRA = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AVGO", "JPM", "V", "WMT", "COST", "UNH", "JNJ", "PG",
]


def fetch_one(ticker: str) -> list[dict]:
    """Return one row per (ticker × period). yfinance returns 4 periods
    per ticker — 0q (current quarter), +1q, 0y (current year), +1y.

    Each period's estimate has 5 columns: current, 7daysAgo, 30daysAgo,
    60daysAgo, 90daysAgo. We compute revision_pct at 30d and 90d horizons.
    """
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        df = t.eps_trend
    except Exception as e:
        print(f"  {ticker}: fetch error: {e}")
        return []
    if df is None or df.empty:
        return []
    rows = []
    for period in df.index:
        try:
            current = df.at[period, "current"]
            d30 = df.at[period, "30daysAgo"]
            d90 = df.at[period, "90daysAgo"]
            if pd.isna(current):
                continue
            r30 = None
            if pd.notna(d30) and d30 != 0:
                r30 = (current - d30) / abs(d30)
            r90 = None
            if pd.notna(d90) and d90 != 0:
                r90 = (current - d90) / abs(d90)
            rows.append({
                "ticker": ticker,
                "period": str(period),
                "current_eps": float(current),
                "eps_30d_ago": float(d30) if pd.notna(d30) else None,
                "revision_30d_pct": float(r30) if r30 is not None else None,
                "eps_90d_ago": float(d90) if pd.notna(d90) else None,
                "revision_90d_pct": float(r90) if r90 is not None else None,
            })
        except (KeyError, ValueError) as e:
            print(f"  {ticker} period={period}: {e}")
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="extended", choices=["wsb", "extended"])
    args = ap.parse_args()

    tickers = WSB_UNIVERSE if args.universe == "wsb" else WSB_UNIVERSE + EXTRA
    print(f"Fetching EPS-trend snapshots for {len(tickers)} tickers...")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows: list[dict] = []
    for t in tickers:
        r = fetch_one(t)
        for row in r:
            row["date"] = today
            rows.append(row)
        time.sleep(0.25)
    if not rows:
        print("WARN: no rows fetched")
        return 1

    new_df = pd.DataFrame(rows)
    # Reorder columns for readability
    col_order = ["date", "ticker", "period", "current_eps",
                 "eps_30d_ago", "revision_30d_pct",
                 "eps_90d_ago", "revision_90d_pct"]
    new_df = new_df[[c for c in col_order if c in new_df.columns]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV)
            # Drop today's previously-recorded rows (idempotent re-run)
            existing = existing[existing["date"] != today]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as e:
            print(f"  WARN: existing CSV read failed ({e}); overwriting")
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(new_df)} new rows (total {len(combined)}) -> "
          f"{OUT_CSV.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
