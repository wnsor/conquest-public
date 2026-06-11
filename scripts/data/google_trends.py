"""Fetch Google Trends per-ticker search volume via pytrends.

Google Trends captures public attention BEFORE it shows up in news cycles
or social media propagation. 1-3 day lead time on retail-driven moves.

Output: storage/conquest/sentiment/google_trends_daily.csv
Schema: date, ticker, search_volume_idx (0-100 normalized within batch)

Used by: retail_attention_cascade strategy (5d/5d velocity computed in main.py)

Limitations:
  - pytrends is free but rate-limited; we batch 5 tickers at a time
  - Returns relative search interest (0-100), not absolute volumes
  - Subject to occasional Google CAPTCHA blocks; ingester handles gracefully
  - We track ticker symbols (e.g., "MSTR") not company names — accuracy varies
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "sentiment" / "google_trends_daily.csv"

WSB_UNIVERSE = [
    "CRDO", "MU", "NBIS", "NOK", "MX", "PL", "DRAM", "LPTH", "SOUN", "RDDT",
    "MSTR", "COIN", "PLTR", "RKLB", "APP", "SMCI",
]


def fetch_batch(tickers: list[str], timeframe: str) -> pd.DataFrame:
    """Pytrends — 5 tickers per call, returns normalized 0-100 series."""
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl="en-US", tz=360, retries=2, backoff_factor=1.0)
    # pytrends caps at 5 keywords per call
    pytrends.build_payload(tickers[:5], timeframe=timeframe, geo="US")
    df = pytrends.interest_over_time()
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", default="today 3-m",
                    help="pytrends timeframe — 'today 3-m', 'today 12-m', 'all', etc.")
    ap.add_argument("--universe", default="wsb")
    args = ap.parse_args()

    tickers = WSB_UNIVERSE
    print(f"Fetching Google Trends for {len(tickers)} tickers (batches of 5)")
    parts = []
    for i in range(0, len(tickers), 5):
        batch = tickers[i:i + 5]
        print(f"  batch {i // 5 + 1}: {batch}")
        try:
            df = fetch_batch(batch, args.timeframe)
            if not df.empty:
                # Reshape wide → long
                df = df.reset_index().melt(id_vars=["date"], var_name="ticker",
                                            value_name="search_volume_idx")
                parts.append(df)
        except Exception as e:
            print(f"  WARN batch {batch}: {e}")
        time.sleep(2.0)  # rate-limit politeness

    if not parts:
        print("ERROR: no Google Trends data fetched")
        return 1
    combined = pd.concat(parts, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.date

    # Merge with existing (append-with-dedupe)
    if OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV)
            existing["date"] = pd.to_datetime(existing["date"]).dt.date
            df = pd.concat([existing, combined], ignore_index=True)
            df.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)
        except Exception:
            df = combined
    else:
        df = combined
    df.sort_values(["date", "ticker"], inplace=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_CSV.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
