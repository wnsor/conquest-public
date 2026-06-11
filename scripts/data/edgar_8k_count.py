"""Fetch SEC EDGAR 8-K filings + compute rolling 14-day count per ticker.

8-K is the "material events" filing — sudden disclosures about M&A,
litigation, regulatory action, restated earnings, executive changes, etc.
Normal companies file 0-1 per year. A BURST (3+ in 14 days) means
something material is brewing.

Output: storage/conquest/insider/edgar_8k_count_daily.csv
Schema: date, ticker, count_14d

Used by: eightk_burst strategy

Source: EDGAR via edgartools — same auth/throttling as Form 4.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "insider" / "edgar_8k_count_daily.csv"


def setup_edgar(email: str):
    from edgar import set_identity
    set_identity(email)


def fetch_8k_filings(start: date, end: date) -> list[dict]:
    """Pull all 8-K filings in [start, end]."""
    from edgar import get_filings
    rows = []
    print(f"  fetching 8-K filings {start} → {end}")
    try:
        # edgartools wants a colon-separated date-range STRING, not a tuple
        # (see edgar_13d.py for the same fix).
        filings = get_filings(form="8-K", filing_date=f"{start}:{end}")
    except Exception as e:
        print(f"  WARN: {e}")
        return rows
    for f in filings:
        ticker = getattr(f, "ticker", "") or ""
        if not ticker:
            continue
        try:
            fd = f.filing_date.isoformat() if hasattr(f.filing_date, "isoformat") else str(f.filing_date)
        except Exception:
            continue
        rows.append({"filing_date": fd, "ticker": ticker.upper()})
    return rows


def compute_rolling_14d(filings_df: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, date), count 8-K filings in past 14 days."""
    if filings_df.empty:
        return pd.DataFrame()
    filings_df["filing_date"] = pd.to_datetime(filings_df["filing_date"]).dt.date
    # Build per-ticker rolling count
    rows = []
    all_dates = sorted(filings_df["filing_date"].unique())
    for ticker, group in filings_df.groupby("ticker"):
        dates_for_ticker = sorted(group["filing_date"].tolist())
        # For each unique date in the universe, count 8-Ks for this ticker in last 14d
        for d in all_dates:
            window_start = d - timedelta(days=14)
            n = sum(1 for fd in dates_for_ticker if window_start < fd <= d)
            if n > 0:
                rows.append({"date": d.isoformat(), "ticker": ticker, "count_14d": n})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=90)
    ap.add_argument("--email", default="conquest-research@example.com")
    ap.add_argument("--incremental", action="store_true")
    args = ap.parse_args()

    setup_edgar(args.email)
    end = date.today()
    start = end - timedelta(days=args.days_back)
    if args.incremental and OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV, parse_dates=["date"])
            if len(existing):
                last = existing["date"].max().date()
                # Re-pull last 14d + new (to refresh rolling window for boundary)
                start = max(start, last - timedelta(days=14))
                print(f"INCREMENTAL: from {start} (existing max {last})")
        except Exception:
            pass

    raw = fetch_8k_filings(start, end)
    if not raw:
        print("WARN: no 8-K filings fetched")
        return 0
    filings_df = pd.DataFrame(raw)
    rolling = compute_rolling_14d(filings_df)

    # Merge with existing
    if OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV)
            df = pd.concat([existing, rolling], ignore_index=True)
            df.drop_duplicates(subset=["date", "ticker"], keep="last", inplace=True)
        except Exception:
            df = rolling
    else:
        df = rolling
    df.sort_values(["date", "ticker"], inplace=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    bursts = (df["count_14d"] >= 3).sum()
    print(f"Wrote {len(df)} (ticker, date) rows -> {OUT_CSV.relative_to(WORKSPACE)}")
    print(f"  burst events (count_14d >= 3): {bursts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
