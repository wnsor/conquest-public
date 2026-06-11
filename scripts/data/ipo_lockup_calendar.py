"""Compute IPO lockup-expiry calendar by scanning EDGAR S-1 filings.

When a company IPOs, insiders + early investors typically agree to a
180-day "lockup" preventing them from selling. Day 181 sees forced
selling pressure as those holders finally liquidate.

This is a calendar-deterministic edge:
  - The lockup expiry date is mathematically certain (IPO date + 180 days)
  - Forced selling = downward pressure for ~5-10 days post-expiry
  - Strategy: buy puts ~7 days before known lockup expiries

Output: storage/conquest/options/ipo_lockup_calendar.csv
Schema: ticker, ipo_date, lockup_expiry_date

Used by: future v_LOCKUP_DRIFT strategy (not yet drafted — depends on this data)

Implementation:
  - Query EDGAR for IPO filings (S-1 = registration statement, 424B4 = prospectus)
  - Extract effective IPO date (when 424B4 was filed)
  - Compute lockup_expiry = ipo_date + 180 days
  - Filter to last 365 days of IPOs (forward-looking 6 months of expiries)
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "options" / "ipo_lockup_calendar.csv"

LOCKUP_DAYS = 180   # standard underwriter agreement


def setup_edgar(email: str):
    from edgar import set_identity
    set_identity(email)


def fetch_ipos(days_back: int) -> list[dict]:
    """Scan EDGAR for recent IPOs via 424B prospectus filings."""
    from edgar import get_filings
    end = date.today()
    start = end - timedelta(days=days_back)
    rows = []
    # 424B4 is the most common final prospectus form
    for ft in ["424B4", "424B1", "424B3"]:
        print(f"  fetching {ft} filings {start} → {end}")
        try:
            # edgartools wants a colon-separated date-range STRING, not a tuple
            # (see edgar_13d.py for the same fix).
            filings = get_filings(form=ft, filing_date=f"{start}:{end}")
        except Exception as e:
            print(f"  WARN {ft}: {e}")
            continue
        for f in filings:
            ticker = getattr(f, "ticker", "") or ""
            company = getattr(f, "company", "") or ""
            if not ticker:
                continue
            try:
                fd = f.filing_date if isinstance(f.filing_date, date) else \
                     datetime.strptime(str(f.filing_date)[:10], "%Y-%m-%d").date()
            except Exception:
                continue
            rows.append({
                "ticker": ticker.upper(),
                "company": company,
                "ipo_date": fd.isoformat(),
                "lockup_expiry_date": (fd + timedelta(days=LOCKUP_DAYS)).isoformat(),
                "form_type": ft,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=365,
                    help="Lookback window for IPO filings — 365d = next 6mo of expiries")
    ap.add_argument("--email", default="conquest-research@example.com")
    args = ap.parse_args()

    setup_edgar(args.email)
    rows = fetch_ipos(args.days_back)
    if not rows:
        print("WARN: no IPO filings fetched")
        return 0

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["ticker", "ipo_date"], inplace=True)
    df.sort_values("lockup_expiry_date", inplace=True)

    # Merge with existing
    if OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV)
            df = pd.concat([existing, df], ignore_index=True)
            df.drop_duplicates(subset=["ticker", "ipo_date"], keep="last", inplace=True)
        except Exception:
            pass
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    # How many expiries are in the next 30 days?
    upcoming = df[
        (pd.to_datetime(df["lockup_expiry_date"]).dt.date >= date.today()) &
        (pd.to_datetime(df["lockup_expiry_date"]).dt.date <= date.today() + timedelta(days=30))
    ]
    print(f"Wrote {len(df)} IPO records -> {OUT_CSV.relative_to(WORKSPACE)}")
    print(f"  upcoming lockup expiries in next 30 days: {len(upcoming)}")
    if len(upcoming):
        print(upcoming[["ticker", "ipo_date", "lockup_expiry_date"]].head(10).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
