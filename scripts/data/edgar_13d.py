"""Fetch SEC EDGAR Form 13D/G filings — activist >5% ownership stakes.

13D filings are MUCH stronger signal than Form 4 because:
  - Filer must own >5% of outstanding shares
  - 13D specifically signals intent to influence (vs 13G = passive)
  - Activist campaigns typically run 30-180 days with public catalysts
  - Public-knowledge edge: market often hasn't fully priced the activist thesis

Output: storage/conquest/insider/edgar_13d_filings.csv
Schema: filing_date, filer_cik, filer_name, ticker, form_type, percent_owned

Used by: activist_drift strategy

Source: EDGAR's full-text search + filings API. We pull last N days of
13D + 13D/A (amendments) + 13G/A filings, parse the cover page for
ticker + percent_owned.

Implementation notes:
  - edgartools library handles the EDGAR API + pagination
  - 13D/G are filed within 10 days of crossing the 5% threshold
  - 13D/A is filed for material changes; 13G/A for passive changes
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "insider" / "edgar_13d_filings.csv"


def setup_edgar(email: str):
    """SEC requires User-Agent with contact email; edgartools handles via set_identity."""
    from edgar import set_identity
    set_identity(email)


def fetch_filings(start: datetime, end: datetime, form_types: list[str]) -> list[dict]:
    """Pull 13D/G filings via edgartools' get_filings API."""
    from edgar import get_filings
    rows = []
    for ft in form_types:
        print(f"  fetching {ft} filings {start.date()} → {end.date()}")
        try:
            # edgartools wants a colon-separated date-range STRING, not a tuple;
            # passing a tuple triggers 'expected string or bytes-like object,
            # got tuple' deep inside the URL builder and returns 0 filings.
            filings = get_filings(form=ft, filing_date=f"{start.date()}:{end.date()}")
        except Exception as e:
            print(f"  WARN {ft}: {e}")
            continue
        for f in filings:
            try:
                ticker = (f.subject_company or "").upper() if hasattr(f, "subject_company") else ""
                # 13D primary subject ticker is in the filing's reporting_owners or cover page;
                # edgartools exposes via .ticker property on some Filing objects
                ticker = getattr(f, "ticker", "") or ticker
                if not ticker:
                    continue
                rows.append({
                    "filing_date": f.filing_date.isoformat() if hasattr(f.filing_date, "isoformat") else str(f.filing_date),
                    "filer_cik": str(f.cik) if hasattr(f, "cik") else "",
                    "filer_name": getattr(f, "company", "") or "",
                    "ticker": ticker,
                    "form_type": ft,
                    "accession_no": getattr(f, "accession_no", ""),
                })
            except Exception:
                continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-back", type=int, default=90, help="Lookback window")
    ap.add_argument("--email", default="conquest-research@example.com")
    ap.add_argument("--incremental", action="store_true",
                    help="Append to existing CSV; only fetch since last filing_date")
    args = ap.parse_args()

    setup_edgar(args.email)
    end = datetime.utcnow()
    start = end - timedelta(days=args.days_back)
    if args.incremental and OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV, parse_dates=["filing_date"])
            if len(existing):
                last = existing["filing_date"].max().to_pydatetime()
                start = max(start, last - timedelta(days=5))  # 5-day overlap safety
                print(f"INCREMENTAL: starting from {start.date()} (existing max {last.date()})")
        except Exception as e:
            print(f"INCREMENTAL: couldn't read existing CSV ({e}); full window")

    rows = fetch_filings(start, end, ["SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"])
    if not rows:
        print("WARN: no filings fetched")
        return 0  # not an error — quiet weeks happen
    new_df = pd.DataFrame(rows)
    new_df.drop_duplicates(subset=["accession_no"], inplace=True)
    # Merge with existing
    if OUT_CSV.exists():
        try:
            existing = pd.read_csv(OUT_CSV)
            df = pd.concat([existing, new_df], ignore_index=True)
            df.drop_duplicates(subset=["accession_no"], inplace=True)
        except Exception:
            df = new_df
    else:
        df = new_df
    df.sort_values(["filing_date", "ticker"], inplace=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} total filings ({len(new_df)} new) -> {OUT_CSV.relative_to(WORKSPACE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
