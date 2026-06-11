"""
Ingest SEC Form 4 (insider transaction) filings from EDGAR.
Classifies each insider as routine vs opportunistic per Cohen-Malloy-Pomorski
2012 ("Decoding Inside Information") and emits a daily CSV of opportunistic
insider BUYS (transaction code "P") for use in cspec.

Output:
    storage/conquest/insider/form4_opportunistic_buys_daily.csv
        columns: filing_date,transaction_date,ticker,insider_cik,insider_name,
                 role,shares,price,dollar_value

Usage:
    python scripts/data/form4.py --start 2018 --end 2026     # full backfill
    python scripts/data/form4.py --smoke                     # 1-quarter, 100-filing smoke test
    python scripts/data/form4.py --quarters-only 2026-Q1     # specific quarter only

Notes
-----
* SEC requires User-Agent header with contact email; max 10 req/sec global.
  edgartools handles both via set_identity() and built-in throttling.
* Backfill 2018-2026 = ~2.5M filings; expect 3-4 days wall clock on first run.
  Incremental daily refresh = <5 min.
* Cache raw parsed transactions to parquet so re-runs skip already-fetched
  quarters.
* CMP classifier: insider is "routine" in year t if she traded the same
  calendar month in each of years (t-1, t-2, t-3). Everyone else with >= 3
  prior years of trading history = "opportunistic". Insiders with <3 years
  = unclassified (excluded from output).

Sources
-------
* Cohen-Malloy-Pomorski 2012 (NBER w16454):
  https://www.nber.org/papers/w16454
* edgartools: https://github.com/dgunning/edgartools
* SEC EDGAR API: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
CACHE_DIR = WORKSPACE / "data" / "alternative" / "edgar"
OUTPUT_CSV = WORKSPACE / "storage" / "conquest" / "insider" / "form4_opportunistic_buys_daily.csv"


def setup_edgar(contact_email: str = "conquest-research@example.com"):
    """Initialize edgartools with required identity."""
    try:
        from edgar import set_identity
    except ImportError:
        sys.exit("edgartools not installed. Run: pip install edgartools")
    set_identity(f"Conquest Research {contact_email}")


def fetch_quarter(year: int, qtr: int, limit: int = None) -> pd.DataFrame:
    """Pull Form 4 filings for one quarter via edgartools.

    Args:
        year, qtr: target quarter
        limit: optional max number of filings to parse (for testing)

    Returns DataFrame with one row per transaction (a single Form 4 may have
    multiple transactions).
    """
    from edgar import get_filings
    cache_key = f"{year}_Q{qtr}" + (f"_lim{limit}" if limit else "")
    cache_path = CACHE_DIR / "quarters" / f"{cache_key}.parquet"
    if cache_path.exists():
        print(f"  cache hit: {cache_path.name}")
        return pd.read_parquet(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {year} Q{qtr} from EDGAR (limit={limit or 'all'})...")
    filings = get_filings(form='4', year=year, quarter=qtr)
    rows = []
    n_total = 0
    n_with_purchases = 0
    for filing in filings:
        n_total += 1
        if limit and n_total > limit:
            break
        try:
            obj = filing.obj()
        except Exception:
            continue
        if obj is None:
            continue
        # Skip private funds (no exchange ticker)
        try:
            issuer_ticker = getattr(obj.issuer, "ticker", None) or ""
        except Exception:
            issuer_ticker = ""
        if not issuer_ticker or issuer_ticker == "NONE":
            continue
        try:
            issuer_cik = filing.cik
            ros = obj.reporting_owners
            if ros is None or len(ros) == 0:
                continue
            ro = ros[0]
            insider_cik = str(getattr(ro, "cik", "") or "")
            insider_name = getattr(ro, "name", "") or ""
            role_parts = []
            if getattr(ro, "is_officer", False): role_parts.append("Officer")
            if getattr(ro, "is_director", False): role_parts.append("Director")
            if getattr(ro, "is_ten_pct_owner", False): role_parts.append("10pct")
            role = ",".join(role_parts)

            purchases = obj.common_stock_purchases
            if purchases is None or len(purchases) == 0:
                continue
            n_with_purchases += 1
            for _, t in purchases.iterrows():
                rows.append({
                    "filing_date": filing.filing_date,
                    "accession": filing.accession_no,
                    "transaction_date": t.get("Date"),
                    "ticker": issuer_ticker,
                    "issuer_cik": issuer_cik,
                    "insider_cik": insider_cik,
                    "insider_name": insider_name,
                    "role": role,
                    "code": t.get("Code", "P"),
                    "shares": float(t.get("Shares", 0) or 0),
                    "price": float(t.get("Price", 0) or 0),
                })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    df.to_parquet(cache_path)
    print(f"  cached {len(df)} txns from {n_total} filings ({n_with_purchases} had P-coded purchases) -> {cache_path.name}")
    return df


def classify_routine(all_txns: pd.DataFrame) -> set:
    """Cohen-Malloy-Pomorski 2012 classifier.

    An insider is ROUTINE in year t if they traded in the same calendar month
    in each of years t-1, t-2, t-3. Everyone else with >=3 years of history
    is OPPORTUNISTIC. Returns a set of (insider_cik, year) tuples that are
    classified ROUTINE; everything else is opportunistic.
    """
    routine = set()
    if all_txns.empty:
        return routine
    df = all_txns.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["year"] = df["filing_date"].dt.year
    df["month"] = df["filing_date"].dt.month
    for insider_cik, group in df.groupby("insider_cik"):
        if not insider_cik:
            continue
        years = sorted(group["year"].unique())
        for y in years:
            window_years = [y - 1, y - 2, y - 3]
            months_per_year = {
                yr: set(group[group["year"] == yr]["month"])
                for yr in window_years
            }
            # All three prior years must have at least one trade
            if not all(months_per_year[yr] for yr in window_years):
                continue
            # Common month across all three years
            common = set.intersection(*[months_per_year[yr] for yr in window_years])
            if common:
                routine.add((insider_cik, y))
    return routine


def emit_opportunistic_buys_csv(all_txns: pd.DataFrame, routine_set: set, out_path: Path):
    """Filter to opportunistic open-market PURCHASES (code='P'), emit CSV."""
    if all_txns.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "filing_date", "transaction_date", "ticker", "insider_cik",
            "insider_name", "role", "shares", "price", "dollar_value"
        ]).to_csv(out_path, index=False)
        return
    df = all_txns.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["year"] = df["filing_date"].dt.year
    # Filter to purchases only
    df = df[df["code"] == "P"].copy()
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "filing_date", "transaction_date", "ticker", "insider_cik",
            "insider_name", "role", "shares", "price", "dollar_value"
        ]).to_csv(out_path, index=False)
        return
    # Mark routine vs opportunistic
    df["routine"] = df.apply(lambda r: (r["insider_cik"], r["year"]) in routine_set, axis=1)
    # Filter to opportunistic
    df = df[~df["routine"]].copy()
    df["dollar_value"] = df["shares"] * df["price"]
    out = df[[
        "filing_date", "transaction_date", "ticker", "insider_cik",
        "insider_name", "role", "shares", "price", "dollar_value"
    ]].copy()
    out["filing_date"] = out["filing_date"].dt.strftime("%Y-%m-%d")
    out["transaction_date"] = pd.to_datetime(out["transaction_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # Deduplicate (same filing/insider/date/ticker/shares/price often appears 2x in raw)
    out.drop_duplicates(
        subset=["filing_date", "transaction_date", "ticker", "insider_cik", "shares", "price"],
        inplace=True,
    )
    # If an existing CSV is present, merge with the new rows to preserve
    # history we didn't re-fetch this run (incremental mode safety net).
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            if len(existing):
                out = pd.concat([existing, out], ignore_index=True)
                out.drop_duplicates(
                    subset=["filing_date", "transaction_date", "ticker",
                            "insider_cik", "shares", "price"],
                    inplace=True,
                )
        except Exception as e:
            print(f"  warn: failed to merge with existing CSV ({e}); writing fresh")
    out.sort_values(["filing_date", "ticker"], inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} unique opportunistic buy rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Ingest SEC Form 4 filings + classify per CMP 2012")
    ap.add_argument("--start", type=int, default=2018, help="Start year (default 2018)")
    ap.add_argument("--end", type=int, default=2026, help="End year (inclusive)")
    ap.add_argument("--email", default="conquest-research@example.com",
                    help="Contact email for SEC User-Agent")
    ap.add_argument("--quarters-only", help="Comma-separated YYYY-Q list to fetch only those")
    ap.add_argument("--limit", type=int, default=None,
                    help="For testing: max filings to parse per quarter")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: fetch most-recent completed quarter, limit 100 filings, "
                         "verify pipeline end-to-end without long backfill.")
    ap.add_argument("--incremental", action="store_true",
                    help="Incremental mode: detect max filing_date in existing output CSV, "
                         "fetch only the quarter containing that date + all later quarters. "
                         "Falls back to full --start/--end backfill if the CSV is missing/empty. "
                         "Designed for daily GitHub Action refresh.")
    args = ap.parse_args()

    setup_edgar(args.email)

    # Build the list of (year, quarter) to fetch
    if args.smoke:
        now = datetime.now()
        prev_q = (now.month - 1) // 3  # 0 in Jan/Feb/Mar → fetch year-1 Q4
        if prev_q == 0:
            quarters = [(now.year - 1, 4)]
        else:
            quarters = [(now.year, prev_q)]
        args.limit = args.limit or 100
        print(f"SMOKE: fetching {quarters[0][0]} Q{quarters[0][1]} with limit={args.limit}")
    elif args.quarters_only:
        quarters = []
        for tok in args.quarters_only.split(","):
            y, q = tok.strip().split("-")
            quarters.append((int(y), int(q.lstrip("Q"))))
    elif args.incremental:
        # Find latest filing_date in existing CSV; refetch from that quarter
        # onward (inclusive) to catch late-arriving filings within the same
        # quarter we'd already partly ingested.
        now = datetime.now()
        max_q = (now.year, (now.month - 1) // 3 + 1)
        start_year, start_q = args.start, 1
        if OUTPUT_CSV.exists():
            try:
                existing = pd.read_csv(OUTPUT_CSV, parse_dates=["filing_date"])
                if len(existing):
                    last = existing["filing_date"].max()
                    start_year, start_q = last.year, (last.month - 1) // 3 + 1
                    print(f"INCREMENTAL: existing CSV max filing_date = {last.date()}; "
                          f"re-fetching from {start_year} Q{start_q} onward")
                else:
                    print(f"INCREMENTAL: existing CSV is empty; full backfill from {start_year} Q1")
            except Exception as e:
                print(f"INCREMENTAL: failed to read existing CSV ({e}); full backfill")
        else:
            print(f"INCREMENTAL: no existing CSV at {OUTPUT_CSV}; full backfill from {start_year}")
        quarters = []
        y, q = start_year, start_q
        while (y, q) <= max_q:
            quarters.append((y, q))
            if q == 4:
                y, q = y + 1, 1
            else:
                q += 1
    else:
        quarters = [(y, q) for y in range(args.start, args.end + 1) for q in (1, 2, 3, 4)]
        # Trim future quarters
        now = datetime.now()
        quarters = [(y, q) for (y, q) in quarters if (y, q) <= (now.year, (now.month - 1) // 3 + 1)]

    print(f"Fetching {len(quarters)} quarter(s) of Form 4 from EDGAR...")
    parts = []
    for (y, q) in quarters:
        try:
            df = fetch_quarter(y, q, limit=args.limit)
            parts.append(df)
        except Exception as e:
            print(f"  WARN: {y} Q{q} failed: {e}")
    if not parts:
        sys.exit("No data fetched")
    all_txns = pd.concat(parts, ignore_index=True)
    print(f"Total transactions: {len(all_txns):,}")

    print("Classifying routine vs opportunistic per Cohen-Malloy-Pomorski 2012...")
    routine_set = classify_routine(all_txns)
    print(f"  {len(routine_set):,} (insider, year) pairs classified ROUTINE")

    emit_opportunistic_buys_csv(all_txns, routine_set, OUTPUT_CSV)
    print("Done.")


if __name__ == "__main__":
    main()
