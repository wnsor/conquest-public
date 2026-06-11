"""FINRA short-volume aggregator (rewrite 2026-05-27).

Background
----------
FINRA's free monthly consolidated short-interest endpoint was pulled
behind a subscription wall (HTTP 403 as of 2026-05). The DAILY short-sale
volume endpoint at:

    https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt

remains free and verified working. This file lists per-ticker short volume
+ total volume for a single trading day in pipe-delimited format:

    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

This script aggregates the last N daily files into a biweekly schema
matching the OLD monthly endpoint's output, so downstream consumers
(`FinraShortInterestLoader.velocity()`) don't need to change.

Mapping (daily volume → biweekly "short interest" proxy)
--------------------------------------------------------
The free daily file gives FLOW (short sales that day), not LEVEL
(outstanding shares short). Without per-day short-cover data, we
approximate the SI LEVEL via 10-day rolling sum of ShortVolume per ticker.
The strategies that consume this (short_squeeze_pure, reflex_ignition_v2)
gate on velocity = (latest_proxy - prior_proxy) / prior_proxy, which is
invariant to constant-factor differences between "flow" and "level"
interpretations.

Output schema (unchanged from old monthly format)
-------------------------------------------------
    settlement_date, ticker, short_interest_shares, days_to_cover, percent_float

  - settlement_date: biweekly synthesized dates (15th + EOM of each month)
  - short_interest_shares: 10-day rolling sum of ShortVolume per ticker
  - days_to_cover: 10d-sum-short / 10d-avg-total-volume (proxy)
  - percent_float: 0.0 (not derivable from daily volume — kept for schema
    compatibility)

Usage
-----
    python scripts/data/finra_short_interest.py --start 2024-01-01 --end 2026-05-27
    python scripts/data/finra_short_interest.py --smoke 1
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_CSV = WORKSPACE / "storage" / "conquest" / "options" / "finra_si_biweekly.csv"

DAILY_URL_TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
USER_AGENT = "Conquest Research (conquest-research@example.com)"
ROLLING_WINDOW = 10


def _fetch_one_day(d: date) -> pd.DataFrame:
    """Fetch + parse one day's CNMSshvol file. Empty DataFrame on miss."""
    url = DAILY_URL_TEMPLATE.format(ymd=d.strftime("%Y%m%d"))
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    except requests.RequestException:
        return pd.DataFrame()
    if r.status_code != 200 or not r.content:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(r.text), sep="|", low_memory=False)
    except Exception:
        return pd.DataFrame()
    if "Symbol" not in df.columns:
        return pd.DataFrame()
    df["ShortVolume"] = pd.to_numeric(df.get("ShortVolume"), errors="coerce")
    df["TotalVolume"] = pd.to_numeric(df.get("TotalVolume"), errors="coerce")
    df = df.dropna(subset=["ShortVolume", "TotalVolume", "Symbol"]).copy()
    df["date"] = d
    return df[["date", "Symbol", "ShortVolume", "TotalVolume"]].rename(
        columns={"Symbol": "ticker"}
    )


def fetch_daily_range(start: date, end: date) -> pd.DataFrame:
    """Pull each weekday between start and end. Skips 404s (holidays etc)."""
    parts = []
    d = start
    fetched = 0
    skipped = 0
    while d <= end:
        if d.weekday() < 5:
            df_d = _fetch_one_day(d)
            if not df_d.empty:
                parts.append(df_d)
                fetched += 1
            else:
                skipped += 1
        d += timedelta(days=1)
    print(f"  fetched {fetched} daily files; skipped {skipped} (holidays / missing)")
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _biweekly_dates_between(start: date, end: date) -> list[date]:
    """Return the 15th + last day of each month in [start, end]."""
    out = []
    cur = start.replace(day=1)
    while cur <= end:
        mid = cur.replace(day=15)
        if start <= mid <= end:
            out.append(mid)
        next_month = (cur + pd.offsets.MonthBegin(1)).date()
        eom = next_month - timedelta(days=1)
        if start <= eom <= end:
            out.append(eom)
        cur = next_month
    return sorted(out)


def aggregate_to_biweekly(daily: pd.DataFrame) -> pd.DataFrame:
    """For each settlement date (biweekly), compute per-ticker rolling-sum
    short_volume + total_volume across the prior ROLLING_WINDOW trading days."""
    if daily.empty:
        return pd.DataFrame()
    daily = daily.sort_values(["ticker", "date"]).copy()
    daily["date"] = pd.to_datetime(daily["date"])

    daily["sv_roll"] = (
        daily.groupby("ticker")["ShortVolume"]
             .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()
             .reset_index(level=0, drop=True)
    )
    daily["tv_roll"] = (
        daily.groupby("ticker")["TotalVolume"]
             .rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).sum()
             .reset_index(level=0, drop=True)
    )

    start_d = daily["date"].min().date()
    end_d = daily["date"].max().date()
    settlement_dates = _biweekly_dates_between(start_d, end_d)
    if not settlement_dates:
        return pd.DataFrame()
    rows = []
    for sd in settlement_dates:
        sd_ts = pd.Timestamp(sd)
        sub = daily[daily["date"] <= sd_ts]
        if sub.empty:
            continue
        latest = sub.groupby("ticker", as_index=False).last()
        latest = latest.dropna(subset=["sv_roll", "tv_roll"])
        for _, r in latest.iterrows():
            sv = float(r["sv_roll"])
            tv = float(r["tv_roll"])
            if sv <= 0 or tv <= 0:
                continue
            avg_daily_vol = tv / ROLLING_WINDOW
            dtc = sv / max(1.0, avg_daily_vol)
            rows.append({
                "settlement_date": sd.isoformat(),
                "ticker": str(r["ticker"]).upper(),
                "short_interest_shares": sv,
                "days_to_cover": dtc,
                "percent_float": 0.0,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="YYYY-MM-DD inclusive", default=None)
    ap.add_argument("--end", help="YYYY-MM-DD inclusive", default=None)
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, fetch only enough days for last N biweekly snapshots")
    args = ap.parse_args()

    today = date.today()
    if args.smoke > 0:
        start = today - timedelta(days=args.smoke * 14 + ROLLING_WINDOW + 7)
        end = today - timedelta(days=1)
    else:
        start = date.fromisoformat(args.start) if args.start else today - timedelta(days=90)
        end = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)

    print(f"Fetching FINRA daily short volume {start} → {end}")
    daily = fetch_daily_range(start, end)
    if daily.empty:
        print("ERROR: no daily data fetched", file=sys.stderr)
        return 1
    print(f"  daily rows: {len(daily):,} across {daily['ticker'].nunique()} tickers")

    biweekly = aggregate_to_biweekly(daily)
    if biweekly.empty:
        print("ERROR: aggregation produced 0 rows", file=sys.stderr)
        return 1

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        try:
            existing = pd.read_csv(OUTPUT_CSV)
            keys_new = set(zip(biweekly["settlement_date"], biweekly["ticker"]))
            existing = existing[~existing.apply(
                lambda r: (r["settlement_date"], r["ticker"]) in keys_new, axis=1
            )]
            combined = pd.concat([existing, biweekly], ignore_index=True)
        except Exception as e:
            print(f"  WARN: failed to merge existing CSV ({e}); overwriting")
            combined = biweekly
    else:
        combined = biweekly
    combined = combined.sort_values(["settlement_date", "ticker"])
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(biweekly)} new rows (total {len(combined)}) -> "
          f"{OUTPUT_CSV.relative_to(WORKSPACE)}")
    print(f"  ticker count: {combined['ticker'].nunique():,}")
    print(f"  date range: {combined['settlement_date'].min()} → "
          f"{combined['settlement_date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
