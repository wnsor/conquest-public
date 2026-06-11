"""
Pull CBOE put/call ratios.

Sources verified 2026-05-23:

  cboe_cdn  Official CBOE CDN archive (free, Mozilla UA needed to bypass
            Cloudflare). 2006-11-01 → 2019-10-04 only — they stopped
            updating the public archive after 2019. Use for historical
            backtests pre-2019.

  manual    Hand-downloaded CSV via CBOE's interactive daily-summary page
            (https://www.cboe.com/us/options/market_statistics/daily/).
            User saves CSV, passes --file. Covers post-2019 if needed.

  yahoo     ^CPC / ^CPCE / ^CPCI — DELISTED on Yahoo as of 2026. Kept for
            forward-compatibility if revived.

What we ruled out (gap is documented in
docs/operations/CONQUEST_OPTIONS_DATA_SOURCES.md):
  - QuantConnect CBOE bundle: VIX only, no put/call.
  - IBKR TWS API: HIGH_OPT_VOLUME_PUT_CALL_RATIO scanner is point-in-time
    only; not tradable so no reqHistoricalData.
  - FRED release rid=200: VIX-style series only, no put/call ratio.
  - Polygon.io Starter $199/mo would have it; deferred per Phase 0.

RECOMMENDED long-term path: compute equity P/C ratio in the conquest_options/
Lean Algorithm directly from QC's `US Equity Option Universe` — sum put-leg
volume / sum call-leg volume each day. Matches the Vasquez/Xiao 2024
methodology and avoids the third-party-source maintenance burden entirely.
This script remains useful for the 2006-2019 archive backfill and any
hand-downloaded gap-fills.

Output:
    storage/conquest/options/cboe_pc_daily.csv
        columns: date, equity_pc, index_pc, total_pc

Usage:
    python scripts/data/cboe_pc_ratio.py --source cboe_cdn              # 2006-2019 archive
    python scripts/data/cboe_pc_ratio.py --source cboe_cdn --smoke      # archive smoke (last 30d in archive)
    python scripts/data/cboe_pc_ratio.py --source manual --file foo.csv
    python scripts/data/cboe_pc_ratio.py --source yahoo --smoke 30d     # legacy; no-op
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_CSV = WORKSPACE / "storage" / "conquest" / "options" / "cboe_pc_daily.csv"

YAHOO_TICKERS = {
    "equity_pc": "^CPCE",
    "index_pc": "^CPCI",
    "total_pc": "^CPC",
}

CBOE_CDN_BASE = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios"
CBOE_CDN_FILES = {
    "equity_pc": "equitypc.csv",
    "index_pc": "indexpcarchive.csv",
    "total_pc": "totalpc.csv",
}
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# --- Yahoo path (kept for compat) ---------------------------------------------

def fetch_yahoo_one(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].rename(ticker)


def fetch_yahoo(start: str, end: str) -> pd.DataFrame:
    out = {}
    for name, tkr in YAHOO_TICKERS.items():
        s = fetch_yahoo_one(tkr, start, end)
        if s.empty:
            print(f"  WARN: yahoo {tkr} no data for {start}..{end}")
        out[name] = s
    df = pd.DataFrame(out)
    df.index.name = "date"
    return df.dropna(how="all").sort_index()


# --- CBOE CDN archive path ----------------------------------------------------

def _parse_cboe_csv(text: str, value_col: str) -> pd.Series:
    """CBOE archive files have a few header rows of disclaimer/product info,
    then a DATE-like first column, CALL/CALLS, PUT/PUTS, TOTAL, P/C Ratio.
    Schemas vary: equity/total use 'DATE', index uses 'Trade_date'. Detect
    either, then return the P/C Ratio as a date-indexed series."""
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines)
         if ln.lower().startswith("date") or ln.lower().startswith("trade_date")),
        None,
    )
    if header_idx is None:
        return pd.Series(dtype=float)
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(body))
    df.columns = [c.strip() for c in df.columns]
    date_col = "DATE" if "DATE" in df.columns else (
        "Trade_date" if "Trade_date" in df.columns else None)
    if date_col is None or "P/C Ratio" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    return df["P/C Ratio"].astype(float).rename(value_col).sort_index()


def fetch_cboe_cdn() -> pd.DataFrame:
    out = {}
    for name, fn in CBOE_CDN_FILES.items():
        url = f"{CBOE_CDN_BASE}/{fn}"
        r = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=30)
        if r.status_code != 200:
            print(f"  WARN: CBOE CDN {fn} → HTTP {r.status_code}")
            continue
        s = _parse_cboe_csv(r.text, name)
        print(f"  {fn}: {len(s)} rows ({s.index.min().date() if len(s) else '—'} → {s.index.max().date() if len(s) else '—'})")
        out[name] = s
    df = pd.DataFrame(out)
    df.index.name = "date"
    return df.dropna(how="all").sort_index()


# --- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pull CBOE put/call ratios")
    ap.add_argument("--source", choices=["cboe_cdn", "yahoo", "manual"],
                    default="cboe_cdn",
                    help="cboe_cdn (default): official archive 2006-2019. "
                         "yahoo: legacy proxy (delisted). "
                         "manual: hand-downloaded CSV via --file.")
    ap.add_argument("--file", help="CSV path when --source=manual")
    ap.add_argument("--start", default="2008-01-01", help="Start date YYYY-MM-DD (filter)")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"),
                    help="End date YYYY-MM-DD (filter)")
    ap.add_argument("--smoke", nargs="?", const=True, default=None,
                    help="Smoke mode. --smoke alone prints last 30 days of fetched data; "
                         "--smoke 30d sets a backward window (relevant for --source=yahoo).")
    args = ap.parse_args()

    if args.source == "manual":
        if not args.file:
            sys.exit("--source=manual requires --file")
        df = pd.read_csv(args.file, parse_dates=["date"]).set_index("date")

    elif args.source == "cboe_cdn":
        print("Fetching CBOE CDN archive (Mozilla UA, 2006-2019 coverage)...")
        df = fetch_cboe_cdn()

    else:  # yahoo
        if isinstance(args.smoke, str):
            unit = args.smoke[-1].lower()
            n = int(args.smoke[:-1])
            days = n if unit == "d" else n * 365 if unit == "y" else None
            if days is None:
                sys.exit(f"Unrecognized --smoke unit '{unit}' (use Nd or Ny)")
            end = datetime.now()
            start = end - timedelta(days=days)
            args.start = start.strftime("%Y-%m-%d")
            args.end = end.strftime("%Y-%m-%d")
            print(f"SMOKE: Yahoo proxy {args.start}..{args.end}")
        df = fetch_yahoo(args.start, args.end)

    # Date-range filter (applies to all sources)
    if not df.empty:
        df = df.loc[(df.index >= pd.Timestamp(args.start)) & (df.index <= pd.Timestamp(args.end))]

    print(f"Fetched {len(df)} rows, columns={list(df.columns)}")
    if df.empty:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["date", "equity_pc", "index_pc", "total_pc"]).to_csv(
            OUTPUT_CSV, index=False)
        print(f"WARNING: no data. Wrote empty stub to {OUTPUT_CSV}")
        sys.exit(2)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, date_format="%Y-%m-%d")
    print(f"Wrote {OUTPUT_CSV}")
    if args.smoke:
        print("\nLast 10 rows:")
        print(df.tail(10).to_string())


if __name__ == "__main__":
    main()
