"""Fetch CBOE VVIX + SKEW daily indices via yfinance.

VVIX (^VVIX): vol-of-vol, leading for vol regime shifts (>130 = elevated).
SKEW (^SKEW): tail-risk pricing, >130 = market paying up for crash insurance.

Both are free yfinance tickers. Used by:
  - tail_hedge_regime (SKEW + VIX percentile)
  - vvix_divergence (VVIX + VIX percentile)

Output: storage/conquest/options/cboe_indices_daily.csv
Schema: date, vvix, skew
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "options" / "cboe_indices_daily.csv"

START_DEFAULT = "2008-01-01"


def fetch(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    vvix = yf.download("^VVIX", start=start, end=end, auto_adjust=False, progress=False)["Close"]
    skew = yf.download("^SKEW", start=start, end=end, auto_adjust=False, progress=False)["Close"]
    if isinstance(vvix, pd.DataFrame): vvix = vvix.iloc[:, 0]
    if isinstance(skew, pd.DataFrame): skew = skew.iloc[:, 0]
    df = pd.concat([vvix.rename("vvix"), skew.rename("skew")], axis=1).dropna(how="all")
    df.index.name = "date"
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START_DEFAULT)
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    args = ap.parse_args()
    print(f"Fetching VVIX + SKEW from yfinance, {args.start} → {args.end}")
    df = fetch(args.start, args.end)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV)
    print(f"Wrote {len(df)} rows -> {OUT_CSV.relative_to(WORKSPACE)}")
    print(f"  vvix range: {df['vvix'].min():.1f} - {df['vvix'].max():.1f}  latest={df['vvix'].iloc[-1]:.1f}")
    print(f"  skew range: {df['skew'].min():.1f} - {df['skew'].max():.1f}  latest={df['skew'].iloc[-1]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
