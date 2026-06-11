"""Export the S&P 500 ticker → GICS sector mapping to Lean's Object Store.

Output: ``storage/conquest/universe/sp500.csv`` (key: ``conquest/universe/sp500.csv``)
Columns: ticker, gics_sector

The cgrowth Lean Algorithm reads this file in `Initialize` to know which
500 names to subscribe to, and to apply per-sector concentration caps in
`rebalance`.

Usage
-----
    python scripts/export_sp500_universe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "universe" / "sp500.csv"


def main() -> int:
    from conquest.data.sp500 import fetch_sp500
    df = fetch_sp500()  # uses cache if present
    out = df[["ticker", "gics_sector"]].copy()
    out.columns = ["ticker", "sector"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    sectors = out["sector"].value_counts()
    print(f"Wrote {len(out)} (ticker, sector) rows to {OUT_CSV}")
    print(f"GICS sector distribution:")
    for sec, count in sectors.items():
        print(f"  {sec:35s} {count:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
