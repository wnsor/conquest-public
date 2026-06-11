"""CLI wrapper around conquest.options.uoa for offline batch screening.

Phase 0 use: read a CSV of (date, contract_id, volume, open_interest) rows,
emit a CSV with an `is_uoa` column. Phase 1 integration: the conquest_options/
Lean Algorithm calls `uoa_flag(...)` directly in `OnData` per contract.

Input CSV schema (header required):
    date,contract_id,volume,open_interest

Output CSV schema:
    date,contract_id,volume,open_interest,is_uoa

Usage:
    python scripts/data/uoa_detector.py --in /path/in.csv --out /path/out.csv
    python scripts/data/uoa_detector.py --in /path/in.csv --vol-multiplier 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from conquest.options.uoa import uoa_flag_series


def main():
    ap = argparse.ArgumentParser(description="Offline UOA batch screener")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV path")
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--oi-window", type=int, default=5)
    ap.add_argument("--vol-multiplier", type=float, default=5.0)
    ap.add_argument("--oi-multiplier", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    needed = {"date", "contract_id", "volume", "open_interest"}
    if not needed.issubset(df.columns):
        sys.exit(f"Input CSV missing required columns: {needed - set(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["contract_id", "date"]).reset_index(drop=True)

    parts = []
    for cid, grp in df.groupby("contract_id"):
        flags = uoa_flag_series(
            grp["volume"], grp["open_interest"],
            vol_window=args.vol_window,
            oi_window=args.oi_window,
            vol_multiplier=args.vol_multiplier,
            oi_multiplier=args.oi_multiplier,
        )
        out = grp.copy()
        out["is_uoa"] = flags.values
        parts.append(out)

    out_df = pd.concat(parts, ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    n_uoa = int(out_df["is_uoa"].sum())
    print(f"Wrote {len(out_df)} rows ({n_uoa} flagged UOA) -> {args.out_csv}")


if __name__ == "__main__":
    main()
