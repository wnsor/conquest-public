"""Export the VIX series cache to Lean's Object Store CSV path.

Pulls ^VIX daily close from yfinance (or the parquet cache populated by
``scripts/rank_models.py --include-v2``) and writes the canonical Object
Store CSV that cstability / cgrowth / cgrowth_options all read:

    storage/conquest/vix/daily.csv

For QC cloud push, follow with:
    lean cloud object-store set --key conquest/vix/daily.csv --path storage/conquest/vix/daily.csv

START is fixed at 2008-01-01 to support PIT-extended 2008-2024 backtests
(cstability v11 Layer 1 cash trigger needs spot VIX back to the GFC). If
the parquet cache doesn't cover the requested range, refetch directly
from yfinance.

Usage
-----
    python scripts/export_vix.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent
VIX_PARQUET = WORKSPACE / "data" / "alternative" / "conquest" / "raw" / "prices" / "vix_daily.parquet"
OUT_CSV = WORKSPACE / "storage" / "conquest" / "vix" / "daily.csv"

START = "2008-01-01"
# END dynamically tracks today so daily cron runs extend the series.
# Pre-2026-05-24 this was hardcoded "2025-12-31" — that froze the signal at YE2025
# despite daily refresh runs, causing 5-month-stale regime/vix/credit feeds in LIVE.
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def _fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    print(f"Downloading ^VIX from yfinance, {start} -> {end} ...")
    raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty for ^VIX")
    closes = raw["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    df = closes.to_frame("VIX")
    df.index.name = "Date"
    return df


def main() -> int:
    df: pd.DataFrame | None = None
    if VIX_PARQUET.exists():
        cached = pd.read_parquet(VIX_PARQUET)
        if cached.index[0] <= pd.Timestamp(START) and cached.index[-1] >= pd.Timestamp(END):
            print(
                f"Using cached VIX from {VIX_PARQUET} "
                f"({cached.index[0].date()} -> {cached.index[-1].date()})."
            )
            df = cached
        else:
            print(
                f"Cache range {cached.index[0].date()} -> {cached.index[-1].date()} "
                f"does not cover {START} -> {END}; refetching from yfinance."
            )

    if df is None:
        df = _fetch_yfinance(START, END)
        VIX_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(VIX_PARQUET)
        print(f"Cached {len(df)} VIX bars to {VIX_PARQUET}.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV)
    print(f"Wrote {len(df)} VIX rows to {OUT_CSV}")
    print(
        f"Range: {df.index[0].date()} -> {df.index[-1].date()},  "
        f"last VIX = {float(df.iloc[-1, 0]):.2f}"
    )
    for date_label, lo, hi in (("2008-10-24", 70, 90), ("2020-03-16", 70, 90)):
        try:
            v = float(df.loc[date_label, "VIX"])
            verdict = "OK" if lo <= v <= hi else "OUT OF RANGE"
            print(f"  Sanity {date_label}: VIX={v:.2f} [{verdict}, expected [{lo},{hi}]]")
        except KeyError:
            print(f"  Sanity {date_label}: not in series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
