"""Read cached FRED data, run the regime classifier, write the daily regime CSV.

Output: ``storage/conquest/regime/daily.csv`` — the Object Store key the
cstability and cgrowth Lean Algorithms read from in `Initialize`.

Usage
-----
    python scripts/classify_regime.py
    python scripts/classify_regime.py --start 2000-01-01 --hysteresis 0.25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from conquest.data import ParquetCache, FredClient
from conquest.regime import RegimeClassifier


WORKSPACE = Path(__file__).resolve().parent.parent
CACHE_ROOT = WORKSPACE / "data" / "alternative" / "conquest" / "raw"
OUT_FILE = WORKSPACE / "storage" / "conquest" / "regime" / "daily.csv"


def _yoy(series: pd.Series, periods: int) -> pd.Series:
    # fill_method=None pin: pandas 3.x will drop the implicit ffill default; be explicit.
    return series.pct_change(periods, fill_method=None) * 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-months", type=int, default=60)
    ap.add_argument("--min-dwell-months", type=int, default=2)
    ap.add_argument("--hysteresis", type=float, default=0.0)
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cache = ParquetCache(CACHE_ROOT)
    if not cache.exists("fred", "GDPC1") or not cache.exists("fred", "CPIAUCSL"):
        print("ERROR: missing FRED cache. Run scripts/refresh_data.py first.", file=sys.stderr)
        return 1

    gdp_vintage = cache.read("fred", "GDPC1")
    cpi_vintage = cache.read("fred", "CPIAUCSL")

    # v1 simplification: use revised data for the rolling z-score normalisation
    # (revisions are typically small; the rolling stats are stable). The publication-lag
    # stamping below is what enforces PIT discipline at consumption time.
    today = pd.Timestamp.today()
    gdp = FredClient.as_of(gdp_vintage, today)
    cpi = FredClient.as_of(cpi_vintage, today)

    gdp_yoy = _yoy(gdp, periods=4)    # quarterly → 4 periods
    cpi_yoy = _yoy(cpi, periods=12)   # monthly → 12 periods

    classifier = RegimeClassifier(
        lookback_months=args.lookback_months,
        min_dwell_months=args.min_dwell_months,
        hysteresis=args.hysteresis,
    )
    daily = classifier.classify_to_daily(gdp_yoy, cpi_yoy)

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today()
    daily = daily.loc[args.start:end]
    daily.index.name = "date"

    # Stamp release date: regime for month-end T is "knowable" ~30 days later
    # (advance GDP estimate ≈ 30 days post-quarter-end is the binding constraint).
    daily["release_date"] = daily.index + pd.Timedelta(days=30)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_FILE)
    print(f"Wrote {len(daily):,} daily rows to {OUT_FILE}")

    counts = daily["regime"].value_counts(dropna=False)
    print("\nRegime distribution:")
    for label, count in counts.items():
        pct = count / len(daily) * 100
        print(f"  {str(label):14s} {count:6d} days ({pct:5.1f}%)")

    print("\nLast 5 daily rows:")
    print(daily[["regime", "gdp_yoy", "cpi_yoy", "gdp_yoy_z", "cpi_yoy_z", "confidence"]].tail(5).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
