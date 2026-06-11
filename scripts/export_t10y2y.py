"""Export T10Y2Y daily yield-curve spread for cstability v10.5a 4-vote ensemble.

v10.5a substitutes the noisy P(Stagflation) GBM forecast vote with a T10Y2Y
yield-curve inversion vote (T10Y2Y < 0). Yield-curve inversion has historically
led recessions by 6-18 months and is a market-determined signal (no model risk
from a probability forecaster).

FRED series: T10Y2Y = 10-Year Treasury Constant Maturity Minus 2-Year Treasury
Constant Maturity. Daily, no licensing restriction (free public series).

Output: storage/conquest/yield_curve/t10y2y.csv  (date, t10y2y)
        Forward-filled to a daily index aligned with the rest of conquest's
        signal CSVs (FRED publishes business-day only).

Cstability reads via Object Store key conquest/yield_curve/t10y2y.csv.
Run after scripts/refresh_data.py (which populates the FRED parquet cache).
Idempotent.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT
sys.path.insert(0, str(ROOT))

from conquest.data import ParquetCache, FredClient

SECRET_FILE = WORKSPACE / "secret.yaml"


def load_secrets() -> dict:
    if not SECRET_FILE.exists():
        return {}
    with open(SECRET_FILE) as f:
        return yaml.safe_load(f) or {}


def export_t10y2y(start: str = "2014-01-01", end: str | None = None):
    # Default end = today so daily cron extends the series. Pre-2026-05-24 was
    # hardcoded "2025-12-31" — froze t10y2y signal at YE2025.
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    out_path = ROOT / "storage" / "conquest" / "yield_curve" / "t10y2y.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    secrets = load_secrets()
    fred_key = secrets.get("fred_api_key", "")
    if not fred_key:
        print("ERROR: fred_api_key not set in secret.yaml", file=sys.stderr)
        sys.exit(1)

    cache = ParquetCache(WORKSPACE / "data" / "alternative" / "conquest" / "raw")
    fred = FredClient(api_key=fred_key, cache=cache)

    print("Fetching T10Y2Y from FRED...")
    vintage = fred.fetch_vintage("T10Y2Y", refresh=False)
    if vintage.empty:
        print("ERROR: T10Y2Y vintage is empty", file=sys.stderr)
        sys.exit(1)

    # T10Y2Y is daily and doesn't revise meaningfully; take current revision.
    series = (
        vintage.sort_values(["date", "realtime_start"])
        .drop_duplicates("date", keep="last")
        .set_index("date")["value"]
        .sort_index()
    )

    # Filter to backtest window + forward-fill to handle FRED holiday gaps so
    # cstability's _latest_value always returns a value when queried on weekdays.
    series = series.loc[start:end].dropna()
    daily_idx = pd.bdate_range(start=series.index.min(), end=series.index.max())
    series = series.reindex(daily_idx).ffill()

    series.index.name = "date"
    series.to_frame(name="t10y2y").to_csv(out_path)
    print(f"  → {out_path.relative_to(ROOT)}")
    print(
        f"     {len(series)} daily rows; range "
        f"{series.index[0].date()} → {series.index[-1].date()}; "
        f"latest value = {series.iloc[-1]:+.3f}"
    )
    inv_count = int((series < 0).sum())
    print(f"     Inverted (<0) on {inv_count}/{len(series)} days ({inv_count/len(series):.0%}).")


if __name__ == "__main__":
    export_t10y2y()
