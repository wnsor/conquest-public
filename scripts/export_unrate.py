"""Export UNRATE (FRED unemployment rate, seasonally adjusted) for the webapp
macro overlay and v11.3-candidate evaluation.

UNRATE is published by BLS via FRED on the first Friday of each month covering
the previous month (release lag ~ 1 week). We use the ALFRED vintage endpoint
so that on any given date the value is the most-recently-released figure as of
that date — i.e., the PIT-correct view a strategy could have used in real time.

Output: storage/conquest/macro/unrate.csv  (date, unrate, release_date, reference_date)
        Forward-filled to a business-day index. release_date = BLS publication
        date for that figure; reference_date = the month being measured.

Read by the webapp Macro overlay (toggle in section 8 chart-controls). Also
the leading vote candidate for the deferred v11.3 4-vote ensemble (CLAUDE.md).

Run after scripts/refresh_data.py (which populates the FRED parquet cache).
Idempotent.
"""
from __future__ import annotations

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


def export_unrate(start: str = "2008-01-01", end: str = "2026-12-31") -> None:
    out_path = ROOT / "storage" / "conquest" / "macro" / "unrate.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    secrets = load_secrets()
    fred_key = secrets.get("fred_api_key", "")
    if not fred_key:
        print("ERROR: fred_api_key not set in secret.yaml", file=sys.stderr)
        sys.exit(1)

    cache = ParquetCache(WORKSPACE / "data" / "alternative" / "conquest" / "raw")
    fred = FredClient(api_key=fred_key, cache=cache)

    print("Fetching UNRATE from FRED (ALFRED vintage)...")
    vintage = fred.fetch_vintage("UNRATE", refresh=False)
    if vintage.empty:
        print("ERROR: UNRATE vintage is empty", file=sys.stderr)
        sys.exit(1)

    # PIT view: for each release event (realtime_start), take the latest value
    # that was published in that event. UNRATE rarely revises but for the few
    # cases it does we want the first publication's value.
    pit = (
        vintage.sort_values(["realtime_start", "date"])
        .drop_duplicates(subset=["realtime_start"], keep="last")
        [["realtime_start", "date", "value"]]
        .rename(columns={"realtime_start": "release_date", "date": "reference_date"})
    )
    pit["release_date"] = pd.to_datetime(pit["release_date"])
    pit["reference_date"] = pd.to_datetime(pit["reference_date"])

    # Index by release_date and keep release_date itself as a column so the
    # FFill carries it forward — every daily row will know when the currently
    # effective figure was first published.
    indexed = (
        pit.assign(release_date_col=pit["release_date"])
        .set_index("release_date")
        .sort_index()
        .rename(columns={"release_date_col": "release_date"})
        [["value", "release_date", "reference_date"]]
    )

    # Forward-fill to a daily business-day index. On each business day the row
    # reflects the most recently-released UNRATE figure (PIT correct).
    daily_idx = pd.bdate_range(start=start, end=end)
    daily = indexed.reindex(daily_idx, method="ffill")
    daily = daily.dropna(subset=["value"])
    daily.index.name = "date"
    daily = daily.rename(columns={"value": "unrate"})
    daily = daily[["unrate", "release_date", "reference_date"]]
    daily.to_csv(out_path)

    print(f"  → {out_path.relative_to(ROOT)}")
    print(
        f"     {len(daily)} daily rows; range "
        f"{daily.index[0].date()} → {daily.index[-1].date()}; "
        f"latest unrate = {daily['unrate'].iloc[-1]:.1f}% "
        f"(reference {daily['reference_date'].iloc[-1].date()})"
    )


if __name__ == "__main__":
    export_unrate()
