"""OECD data-explorer CSV parser.

The user pulls Composite Leading Indicator / Consumer Confidence / Business
Confidence CSVs manually from https://data-explorer.oecd.org/ and we parse them
here. Phase 2 may automate via SDMX; the manual flow is fine for v1.

Column expectations (default OECD CSV export):
- TIME_PERIOD: ISO date (e.g. "2024-08")
- OBS_VALUE:   numeric observation
- REF_AREA:    country/aggregate code (USA, OECD, etc.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from conquest.data.cache import ParquetCache


def parse_oecd_csv(path: Path) -> pd.DataFrame:
    """Parse a CSV from the OECD data-explorer; return a tidy DataFrame.

    Columns of the returned frame: period_date, value, ref_area.
    """
    raw = pd.read_csv(path)
    cols = {c.upper(): c for c in raw.columns}
    needed = {"TIME_PERIOD", "OBS_VALUE", "REF_AREA"}
    missing = needed - cols.keys()
    if missing:
        raise ValueError(f"OECD CSV {path} missing columns: {missing}")
    df = raw.rename(columns={
        cols["TIME_PERIOD"]: "period",
        cols["OBS_VALUE"]: "value",
        cols["REF_AREA"]: "ref_area",
    })
    df["period_date"] = pd.to_datetime(df["period"], errors="coerce")
    df = df.dropna(subset=["period_date"]).sort_values("period_date").reset_index(drop=True)
    return df[["period_date", "value", "ref_area"]]


def cache_oecd_csv(path: Path, key: str, cache: ParquetCache) -> pd.DataFrame:
    """Parse an OECD CSV and write the parsed frame into the parquet cache."""
    df = parse_oecd_csv(path)
    cache.write("oecd", key, df)
    return df
