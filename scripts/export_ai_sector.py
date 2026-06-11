"""AI/tech-sector trend signal for the options-sleeve monitor — the broad-gate BLIND SPOT
(a narrow AI decline while the S&P holds, which the breadth gate misses).

NASDAQ Composite vs its 200-day moving average from FRED (NASDAQCOM); <0 = AI/tech below
its 200DMA = downtrend. Output: storage/conquest/macro/ai_sector.csv (date, ndx_vs_200dma).

SELF-CONTAINED by design: fetches NASDAQCOM directly from the FRED REST API (one call, only
`requests`), with NO `conquest.data`/ParquetCache/pyarrow dependency — so it runs in the bare
GH-Action environment (numpy/pandas/pyyaml/requests). NASDAQCOM is an index close (never
revised), so no ALFRED/PIT-vintage machinery is needed; the live close == the point-in-time value.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent


def fetch_fred_series(code: str, key: str) -> pd.Series:
    """One FRED REST call -> a clean float Series indexed by date (drops '.'/missing)."""
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params={"series_id": code, "api_key": key, "file_type": "json"}, timeout=60)
    r.raise_for_status()
    obs = r.json().get("observations", []) or []
    data = {pd.to_datetime(o["date"]): float(o["value"])
            for o in obs if o.get("value") not in (".", "", None)}
    return pd.Series(data).sort_index()


def main():
    out = ROOT / "storage" / "conquest" / "macro" / "ai_sector.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sec = yaml.safe_load(open(ROOT / "secret.yaml")) if (ROOT / "secret.yaml").exists() else {}
    key = (sec or {}).get("fred_api_key", "")
    if not key:
        print("ERROR: fred_api_key missing", file=sys.stderr); sys.exit(1)
    print("Fetching NASDAQCOM from FRED...")
    ndx = fetch_fred_series("NASDAQCOM", key)
    if ndx.empty:
        print("ERROR: NASDAQCOM empty", file=sys.stderr); sys.exit(1)
    rs = (np.log(ndx) - np.log(ndx.rolling(200).mean())).dropna()   # NDX vs its 200DMA; <0 = downtrend
    rs.index.name = "date"; rs.name = "ndx_vs_200dma"
    rs.to_csv(out)
    print(f"  -> {out.relative_to(ROOT)}  ({len(rs)} rows, {rs.index[0].date()} -> {rs.index[-1].date()}, "
          f"latest {rs.iloc[-1]:+.3f} = AI/tech {'BELOW 200DMA (weak)' if rs.iloc[-1] < 0 else 'above 200DMA'})")


if __name__ == "__main__":
    main()
