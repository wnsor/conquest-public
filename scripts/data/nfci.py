"""Fetch Chicago Fed National Financial Conditions Index (NFCI) from FRED.

NFCI is a comprehensive weekly composite of 100+ financial conditions
indicators (credit, equity, liquidity, leverage). Single best leading
indicator for risk-asset regime — typically leads equity by 4-12 weeks.

Series IDs on FRED:
  NFCI          — main NFCI (risk-weighted standardized score; 0 = average)
  ANFCI         — Adjusted NFCI (controls for GDP/inflation)
  NFCICREDIT    — Credit sub-index
  NFCILEVERAGE  — Leverage sub-index
  NFCIRISK      — Risk sub-index

Output: storage/conquest/macro/nfci_weekly.csv
Schema: date, nfci, anfci, nfci_credit, nfci_leverage, nfci_risk

Interpretation:
  NFCI < 0    = financial conditions LOOSER than average (risk-on)
  NFCI > 0    = TIGHTER than average (risk-off)
  Δ NFCI > +0.2 in 4 weeks → coming tightening; equity stress in 1-2 months
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

WORKSPACE = Path(__file__).resolve().parent.parent.parent
OUT_CSV = WORKSPACE / "storage" / "conquest" / "macro" / "nfci_weekly.csv"
SECRET_FILE = WORKSPACE / "secret.yaml"

FRED_SERIES = ["NFCI", "ANFCI", "NFCICREDIT", "NFCILEVERAGE", "NFCIRISK"]


def load_fred_key() -> str:
    import os
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    if SECRET_FILE.exists():
        data = yaml.safe_load(SECRET_FILE.read_text()) or {}
        return data.get("fred_api_key", "")
    return ""


def fetch_series(series_id: str, api_key: str, start: str) -> pd.Series:
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id, observation_start=start)
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s.rename(series_id.lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2008-01-01")
    args = ap.parse_args()
    api_key = load_fred_key()
    if not api_key:
        print("ERROR: FRED_API_KEY env var or secret.yaml:fred_api_key required", file=sys.stderr)
        return 1
    print(f"Fetching NFCI sub-indices from FRED, start={args.start}")
    cols = [fetch_series(sid, api_key, args.start) for sid in FRED_SERIES]
    df = pd.concat(cols, axis=1).dropna(how="all")
    # Rename for clarity
    df.columns = ["nfci", "anfci", "nfci_credit", "nfci_leverage", "nfci_risk"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV)
    print(f"Wrote {len(df)} weekly rows -> {OUT_CSV.relative_to(WORKSPACE)}")
    print(f"  NFCI range: {df['nfci'].min():.3f} - {df['nfci'].max():.3f}  latest={df['nfci'].iloc[-1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
