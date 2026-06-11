"""Pull macro data from FRED (vintage-aware) + BLS into the local parquet cache.

Idempotent: re-runs are no-ops if the cache already has the series. Pass
``--refresh`` to force re-download.

Usage
-----
    python scripts/refresh_data.py
    python scripts/refresh_data.py --refresh
    python scripts/refresh_data.py --refresh --series gdp_real cpi_headline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from conquest.data import ParquetCache, FredClient, BlsClient, REGISTRY


WORKSPACE = Path(__file__).resolve().parent.parent
CACHE_ROOT = WORKSPACE / "data" / "alternative" / "conquest" / "raw"
SECRET_FILE = WORKSPACE / "secret.yaml"


def load_secrets() -> dict:
    if not SECRET_FILE.exists():
        raise FileNotFoundError(f"{SECRET_FILE} not found.")
    with open(SECRET_FILE) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Force re-download even if cached")
    ap.add_argument("--series", nargs="*", default=None,
                    help="Only refresh these registry names (default: all)")
    args = ap.parse_args()

    secrets = load_secrets()
    fred_key = secrets.get("fred_api_key", "")
    bls_key = secrets.get("bls_api_key") or None
    if not fred_key:
        print("ERROR: fred_api_key not set in secret.yaml", file=sys.stderr)
        return 1

    cache = ParquetCache(CACHE_ROOT)
    fred = FredClient(api_key=fred_key, cache=cache)
    bls = BlsClient(api_key=bls_key, cache=cache)

    targets = (
        list(REGISTRY.values()) if args.series is None
        else [REGISTRY[name] for name in args.series if name in REGISTRY]
    )
    if not targets:
        print("Nothing to fetch.", file=sys.stderr)
        return 1

    for spec in targets:
        if spec.source == "fred":
            print(f"FRED   {spec.series_id:12s} ({spec.name}) ...", end=" ", flush=True)
            df = fred.fetch_vintage(spec.series_id, refresh=args.refresh)
            print(f"{len(df):,} vintage rows")
        elif spec.source == "bls":
            print(f"BLS    {spec.series_id:12s} ({spec.name}) ...", end=" ", flush=True)
            df = bls.fetch_series(spec.series_id, start_year=2000, end_year=2025, refresh=args.refresh)
            print(f"{len(df):,} rows")
        else:
            print(f"SKIP   {spec.name} (source={spec.source}; not yet wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
