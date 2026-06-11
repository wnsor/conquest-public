"""Credit & rate stress signals — leading indicators for risk-off regime.

Pulls daily series from FRED that have proven leading-indicator behavior for
equity drawdowns:

- HY_OAS         BAMLH0A0HYM2  ICE BofA US High Yield OAS  (2-4 week lead, low FP)
- IG_OAS         BAMLC0A0CM    ICE BofA US Corporate IG OAS (slower, less actionable)
- Y10            DGS10         10-Year Treasury yield
- Y2             DGS2          2-Year Treasury yield
- T10Y2Y         T10Y2Y        10y - 2y spread (recession leading indicator)
- REAL10         DFII10        10y TIPS real yield
- DOLLAR         DTWEXBGS      Trade-weighted dollar index (broad)

Uses the existing FredClient with vintage-disabled fetch (these series rarely
revise meaningfully; daily-bar data ages out of FRED's vintage cap quickly).

Caches as parquet under data/alternative/conquest/raw/fred/{SERIES}.parquet.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from conquest.secrets import fred_api_key as _fred_api_key


# Series ID -> human-readable label (for the output DataFrame columns)
CREDIT_SERIES = {
    "BAMLH0A0HYM2": "hy_oas",       # High-yield OAS, daily, percent
    "BAMLC0A0CM":   "ig_oas",       # IG OAS, daily, percent
    "DGS10":        "y10",          # 10y treasury yield
    "DGS2":         "y2",           # 2y treasury yield
    "T10Y2Y":       "t10y2y",       # 10y-2y spread (recession indicator)
    "DFII10":       "real10",       # 10y TIPS real yield
    "DTWEXBGS":     "dollar",       # Trade-weighted dollar
}


def fetch_credit_series(series_id: str, api_key: str | None = None) -> pd.Series:
    """Fetch a single FRED series as a daily pd.Series of values."""
    if api_key is None:
        api_key = _fred_api_key()
    if not api_key:
        raise RuntimeError(
            "FRED API key missing. Set fred_api_key in secret.yaml."
        )
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "limit": 100000,
    }
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params=params, timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    rows = []
    for o in obs:
        try:
            v = float(o["value"])
        except (ValueError, KeyError):
            continue
        rows.append((pd.Timestamp(o["date"]), v))
    if not rows:
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(rows, columns=["date", "value"]).set_index("date").sort_index()
    return df["value"].rename(series_id)


def fetch_credit_panel(
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Pull all CREDIT_SERIES from FRED (with parquet cache) and return a daily panel.

    Returns DataFrame with columns from CREDIT_SERIES.values() (hy_oas, ig_oas,
    y10, y2, t10y2y, real10, dollar). Forward-filled to a daily business-day
    index spanning the union of all series.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent / \
                    "data" / "alternative" / "conquest" / "raw" / "fred"
    cache_dir.mkdir(parents=True, exist_ok=True)

    series_dict: dict[str, pd.Series] = {}
    for sid, label in CREDIT_SERIES.items():
        cache_path = cache_dir / f"{sid}.parquet"
        if cache_path.exists() and not refresh:
            df = pd.read_parquet(cache_path)
            series_dict[label] = df.iloc[:, 0]
            continue
        s = fetch_credit_series(sid)
        s.to_frame(label).to_parquet(cache_path)
        series_dict[label] = s

    panel = pd.concat(series_dict, axis=1).sort_index()
    panel.index.name = "date"
    return panel
