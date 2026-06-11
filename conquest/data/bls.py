"""BLS API v2 client. The BLS public API returns release timestamps natively,
so its published values are point-in-time correct without an ALFRED-equivalent.
"""
from __future__ import annotations

import requests
import pandas as pd

from conquest.data.cache import ParquetCache


class BlsClient:
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    MAX_YEARS_PER_REQUEST = 20  # BLS API per-request span limit

    def __init__(self, api_key: str | None = None, cache: ParquetCache | None = None):
        # Public access works without a key but with stricter rate limits;
        # registered keys raise the ceiling. Set bls_api_key in secret.yaml when ready.
        self.api_key = api_key
        self.cache = cache
        self.session = requests.Session()

    def fetch_series(
        self,
        series_id: str,
        start_year: int = 2000,
        end_year: int = 2025,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Pull a BLS time series and return a DataFrame with cols
        (period_date, value, footnotes). Cached by (series_id, start_year, end_year)."""
        cache_key = f"{series_id}_{start_year}_{end_year}"
        if self.cache is not None and not refresh and self.cache.exists("bls", cache_key):
            return self.cache.read("bls", cache_key)

        rows: list[dict] = []
        # Chunk requests across the BLS span limit
        for s in range(start_year, end_year + 1, self.MAX_YEARS_PER_REQUEST):
            e = min(s + self.MAX_YEARS_PER_REQUEST - 1, end_year)
            payload: dict = {
                "seriesid": [series_id],
                "startyear": str(s),
                "endyear": str(e),
            }
            if self.api_key:
                payload["registrationkey"] = self.api_key
            resp = self.session.post(self.BASE_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(f"BLS request failed: {data.get('message')}")
            for series in data.get("Results", {}).get("series", []):
                for item in series.get("data", []):
                    period = item.get("period", "")
                    year = int(item["year"])
                    if period.startswith("M") and period[1:].isdigit():
                        month = int(period[1:])
                        if month > 12:        # M13 = annual average; skip
                            continue
                        period_date = pd.Timestamp(year=year, month=month, day=1)
                    elif period.startswith("Q") and period[1:].isdigit():
                        quarter = int(period[1:])
                        period_date = pd.Timestamp(year=year, month=quarter * 3 - 2, day=1)
                    else:
                        continue
                    rows.append({
                        "period_date": period_date,
                        "value": float(item["value"]),
                        "footnotes": str(item.get("footnotes", "")),
                    })

        df = (
            pd.DataFrame(rows)
            .drop_duplicates("period_date")
            .sort_values("period_date")
            .reset_index(drop=True)
        )
        if self.cache is not None:
            self.cache.write("bls", cache_key, df)
        return df
