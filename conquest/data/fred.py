"""FRED + ALFRED client. Vintage-aware by default — for any historical date
the client returns only what was knowable then, not later revisions.
"""
from __future__ import annotations

import time

import requests
import pandas as pd

from conquest.data.cache import ParquetCache


class FredClient:
    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str, cache: ParquetCache | None = None):
        if not api_key:
            raise ValueError(
                "FRED API key required (set fred_api_key in secret.yaml; see https://fredaccount.stlouisfed.org/apikeys)."
            )
        self.api_key = api_key
        self.cache = cache
        self.session = requests.Session()

    def _fred_get(self, params: dict, timeout: int = 30, max_retries: int = 6):
        """GET /series/observations with retry + backoff on FRED rate-limits
        (HTTP 429) and transient 5xx. FRED caps at ~120 req/min, so a cold
        full-history refresh of many series bursts past the limit; without this,
        a single 429 aborts the entire macro refresh (and cascades to the
        regime classifier + 4-vote that depend on it)."""
        delay = 5.0
        resp = None
        for attempt in range(max_retries):
            resp = self.session.get(
                f"{self.BASE_URL}/series/observations", params=params, timeout=timeout
            )
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                ra = resp.headers.get("Retry-After", "")
                wait = float(ra) if ra.isdigit() else delay
                time.sleep(min(wait, 60.0))
                delay = min(delay * 2, 60.0)
                continue
            return resp
        return resp

    def fetch_vintage(self, series_id: str, refresh: bool = False) -> pd.DataFrame:
        """Pull the full real-time period history of a series from ALFRED.

        Returns a DataFrame with columns:
            date            — reference date of the observation
            realtime_start  — first date this revision was the published value
            realtime_end    — first date a new revision superseded it (exclusive)
            value           — the published value (NaN if FRED reports '.')

        Use `as_of(df, t)` to slice to what was knowable by date `t`.
        """
        if self.cache is not None and not refresh and self.cache.exists("fred", series_id):
            return self.cache.read("fred", series_id)

        # FRED rejects realtime_end values other than today or its magic "9999-12-31" sentinel.
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "realtime_start": "1776-07-04",   # FRED epoch
            "realtime_end": "9999-12-31",
            "limit": 100000,
        }
        resp = self._fred_get(params)
        if resp.status_code == 400:
            try:
                msg = resp.json().get("error_message", "")
            except Exception:
                msg = ""
            # Daily series (e.g. DFF) can exceed FRED's 2000-vintage cap. Fall back to
            # current-revision-only — fine because such series rarely revise meaningfully.
            if "vintage dates" in msg and "exceeds the maximum" in msg:
                fallback = {k: v for k, v in params.items()
                            if k not in ("realtime_start", "realtime_end")}
                resp = self._fred_get(fallback)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if not obs:
            df = pd.DataFrame(columns=["date", "realtime_start", "realtime_end", "value"])
        else:
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df["realtime_start"] = pd.to_datetime(df["realtime_start"])
            # FRED's open-ended realtime_end is "9999-12-31" — beyond pandas'
            # nanosecond Timestamp range. Substitute a far-future sentinel.
            df["realtime_end"] = pd.to_datetime(
                df["realtime_end"].replace("9999-12-31", "2099-12-31")
            )
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df[["date", "realtime_start", "realtime_end", "value"]].sort_values(
                ["date", "realtime_start"]
            ).reset_index(drop=True)

        if self.cache is not None:
            self.cache.write("fred", series_id, df)
        return df

    @staticmethod
    def as_of(vintage_df: pd.DataFrame, asof_date) -> pd.Series:
        """Slice the vintage DataFrame to values that were known by `asof_date`.

        For each reference date, return the most-recent revision whose
        ``realtime_start <= asof_date < realtime_end``.
        """
        asof = pd.Timestamp(asof_date)
        valid = vintage_df[
            (vintage_df["realtime_start"] <= asof) & (vintage_df["realtime_end"] > asof)
        ]
        return (
            valid.sort_values("realtime_start")
            .drop_duplicates("date", keep="last")
            .set_index("date")["value"]
            .sort_index()
        )
