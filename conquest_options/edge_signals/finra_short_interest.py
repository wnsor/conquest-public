"""Object Store-backed FINRA short interest velocity lookup.

The daily GH Action refreshes FINRA biweekly SI data into:
    storage/conquest/options/finra_si_biweekly.csv (key: conquest/options/finra_si_biweekly.csv)
    columns: settlement_date, ticker, short_interest_shares, days_to_cover, percent_float

This loader provides the LEADING signal: short_interest_velocity, the
week-over-week (biweekly-over-biweekly) % change in shares held short.

Used by v_REFLEX_v2, v_SHORT_SQUEEZE_PURE, v_TRIPLE_CONFLUENCE.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from io import StringIO


class FinraShortInterestLoader:
    OBJECT_STORE_KEY = "conquest/options/finra_si_biweekly.csv"

    def __init__(self):
        # ticker → sorted list of (settlement_date, short_interest_shares,
        #                          days_to_cover, percent_float)
        self._by_ticker: dict[
            str, list[tuple[date, float, float, float]]
        ] = defaultdict(list)

    @classmethod
    def from_csv_text(cls, csv_text: str) -> "FinraShortInterestLoader":
        import csv
        loader = cls()
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            sd_str = (row.get("settlement_date") or "").strip()
            if not ticker or not sd_str:
                continue
            try:
                sd = date.fromisoformat(sd_str[:10])
            except Exception:
                continue
            try:
                si = float(row.get("short_interest_shares", 0) or 0)
                dtc = float(row.get("days_to_cover", 0) or 0)
                pf = float(row.get("percent_float", 0) or 0)
            except Exception:
                continue
            loader._by_ticker[ticker].append((sd, si, dtc, pf))
        for t in loader._by_ticker:
            loader._by_ticker[t].sort()
        return loader

    def velocity(self, ticker: str, today: date) -> float | None:
        """Returns (latest_si - prior_si) / prior_si, or None if insufficient history.

        Uses the two most recent reports on or before `today`. FINRA publishes
        biweekly, so "velocity" here is the 2-week % change in shares short.
        > +0.20 = SI growing 20% biweekly = squeeze precursor (per v_SHORT_SQUEEZE_PURE).
        """
        series = self._by_ticker.get(ticker.upper())
        if not series:
            return None
        # Get the two most recent reports on/before today
        relevant = [(d, si) for d, si, _, _ in series if d <= today]
        if len(relevant) < 2:
            return None
        relevant.sort()
        prior_d, prior_si = relevant[-2]
        latest_d, latest_si = relevant[-1]
        if prior_si <= 0:
            return None
        return (latest_si - prior_si) / prior_si

    def days_to_cover(self, ticker: str, today: date) -> float | None:
        """Most recent days-to-cover on or before today, or None."""
        series = self._by_ticker.get(ticker.upper())
        if not series:
            return None
        for d, _si, dtc, _pf in reversed(series):
            if d <= today:
                return dtc
        return None
