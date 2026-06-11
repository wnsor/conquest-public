"""Object-store-backed earnings calendar lookup.

The algorithm calls load_from_object_store(self.ObjectStore) once at
Initialize. Strategies query within_n_days(ticker, n) per OnData to decide
whether to enter an earnings-vol play (C-category).

Object Store key: `conquest/options/earnings_calendar.csv`
Schema (matches scripts/data/earnings_calendar.py):
    ticker, earnings_date, time_of_day, eps_estimate, eps_actual, surprise_pct
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from io import StringIO


class EarningsCalendar:
    OBJECT_STORE_KEY = "conquest/options/earnings_calendar.csv"

    def __init__(self):
        # ticker → sorted list of earnings dates
        self._by_ticker: dict[str, list[date]] = defaultdict(list)
        # (ticker, date) → surprise_pct (None if not yet known)
        self._surprise: dict[tuple[str, date], float | None] = {}

    @classmethod
    def from_csv_text(cls, csv_text: str) -> "EarningsCalendar":
        import csv
        cal = cls()
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            ed = row.get("earnings_date", "").strip()
            if not ticker or not ed:
                continue
            try:
                d = date.fromisoformat(ed[:10])
            except Exception:
                continue
            cal._by_ticker[ticker].append(d)
            sp = row.get("surprise_pct", "")
            try:
                cal._surprise[(ticker, d)] = float(sp) if sp not in ("", None) else None
            except Exception:
                cal._surprise[(ticker, d)] = None
        for t in cal._by_ticker:
            cal._by_ticker[t].sort()
        return cal

    def next_earnings(self, ticker: str, today: date) -> date | None:
        dates = self._by_ticker.get(ticker.upper())
        if not dates:
            return None
        for d in dates:
            if d >= today:
                return d
        return None

    def within_n_days(self, ticker: str, today: date, n: int) -> bool:
        nxt = self.next_earnings(ticker, today)
        if nxt is None:
            return False
        return 0 <= (nxt - today).days <= n

    def last_earnings(self, ticker: str, today: date) -> date | None:
        dates = self._by_ticker.get(ticker.upper())
        if not dates:
            return None
        past = [d for d in dates if d < today]
        return past[-1] if past else None

    def days_since_last_earnings(self, ticker: str, today: date) -> int | None:
        last = self.last_earnings(ticker, today)
        if last is None:
            return None
        return (today - last).days

    def last_surprise(self, ticker: str, today: date) -> float | None:
        last = self.last_earnings(ticker, today)
        if last is None:
            return None
        return self._surprise.get((ticker.upper(), last))
