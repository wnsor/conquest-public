"""Object Store-backed GDELT news sentiment + article-volume lookup.

Tier1 Signal 1 — populated daily from `scripts/ingest_gdelt_sentiment.py`,
which writes `storage/conquest/sentiment/gdelt_daily.csv` with two columns
per ticker: `<TICKER>_tone` and `<TICKER>_count`. The algorithm uploads it
to Object Store at key `conquest/sentiment/gdelt_daily.csv` and consults it
in `_build_context()` to populate two ctx fields:

  - `ctx.news_sentiment_24h[ticker]` — most recent day's mean tone,
    rescaled from GDELT's [-100, +100] to [-1.0, +1.0].
  - `ctx.news_volume_spike[ticker]` — today's article count divided by
    the prior 30-day mean article count.

Both lookups return None when data is missing (no row for `today`, or
insufficient history to compute a spike); the consumer should `.get(...)`
with a default.

Schema (CSV produced by scripts/ingest_gdelt_sentiment.py):
    date, <TICKER>_tone, <TICKER>_count, ... × N tickers

Pre-2015 dates have all-zero rows because GDELT v2 begins 2015-02-18.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from io import StringIO


class GDELTSentimentLoader:
    OBJECT_STORE_KEY = "conquest/sentiment/gdelt_daily.csv"
    LOOKBACK_DAYS = 30  # for news_volume_spike denominator

    def __init__(self):
        # ticker → sorted list of (date, tone, count)
        self._by_ticker: dict[str, list[tuple[date, float, int]]] = defaultdict(list)
        # ticker → date → 30d-prior mean article count (precomputed at parse time)
        self._volume_baseline: dict[str, dict[date, float]] = defaultdict(dict)

    @classmethod
    def from_csv_text(cls, csv_text: str) -> "GDELTSentimentLoader":
        cal = cls()
        if not csv_text:
            return cal
        reader = csv.DictReader(StringIO(csv_text))
        # Discover tickers from header: any "<TICKER>_tone" column.
        fields = reader.fieldnames or []
        tickers = [f[:-5] for f in fields if f.endswith("_tone")]
        for row in reader:
            d_str = (row.get("date") or "").strip()
            if not d_str:
                continue
            try:
                d = date.fromisoformat(d_str[:10])
            except ValueError:
                continue
            for t in tickers:
                tone_raw = row.get(f"{t}_tone", "")
                count_raw = row.get(f"{t}_count", "")
                try:
                    tone = float(tone_raw) if tone_raw not in (None, "") else 0.0
                except ValueError:
                    tone = 0.0
                try:
                    count = int(float(count_raw)) if count_raw not in (None, "") else 0
                except ValueError:
                    count = 0
                cal._by_ticker[t].append((d, tone, count))
        # Sort by date and precompute the 30d trailing mean count per (ticker, date).
        for t in cal._by_ticker:
            cal._by_ticker[t].sort()
            cal._precompute_volume_baseline(t)
        return cal

    def _precompute_volume_baseline(self, ticker: str) -> None:
        """30-day trailing mean article count for each calendar date in the series.

        Excludes the current day from the average (parity with conquest's
        existing volume_spike convention).
        """
        series = self._by_ticker[ticker]
        if len(series) < self.LOOKBACK_DAYS + 1:
            return
        # Maintain a running sum across the 30 prior entries.
        # series is sorted by date.
        baseline: dict[date, float] = {}
        counts = [c for _, _, c in series]
        cum = 0.0
        for i in range(len(series)):
            if i < self.LOOKBACK_DAYS:
                cum += counts[i]
                continue
            window = counts[i - self.LOOKBACK_DAYS:i]
            mean = sum(window) / self.LOOKBACK_DAYS
            baseline[series[i][0]] = mean
        self._volume_baseline[ticker] = baseline

    def tone(self, ticker: str, today: date) -> float | None:
        """Most recent tone on or before `today`, or None if no row.

        Returns raw GDELT scale [-100, +100]. Callers rescale to [-1, +1]
        if they want a normalized signal.
        """
        series = self._by_ticker.get(ticker.upper())
        if not series:
            return None
        # Binary-search-ish: walk back from the end (series is sorted).
        for d, tone, _count in reversed(series):
            if d <= today:
                return tone
        return None

    def volume_spike(self, ticker: str, today: date) -> float | None:
        """Today's article count / 30d-prior mean. None if no baseline yet
        or if baseline is zero."""
        series = self._by_ticker.get(ticker.upper())
        if not series:
            return None
        baseline = self._volume_baseline.get(ticker.upper(), {})
        # Find the most recent (date, count) entry ≤ today
        today_count = None
        today_date = None
        for d, _tone, count in reversed(series):
            if d <= today:
                today_date = d
                today_count = count
                break
        if today_date is None:
            return None
        base = baseline.get(today_date)
        if base is None or base <= 0:
            return None
        return today_count / base

    def propagation_5d(self, ticker: str, today: date) -> float | None:
        """5d-over-5d attention propagation ratio (leading signal).

        Returns (mean count last 5d) / (mean count prior 5d). > 1.50 means
        narrative is accelerating BEFORE the price move (per v_REFLEX_v2
        thesis). Returns None if insufficient history (< 10d of data
        on/before today).
        """
        series = self._by_ticker.get(ticker.upper())
        if not series:
            return None
        # Get counts for days <= today, sorted ascending
        relevant = [(d, count) for d, _tone, count in series if d <= today]
        if len(relevant) < 10:
            return None
        relevant.sort()
        last_5 = [c for _, c in relevant[-5:]]
        prior_5 = [c for _, c in relevant[-10:-5]]
        mean_last = sum(last_5) / 5.0
        mean_prior = sum(prior_5) / 5.0
        if mean_prior <= 0:
            return None
        return mean_last / mean_prior
