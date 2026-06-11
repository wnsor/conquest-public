"""Object Store-backed insider Form 4 opportunistic-buy lookup.

Phase 0 produces `storage/conquest/insider/form4_opportunistic_buys_daily.csv`
via scripts/data/form4.py (CMP 2012 classifier). The algorithm uploads it to
Object Store at key `conquest/insider/form4_opportunistic_buys_daily.csv`
and consults it per OnData.

Strategy A6 queries:
  - Did <ticker> have an opportunistic Officer/Director/10pct buy in the
    last N trading days? → fires a 45-DTE call.

Schema (matches Phase 0 fetcher):
    filing_date, transaction_date, ticker, insider_cik, insider_name, role,
    shares, price, dollar_value
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from io import StringIO


class InsiderForm4Calendar:
    OBJECT_STORE_KEY = "conquest/insider/form4_opportunistic_buys_daily.csv"

    # Tier1 Signal 3 — role weight for cluster score
    ROLE_WEIGHTS = {"officer": 2.0, "director": 1.5, "10pct": 1.0}

    def __init__(self):
        # ticker → sorted list of (transaction_date, role, dollar_value, insider_cik)
        self._by_ticker: dict[
            str, list[tuple[date, str, float, str]]
        ] = defaultdict(list)

    @classmethod
    def from_csv_text(cls, csv_text: str) -> "InsiderForm4Calendar":
        import csv
        cal = cls()
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            td_str = (row.get("transaction_date") or row.get("filing_date") or "").strip()
            if not ticker or not td_str:
                continue
            try:
                td = date.fromisoformat(td_str[:10])
            except Exception:
                continue
            role = (row.get("role") or "").strip()
            try:
                dollar = float(row.get("dollar_value", 0) or 0)
            except Exception:
                dollar = 0.0
            insider_cik = (row.get("insider_cik") or "").strip()
            cal._by_ticker[ticker].append((td, role, dollar, insider_cik))
        for t in cal._by_ticker:
            cal._by_ticker[t].sort()
        return cal

    def buys_within_n_days(
        self, ticker: str, today: date, n: int,
        min_dollar: float = 25_000.0,
        require_officer_or_director: bool = True,
    ) -> list[tuple[date, str, float, str]]:
        """Return list of buys for this ticker within last n days, filtered.

        Returns 4-tuples (transaction_date, role, dollar_value, insider_cik).
        Existing index-2 (dollar) and index-0/1 (date/role) consumers are
        unaffected.

        min_dollar: ignore tiny buys (default $25k).
        require_officer_or_director: drop pure 10pct-owner filings (often
        funds, not insiders).
        """
        buys = self._by_ticker.get(ticker.upper(), [])
        if not buys:
            return []
        cutoff = today - timedelta(days=n)
        out = []
        for td, role, dollar, cik in buys:
            if td > today:
                break  # rest are future-dated relative to walk-forward
            if td < cutoff:
                continue
            if dollar < min_dollar:
                continue
            if require_officer_or_director:
                rl = role.lower()
                if "officer" not in rl and "director" not in rl:
                    continue
            out.append((td, role, dollar, cik))
        return out

    def tickers_with_recent_buys(
        self, today: date, n: int = 5,
        min_dollar: float = 25_000.0,
    ) -> set[str]:
        result = set()
        for ticker in self._by_ticker:
            if self.buys_within_n_days(ticker, today, n, min_dollar=min_dollar):
                result.add(ticker)
        return result

    def distinct_insider_count(
        self, ticker: str, today: date, n_days: int = 5,
        min_dollar: float = 25_000.0,
    ) -> int:
        """Number of DISTINCT insider CIKs with qualifying buys in last n days.

        Used by v_REFLEX_v2 and v_TRIPLE_CONFLUENCE — leading signal that
        an insider cluster is forming around a ticker (3+ insiders = active
        accumulation). Doesn't weight by role; counts each unique CIK once.
        """
        buys = self._by_ticker.get(ticker.upper(), [])
        if not buys:
            return 0
        cutoff = today - timedelta(days=n_days)
        ciks: set[str] = set()
        for td, role, dollar, cik in buys:
            if td > today:
                break
            if td < cutoff:
                continue
            if dollar < min_dollar:
                continue
            if cik:
                ciks.add(cik)
        return len(ciks)

    def cluster_score(
        self, ticker: str, today: date, n_days: int = 5,
        min_dollar: float = 25_000.0,
    ) -> float:
        """Tier1 Signal 3 — weighted distinct-insider count over last n days.

        Each insider_cik with at least one qualifying buy contributes its
        role weight (Officer 2.0, Director 1.5, 10pct 1.0). Multiple buys
        from the same CIK count once at the maximum-weighted role observed
        (an Officer who also happens to file as 10pct counts as Officer).

        Comma-separated role strings (e.g. "Officer,Director") match each
        token independently and use the maximum.

        Returns 0.0 if no qualifying buys or ticker not in calendar.
        """
        buys = self._by_ticker.get(ticker.upper(), [])
        if not buys:
            return 0.0
        cutoff = today - timedelta(days=n_days)
        # cik → max role weight observed
        by_cik: dict[str, float] = {}
        for td, role, dollar, cik in buys:
            if td > today:
                break
            if td < cutoff:
                continue
            if dollar < min_dollar:
                continue
            if not cik:
                # Without a CIK we can't dedupe across filings — skip rather
                # than risk inflating the score.
                continue
            rl = role.lower()
            best = 0.0
            for token, w in self.ROLE_WEIGHTS.items():
                if token in rl and w > best:
                    best = w
            if best <= 0:
                continue
            prev = by_cik.get(cik, 0.0)
            if best > prev:
                by_cik[cik] = best
        return sum(by_cik.values())
