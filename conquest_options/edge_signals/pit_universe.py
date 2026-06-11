"""Point-in-time tradeable-universe gate (survivorship-bias kill) for conquest_options.

Why this exists
---------------
The WSB single-stock universe (NVDA, SMCI, PLTR, NBIS, CRWV, COIN, RKLB, ...) is a
2026 hand-pick of names that ALREADY mooned. Backtesting 2018-2026 on that fixed
forward list trades tickers we only know mattered *because* they won — textbook
survivorship / selection-on-outcome bias (Day-2 handover, bias #1). The
``options_screen_daily.csv`` is also a single forward snapshot, not point-in-time.

This module enforces PIT discipline: on date *T* a strategy may only trade names
that were *index members as of T*. Membership is reconstructed by the same
S&P-1500 changes-log replay as ``scripts/sp500_pit_membership.py`` but materialized
into a monthly-snapshot CSV (``conquest/universe/sp500_pit_monthly.csv``, columns
``as_of,ticker`` — the schema ``scripts/train_xgb_m1.py`` already consumes) that the
Lean algorithm can read from the Object Store at runtime.

Pure stdlib (csv + datetime) — no pandas — so it runs inside the Lean algorithm and
is unit-testable on synthetic CSV text with no QC / network / data dependency.

Runtime use (inside Algorithm.initialize, then at entry time):
    txt = self.object_store.read(PitUniverse.OBJECT_STORE_KEY)
    self._pit = PitUniverse.from_csv_text(txt)
    ...
    if ticker not in self._pit.members_asof(self.Time):   # gate each entry
        continue   # not PIT-tradeable on this date → skip (survivorship guard)

``select_pit_top_n`` composes the gate with any ranker exposing ``.top_n(inputs, n)``
(e.g. ``edge_signals.stock_picker.StockPicker``) so the candidate pool is PIT-gated
BEFORE ranking — the output can never contain a name that was not a member on the
trade date, regardless of how strong its (forward-known) momentum looks.
"""
from __future__ import annotations

import csv
import io
from datetime import date, datetime
from typing import Iterable, Mapping


def _to_date(value) -> "date | None":
    """Coerce str / date / datetime to a date; tolerate 'YYYY-MM-DD[ HH:MM:SS]'."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s:
        return None
    s = s.split()[0].split("T")[0]
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


class PitUniverse:
    """Point-in-time membership from monthly snapshots (``as_of,ticker`` long form)."""

    OBJECT_STORE_KEY = "conquest/universe/sp500_pit_monthly.csv"
    _DATE_COLS = ("as_of", "date", "asof")
    _TICKER_COLS = ("ticker", "symbol")

    def __init__(self, snapshots: "Mapping[date, set[str]]"):
        self._dates: list[date] = sorted(snapshots)
        self._snapshots: dict[date, set[str]] = {d: set(snapshots[d]) for d in self._dates}

    @classmethod
    def from_csv_text(cls, text: str) -> "PitUniverse":
        reader = csv.DictReader(io.StringIO(text))
        fields = [f.strip() for f in (reader.fieldnames or [])]
        date_col = next((c for c in cls._DATE_COLS if c in fields), None)
        tick_col = next((c for c in cls._TICKER_COLS if c in fields), None)
        if date_col is None or tick_col is None:
            raise ValueError(
                f"PIT CSV needs a date column in {cls._DATE_COLS} and a ticker column "
                f"in {cls._TICKER_COLS}; got header {fields}"
            )
        snapshots: dict[date, set[str]] = {}
        for row in reader:
            d = _to_date(row.get(date_col))
            tk = (row.get(tick_col) or "").strip().upper()
            if d is None or not tk:
                continue
            snapshots.setdefault(d, set()).add(tk)
        return cls(snapshots)

    @classmethod
    def from_path(cls, path) -> "PitUniverse":
        with open(path, "r", newline="") as f:
            return cls.from_csv_text(f.read())

    def members_asof(self, target) -> "set[str]":
        """Tickers in the latest snapshot at-or-before ``target`` (carry-forward).

        Returns an empty set if ``target`` precedes the earliest snapshot — with no
        PIT evidence the name was tradeable, it is (conservatively) untradeable.
        """
        t = _to_date(target)
        if t is None or not self._dates:
            return set()
        latest: "date | None" = None
        for d in self._dates:
            if d <= t:
                latest = d
            else:
                break
        if latest is None:
            return set()
        return set(self._snapshots[latest])

    @property
    def snapshot_dates(self) -> "list[date]":
        return list(self._dates)

    def __len__(self) -> int:
        return len(self._dates)


def select_pit_top_n(
    target_date,
    inputs: "Mapping[str, object]",
    picker,
    universe: PitUniverse,
    n: int = 20,
) -> "list[str]":
    """PIT-gated top-N: rank ONLY names that were members as of ``target_date``.

    Survivorship-free by construction — a ticker absent from
    ``universe.members_asof(target_date)`` is dropped from the candidate pool BEFORE
    ranking, so it can never appear in the result regardless of how strong its
    (forward-known) momentum looks. ``picker`` is any object exposing
    ``top_n(inputs, n) -> list[str]`` (duck-typed; no hard dependency on StockPicker).
    """
    eligible = universe.members_asof(target_date)
    if not eligible:
        return []
    filtered = {tk: ti for tk, ti in inputs.items() if str(tk).strip().upper() in eligible}
    if not filtered:
        return []
    return picker.top_n(filtered, n)


def pit_filter(target_date, tickers: "Iterable[str]", universe: PitUniverse) -> "list[str]":
    """Lightweight gate for the main.py entry path: keep only PIT-member tickers.

    Use when a strategy already holds candidate tickers and just needs the
    survivorship gate (no re-ranking). Preserves input order.
    """
    eligible = universe.members_asof(target_date)
    return [t for t in tickers if str(t).strip().upper() in eligible]
