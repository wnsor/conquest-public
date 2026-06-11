"""Regression tests for the once-per-day sampling invariant in
conquest_options/main.py.

CONTEXT — the bug this guards against (2026-05-24):
  At MINUTE resolution, `on_data` fires ~390 times per trading day. Five
  trackers used `lookback_days=252` / `maxlen=20` style deques that grew
  per-minute, so any "252-day percentile" or "20-day baseline" actually
  spanned <1 trading day. D2 Tepper never triggered on COVID V-bottom;
  CrisisDetector never reached "rebound"; A_GEX trend gate was random.

  Fix: in main.py, the once-per-day samplers (price_history, UOA history,
  P/C ratio, ATM IV sample, skew) are gated behind a per-day check —
  they run on the FIRST on_data tick of each calendar day where
  time.hour >= 15 (i.e. once per trading day, in the last hour, when
  intraday volume aggregates are mostly settled).

These tests don't instantiate the full QC algorithm (too heavy) — they
test the simpler invariants that, if violated, would re-introduce the
bug pattern in future strategies/trackers.
"""
from __future__ import annotations

from collections import deque
from datetime import date, datetime, timedelta


def _simulate_on_data_ticks(start_dt: datetime, n_days: int,
                              minutes_per_day: int = 390) -> list[datetime]:
    """Produce an `on_data` timestamp sequence for `n_days` trading days
    at minute resolution (390 ticks/day, 9:30am - 4:00pm)."""
    ticks = []
    for d in range(n_days):
        day = start_dt + timedelta(days=d)
        # Trading session 9:30 AM - 4:00 PM = 390 minutes
        for m in range(minutes_per_day):
            ticks.append(day.replace(hour=9, minute=30) + timedelta(minutes=m))
    return ticks


class _MockDailySampler:
    """Mirror of main.py's per-day gate logic. Tests that this pattern
    produces the right cadence regardless of input tick density."""

    def __init__(self):
        self._last_daily_pass: date | None = None
        self._price_history: deque[float] = deque(maxlen=260)

    def on_data(self, ts: datetime, price: float) -> None:
        today = ts.date()
        # PER-TICK: nothing in this mock.
        # ONCE-PER-DAY: first tick at hour >= 15 per calendar day.
        if self._last_daily_pass != today and ts.hour >= 15:
            self._last_daily_pass = today
            self._price_history.append(price)


def test_daily_sampler_appends_once_per_trading_day():
    """The price_history bug fix: at minute resolution, 390 on_data ticks
    in one day must yield ≤ 1 history append (not 390)."""
    sampler = _MockDailySampler()
    start = datetime(2024, 1, 2)  # arbitrary trading-Tuesday
    ticks = _simulate_on_data_ticks(start, n_days=1)
    assert len(ticks) == 390, "minute-bar session should be 390 ticks"

    for i, ts in enumerate(ticks):
        sampler.on_data(ts, price=100.0 + i * 0.01)

    # MUST be exactly 1: one day → one history sample.
    assert len(sampler._price_history) == 1, (
        f"per-day gate broken — got {len(sampler._price_history)} samples "
        f"in one trading day; expected 1"
    )


def test_daily_sampler_grows_by_one_per_day_across_week():
    """5 trading days → exactly 5 history entries (not 5×390 = 1950)."""
    sampler = _MockDailySampler()
    start = datetime(2024, 1, 2)
    ticks = _simulate_on_data_ticks(start, n_days=5)
    for i, ts in enumerate(ticks):
        sampler.on_data(ts, price=100.0 + i * 0.01)
    assert len(sampler._price_history) == 5, (
        f"5 trading days → got {len(sampler._price_history)} samples; expected 5"
    )


def test_daily_sampler_handles_pre_15h_ticks_without_growing():
    """Ticks before 15:00 must NOT advance the sampler — the gate is
    `hour >= 15`, and pre-15h calls are no-ops. Otherwise we'd snapshot
    09:30 chain (0 volume so far) instead of late-day settled aggregates."""
    sampler = _MockDailySampler()
    start = datetime(2024, 1, 2)
    # 300 ticks from 9:30 to 14:30 — none should trigger
    pre_ticks = [start.replace(hour=9, minute=30) + timedelta(minutes=m)
                 for m in range(300)]  # 9:30 → 14:30
    for ts in pre_ticks:
        sampler.on_data(ts, price=100.0)
    assert len(sampler._price_history) == 0, (
        "pre-15:00 ticks must not advance the daily sampler"
    )
    # Now one tick at 15:00 — should trigger
    sampler.on_data(start.replace(hour=15, minute=0), price=100.0)
    assert len(sampler._price_history) == 1


def test_daily_sampler_handles_252_day_window_correctly():
    """Over 1 calendar year of trading days (~252), price_history should
    grow to maxlen=260 but stop there — and represent 1 sample per day."""
    sampler = _MockDailySampler()
    start = datetime(2024, 1, 2)
    ticks = _simulate_on_data_ticks(start, n_days=300)  # 300 calendar days
    for i, ts in enumerate(ticks):
        sampler.on_data(ts, price=100.0 + i * 0.001)
    # Capped at maxlen=260
    assert len(sampler._price_history) == 260
    # 300 days × 1 sample/day = capped at 260 most-recent (oldest evicted)
    # Sanity: the buffer is dense with daily-sampled prices.


def test_buggy_pattern_would_have_failed_invariant():
    """Sanity test: an unguarded per-tick append (the original bug) would
    have produced 390 samples in 1 day, blowing past any maxlen=260. This
    test demonstrates the failure mode the fix prevents."""
    history: deque[float] = deque(maxlen=260)
    start = datetime(2024, 1, 2)
    ticks = _simulate_on_data_ticks(start, n_days=1)
    for i, ts in enumerate(ticks):
        history.append(100.0 + i * 0.01)  # ← unguarded: every tick
    # maxlen=260 capped it, but the FULL day filled it: 260 most-recent of 390.
    # Means by end of day 1, "252d high" actually = max of last 260 minutes.
    assert len(history) == 260
    # By day 2 the entire deque has been overwritten with day-2 minutes —
    # so no day-1 data survives. Indicators would be intraday-only.
