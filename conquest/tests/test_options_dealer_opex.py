"""Unit tests for v_DEALER_OPEX (dealer_opex_squeeze) — calendar-anchored
forced-flow trade around monthly OPEX (3rd Friday of month)."""
from __future__ import annotations

from datetime import date, datetime

from strategies.dealer_opex_squeeze import (
    DealerOpexSqueeze,
    _third_friday,
    _is_opex_window,
    _days_to_opex,
)
from strategies.momentum_otm_calls import WSB_UNIVERSE
from strategies.base import StrategyContext


class TestOpexCalendarHelpers:
    def test_third_friday_known_dates(self):
        # Sanity-check against known OPEX calendar
        assert _third_friday(2026, 1) == date(2026, 1, 16)
        assert _third_friday(2026, 2) == date(2026, 2, 20)
        assert _third_friday(2026, 3) == date(2026, 3, 20)
        # Edge case: month starts on Friday
        # Jan 2027: 1st is a Friday, so 3rd Friday is Jan 15
        assert _third_friday(2027, 1) == date(2027, 1, 15)

    def test_is_opex_window_inside(self):
        # OPEX Jan 16, 2026 (3rd Friday). Window: 7 days before to 1 day after
        # so Jan 9 .. Jan 17 should be inside.
        assert _is_opex_window(date(2026, 1, 9))
        assert _is_opex_window(date(2026, 1, 15))   # day before OPEX
        assert _is_opex_window(date(2026, 1, 16))   # OPEX itself
        assert _is_opex_window(date(2026, 1, 17))   # day after

    def test_is_opex_window_outside(self):
        # Mid-month before window (early in month)
        assert not _is_opex_window(date(2026, 1, 5))
        # Just after window
        assert not _is_opex_window(date(2026, 1, 19))

    def test_days_to_opex(self):
        # Jan 9 → Jan 16 = 7 days
        assert _days_to_opex(date(2026, 1, 9)) == 7
        # On OPEX day → 0
        assert _days_to_opex(date(2026, 1, 16)) == 0
        # Day after OPEX → next month's OPEX
        next_opex = _third_friday(2026, 2)
        assert _days_to_opex(date(2026, 1, 17)) == (next_opex - date(2026, 1, 17)).days


def _opex_window_ctx(ticker="MSTR", days_before_opex=5, **kw):
    """Build a context inside the OPEX window with all gates passing."""
    # Pick Jan 11 2026 (5 days before Jan 16 OPEX)
    target_day = _third_friday(2026, 1) - __import__("datetime").timedelta(days=days_before_opex)
    return StrategyContext(
        timestamp=datetime(target_day.year, target_day.month, target_day.day, 16, 0),
        vix=18.0,
        term_regime="contango",
        gex_regime="short_gamma",                    # dealers forced to buy
        uoa_active={ticker},                         # heavy OTM call OI proxy
        underlying_momentum_30d={ticker: 1.08},      # rally in progress
        **kw,
    )


class TestDealerOpexSqueeze:
    def test_fires_in_opex_window_with_all_gates(self):
        s = DealerOpexSqueeze()
        ctx = _opex_window_ctx("MSTR", days_before_opex=5)
        sigs = s.on_data(ctx)
        assert any(sig.underlying == "MSTR" for sig in sigs)
        mstr = next(sig for sig in sigs if sig.underlying == "MSTR")
        assert mstr.side == "call"
        assert mstr.target_otm_pct == 0.10
        assert mstr.stop_loss_pct == -0.40
        assert mstr.max_per_trade_pct_nav == 0.06
        assert "OPEX_d-5" in mstr.notes

    def test_no_fire_outside_opex_window(self):
        s = DealerOpexSqueeze()
        # Early in month, well before OPEX
        ctx = _opex_window_ctx("MSTR", days_before_opex=20)  # Dec 27 prior
        # The helper subtracts days; day=20 before Jan 16 = Dec 27 2025
        sigs = s.on_data(ctx)
        # Dec 27 is INSIDE Dec OPEX window (Dec OPEX = Dec 19; this is after)
        # so this case is actually outside Dec window AND outside Jan window
        # (Jan window starts Jan 9). Let me make this explicit.
        from datetime import datetime as dt
        outside = StrategyContext(
            timestamp=dt(2026, 1, 5, 16, 0),    # 11 days before Jan 16 OPEX
            vix=18.0, term_regime="contango",
            gex_regime="short_gamma",
            uoa_active={"MSTR"},
            underlying_momentum_30d={"MSTR": 1.08},
        )
        assert s.on_data(outside) == []

    def test_no_fire_when_gex_long_gamma(self):
        """Long_gamma means dealers SELL into rallies — no squeeze setup."""
        s = DealerOpexSqueeze()
        ctx = _opex_window_ctx("MSTR")
        # Manually override gex_regime
        from datetime import datetime as dt
        target = _third_friday(2026, 1) - __import__("datetime").timedelta(days=5)
        ctx = StrategyContext(
            timestamp=dt(target.year, target.month, target.day, 16, 0),
            vix=18.0, term_regime="contango",
            gex_regime="long_gamma",   # ← wrong regime
            uoa_active={"MSTR"},
            underlying_momentum_30d={"MSTR": 1.08},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_no_uoa(self):
        s = DealerOpexSqueeze()
        from datetime import datetime as dt
        target = _third_friday(2026, 1) - __import__("datetime").timedelta(days=5)
        ctx = StrategyContext(
            timestamp=dt(target.year, target.month, target.day, 16, 0),
            vix=18.0, term_regime="contango",
            gex_regime="short_gamma",
            uoa_active=set(),   # ← no UOA
            underlying_momentum_30d={"MSTR": 1.08},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_vix_high(self):
        s = DealerOpexSqueeze()
        from datetime import datetime as dt
        target = _third_friday(2026, 1) - __import__("datetime").timedelta(days=5)
        ctx = StrategyContext(
            timestamp=dt(target.year, target.month, target.day, 16, 0),
            vix=28.0,   # ← high VIX, no gamma squeeze
            term_regime="contango",
            gex_regime="short_gamma",
            uoa_active={"MSTR"},
            underlying_momentum_30d={"MSTR": 1.08},
        )
        assert s.on_data(ctx) == []

    def test_no_fire_when_momentum_flat(self):
        s = DealerOpexSqueeze()
        from datetime import datetime as dt
        target = _third_friday(2026, 1) - __import__("datetime").timedelta(days=5)
        ctx = StrategyContext(
            timestamp=dt(target.year, target.month, target.day, 16, 0),
            vix=18.0, term_regime="contango",
            gex_regime="short_gamma",
            uoa_active={"MSTR"},
            underlying_momentum_30d={"MSTR": 1.005},   # ← below 1.02 trigger
        )
        assert s.on_data(ctx) == []

    def test_target_dte_matches_days_to_opex(self):
        """Contract chosen so it expires at OPEX cycle close."""
        s = DealerOpexSqueeze()
        ctx = _opex_window_ctx("MSTR", days_before_opex=7)
        sig = next(s for s in s.on_data(ctx) if s.underlying == "MSTR")
        # 7d to OPEX → target_dte=8 (max with d_opex+1)
        assert sig.target_dte == 8

    def test_cooldown_blocks_re_entry(self):
        """Once per month — same name shouldn't fire twice in same OPEX cycle."""
        s = DealerOpexSqueeze()
        from datetime import datetime as dt
        opex1 = _third_friday(2026, 1)
        opex2 = _third_friday(2026, 2)
        # Trade #1 — 5 days before Jan OPEX
        d1 = opex1 - __import__("datetime").timedelta(days=5)
        # Trade #2 — same window 2 days later
        d2 = opex1 - __import__("datetime").timedelta(days=3)
        ctx1 = StrategyContext(
            timestamp=dt(d1.year, d1.month, d1.day, 16, 0),
            vix=18.0, term_regime="contango", gex_regime="short_gamma",
            uoa_active={"MSTR"}, underlying_momentum_30d={"MSTR": 1.08})
        ctx2 = StrategyContext(
            timestamp=dt(d2.year, d2.month, d2.day, 16, 0),
            vix=18.0, term_regime="contango", gex_regime="short_gamma",
            uoa_active={"MSTR"}, underlying_momentum_30d={"MSTR": 1.08})
        assert any(s.underlying == "MSTR" for s in s.on_data(ctx1))
        assert not any(s.underlying == "MSTR" for s in s.on_data(ctx2))
        # New OPEX cycle (~30d later) — fires again
        d3 = opex2 - __import__("datetime").timedelta(days=5)
        ctx3 = StrategyContext(
            timestamp=dt(d3.year, d3.month, d3.day, 16, 0),
            vix=18.0, term_regime="contango", gex_regime="short_gamma",
            uoa_active={"MSTR"}, underlying_momentum_30d={"MSTR": 1.08})
        assert any(s.underlying == "MSTR" for s in s.on_data(ctx3))

    def test_universe_matches_momentum_otm_calls(self):
        s = DealerOpexSqueeze()
        assert s.universe == WSB_UNIVERSE

    def test_no_signal_when_timestamp_missing(self):
        s = DealerOpexSqueeze()
        ctx = StrategyContext(timestamp=None)
        assert s.on_data(ctx) == []
