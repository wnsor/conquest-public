"""Phase 2.1 tests — new edge signals (GEX, term_structure, skew) +
new strategies (A5b midcap, A6 insider) + confluence updates to A1/A2/A8.
"""
from __future__ import annotations

from datetime import datetime, date, timedelta
from types import SimpleNamespace

from edge_signals.gex import (
    compute_gex_contributions, classify_gex_regime,
)
from edge_signals.term_structure import (
    compute_term_ratio, classify_term_regime, is_acute_stress,
)
from edge_signals.skew import (
    compute_skew, SkewTracker,
)
from edge_signals.insider_lookup import InsiderForm4Calendar
from strategies.base import StrategyContext
from strategies.momentum_otm_calls import MomentumOtmCalls, WSB_UNIVERSE
from strategies.pead_midcap import PeadMidcap, MIDCAP_UNIVERSE
from strategies.insider_buy_calls import InsiderBuyCalls, INSIDER_UNIVERSE


# ---------------------------------------------------------------------------
# GEX
# ---------------------------------------------------------------------------

def _fake_contract(strike, right, oi, gamma):
    g = SimpleNamespace(Gamma=gamma)
    return SimpleNamespace(
        Strike=strike, Right=right, OpenInterest=oi, Greeks=g,
    )


class TestGEX:
    def test_basic_long_gamma(self):
        # All calls, dealers short calls → positive GEX
        chain = [_fake_contract(500, 0, 10_000, 0.02)]
        r = compute_gex_contributions(chain, spot=500)
        assert r["gex_total"] > 0
        assert r["count_used"] == 1

    def test_all_puts_negative_gex(self):
        # All puts, dealers long puts → negative GEX
        chain = [_fake_contract(500, 1, 10_000, 0.02)]
        r = compute_gex_contributions(chain, spot=500)
        assert r["gex_total"] < 0

    def test_balanced_neutral(self):
        chain = [
            _fake_contract(500, 0, 1000, 0.01),
            _fake_contract(500, 1, 1000, 0.01),
        ]
        r = compute_gex_contributions(chain, spot=500)
        assert abs(r["gex_total"]) < 1e-9

    def test_zero_oi_skipped(self):
        chain = [_fake_contract(500, 0, 0, 0.02)]
        r = compute_gex_contributions(chain, spot=500)
        assert r["count_used"] == 0

    def test_classify_regime(self):
        assert classify_gex_regime(2.0) == "long_gamma"
        assert classify_gex_regime(-2.0) == "short_gamma"
        assert classify_gex_regime(0.1) == "flip_zone"


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------

class TestTermStructure:
    def test_contango(self):
        assert classify_term_regime(15, 18) == "contango"
        assert compute_term_ratio(15, 18) < 1.0

    def test_backwardation(self):
        assert classify_term_regime(30, 22) == "backwardation"
        assert compute_term_ratio(30, 22) > 1.0

    def test_flat_band(self):
        # Default flat band 2%
        assert classify_term_regime(20, 20.1) == "flat"

    def test_unknown(self):
        assert classify_term_regime(None, 20) == "unknown"
        assert classify_term_regime(20, None) == "unknown"
        assert compute_term_ratio(None, 20) is None

    def test_acute_stress(self):
        # VIX9D > VIX = panic-now
        assert is_acute_stress(vix9d=35, vix=30) is True
        assert is_acute_stress(vix9d=15, vix=20) is False
        assert is_acute_stress(None, 20) is False


# ---------------------------------------------------------------------------
# Skew
# ---------------------------------------------------------------------------

def _fake_iv_contract(strike, right, delta, iv, expiry_offset_days=14):
    g = SimpleNamespace(Delta=delta, ImpliedVolatility=iv)
    expiry = date.today() + timedelta(days=expiry_offset_days)
    return SimpleNamespace(
        Strike=strike, Right=right, Expiry=expiry, Greeks=g,
    )


class TestSkew:
    def test_compute_basic(self):
        # 25Δ put IV=0.30, 25Δ call IV=0.25 → skew = +0.05
        chain = [
            _fake_iv_contract(95, 1, -0.25, 0.30),   # put
            _fake_iv_contract(105, 0, 0.25, 0.25),   # call
        ]
        s = compute_skew(chain, now_date=date.today())
        assert abs(s - 0.05) < 1e-9

    def test_missing_leg_returns_none(self):
        chain = [_fake_iv_contract(95, 1, -0.25, 0.30)]
        assert compute_skew(chain, now_date=date.today()) is None

    def test_tracker_z_score(self):
        t = SkewTracker(lookback_days=10)
        for v in [0.01, 0.02, 0.03, 0.02, 0.01, 0.015, 0.02, 0.018, 0.022, 0.019, 0.05]:
            t.update("AAPL", v)
        # Last update was 0.05 — well above the mean
        z = t.z_score("AAPL")
        assert z is not None
        assert z > 1.0

    def test_tracker_needs_warmup(self):
        t = SkewTracker(lookback_days=252)
        for _ in range(3):
            t.update("AAPL", 0.02)
        assert t.z_score("AAPL") is None  # needs >= 5 samples


# ---------------------------------------------------------------------------
# Insider lookup
# ---------------------------------------------------------------------------

class TestInsiderCalendar:
    def test_buys_within_window(self):
        csv_text = (
            "filing_date,transaction_date,ticker,insider_cik,insider_name,role,shares,price,dollar_value\n"
            "2026-01-15,2026-01-14,AAPL,123,Jane Doe,Officer,1000,150.0,150000\n"
            "2026-01-10,2026-01-08,AAPL,124,Bob,Director,500,148.0,74000\n"
            "2026-01-01,2026-01-01,AAPL,125,Tiny,Officer,10,150.0,1500\n"
        )
        cal = InsiderForm4Calendar.from_csv_text(csv_text)
        buys = cal.buys_within_n_days("AAPL", date(2026, 1, 15), n=10)
        # The $1.5k buy is below min_dollar=$25k default → filtered
        assert len(buys) == 2

    def test_filters_pure_10pct_owner(self):
        csv_text = (
            "filing_date,transaction_date,ticker,insider_cik,insider_name,role,shares,price,dollar_value\n"
            "2026-01-15,2026-01-14,AAPL,123,Fund,10pct,10000,150.0,1500000\n"
        )
        cal = InsiderForm4Calendar.from_csv_text(csv_text)
        buys = cal.buys_within_n_days("AAPL", date(2026, 1, 15), n=10)
        assert len(buys) == 0  # 10pct-only filtered

    def test_outside_window(self):
        csv_text = (
            "filing_date,transaction_date,ticker,insider_cik,insider_name,role,shares,price,dollar_value\n"
            "2026-01-01,2026-01-01,AAPL,123,Jane,Officer,1000,150.0,150000\n"
        )
        cal = InsiderForm4Calendar.from_csv_text(csv_text)
        # n=5 day window from 2026-01-15 → cutoff 2026-01-10 → 2026-01-01 outside
        buys = cal.buys_within_n_days("AAPL", date(2026, 1, 15), n=5)
        assert len(buys) == 0


# ---------------------------------------------------------------------------
# Strategy: A1 term-backwardation gate
# ---------------------------------------------------------------------------

def _ctx(**overrides) -> StrategyContext:
    base = dict(
        timestamp=datetime(2026, 1, 15, 10, 0),
        underlying_prices={},
        vix=20.0,
        cstability_vote_count=0,
        iv_rank={},
        earnings_today=set(),
        earnings_within_5d=set(),
        last_earnings_surprise_pct={},
        days_since_last_earnings={},
        underlying_momentum_30d={},
        underlying_momentum_60d={},
        uoa_active=set(),
    )
    base.update(overrides)
    return StrategyContext(**base)


class TestA1TermStructureGate:
    def test_backwardation_blocks(self):
        s = MomentumOtmCalls()
        ctx = _ctx(
            vix=20.0,
            term_regime="backwardation",
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: 1.10 for t in WSB_UNIVERSE},
        )
        assert s.on_data(ctx) == []

    def test_acute_stress_blocks(self):
        s = MomentumOtmCalls()
        ctx = _ctx(
            vix=20.0,
            vix9d_vix_ratio=1.10,   # VIX9D > VIX → panic
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: 1.10 for t in WSB_UNIVERSE},
        )
        assert s.on_data(ctx) == []

    def test_contango_passes(self):
        s = MomentumOtmCalls()
        ctx = _ctx(
            vix=20.0,
            term_regime="contango",
            iv_rank={t: 20.0 for t in WSB_UNIVERSE},
            underlying_momentum_30d={t: 1.10 for t in WSB_UNIVERSE},
        )
        assert len(s.on_data(ctx)) == len(WSB_UNIVERSE)


# ---------------------------------------------------------------------------
# Strategy: A5b midcap PEAD
# ---------------------------------------------------------------------------

class TestA5bPeadMidcap:
    def test_fires_on_midcap_universe_only(self):
        s = PeadMidcap()
        # Earnings hit for both midcap (RH) and megacap (AAPL); only RH fires
        ctx = _ctx(
            days_since_last_earnings={"RH": 2, "AAPL": 2},
            last_earnings_surprise_pct={"RH": 12.0, "AAPL": 8.0},
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == 1
        assert sigs[0].underlying == "RH"
        assert sigs[0].strategy_id == "pead_midcap"


# ---------------------------------------------------------------------------
# Strategy: A6 insider Form 4
# ---------------------------------------------------------------------------

class TestA6InsiderCall:
    def test_fires_on_recent_buy(self):
        s = InsiderBuyCalls()
        ctx = _ctx(
            insider_recent_buys={"AAPL": 250_000.0},
            vix=20.0,
            term_regime="contango",
        )
        sigs = s.on_data(ctx)
        assert len(sigs) == 1
        assert sigs[0].underlying == "AAPL"
        assert sigs[0].target_dte == 45
        assert sigs[0].target_otm_pct == 0.07
        assert sigs[0].take_profit_pct == 1.5

    def test_no_buy_no_signal(self):
        s = InsiderBuyCalls()
        ctx = _ctx(insider_recent_buys={})
        assert s.on_data(ctx) == []

    def test_per_ticker_cooldown(self):
        s = InsiderBuyCalls()
        ctx_args = dict(insider_recent_buys={"NVDA": 100_000.0})
        d1 = _ctx(timestamp=datetime(2026, 1, 1), **ctx_args)
        d15 = _ctx(timestamp=datetime(2026, 1, 15), **ctx_args)
        d40 = _ctx(timestamp=datetime(2026, 2, 10), **ctx_args)
        assert len(s.on_data(d1)) == 1
        assert s.on_data(d15) == []   # within 30-day cooldown
        assert len(s.on_data(d40)) == 1

    def test_dollar_size_scales_edge(self):
        s_small = InsiderBuyCalls()
        small_ctx = _ctx(insider_recent_buys={"AAPL": 30_000.0})  # near $25k floor
        sigs_small = s_small.on_data(small_ctx)
        s_big = InsiderBuyCalls()
        big_ctx = _ctx(insider_recent_buys={"AAPL": 5_000_000.0})  # large
        sigs_big = s_big.on_data(big_ctx)
        assert sigs_small[0].edge_score < sigs_big[0].edge_score

    def test_not_in_universe_skipped(self):
        s = InsiderBuyCalls()
        # SHOP isn't in INSIDER_UNIVERSE
        ctx = _ctx(insider_recent_buys={"SHOP": 200_000.0})
        assert s.on_data(ctx) == []
