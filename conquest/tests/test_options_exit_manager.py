"""Unit tests for conquest_options.exit_manager."""
from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

from exit_manager import ExitManager, compute_current_prices
from strategies.base import StrategySignal


def _signal(**kw) -> StrategySignal:
    base = dict(
        strategy_id="t", underlying="SPY", side="call",
        target_dte=30, edge_score=0.8, target_delta=0.5,
    )
    base.update(kw)
    return StrategySignal(**base)


def _register(em, sig, *, sym="SPY_30D_500C", entry_premium=2.0, expiry_offset=30,
              symbol=None):
    em.register(
        sym, sig,
        entry_time=datetime(2026, 1, 1, 10, 0),
        expiry=date(2026, 1, 1).replace(month=2),
        entry_premium_per_share=entry_premium,
        contracts=10,
        symbol=symbol,
    )
    return sym


class TestTakeProfit:
    def test_tp_triggers(self):
        em = ExitManager()
        sym = _register(em, _signal(take_profit_pct=1.0))   # +100%
        # +120% PnL → TP fires
        closes = em.positions_to_close({sym: 4.4}, date(2026, 1, 15))
        assert (sym, "take_profit") in closes

    def test_tp_below_threshold_does_not_fire(self):
        em = ExitManager()
        sym = _register(em, _signal(take_profit_pct=1.0))
        # +50% PnL → no exit
        closes = em.positions_to_close({sym: 3.0}, date(2026, 1, 15))
        assert closes == []


class TestStopLoss:
    def test_sl_triggers(self):
        em = ExitManager()
        sym = _register(em, _signal(stop_loss_pct=-0.5))
        # -60% PnL → SL fires
        closes = em.positions_to_close({sym: 0.8}, date(2026, 1, 15))
        assert (sym, "stop_loss") in closes

    def test_sl_above_threshold_does_not_fire(self):
        em = ExitManager()
        sym = _register(em, _signal(stop_loss_pct=-0.5))
        closes = em.positions_to_close({sym: 1.5}, date(2026, 1, 15))
        assert closes == []


class TestTimeStop:
    def test_time_stop_triggers(self):
        em = ExitManager()
        sym = _register(em, _signal(time_stop_dte=5))
        # 4 DTE remaining
        closes = em.positions_to_close({sym: 2.0}, date(2026, 1, 28))
        assert (sym, "time_stop") in closes


class TestExpiry:
    def test_expiry_triggers(self):
        em = ExitManager()
        sym = _register(em, _signal())
        # Past expiry
        closes = em.positions_to_close({sym: 0.1}, date(2026, 3, 1))
        assert (sym, "expiry") in closes


class TestForceClose:
    def test_force_close_overrides(self):
        em = ExitManager()
        sym = _register(em, _signal())
        em.force_close(sym, "signal_exit")
        closes = em.positions_to_close({sym: 2.1}, date(2026, 1, 10))
        assert (sym, "signal_exit") in closes


class TestNoPriceData:
    def test_missing_price_does_not_crash(self):
        em = ExitManager()
        sym = _register(em, _signal(take_profit_pct=1.0, stop_loss_pct=-0.5))
        # No price → no exit (defer)
        closes = em.positions_to_close({}, date(2026, 1, 10))
        assert closes == []

    def test_zero_price_does_not_crash(self):
        em = ExitManager()
        sym = _register(em, _signal(stop_loss_pct=-0.5))
        closes = em.positions_to_close({sym: 0.0}, date(2026, 1, 10))
        assert closes == []


class TestUnregister:
    def test_unregister_removes_position(self):
        em = ExitManager()
        sym = _register(em, _signal())
        assert em.is_tracked(sym)
        em.unregister(sym)
        assert not em.is_tracked(sym)
        assert em.n_open == 0


# ---------------------------------------------------------------------------
# v25 (v13 fix): compute_current_prices uses Securities directly, not chain.
# ---------------------------------------------------------------------------

class TestComputeCurrentPricesV13Fix:
    """v13 fix tests: validate exit-feed price discovery via Securities collection
    rather than option_chain. This is the root-cause fix for the v11x_exitlog
    finding (7/8 trades exited 'manual' because chain dropped deep-OTM contracts
    so SL price check was skipped, position rode to expiration)."""

    def test_securities_lookup_populates_prices_for_tracked_position(self):
        """Simulates: contract dropped from chain (deep OTM), but Securities
        still has price data. The fix should populate current_prices via the
        Securities lookup."""
        em = ExitManager()
        # Use a sentinel object as the Symbol stand-in (only need identity)
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4),
                            entry_premium=1.0, symbol=mock_symbol)

        # Mock Securities lookup: chain has dropped this contract, but
        # securities[sym] still returns a (decayed) price.
        fake_sec = SimpleNamespace(
            has_data=True,
            price=0.30,                                      # -70% from $1.00 entry
            greeks=SimpleNamespace(delta=0.04),              # near-worthless
        )
        lookup = lambda s: fake_sec if s is mock_symbol else None

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices[sym_str] == 0.30
        assert deltas[sym_str] == 0.04

    def test_legacy_position_without_symbol_is_skipped(self):
        """Backward compat: positions registered before v25 have symbol=None
        and should be silently skipped (no crash). They fall back to whatever
        external code populated current_prices for them."""
        em = ExitManager()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4), symbol=None)

        # Lookup should never be called for None-symbol positions
        calls = []
        lookup = lambda s: calls.append(s) or None

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices == {}
        assert deltas == {}
        assert calls == []  # lookup not invoked

    def test_unsubscribed_symbol_returns_none_skipped(self):
        """If Securities.get(sym) returns None (subscription gone), skip."""
        em = ExitManager()
        mock_symbol = object()
        _register(em, _signal(stop_loss_pct=-0.4), symbol=mock_symbol)

        lookup = lambda s: None  # symbol not in securities collection

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices == {}
        assert deltas == {}

    def test_security_without_has_data_is_skipped(self):
        """If sec.has_data is False (no quote arrived yet), skip — don't
        populate stale or zero price."""
        em = ExitManager()
        mock_symbol = object()
        _register(em, _signal(stop_loss_pct=-0.4), symbol=mock_symbol)

        fake_sec = SimpleNamespace(has_data=False, price=0.30, greeks=None)
        lookup = lambda s: fake_sec

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices == {}
        assert deltas == {}

    def test_zero_price_does_not_populate_prices_but_still_captures_delta(self):
        """Defensive: if has_data is True but price is 0 (dead options
        sometimes), treat price as missing but still capture delta — the
        delta-died check (delta < 0.05) is independent of price availability."""
        em = ExitManager()
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4), symbol=mock_symbol)

        fake_sec = SimpleNamespace(has_data=True, price=0.0,
                                    greeks=SimpleNamespace(delta=0.02))
        lookup = lambda s: fake_sec

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices == {}                # zero price → not populated
        assert deltas[sym_str] == 0.02     # delta still captured

    def test_missing_greeks_does_not_crash(self):
        """If security has price but no greeks attribute, populate prices
        only."""
        em = ExitManager()
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4), symbol=mock_symbol)

        fake_sec = SimpleNamespace(has_data=True, price=0.50)  # no .greeks
        lookup = lambda s: fake_sec

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices[sym_str] == 0.50
        assert deltas == {}

    def test_negative_delta_absolute_valued(self):
        """For puts, delta is negative; we store absolute value for the
        delta-died threshold (delta < 0.05 means |delta| < 0.05)."""
        em = ExitManager()
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4, side="put"),
                            symbol=mock_symbol)

        fake_sec = SimpleNamespace(has_data=True, price=0.10,
                                    greeks=SimpleNamespace(delta=-0.03))
        lookup = lambda s: fake_sec

        prices, deltas, _diag = compute_current_prices(em.positions(), lookup)
        assert prices[sym_str] == 0.10
        assert deltas[sym_str] == 0.03  # absolute value

    def test_end_to_end_sl_fires_via_securities_lookup(self):
        """Integration: securities lookup → current_prices → positions_to_close
        → SL fires. This is the path v12 expected to work but didn't because
        it used chain iteration instead. v13 uses securities lookup → SL
        actually fires."""
        em = ExitManager()
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4),
                            entry_premium=1.0, symbol=mock_symbol)

        # Decayed deep-OTM price: -70% from entry
        fake_sec = SimpleNamespace(has_data=True, price=0.30, greeks=None)
        lookup = lambda s: fake_sec

        prices, _, _ = compute_current_prices(em.positions(), lookup)
        closes = em.positions_to_close(prices, date(2026, 1, 10))
        assert (sym_str, "stop_loss") in closes


# ---------------------------------------------------------------------------
# v26 (v14 diag): portfolio fallback + diagnostic counters
# ---------------------------------------------------------------------------

class TestComputeCurrentPricesV14Diag:
    """Diagnostic counter tests + portfolio fallback for v14."""

    def test_diag_counts_securities_priced(self):
        em = ExitManager()
        mock_symbol = object()
        _register(em, _signal(stop_loss_pct=-0.4), symbol=mock_symbol)
        fake_sec = SimpleNamespace(has_data=True, price=0.30, greeks=None)
        prices, _, diag = compute_current_prices(em.positions(),
                                                  lambda s: fake_sec)
        assert diag['n_tracked'] == 1
        assert diag['n_sec_priced'] == 1
        assert diag['n_sec_returns_none'] == 0
        assert diag['n_port_priced'] == 0

    def test_diag_counts_securities_none(self):
        em = ExitManager()
        _register(em, _signal(stop_loss_pct=-0.4), symbol=object())
        prices, _, diag = compute_current_prices(em.positions(), lambda s: None)
        assert diag['n_sec_returns_none'] == 1
        assert diag['n_sec_priced'] == 0

    def test_diag_counts_no_data(self):
        em = ExitManager()
        _register(em, _signal(stop_loss_pct=-0.4), symbol=object())
        fake_sec = SimpleNamespace(has_data=False, price=0.30, greeks=None)
        prices, _, diag = compute_current_prices(em.positions(),
                                                  lambda s: fake_sec)
        assert diag['n_sec_no_data'] == 1
        assert diag['n_sec_priced'] == 0

    def test_portfolio_fallback_used_when_securities_empty(self):
        """If securities returns None (subscription dropped), portfolio
        fallback should populate price."""
        em = ExitManager()
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4),
                            entry_premium=1.0, symbol=mock_symbol)
        sec_lookup = lambda s: None
        # Portfolio holding has mark-to-market price
        fake_holding = SimpleNamespace(invested=True, price=0.25)
        port_lookup = lambda s: fake_holding

        prices, _, diag = compute_current_prices(em.positions(), sec_lookup,
                                                  portfolio_lookup=port_lookup)
        assert prices[sym_str] == 0.25
        assert diag['n_port_priced'] == 1
        assert diag['n_sec_returns_none'] == 1   # securities failed

    def test_portfolio_fallback_skipped_when_not_invested(self):
        em = ExitManager()
        _register(em, _signal(stop_loss_pct=-0.4), symbol=object())
        fake_holding = SimpleNamespace(invested=False, price=0.25)
        prices, _, diag = compute_current_prices(em.positions(), lambda s: None,
                                                  portfolio_lookup=lambda s: fake_holding)
        assert prices == {}
        assert diag['n_port_priced'] == 0

    def test_trailing_sl_ladder_v15d(self):
        """v15d (v28): trailing SL ratchets up as peak_pnl_seen accumulates."""
        em = ExitManager()
        sym = _register(em, _signal(stop_loss_pct=-0.4),
                        entry_premium=1.0, symbol=object())
        pos = em.get(sym)
        # Without gains: original SL
        assert pos.trailing_effective_sl() == -0.4
        # After +50% peak: SL ratchets to breakeven
        pos.peak_pnl_seen = 0.6
        assert pos.trailing_effective_sl() == 0.0
        # After +100% peak: SL ratchets to +50%
        pos.peak_pnl_seen = 1.2
        assert pos.trailing_effective_sl() == 0.5
        # After +200% peak: SL ratchets to +100%
        pos.peak_pnl_seen = 2.5
        assert pos.trailing_effective_sl() == 1.0
        # After +400% peak: SL ratchets to +200%
        pos.peak_pnl_seen = 4.5
        assert pos.trailing_effective_sl() == 2.0

    def test_trailing_sl_fires_after_winner_reverses(self):
        """Position gains +100%, then drops back to +30% → trailing SL fires
        at +50% (the locked-in level)."""
        em = ExitManager()
        sym = _register(em, _signal(stop_loss_pct=-0.4),
                        entry_premium=1.0, symbol=object())
        # Tick 1: price = 2.50 → pnl = +150% (peak set to +150% → SL = +50%)
        em.positions_to_close({sym: 2.50}, date(2026, 1, 5))
        # Tick 2: price drops to 1.40 → pnl = +40% → trailing SL @ +50% NOT yet hit
        closes = em.positions_to_close({sym: 1.40}, date(2026, 1, 10))
        assert (sym, "stop_loss") in closes  # +40% < +50% trailing SL → fires

    def test_no_tp_when_take_profit_is_none(self):
        """v15d strategies set take_profit_pct=None → no fixed TP cap. Only
        trailing SL controls exits."""
        em = ExitManager()
        sym = _register(em, _signal(take_profit_pct=None, stop_loss_pct=-0.4),
                        entry_premium=1.0, symbol=object())
        # +500% gain — no TP fires
        closes = em.positions_to_close({sym: 6.0}, date(2026, 1, 10))
        # No TP exit logged
        assert (sym, "take_profit") not in closes
        # Peak should now be +500%, trailing SL = +200%
        pos = em.get(sym)
        assert pos.peak_pnl_seen == 5.0
        assert pos.trailing_effective_sl() == 2.0

    def test_portfolio_takes_priority_over_securities_in_v15b(self):
        """v15b: portfolio.holdings.price is PRIMARY (not fallback).
        Securities is backup + greeks source. v15a showed securities
        returning stale values for deep-OTM held contracts."""
        em = ExitManager()
        mock_symbol = object()
        sym_str = _register(em, _signal(stop_loss_pct=-0.4),
                            entry_premium=1.0, symbol=mock_symbol)
        fake_sec = SimpleNamespace(has_data=True, price=0.30, greeks=None)
        fake_holding = SimpleNamespace(invested=True, price=0.20)  # different
        prices, _, diag = compute_current_prices(em.positions(),
                                                  lambda s: fake_sec,
                                                  portfolio_lookup=lambda s: fake_holding)
        assert prices[sym_str] == 0.20  # portfolio value (mark-to-market)
        assert diag['n_port_priced'] == 1
        assert diag['n_sec_priced'] == 1   # securities also counted (for diag)
        assert diag['n_diverged_sec_vs_port'] == 1  # 0.30 vs 0.20 = 50% diverge
