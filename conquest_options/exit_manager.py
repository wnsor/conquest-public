"""Per-position exit-rule tracking.

When the main algorithm opens a position from a StrategySignal, it calls
register(symbol, signal). On each OnData tick, the algorithm calls
positions_to_close(slice) which returns the symbols whose exit conditions
are met, with the reason.

Exit conditions evaluated each tick (in order):
  1. take_profit: PnL% >= signal.take_profit_pct
  2. stop_loss: PnL% <= signal.stop_loss_pct
  3. time_stop: DTE drops below signal.time_stop_dte
  4. expiry: contract expires today (defensive — Lean handles this too)

Strategies can also force a signal_exit by passing the symbol to
force_close(symbol, reason).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date

from strategies.base import ExitReason, StrategySignal


@dataclass
class TrackedPosition:
    symbol_str: str
    strategy_id: str
    entry_time: datetime
    expiry: date
    entry_premium_per_share: float       # per-share mid price at entry
    contracts: int
    side_is_long: bool                   # True for long-only positions (always True in Conquest)
    take_profit_pct: float | None
    stop_loss_pct: float | None
    time_stop_dte: int | None
    max_hold_days: int | None
    edge_score: float
    # v25 (2026-05-25): QC Symbol object for direct securities[sym] lookup
    # during _process_exits. Required because the option_chain filter drops
    # contracts once they go deep OTM or DTE ≤ 14 — the previous
    # chain-iteration approach for price discovery left dropped contracts
    # invisible to SL evaluation. Defaults to None for backward-compat with
    # any caller that doesn't pass it.
    symbol: object | None = None
    # v28 (v15d): trailing-SL state. peak_pnl_seen tracks the highest pnl_pct
    # this position has reached. Once peak exceeds tiers (0.5, 1.0, 2.0, 4.0),
    # the effective SL ratchets up to lock in gains. See trailing_effective_sl().
    peak_pnl_seen: float = 0.0

    def pnl_pct(self, current_premium_per_share: float) -> float:
        """Return PnL as fraction of premium paid. 1.0 = +100%, -0.5 = -50%."""
        if self.entry_premium_per_share <= 0:
            return 0.0
        return (current_premium_per_share - self.entry_premium_per_share) / self.entry_premium_per_share

    def dte_remaining(self, now: date) -> int:
        return (self.expiry - now).days

    def trailing_effective_sl(self) -> float | None:
        """v28 (v15d): compute effective stop_loss given peak_pnl_seen.
        Ratchets the SL up as gains accumulate to lock in profit and let
        winners run uncapped. Returns None if stop_loss_pct is None.

        Ladder:
          peak >= +400% (4.0) → SL = +200% (2.0)  (lock in 200%)
          peak >= +200% (2.0) → SL = +100% (1.0)  (lock in 100%)
          peak >= +100% (1.0) → SL = +50%  (0.5)  (lock in 50%)
          peak >= +50%  (0.5) → SL = 0.0          (breakeven — no loss possible)
          peak <  +50%        → original stop_loss_pct (e.g., -0.4)
        """
        if self.stop_loss_pct is None:
            return None
        if self.peak_pnl_seen >= 4.0:
            return 2.0
        if self.peak_pnl_seen >= 2.0:
            return 1.0
        if self.peak_pnl_seen >= 1.0:
            return 0.5
        if self.peak_pnl_seen >= 0.5:
            return 0.0
        return self.stop_loss_pct


def compute_current_prices(positions, securities_lookup, portfolio_lookup=None):
    """v27 (v15b): prefer portfolio.holdings.price as PRIMARY source for held
    options — securities.price was returning populated-but-stale values for
    deep-OTM contracts (v15a result: 5393/5393 priced, 0 SL fires, E -32%
    confirms pnl never triggered SL math). portfolio.holdings.price is
    explicit mark-to-market of HELD positions.

    Args:
        positions: iterable of TrackedPosition
        securities_lookup: callable mapping Symbol → security-like object
        portfolio_lookup: callable mapping Symbol → holding-like object with
            .holdings.price (mark-to-market) and .invested (bool)

    Returns: (current_prices, current_deltas, diag).
    """
    current_prices: dict[str, float] = {}
    current_deltas: dict[str, float] = {}
    diag: dict[str, int] = {
        'n_tracked': 0,
        'n_legacy_skip': 0,
        'n_sec_returns_none': 0,
        'n_sec_no_data': 0,
        'n_sec_zero_price': 0,
        'n_sec_priced': 0,
        'n_port_priced': 0,
        'n_port_zero_or_uninvested': 0,
        'n_delta': 0,
        'n_diverged_sec_vs_port': 0,    # both populated but >5% apart
    }
    for pos in positions:
        diag['n_tracked'] += 1
        if pos.symbol is None:
            diag['n_legacy_skip'] += 1
            continue

        # v15b: PORTFOLIO FIRST (mark-to-market of actual holding)
        port_price = None
        if portfolio_lookup is not None:
            holding = portfolio_lookup(pos.symbol)
            if holding is not None and getattr(holding, "invested", False):
                p2 = float(getattr(holding, "price", 0) or 0)
                if p2 > 0:
                    port_price = p2
                    current_prices[pos.symbol_str] = p2
                    diag['n_port_priced'] += 1
                else:
                    diag['n_port_zero_or_uninvested'] += 1
            else:
                diag['n_port_zero_or_uninvested'] += 1

        # SECURITIES as backup + always inspect for diag + greeks
        sec = securities_lookup(pos.symbol)
        if sec is None:
            diag['n_sec_returns_none'] += 1
        elif not getattr(sec, "has_data", False):
            diag['n_sec_no_data'] += 1
        else:
            sec_price = float(getattr(sec, "price", 0) or 0)
            if sec_price > 0:
                diag['n_sec_priced'] += 1
                # Use as fallback if portfolio didn't populate
                if pos.symbol_str not in current_prices:
                    current_prices[pos.symbol_str] = sec_price
                # Divergence check: portfolio vs securities
                elif port_price is not None and abs(sec_price - port_price) / port_price > 0.05:
                    diag['n_diverged_sec_vs_port'] += 1
            else:
                diag['n_sec_zero_price'] += 1
            greeks = getattr(sec, "greeks", None)
            if greeks is not None:
                d_val = getattr(greeks, "delta", None)
                if d_val is not None:
                    current_deltas[pos.symbol_str] = abs(float(d_val))
                    diag['n_delta'] += 1
    return current_prices, current_deltas, diag


class ExitManager:
    def __init__(self):
        self._positions: dict[str, TrackedPosition] = {}
        self._forced_exits: dict[str, ExitReason] = {}
        # v22 Fix 2: once TP/SL triggers, LOCK the exit so a subsequent
        # price reversal doesn't drop the exit attempt. Without this, a TP
        # spike that retraces within seconds (common on high-vol momentum
        # names) would un-trigger the exit and we'd hold past the target.
        self._triggered_exits: dict[str, ExitReason] = {}

    def register(self, symbol_str: str, signal: StrategySignal, *,
                 entry_time: datetime, expiry: date,
                 entry_premium_per_share: float, contracts: int,
                 symbol: object | None = None) -> None:
        """Register a new position. v25: now accepts optional Symbol object
        for the v13 exit-feed fix (allows direct securities[sym] lookup,
        bypassing the filter-truncated option chain)."""
        self._positions[symbol_str] = TrackedPosition(
            symbol_str=symbol_str,
            strategy_id=signal.strategy_id,
            entry_time=entry_time,
            expiry=expiry,
            entry_premium_per_share=entry_premium_per_share,
            contracts=contracts,
            side_is_long=True,
            take_profit_pct=signal.take_profit_pct,
            stop_loss_pct=signal.stop_loss_pct,
            time_stop_dte=signal.time_stop_dte,
            max_hold_days=signal.max_hold_days,
            edge_score=signal.edge_score,
            symbol=symbol,
        )

    def unregister(self, symbol_str: str) -> TrackedPosition | None:
        pos = self._positions.pop(symbol_str, None)
        self._forced_exits.pop(symbol_str, None)
        self._triggered_exits.pop(symbol_str, None)
        return pos

    def force_close(self, symbol_str: str, reason: ExitReason) -> None:
        if symbol_str in self._positions:
            self._forced_exits[symbol_str] = reason

    def is_tracked(self, symbol_str: str) -> bool:
        return symbol_str in self._positions

    def get(self, symbol_str: str) -> TrackedPosition | None:
        return self._positions.get(symbol_str)

    def positions_to_close(
        self,
        current_prices: dict[str, float],
        now: date,
    ) -> list[tuple[str, ExitReason]]:
        """Inspect every tracked position; return list of (symbol, reason) to close.

        v22 Fix 2: TP/SL triggers are LOCKED — once a position hits the target
        once, it stays in the close list on subsequent ticks until the actual
        exit order fills (and the caller unregisters the position). This
        prevents brief price reversals from de-triggering the exit.
        """
        to_close: list[tuple[str, ExitReason]] = []
        for sym, pos in self._positions.items():
            # v22 Fix 2: if already triggered, keep returning the close intent
            # until the position is unregistered (i.e. the exit fills).
            if sym in self._triggered_exits:
                to_close.append((sym, self._triggered_exits[sym]))
                continue
            if sym in self._forced_exits:
                to_close.append((sym, self._forced_exits[sym]))
                continue
            # Expiry check (defensive)
            if now >= pos.expiry:
                to_close.append((sym, "expiry"))
                continue
            # Max-hold (calendar-day exit, for pre-event strategies like C1/C2)
            if pos.max_hold_days is not None:
                held = (now - pos.entry_time.date()).days
                if held >= pos.max_hold_days:
                    self._triggered_exits[sym] = "time_stop"  # lock
                    to_close.append((sym, "time_stop"))
                    continue
            # DTE / time_stop (option-expiry-relative)
            if pos.time_stop_dte is not None and pos.dte_remaining(now) <= pos.time_stop_dte:
                self._triggered_exits[sym] = "time_stop"  # lock
                to_close.append((sym, "time_stop"))
                continue
            # Price-based exits — need current price
            cur = current_prices.get(sym)
            if cur is None or cur <= 0:
                continue
            pnl = pos.pnl_pct(cur)
            # v28 (v15d): update peak_pnl_seen FIRST (immutably swap pos).
            # TrackedPosition is mutable (dataclass without frozen=True), so
            # we just assign. peak_pnl_seen is used by trailing_effective_sl().
            if pnl > pos.peak_pnl_seen:
                pos.peak_pnl_seen = pnl
            # TP: only fires if take_profit_pct is explicitly set (None = no cap)
            if pos.take_profit_pct is not None and pnl >= pos.take_profit_pct:
                self._triggered_exits[sym] = "take_profit"  # lock
                to_close.append((sym, "take_profit"))
                continue
            # SL: use trailing_effective_sl which ratchets up as gains lock in.
            # peak <+50% → original SL (e.g., -0.4)
            # peak ≥+50% → SL = 0 (breakeven)
            # peak ≥+100% → SL = +50%
            # peak ≥+200% → SL = +100%
            # peak ≥+400% → SL = +200%
            eff_sl = pos.trailing_effective_sl()
            if eff_sl is not None and pnl <= eff_sl:
                self._triggered_exits[sym] = "stop_loss"  # lock
                to_close.append((sym, "stop_loss"))
                continue
        return to_close

    @property
    def n_open(self) -> int:
        return len(self._positions)

    def positions(self) -> list[TrackedPosition]:
        return list(self._positions.values())
