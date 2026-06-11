"""v_DEALER_OPEX — dealer forced-flow squeeze trade (2026-05-26).

Thesis (mechanical, not predictive)
===================================
Market makers run delta-neutral. When small/mid-cap names accumulate
heavy short-dated OTM call open interest, dealers are SHORT GAMMA —
they MUST buy underlying as price rises and sell as it falls. This
creates a *self-reinforcing rally* within the gamma window (until
expiry strips the OI). The pattern is well-known to professional flow
desks (GEX/dealer-positioning research) but rarely traded systematically
at retail because:

  (a) Most retail tools don't compute per-ticker gamma (just SPX/SPY)
  (b) The asymmetric window is short — 7 days before to 1 day after
      monthly OPEX (3rd Friday of month)
  (c) Setup requires specific OI structure most quants don't filter for

We trade the SAME side as the dealer's forced hedge (long underlying
proxy = long calls), with a calendar-anchored entry/exit. The edge is
PURELY mechanical — dealers can't stop hedging, and the calendar
doesn't move. No prediction required.

Entry condition (ALL must fire):
  (1) Today's date is within 3-7 trading days BEFORE monthly OPEX
      (3rd Friday of the calendar month)
  (2) Ticker has UOA active (proxy for heavy short-dated OTM call OI)
  (3) SPY GEX regime is "long_gamma" or "flip_zone"
      (dealers under-hedged, force-buy on rally is highest)
  (4) Underlying is up >2% over last 5 days (rally already started;
      dealer hedging is reactive, not anticipatory)
  (5) VIX < 25 (gamma squeezes require risk-on tape)

Exit: hard-coded calendar exit on monthly OPEX Friday close.
      Trailing SL ladder still active in case of pre-OPEX collapse.

Frequency: ~10-12 monthly OPEX events per year. Setup criteria reduce
to ~2-5 fires/yr per name. Across the WSB universe (16 names) that's
~30-60 fires/yr — enough for sample size.

Universe: same as momentum_otm_calls. Names that produce gamma squeezes
have: smaller float, narrative-driven retail interest, illiquid options
chain that amplifies dealer hedge impact.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE, WSB_TICKER_INCEPTION


def _third_friday(year: int, month: int) -> date:
    """Return the 3rd Friday of (year, month) — monthly OPEX date."""
    d = date(year, month, 1)
    # weekday(): Monday=0 ... Friday=4
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def _is_opex_window(today: date, days_before: int = 7, days_after: int = 1) -> bool:
    """True if today is within OPEX window of the current or next month."""
    # Check current month's 3rd Friday
    current_opex = _third_friday(today.year, today.month)
    if -days_before <= (today - current_opex).days <= days_after:
        return True
    # Edge case: late month, next month's OPEX is close (rare for 7d window)
    next_month = today.month % 12 + 1
    next_year = today.year + (1 if today.month == 12 else 0)
    next_opex = _third_friday(next_year, next_month)
    if -days_before <= (today - next_opex).days <= 0:
        return True
    return False


def _days_to_opex(today: date) -> int:
    """Return signed days from today to nearest monthly OPEX. Positive = future."""
    current_opex = _third_friday(today.year, today.month)
    if today <= current_opex:
        return (current_opex - today).days
    # Past this month's OPEX; next month
    next_month = today.month % 12 + 1
    next_year = today.year + (1 if today.month == 12 else 0)
    next_opex = _third_friday(next_year, next_month)
    return (next_opex - today).days


class DealerOpexSqueeze:
    id = "dealer_opex_squeeze"
    enabled = True
    universe = WSB_UNIVERSE

    DAYS_BEFORE_OPEX = 7        # window opens 7 trading days pre-OPEX
    DAYS_AFTER_OPEX = 1         # close window 1 day after OPEX
    MIN_5D_RETURN = 0.02        # ticker up 2%+ in last 5 days
    MAX_VIX = 25.0              # only in risk-on tape
    # v2 (2026-05-27): UOA gate produced only 1 fire / 8mo window. Relaxed
    # to OR-logic: UOA active OR strong 30d momentum (mom30 >= MOM_HIGH).
    # Thesis: heavy short-dated call OI (UOA) AND strong rally are BOTH
    # signs of dealer-hedging stress. Either alone is sufficient.
    MOM_HIGH = 1.08             # mom30 >= 8% = strong rally → dealer pressure

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        # OPEX fires ~12x/yr; cooldown to one fire per OPEX cycle per ticker
        self._cooldown_days = 25
        # Diagnostic counters — emitted as runtime stats so we can see WHICH gate
        # is rejecting the most days. Resets across BTs (no persistence).
        self._diag = {
            "vix_block": 0, "term_block": 0, "opex_window_miss": 0,
            "gex_block": 0, "uoa_miss": 0, "mom_miss": 0, "cooldown": 0,
            "fired": 0, "ticks_total": 0,
        }

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts
        self._diag["ticks_total"] += 1

        # Hard regime gates
        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            self._diag["vix_block"] += 1
            return []
        if ctx.term_regime == "backwardation":
            self._diag["term_block"] += 1
            return []

        # Calendar gate — OPEX window only
        if not _is_opex_window(today, self.DAYS_BEFORE_OPEX, self.DAYS_AFTER_OPEX):
            self._diag["opex_window_miss"] += 1
            return []
        d_opex = _days_to_opex(today)

        # Dealer-positioning gate — only when dealers under-hedged
        # long_gamma = dealers long gamma, forced to SELL into rallies (bad for us)
        # short_gamma = dealers short gamma, forced to BUY into rallies (good for us)
        # flip_zone = dealer position transitioning, asymmetric hedge demand likely
        # NOTE: gex_regime is computed for SPY in main.py — used as PROXY for
        # broader dealer-hedging stress. Per-ticker GEX would be better.
        # 2026-05-27 FIX: at DAILY resolution, SPY chain rarely has the >=10
        # contracts main.py requires to compute gex. _gex_regime stays None,
        # gating the strategy to 0 fires. Treat None as "flip_zone" (permissive
        # default) so OPEX-window + UOA + momentum gates take over.
        gex_eff = ctx.gex_regime if ctx.gex_regime is not None else "flip_zone"
        if gex_eff not in ("short_gamma", "flip_zone"):
            self._diag["gex_block"] += 1
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            # v17 IPO-date guard (see momentum_otm_calls.py docstring)
            min_date = WSB_TICKER_INCEPTION.get(ticker)
            if min_date is not None and today < min_date:
                continue
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                self._diag["cooldown"] += 1
                continue

            # v2 (2026-05-27): OR-logic confirmation — UOA active OR strong
            # 30d rally. Either signals dealer-hedging stress sufficient for
            # the gamma-squeeze setup. Prior version's UOA-only gate produced
            # 1 fire / 8mo because daily-resolution UOA is sparse for WSB
            # universe names.
            uoa_active = ticker in ctx.uoa_active
            mom30 = ctx.underlying_momentum_30d.get(ticker, 1.0)
            strong_rally = mom30 >= self.MOM_HIGH

            if not (uoa_active or strong_rally):
                self._diag["uoa_miss"] += 1
                continue

            # Even with OR-logic confirm, require SOME positive momentum
            # (otherwise we'd buy calls on falling tickers right before OPEX)
            if mom30 < 1.02:
                self._diag["mom_miss"] += 1
                continue

            self._last_fired[ticker] = today
            self._diag["fired"] += 1

            # Position structure:
            # - DTE = days to OPEX (so contract expires AT the OPEX cycle close)
            # - 10% OTM: cheap enough to benefit from gamma squeeze, not so far
            #   that strike is unreachable in the 7-day window
            # - Hard time-stop at OPEX+1: dealer hedging unwinds after expiry,
            #   no edge remains
            target_dte = max(7, d_opex + 1)
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=target_dte,
                edge_score=0.7,        # mechanical setup; tight bounds expected
                target_otm_pct=0.10,   # closer to spot — gamma window is tight
                take_profit_pct=None,  # trailing SL handles
                stop_loss_pct=-0.40,
                # Calendar exit: 1 day after OPEX (covers Mon settle in worst case)
                max_hold_days=max(2, d_opex + 1),
                max_per_trade_pct_nav=0.06,   # smaller — high-freq, lower-conviction
                notes=f"OPEX_d-{d_opex} gex={ctx.gex_regime} mom30={mom30:.2f}",
            ))
        return signals
