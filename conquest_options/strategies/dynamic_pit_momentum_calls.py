"""Dynamic point-in-time momentum OTM-calls (survivorship-bias #1 kill).

The static ``momentum_otm_calls`` trades a fixed, hand-picked 24-name "WSB
universe" of stocks that ALREADY mooned by 2026 — textbook survivorship /
selection-on-outcome bias (Day-2 handover, bias #1). This strategy removes the
hand-pick entirely: each month ``main.py`` selects the top-N S&P-500 names by
180-day momentum, restricted to **point-in-time index members** (so a name is
only tradeable on date T if it was a member as of T), and exposes that selection
on ``ctx.active_universe``. This strategy then buys OTM calls on those names,
gated ONLY by the market-regime risk filters — survivorship-free by
construction.

Deliberately parsimonious (bias #3 kill): the WSB-tuned catalyst-OR-gate and
13-point confluence scoring of ``momentum_otm_calls`` are DROPPED. Those are
overfit to the 24 curated names and depend on signals (insider clusters, news
sentiment) that simply don't exist for an arbitrary rotating S&P-500 name. The
only gates here are the universal market-regime filters (VIX level, VIX-term
backwardation, acute VIX9D stress) plus a per-ticker cooldown — every one of
which is computable for any ticker.

Entry/exit/sizing params are reused verbatim from ``momentum_otm_calls`` v16 so
the dynamic and static results are directly comparable: the only intended
difference is *which names are eligible*, not how each trade is structured.

Universe is empty: the candidate names arrive dynamically via
``ctx.active_universe`` (populated by ``main.py``'s monthly PIT rebalance), so
this strategy declares no static subscription list of its own.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class DynamicPitMomentumCalls:
    id = "dynamic_pit_momentum_calls"
    enabled = True
    universe: list[str] = []   # dynamic — names come from ctx.active_universe

    def __init__(self):
        self._cooldown_days = 21   # same cadence as momentum_otm_calls v16

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Universal market-regime filters — verbatim from momentum_otm_calls
        # v16 (lines 104-113). These are market-wide, so they work for any
        # rotating ticker (no per-name catalyst data required).
        if ctx.vix is None or ctx.vix >= 25:
            return []
        if ctx.term_regime == "backwardation":
            return []
        if ctx.vix9d_vix_ratio is not None and ctx.vix9d_vix_ratio > 1.0:
            return []

        signals: list[StrategySignal] = []
        for ticker in ctx.active_universe:
            # Cooldown keyed on the last ENTRY (from main.py), not on emit: an
            # emit that fails to fill (chain absent that bar) must not lock the
            # name out for the cycle. main.py also enforces 1 concurrent position
            # per underlying, so this just prevents rapid re-entry after a close.
            last = ctx.last_entry_date.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                continue
            # Parsimonious: no catalyst gate, no confluence score. Membership in
            # ctx.active_universe already means PIT-eligible + top-N momentum.
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=28,         # same as v16
                edge_score=0.6,        # momentum-only conviction
                target_otm_pct=0.15,   # same as v16
                take_profit_pct=None,  # trailing exit from v15d
                stop_loss_pct=-0.4,    # same as v16
                max_hold_days=21,      # same as v16
                max_per_trade_pct_nav=0.08,
                notes="dyn_pit_momentum",
            ))
        return signals
