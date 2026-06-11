"""C2 — Pre-earnings 5% OTM strangle (long).

Same edge as C1 (IV expansion pre-earnings) but with OTM legs — cheaper
premium per leg, higher leverage to a large move, lower break-even
probability for "no-vol-pump" scenarios. The cheaper-entry trade-off is
that you need more vol-expansion to recover the slightly-lower per-leg
exposure.

Universe identical to C1 for direct per-trade comparison: any per-trade
delta in metrics should be attributable to the strike-selection
difference, not universe.

Mechanics:
  Trigger: days_until_next_earnings ∈ [2, 3]
  Entry: 5% OTM call + 5% OTM put, 14-30 DTE
  Exit: max_hold_days=1
  TP: +150% (need bigger pump to make OTM legs worth it), SL: -60%
"""
from __future__ import annotations

from strategies.base import StrategyContext, StrategySignal
from strategies.earnings_straddle import STRADDLE_UNIVERSE


class EarningsStrangle:
    id = "earnings_strangle"
    enabled = True
    universe = STRADDLE_UNIVERSE

    def __init__(self):
        self._fired_event: dict[str, int] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        signals: list[StrategySignal] = []
        for ticker in STRADDLE_UNIVERSE:
            d = ctx.days_until_next_earnings.get(ticker)
            if d is None:
                continue
            if d not in (2, 3):
                if d > 7 and ticker in self._fired_event:
                    del self._fired_event[ticker]
                continue
            if ticker in self._fired_event:
                continue
            self._fired_event[ticker] = d

            confluences = 1
            iv = ctx.iv_rank.get(ticker)
            if iv is not None and iv < 60:
                confluences += 1
            if ctx.vix is not None and ctx.vix < 25:
                confluences += 1
            if ctx.term_regime in ("contango", "flat"):
                confluences += 1
            edge = min(1.0, confluences / 4.0)

            common = dict(
                strategy_id=self.id,
                underlying=ticker,
                target_dte=21,
                edge_score=edge,
                take_profit_pct=1.5,
                stop_loss_pct=-0.6,
                max_hold_days=1,
                notes=f"earnings T-{d}, iv_rank={iv}, conf={confluences}",
            )
            # Two-leg strangle: 5% OTM call + 5% OTM put
            signals.append(StrategySignal(side="call", target_otm_pct=0.05, **common))
            signals.append(StrategySignal(side="put", target_otm_pct=0.05, **common))
        return signals
