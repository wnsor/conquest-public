"""v_EARNINGS_REVISION_MOMENTUM — analyst upgrade cascade (Womack 1996).

Trigger thesis (leading):
  Analyst EPS estimate revisions PRECEDE the next earnings beat by 30-90
  days. When 30-day consensus EPS estimate rises >+5%, the stock typically
  drifts up into the next earnings report (PEAD's pre-earnings cousin).

  L1: earnings_revision_velocity > +5% in last 30 days
  L2: days_until_next_earnings ≤ 30 (must be in the drift window)

Confirmation (1-of-2):
  C1: volume_spike > 2.0 (institutional flow confirming)
  C2: insider_count_5d >= 1 (insider not selling into the upgrade)

Position: 30-DTE calls timed to expire AFTER earnings (capture both the
drift up + the earnings beat itself). 10% OTM = conservative on direction
but cheap enough for leverage.

Universe: WSB_UNIVERSE (could extend to S&P 500 if proven).
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class EarningsRevisionMomentum:
    id = "earnings_revision_momentum"
    enabled = True
    universe = WSB_UNIVERSE

    REVISION_THRESHOLD = 0.05    # +5% 30-day revision
    MAX_DAYS_TO_EARNINGS = 30    # within drift window
    MIN_DAYS_TO_EARNINGS = 5     # not too close (IV crush risk)
    VOLUME_SPIKE_THRESHOLD = 2.0
    INSIDER_FLOOR = 1
    MAX_VIX = 30.0
    COOLDOWN_DAYS = 30

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                continue

            rev = ctx.earnings_revision_velocity.get(ticker)
            if rev is None or rev < self.REVISION_THRESHOLD:
                continue

            d_earn = ctx.days_until_next_earnings.get(ticker, 999)
            if not (self.MIN_DAYS_TO_EARNINGS <= d_earn <= self.MAX_DAYS_TO_EARNINGS):
                continue

            vs = ctx.volume_spike.get(ticker, 1.0)
            ins = ctx.insider_count_5d.get(ticker, 0)
            confirm = (vs >= self.VOLUME_SPIKE_THRESHOLD) or (ins >= self.INSIDER_FLOOR)
            if not confirm:
                continue

            # DTE: position to expire ~5d after earnings to capture both drift + beat
            target_dte = max(15, d_earn + 7)
            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=target_dte,
                edge_score=0.75,
                target_otm_pct=0.10,
                take_profit_pct=None,
                stop_loss_pct=-0.45,
                max_hold_days=d_earn + 7,
                max_per_trade_pct_nav=0.06,
                notes=f"EARN_REV rev={rev:.2%} d_earn={d_earn} vol={vs:.1f} ins={ins}",
            ))
        return signals
