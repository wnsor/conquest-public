"""B1 — SPY crisis-trigger long put.

Thesis (v8 Ackman reference, per project_options_crisis_iteration_2026_05_19):
  Long puts STANDALONE perform well when fired only during the early-warning
  phase of a market crisis. Reference design got 6/6 wins in real crises
  (2008, 2010, 2015, 2018, 2020, 2022) with PF 2.07, 50% WR.

Trigger: CrisisDetector signals 'warning' OR 'crash' (early-cycle stress)
         AND no fire within last 180 days (per-crisis lock)

Position: 90-DTE SPY put, 5% OTM
Exit: TP+200%, NO stop-loss (let losers expire to capture full convex tail),
      max_hold 60 days

Sized larger than baseline strategies — fires rarely (≤2/yr), each fire is
high-conviction. Position sizer will cap via NAV %.

Why not buy at 'capitulation'? Too late — IV peaked, puts expensive, limited
upside. Why not at 'rebound'? Wrong side — that's CrisisReboundBasket's job.
The early-cycle 'warning'/'crash' window is when puts are cheap relative to
realized drawdown the crisis will deliver.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class SpyCrisisPut:
    id = "spy_crisis_put"
    enabled = True
    universe = ["SPY"]

    def __init__(self):
        self._last_fired = None
        self._cooldown_days = 180  # one fire per crisis cycle

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Per-crisis cooldown — don't re-fire during same event
        if (self._last_fired is not None and
                (today - self._last_fired) < timedelta(days=self._cooldown_days)):
            return []

        # Early-cycle crisis trigger only — skip if we're past peak stress
        if ctx.crisis_state not in ("warning", "crash"):
            return []

        # Sanity gate: VIX must be elevated but not yet capitulation-peak
        # (avoid buying puts at the absolute top of IV)
        if ctx.vix is None or ctx.vix < 22 or ctx.vix > 50:
            return []

        # Sanity: term must be flat-to-backwardated (vol surface is reacting)
        if ctx.term_regime == "contango":
            return []

        edge = 0.8  # high-conviction trigger (rare event, asymmetric payoff)
        if ctx.cstability_vote_count is not None and ctx.cstability_vote_count >= 2:
            edge = 1.0  # cstability also confirming risk-off

        self._last_fired = today
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="put",
                target_dte=90,
                edge_score=edge,
                target_otm_pct=0.05,       # 5% OTM put
                take_profit_pct=2.0,        # +200% — Ackman pattern
                stop_loss_pct=None,         # NO SL — let losers expire
                max_hold_days=60,
                notes=(f"crisis={ctx.crisis_state} vix={ctx.vix:.1f} "
                       f"term={ctx.term_regime} vote={ctx.cstability_vote_count}"),
            )
        ]
