"""A_gex_spy_baseline — non-selective companion to A_gex_spy_call (v12).

Purpose: permanent A/B reference point. Same GEX-driven SPY thesis as
A_GexSpyCall but with the original v1 (looser) gates — fires whenever
hard gates pass, no edge_score threshold. Carried alongside the selective
variant so we can track over time whether selectivity actually adds
per-trade alpha.

The v1 logic we're preserving:
  Hard gates: gex_regime == 'short_gamma' AND SPY 5MA cross > 20MA AND
              VIX<25 AND term_regime != 'backwardation'
  Confluences (3-5 of 5):
    - cstability_vote_count == 0  (+1)
    - SPY 30d momentum > 1.0      (+1)
  Entry: 35-DTE 5% OTM call
  Exit: TP+200%, SL-50%, max_hold 15d
  Cooldown: 15 days per-strategy

Use ACTIVE_STRATEGY_IDS to run one OR both:
  A_gex_spy_call       → selective (v12, requires edge ≥ 0.6)
  A_gex_spy_baseline   → non-selective (this file)
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class GexSpyBaseline:
    id = "gex_spy_baseline"
    enabled = True
    universe = ["SPY"]

    def __init__(self):
        self._last_fired = None
        self._cooldown_days = 15

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts
        if self._last_fired is not None and (today - self._last_fired) < timedelta(days=self._cooldown_days):
            return []

        # Hard gates (same as v1)
        if ctx.gex_regime != "short_gamma":
            return []
        if not ctx.underlying_5ma_above_20ma.get("SPY", False):
            return []
        if ctx.vix is None or ctx.vix >= 25:
            return []
        if ctx.term_regime == "backwardation":
            return []

        # v1 confluences — fire as long as hard gates pass (no edge threshold)
        confluences = 3
        if (m30 := ctx.underlying_momentum_30d.get("SPY")) is not None and m30 > 1.0:
            confluences += 1
        if ctx.cstability_vote_count is not None and ctx.cstability_vote_count == 0:
            confluences += 1
        edge = min(1.0, confluences / 5.0)

        self._last_fired = today
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="call",
                target_dte=35,
                edge_score=edge,
                target_otm_pct=0.05,       # v1: 5% OTM
                take_profit_pct=2.0,        # v1: +200%
                stop_loss_pct=-0.5,         # v1: -50%
                max_hold_days=15,            # v1: 15d
                notes=f"baseline gex_short 5MA>20 vix={ctx.vix:.1f} conf={confluences}/5",
            )
        ]
