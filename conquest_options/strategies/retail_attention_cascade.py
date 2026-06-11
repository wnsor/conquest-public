"""v_RETAIL_ATTENTION_CASCADE — fish_-style retail-driven momentum capture.

The Roaring Kitty / WSB-AMC / fish_ archetype: a thesis spreads through
r/WSB and Google search before the price fully reflects it. We detect the
spread BEFORE the price catches up.

Trigger thesis (leading, retail-information cascade):
  L1: wsb_mention_velocity > 3.0  (5d/5d ratio — WSB chatter tripling)
  L2: google_trends_velocity > 2.0  (search interest doubling)

Need 2-of-2 leading triggers. The Google + Reddit combo is hard to fake
(actual humans typing both queries) — pure WSB pump-and-dump would show
in Reddit but not Google.

Confirmation (1 of 2):
  C1: volume_spike > 3.0
  C2: skew_z not panicking (≤ 0.5)

Position: 30-DTE 15% OTM calls. fish_-style.

Universe: WSB_UNIVERSE. The whole thesis is retail-attention-driven
small/mid-caps with narrative.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class RetailAttentionCascade:
    id = "retail_attention_cascade"
    enabled = True
    universe = WSB_UNIVERSE

    WSB_VELOCITY_THRESHOLD = 3.0
    GTRENDS_VELOCITY_THRESHOLD = 2.0
    VOLUME_SPIKE_THRESHOLD = 3.0
    MAX_VIX = 28.0
    MAX_SKEW_Z = 0.5
    COOLDOWN_DAYS = 21

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            return []
        if ctx.term_regime == "backwardation":
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                continue

            wsb = ctx.wsb_mention_velocity.get(ticker)
            gtr = ctx.google_trends_velocity.get(ticker)
            if wsb is None or wsb < self.WSB_VELOCITY_THRESHOLD:
                continue
            if gtr is None or gtr < self.GTRENDS_VELOCITY_THRESHOLD:
                continue

            sz = ctx.skew_z.get(ticker)
            if sz is not None and sz > self.MAX_SKEW_Z:
                continue

            vs = ctx.volume_spike.get(ticker, 1.0)
            if vs < self.VOLUME_SPIKE_THRESHOLD:
                continue

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=30,
                edge_score=0.80,
                target_otm_pct=0.15,
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=30,
                max_per_trade_pct_nav=0.08,
                notes=f"RETAIL_CASCADE wsb={wsb:.2f} gtr={gtr:.2f} vol={vs:.1f}",
            ))
        return signals
