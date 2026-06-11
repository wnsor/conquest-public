"""v_ACTIVIST_DRIFT — Form 13D/G activist-stake follow strategy.

Trigger thesis (leading, mechanistic):
  Form 13D = activist >5% stake filed with intent to influence management
  (vs 13G = passive). 13D filings have a documented forward-return premium:
  the activist will publicly push for changes; the market often hasn't
  fully priced their thesis at filing time.

  L1: ticker in recent_13d_filing (last 14 days)

Confirmation (1-of-2):
  C1: insider_cluster_score >= 1.5 (existing insiders not bailing)
  C2: volume_spike >= 2.0 (flow building post-filing)

Position: 60-DTE 10% OTM calls. Activist campaigns take 30-180 days to
play out; 60-DTE captures the medium-term drift. Conservative 10% OTM
because we don't need a moonshot — just the activist-premium drift.

Universe: WSB_UNIVERSE for now (could extend to S&P 500 later — activists
target larger names too).
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class ActivistDrift:
    id = "activist_drift"
    enabled = True
    universe = WSB_UNIVERSE

    INSIDER_SCORE_FLOOR = 1.5
    VOLUME_SPIKE_THRESHOLD = 2.0
    MAX_VIX = 30.0
    COOLDOWN_DAYS = 180   # one campaign per ticker per 6 months

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            return []
        if not ctx.recent_13d_filing:
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            if ticker not in ctx.recent_13d_filing:
                continue
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                continue

            ins_score = ctx.insider_cluster_score.get(ticker, 0.0)
            vs = ctx.volume_spike.get(ticker, 1.0)
            confirm = (ins_score >= self.INSIDER_SCORE_FLOOR) or (vs >= self.VOLUME_SPIKE_THRESHOLD)
            if not confirm:
                continue

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=60,
                edge_score=0.80,
                target_otm_pct=0.10,
                take_profit_pct=None,
                stop_loss_pct=-0.45,
                max_hold_days=90,
                max_per_trade_pct_nav=0.10,
                notes=f"ACTIVIST_13D ins={ins_score:.1f} vol={vs:.1f}",
            ))
        return signals
