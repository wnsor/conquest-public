"""v_8K_BURST — material disclosure velocity (information cascade).

Trigger thesis (leading):
  Form 8-K is the SEC's "material events" filing. Normal companies file
  0-1 8-Ks per year. A BURST (3+ in 14 days) means something material is
  brewing — M&A talks, regulatory action, executive changes, restated
  earnings, etc.

  The market often doesn't fully process the implications until the
  story consolidates. Catching the burst gives a 5-15 day head start.

  L1: recent_8k_count >= 3 in last 14 days
  L2: insider_count_5d >= 1  (insiders not selling into the disclosures)

Confirmation (1-of-2):
  C1: volume_spike >= 2.0
  C2: skew_z neutral or negative (no put-side panic)

Position: 30-DTE 10% OTM straddle — direction unknown until the story
resolves, but the magnitude is likely material.

Actually use side="call" with conservative OTM since most 8-K bursts
that surface insiders-still-buying resolve UP (the bad ones see insider
selling early).

Universe: WSB_UNIVERSE + could extend to S&P 500.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class EightKBurst:
    id = "eightk_burst"
    enabled = True
    universe = WSB_UNIVERSE

    EIGHTK_BURST_THRESHOLD = 3
    INSIDER_FLOOR = 1
    VOLUME_SPIKE_THRESHOLD = 2.0
    MAX_SKEW_Z = 0.5
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

            n8k = ctx.recent_8k_count.get(ticker, 0)
            if n8k < self.EIGHTK_BURST_THRESHOLD:
                continue

            ins = ctx.insider_count_5d.get(ticker, 0)
            if ins < self.INSIDER_FLOOR:
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
                edge_score=0.70,
                target_otm_pct=0.10,
                take_profit_pct=None,
                stop_loss_pct=-0.45,
                max_hold_days=30,
                max_per_trade_pct_nav=0.06,
                notes=f"8K_BURST n8k={n8k} ins={ins} vol={vs:.1f}",
            ))
        return signals
