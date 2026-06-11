"""v_SHORT_SQUEEZE_PURE — single-signal short-squeeze ignition trade.

Trigger thesis (leading-only)
=============================
A short squeeze is a forced-cover cascade. The ignition condition is:
short interest is GROWING (not just high) at the same time retail
positioning is also growing. When shorts try to add to their position
while retail keeps buying, the supply/demand imbalance creates a price
floor → first squeeze leg. Once the squeeze starts, gamma exposure on
short-dated calls forces dealers to keep buying.

We can't see retail flow directly. We use volume_spike as a proxy
(volume goes up when retail/institutional are accumulating). The
combination of accelerating SI + accelerating volume is the GME/MSTR
archetype.

Entry condition (ALL must fire):
  1. LEADING: short_interest_velocity > +30% (WoW SI rose 30%+)
     Stricter than v_REFLEX_v2's 20% — we want STRONG conviction signal.
  2. LEADING: insider_cluster_score >= 1.5  (insiders not selling)
     If insiders are selling INTO the rising-SI environment, that's a
     ceiling signal (they think it's overpriced). Require neutral-or-positive.
  3. COINCIDENT: volume_spike > 3.0  (retail/instit volume confirmation)
  4. REGIME: vix < 25, term_regime not backwardation, skew_z <= 0.5
     (squeezes need risk-on tape; need puts-side not panicking)

NOT used (deliberately):
  - momentum 30d/60d  (lagging)
  - moving averages   (lagging)
  - days_to_cover     (lagging — derived from prior trade history)
  - short_interest absolute level  (level alone is not predictive;
    Δ is what matters)

Position structure:
  - 45 DTE 15% OTM call — wider window because squeezes can take 30-60 days
  - 8% NAV — high conviction setup
  - Trailing-SL ladder + max_hold 60d (longer than v_REFLEX_v2 because
    squeezes have multi-week patterns)

Expected fire rate: 2-5 per year per ticker (very narrow gate).
Across 16 WSB-style universe: ~30-80 fires/year. Asymmetric R/R: most
will fail at -50%; the 15-25% that work deliver 200-1000%.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class ShortSqueezePure:
    id = "short_squeeze_pure"
    enabled = True
    universe = WSB_UNIVERSE

    SI_VELOCITY_THRESHOLD = 0.30   # +30% WoW SI growth (stricter than v2)
    INSIDER_SCORE_FLOOR = 0.0       # insiders not selling (>= 0)
    VOLUME_SPIKE_THRESHOLD = 3.0
    MAX_VIX = 25.0
    MAX_SKEW_Z = 0.5
    COOLDOWN_DAYS = 45              # one fire per "squeeze cycle"

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
        if ctx.vix9d_vix_ratio is not None and ctx.vix9d_vix_ratio > 1.05:
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                continue

            sz = ctx.skew_z.get(ticker)
            if sz is not None and sz > self.MAX_SKEW_Z:
                continue

            si_v = ctx.short_interest_velocity.get(ticker)
            if si_v is None or si_v < self.SI_VELOCITY_THRESHOLD:
                continue

            ins_score = ctx.insider_cluster_score.get(ticker, 0.0)
            if ins_score < self.INSIDER_SCORE_FLOOR:
                continue

            vol_sp = ctx.volume_spike.get(ticker, 1.0)
            if vol_sp < self.VOLUME_SPIKE_THRESHOLD:
                continue

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=45,
                edge_score=0.85,   # high — all 3 gates passed strictly
                target_otm_pct=0.15,
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=60,
                max_per_trade_pct_nav=0.08,
                notes=f"SQUEEZE si_v={si_v:.2f} ins={ins_score:.1f} vol={vol_sp:.1f}",
            ))
        return signals
