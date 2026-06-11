"""v_TRIPLE_CONFLUENCE — max-edge strict version of v_REFLEX_v2.

Same data + thesis as v_REFLEX_v2 but with STRICTER thresholds. Only
fires when ALL 3 leading triggers + BOTH coincident confirms align in
the same window. The bet: when this rare confluence occurs, the setup
is near-certain to produce a step-function outcome (MSTR-Dec23 archetype).

Entry condition (ALL must fire):
  L1: short_interest_velocity ≥ +25%       (vs 20% in v2)
  L2: insider_count_5d ≥ 3 distinct insiders
  L3: news_propagation_5d ≥ +50%
  C1: volume_spike > 3.0
  C2: implied_move_vs_realized > 2.0
  Regime: VIX < 30, term in (contango/flat), vix9d/vix < 1.05,
          skew_z <= 0.0 (strict no-panic)

Position structure:
  - 60 DTE 25% OTM call — wide because we expect a BIG move
  - 12% NAV per trade — higher conviction → bigger bet
  - max_hold 90d — let the reflexivity loop run to completion
  - Trailing-SL ladder captures the asymmetric tail

Expected fire rate: 0-3 per year per ticker (very rare).
Across 16 names: ~5-30 fires/year. The point isn't frequency — it's
that each fire is the strongest signal in the system.

Why a separate strategy vs. just raising v_REFLEX_v2 thresholds:
  Parallel testing. We run v2 (2-of-3 + 1-of-2) AND TRIPLE (3-of-3 + 2-of-2)
  side by side. Compares fire frequency vs. hit rate to determine
  whether stricter confluence improves R/R or just reduces sample size.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class TripleConfluence:
    id = "triple_confluence"
    enabled = True
    universe = WSB_UNIVERSE

    SI_VELOCITY_THRESHOLD = 0.25
    INSIDER_COUNT_THRESHOLD = 3
    NEWS_PROPAGATION_THRESHOLD = 1.50
    VOLUME_SPIKE_THRESHOLD = 3.0
    IM_REALIZED_THRESHOLD = 2.0
    MAX_VIX = 30.0
    MAX_VIX9D_VIX_RATIO = 1.05
    MAX_SKEW_Z = 0.0   # stricter — must be neutral or negative

    COOLDOWN_DAYS = 60

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
        if (ctx.vix9d_vix_ratio is not None
                and ctx.vix9d_vix_ratio > self.MAX_VIX9D_VIX_RATIO):
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                continue

            sz = ctx.skew_z.get(ticker)
            if sz is not None and sz > self.MAX_SKEW_Z:
                continue

            # ALL 3 leading triggers required
            si_v = ctx.short_interest_velocity.get(ticker)
            if si_v is None or si_v < self.SI_VELOCITY_THRESHOLD:
                continue
            if ctx.insider_count_5d.get(ticker, 0) < self.INSIDER_COUNT_THRESHOLD:
                continue
            news_p = ctx.news_propagation_5d.get(ticker)
            if news_p is None or news_p < self.NEWS_PROPAGATION_THRESHOLD:
                continue

            # BOTH coincident confirms required
            vs = ctx.volume_spike.get(ticker, 1.0)
            if vs < self.VOLUME_SPIKE_THRESHOLD:
                continue
            im_r = ctx.implied_move_vs_realized.get(ticker)
            if im_r is None or im_r < self.IM_REALIZED_THRESHOLD:
                continue

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=60,
                edge_score=1.0,
                target_otm_pct=0.25,   # wide — expecting BIG move
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=90,
                max_per_trade_pct_nav=0.12,   # higher conviction
                notes=(
                    f"TRIPLE si_v={si_v:.2f} ins={ctx.insider_count_5d[ticker]} "
                    f"news={news_p:.2f} vol={vs:.1f} im={im_r:.2f}"
                ),
            ))
        return signals
