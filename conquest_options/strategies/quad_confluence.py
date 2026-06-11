"""v_QUAD_CONFLUENCE — 4 leading signals must align.

Mix-and-match strategy. Uses 4 DIFFERENT leading mechanisms — institutional
positioning (SI velocity) + insider buying (Form 4) + news cascade (GDELT) +
retail attention (WSB). When all 4 fire simultaneously, the setup is the
"perfect storm" for a reflexivity ignition (rare but maximum conviction).

Trigger thesis (all 4 LEADING must fire):
  L1: short_interest_velocity > +25%   (institutional short crowding)
  L2: insider_count_5d >= 3            (insider cluster forming)
  L3: news_propagation_5d > 1.5        (mainstream picking up the story)
  L4: wsb_mention_velocity > 2.5       (retail joining the chase)

When all 4 align, every information layer (institutional + insider +
mainstream + retail) is converging on the same name. Historical pattern:
GME-Jan21, MSTR-Dec23, NVDA-Feb24 setups all showed this confluence.

Confirmation (need 1 of 2):
  C1: volume_spike > 3.0
  C2: implied_move_vs_realized > 1.8

Position:
  - 60-DTE 25% OTM calls (let the reflexivity loop play out)
  - 15% NAV per trade (very high — this fires <5x per year per universe)
  - max_hold 90d (these moves take weeks to develop)

Expected fire rate: 0-3 per year across the WSB universe. Sample-size
constrained but per-fire R:R is the highest of any strategy we have.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class QuadConfluence:
    id = "quad_confluence"
    enabled = True
    universe = WSB_UNIVERSE

    SI_VELOCITY_THRESHOLD = 0.25
    INSIDER_COUNT_THRESHOLD = 3
    NEWS_PROPAGATION_THRESHOLD = 1.5
    WSB_VELOCITY_THRESHOLD = 2.5
    VOLUME_SPIKE_THRESHOLD = 3.0
    IM_REALIZED_THRESHOLD = 1.8
    MAX_VIX = 30.0
    MAX_VIX9D_VIX_RATIO = 1.05
    MAX_SKEW_Z = 0.0     # strict no-panic
    COOLDOWN_DAYS = 90

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

            # ALL 4 leading required
            si_v = ctx.short_interest_velocity.get(ticker)
            if si_v is None or si_v < self.SI_VELOCITY_THRESHOLD:
                continue
            if ctx.insider_count_5d.get(ticker, 0) < self.INSIDER_COUNT_THRESHOLD:
                continue
            np5 = ctx.news_propagation_5d.get(ticker)
            if np5 is None or np5 < self.NEWS_PROPAGATION_THRESHOLD:
                continue
            wsb = ctx.wsb_mention_velocity.get(ticker)
            if wsb is None or wsb < self.WSB_VELOCITY_THRESHOLD:
                continue

            # Confirmation: 1 of 2
            vs = ctx.volume_spike.get(ticker, 1.0)
            im_r = ctx.implied_move_vs_realized.get(ticker)
            c1 = vs >= self.VOLUME_SPIKE_THRESHOLD
            c2 = im_r is not None and im_r >= self.IM_REALIZED_THRESHOLD
            if not (c1 or c2):
                continue

            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=60,
                edge_score=1.0,
                target_otm_pct=0.25,
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=90,
                max_per_trade_pct_nav=0.15,
                notes=(f"QUAD si={si_v:.2f} ins={ctx.insider_count_5d[ticker]} "
                       f"news={np5:.2f} wsb={wsb:.2f}"),
            ))
        return signals
