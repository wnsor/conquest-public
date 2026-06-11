"""v_REFLEX_v2 — leading-indicator-only reflexivity ignition detector.

Replaces reflex_ignition v1 (which mixed coincident + lagging accelerators).
v2 design principle, per user directive 2026-05-26:

  "Lagging indicators are not sufficient for options. Use only leading."

Cleaner restatement:
  - Entry trigger must be LEADING (signal fires BEFORE the price move).
  - Confirmation may be COINCIDENT (signal aligns WITH today's market activity).
  - NEVER use lagging signals (signal reflects past price action).

Architecture
============

  ┌──────── LEADING TRIGGER (need 2 of 3) ─────────┐
  │  L1: short_interest_velocity ≥ +20% WoW       │  ← FINRA biweekly
  │  L2: insider_count_5d ≥ 3 distinct insiders   │  ← SEC Form 4
  │  L3: news_propagation_5d ≥ +50% (5d/5d ratio) │  ← GDELT
  └────────────────────────────────────────────────┘
                       ↓
  ┌──── COINCIDENT CONFIRMATION (need 1 of 2) ─────┐
  │  C1: volume_spike > 3× 20d avg                 │
  │  C2: implied_move_vs_realized > 2.0            │  ← chain math
  └────────────────────────────────────────────────┘
                       ↓
  ┌────────── REGIME FILTERS (all must pass) ──────┐
  │  R1: VIX < 30                                   │
  │  R2: term_regime in (contango, flat)            │
  │  R3: vix9d_vix_ratio < 1.05                     │
  │  R4: skew_z ≤ 0.5  (no puts-side panic)         │
  └────────────────────────────────────────────────┘

Why each leading signal qualifies as "leading":

  L1 — Short interest velocity:
       SI is reported biweekly with ~10 day lag. When SI is GROWING
       (not just high), short sellers are increasing pressure. Combined
       with a price floor or accumulating retail flow, this is the
       canonical setup for a squeeze (GME 2021, MSTR 2024, RKLB 2024).
       The Δ matters more than absolute level — a stock with 30% SI
       that's been stable for months has equilibrium; one going from
       20%→25%→30% in 4 weeks has accelerating disagreement.

  L2 — Insider cluster (Cohen-Malloy-Pomorski 2012):
       Insiders position 30-90 days before public catalysts. 3+ DISTINCT
       insiders (officers, directors, 10% owners) all buying in a 5-day
       window is statistically rare and historically forward-predictive.
       This is the strongest single insider-flow signal.

  L3 — News propagation (5d/5d ratio):
       Attention SPREADING across distinct sources/days is leading.
       Static high attention is not leading (it's coincident).
       Growth rate (this 5d vs prior 5d) is the leading component.

C1 (volume_spike) and C2 (implied_move_vs_realized) are confirmations,
NOT triggers — they make sure the leading signal is being corroborated
by current market behavior. C1 fires when institutional money is moving
today. C2 fires when options sellers are pricing in a move that hasn't
shown up in realized vol — informed positioning.

Why NO momentum/MA/Bollinger/RSI etc:
  These are all lagging by construction. They confirm trends that have
  already happened. Per the user directive, never use them as entry
  triggers.

Sample size & expectation:
  Confluence of 2/3 leading + 1/2 confirmation in regime is RARE.
  Expected: 2-8 fires per ticker per year. Across the WSB universe
  (16 names), 30-120 fires per year. Trailing-SL ladder caps avg loss
  at -3% (per v15d data); the bet is on the ~15-25% of fires that
  become 5x-10x asymmetric tails.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class ReflexIgnitionV2:
    id = "reflex_ignition_v2"
    enabled = True
    universe = WSB_UNIVERSE

    # ── Tuning thresholds — chosen from theory + literature, NOT BT-fit ──

    # LEADING triggers
    SI_VELOCITY_THRESHOLD = 0.20       # SI must rise 20%+ WoW
    INSIDER_COUNT_THRESHOLD = 3        # 3+ distinct insiders in 5d
    NEWS_PROPAGATION_THRESHOLD = 1.50  # 5d attention 50%+ above prior 5d

    # Confirmation
    VOLUME_SPIKE_THRESHOLD = 3.0       # $-volume 3× 20d avg
    IM_REALIZED_THRESHOLD = 2.0        # IM_30d / HV_30d > 2.0

    # Regime gates
    MAX_VIX = 30.0
    MAX_VIX9D_VIX_RATIO = 1.05
    MAX_SKEW_Z = 0.5                   # puts-side not panicking

    # Confluence threshold: 2 of 3 LEADING + 1 of 2 CONFIRMING
    MIN_LEADING = 2
    MIN_CONFIRMATION = 1

    COOLDOWN_DAYS = 30

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # ── Regime gates ───────────────────────────────────────────────
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

            # ── Per-ticker regime check (skew_z) ────────────────────────
            sz = ctx.skew_z.get(ticker)
            if sz is not None and sz > self.MAX_SKEW_Z:
                continue   # puts-side panicking, kill signal

            # ── LEADING triggers ──────────────────────────────────────
            si_v = ctx.short_interest_velocity.get(ticker)
            l1 = si_v is not None and si_v >= self.SI_VELOCITY_THRESHOLD

            ins_n = ctx.insider_count_5d.get(ticker, 0)
            l2 = ins_n >= self.INSIDER_COUNT_THRESHOLD

            news_p = ctx.news_propagation_5d.get(ticker)
            l3 = news_p is not None and news_p >= self.NEWS_PROPAGATION_THRESHOLD

            leading_count = sum([l1, l2, l3])
            if leading_count < self.MIN_LEADING:
                continue

            # ── COINCIDENT confirmation ───────────────────────────────
            vs = ctx.volume_spike.get(ticker, 1.0)
            c1 = vs >= self.VOLUME_SPIKE_THRESHOLD

            im_r = ctx.implied_move_vs_realized.get(ticker)
            c2 = im_r is not None and im_r >= self.IM_REALIZED_THRESHOLD

            confirm_count = sum([c1, c2])
            if confirm_count < self.MIN_CONFIRMATION:
                continue

            # Confluence met → fire
            self._last_fired[ticker] = today
            # Edge score = leading_count / 3 (full leading) * (1 + confirm_count/2)
            # 2/3 lead + 1/2 confirm → 0.67 * 1.5 = 1.00... clamp
            edge = min(1.0, (leading_count / 3.0) * (1.0 + confirm_count / 2.0))

            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                # Match v_REFLEX_v1 parameters so comparison is clean
                target_dte=45,
                edge_score=edge,
                target_otm_pct=0.20,
                take_profit_pct=None,
                stop_loss_pct=-0.50,
                max_hold_days=45,
                max_per_trade_pct_nav=0.08,
                notes=(
                    f"REFLEX_v2 L={leading_count}/3 (si={l1},ins={l2},news={l3}) "
                    f"C={confirm_count}/2 (vol={c1},im={c2}) "
                    f"edge={edge:.2f}"
                ),
            ))
        return signals
