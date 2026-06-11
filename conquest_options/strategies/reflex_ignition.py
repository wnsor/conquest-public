"""v_REFLEX — reflexivity ignition detector (2026-05-26).

Thesis (abstract, not literature-derived)
========================================
A small subset of trades produce step-function 10x+ outcomes (MSTR Dec23+,
GME Jan21, NVDA Feb24, PLTR Nov20, AMC May21). These outcomes are NOT
predicted by any stationary academic model — they require *path-dependent*
narrative-price feedback loops (Soros reflexivity).

The loop is observable BEFORE the explosive move when these acceleratorss
fire SIMULTANEOUSLY in a tight window (NOT independently):

  (1) Attention acceleration   — news_volume_spike >> trailing mean
                                 (institutions + retail focusing in)
  (2) Conviction asymmetry      — news_sentiment_24h positive + uoa_active
                                 (option flow leaning call-side)
  (3) Price-volume confirmation — volume_spike >> 3x + price up 5-20%
                                 (move already started, dealers hedging)
  (4) Skew compression          — skew_z near 0 or negative
                                 (NO put-buying = no doubt, no hedge demand)
  (5) Momentum acceleration     — 30d-mom > 60d-mom AND >1.10
                                 (rate-of-change rising, not just up)

When 4 of 5 fire in the SAME ticker in the SAME 5-day window, the
narrative-price loop is *activating*. Most academic models can't see
this because:
  - Sentiment alone has weak alpha
  - Volume alone has weak alpha
  - Each signal independently is widely studied
  - The CONJUNCTION + SIMULTANEITY in a tight window is what matters

False positives: a single signal firing alone is meaningless. Hence
the 4-of-5 confluence requirement.

Edge thesis: capture asymmetric tails. We don't care about hit rate;
we care about being IN the trade when the loop runs to completion.
Expected: WR 15-25%, AvgWin 100-500%, AvgLoss capped at -30% via
trailing SL ladder.

Sizing: 8% NAV per trade (same as v16) to leave room for multiple
concurrent ignitions during regime-favorable windows.

Universe: same as momentum_otm_calls (WSB-style names where reflexivity
is most active — low-mid cap, narrative-driven, retail-engaged).
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class ReflexIgnition:
    id = "reflex_ignition"
    enabled = True
    universe = WSB_UNIVERSE   # reflexivity is concentrated in this archetype

    # Tuning constants — picked from theoretical considerations, not from
    # backtest overfit. Re-evaluate AFTER first BT result.
    MIN_CONFLUENCE = 4      # out of 5 accelerators
    COOLDOWN_DAYS = 30      # ignitions are rare; don't re-fire too soon

    # Per-accelerator thresholds
    NEWS_VOLUME_TRIGGER = 3.0       # articles today / 30d avg
    NEWS_SENTIMENT_TRIGGER = 0.20   # mean tone 24h (positive)
    VOLUME_SPIKE_TRIGGER = 3.0      # $-volume today / 20d mean
    MOM_30D_TRIGGER = 1.10          # 30d underlying return
    MOM_ACCEL_REQUIRED = True       # 30d-mom > 60d-mom (rate accelerating)
    SKEW_Z_MAX = 0.5                # skew not yet panicking puts (no doubt)

    def __init__(self):
        self._last_fired: dict[str, object] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Regime filters — reflexivity loops DIE in true risk-off (forced
        # liquidation breaks the price-narrative feedback).
        if ctx.vix is None or ctx.vix >= 30:
            return []
        if ctx.term_regime == "backwardation":
            return []
        if ctx.vix9d_vix_ratio is not None and ctx.vix9d_vix_ratio > 1.10:
            return []

        signals: list[StrategySignal] = []
        for ticker in self.universe:
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
                continue

            # Compute 5 accelerator signals — each TRUE or FALSE
            accel = self._compute_accelerators(ctx, ticker)
            confluence = sum(1 for v in accel.values() if v)

            if confluence < self.MIN_CONFLUENCE:
                continue

            # Reflexivity ignition detected — fire OTM call
            self._last_fired[ticker] = today
            edge = confluence / 5.0   # 0.8 or 1.0

            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=45,        # 45 DTE — longer to let the loop unwind
                edge_score=edge,
                # 20% OTM: cheaper leverage, bigger payoff if the loop runs to
                # full reflexivity (CRDO Jan26 +94%, NBIS +135%, MSTR Dec23 +200%)
                target_otm_pct=0.20,
                # No fixed TP — trailing SL ladder captures asymmetric tail
                take_profit_pct=None,
                # Wider SL than v16 — reflexive setups need time to develop;
                # tighter SL kills the long-tail outcomes
                stop_loss_pct=-0.50,
                # Longer hold than v16 — the rip happens 10-30d after ignition
                max_hold_days=45,
                max_per_trade_pct_nav=0.08,
                notes=(
                    f"REFLEX_n={confluence}/5 "
                    f"newsvol={accel['news_vol']} sentp={accel['sentiment_p']} "
                    f"volspk={accel['vol_spike']} mom={accel['mom_accel']} "
                    f"skewok={accel['skew_ok']}"
                ),
            ))
        return signals

    def _compute_accelerators(self, ctx: StrategyContext, ticker: str) -> dict[str, bool]:
        """Return the 5 accelerator booleans for ticker. Conservative on
        missing data — None/missing counts as FALSE (don't fire on partial
        info; reflexivity requires confluence, not luck)."""
        # (1) News attention acceleration
        nvol = ctx.news_volume_spike.get(ticker, 1.0)
        news_vol = nvol > self.NEWS_VOLUME_TRIGGER

        # (2) Conviction asymmetry: positive sentiment + UOA active
        sent = ctx.news_sentiment_24h.get(ticker, 0.0)
        uoa = ticker in ctx.uoa_active
        sentiment_p = (sent > self.NEWS_SENTIMENT_TRIGGER) or uoa

        # (3) Price-volume confirmation: institutional dollar flow
        vs = ctx.volume_spike.get(ticker, 1.0)
        mom30 = ctx.underlying_momentum_30d.get(ticker, 1.0)
        vol_spike = (vs > self.VOLUME_SPIKE_TRIGGER) and (mom30 > 1.05)

        # (4) Skew not panicking: no put-side hedging demand
        sz = ctx.skew_z.get(ticker)
        skew_ok = (sz is None) or (sz <= self.SKEW_Z_MAX)

        # (5) Momentum acceleration (rate of change rising)
        mom60 = ctx.underlying_momentum_60d.get(ticker, 1.0)
        mom_accel = (
            mom30 > self.MOM_30D_TRIGGER
            and (mom30 > mom60 if self.MOM_ACCEL_REQUIRED else True)
        )

        return {
            "news_vol": news_vol,
            "sentiment_p": sentiment_p,
            "vol_spike": vol_spike,
            "skew_ok": skew_ok,
            "mom_accel": mom_accel,
        }
