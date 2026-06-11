"""TEST 1 baseline (2026-05-26): random-entry — fires on EVERY universe ticker
on EVERY day (subject to cooldown + regime filter), with NO catalyst gate.

Purpose: test whether our catalyst gates (insider/earnings/skew/uoa) add
alpha vs. simple random entry on the same universe.

Falsification rule:
  - If this baseline produces WR/E SIMILAR to catalyst-gated momentum_otm_calls
    on the same windows → catalysts are NOT predictive; we've been optimizing
    nothing. Strategy pivot required.
  - If baseline WR/E is meaningfully WORSE (e.g., -10pp WR or worse) → catalyst
    gates DO add alpha; continue refining (v17 earnings revisions, etc.).

Universe + exits + sizing are IDENTICAL to v16 momentum_otm_calls to make the
comparison clean. Only difference: NO catalyst confluence check.
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal
from strategies.momentum_otm_calls import WSB_UNIVERSE


class MomentumNoCatalystBaseline:
    id = "momentum_no_catalyst_baseline"
    enabled = True
    universe = WSB_UNIVERSE   # same as v16 gated version

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        self._cooldown_days = 21   # same cadence as v16

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Regime filters — IDENTICAL to v16
        if ctx.vix is None or ctx.vix >= 25:
            return []
        if ctx.term_regime == "backwardation":
            return []
        if ctx.vix9d_vix_ratio is not None and ctx.vix9d_vix_ratio > 1.0:
            return []

        signals: list[StrategySignal] = []
        for ticker in WSB_UNIVERSE:
            # Per-ticker cooldown — same as v16
            last = self._last_fired.get(ticker)
            if last is not None and (today - last) < timedelta(days=self._cooldown_days):
                continue
            # NO catalyst gate — fire on every ticker every cooldown cycle
            self._last_fired[ticker] = today
            signals.append(StrategySignal(
                strategy_id=self.id,
                underlying=ticker,
                side="call",
                target_dte=28,         # same as v16
                edge_score=0.5,        # neutral baseline
                target_otm_pct=0.15,   # same as v16
                take_profit_pct=None,  # trailing exit from v15d
                stop_loss_pct=-0.4,    # same as v16
                max_hold_days=21,      # same as v16
                max_per_trade_pct_nav=0.08,
                notes="TEST_1_baseline_no_catalyst",
            ))
        return signals
