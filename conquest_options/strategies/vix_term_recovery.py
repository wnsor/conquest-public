"""v_VIX_TERM_RECOVERY — buy SPY/QQQ calls when VIX9D inverts back below VIX.

Trigger thesis (leading-only)
=============================
The VIX9D/VIX ratio measures front-end vs spot vol expectations. When
it INVERTS FROM ELEVATED back to normal, it means short-term stress is
resolving FASTER than long-term — the recovery has begun in the
options market BEFORE it shows up in price.

Specifically:
  - VIX9D/VIX > 1.10 = acute short-term stress (recent spike)
  - VIX9D/VIX < 0.95 from > 1.10 in past 5d = stress resolving
  - The transition is LEADING because vol traders react to incoming
    flow they see before spot does

This is the "bear-fade" signal that captures V-shaped recoveries
(COVID March 2020, COVID-rebound May 2020, Selloff22 fade, banking-
crisis March 2023 fade). Tepper's "V-bottom" instinct, formalized.

Entry condition:
  1. PRIOR STRESS: max(vix9d_vix_ratio over past 5d) >= 1.10
     Required — without prior stress, this isn't a "recovery", just noise
  2. RESOLUTION: today's vix9d_vix_ratio < 0.95
     The transition: front-end vol normalizing
  3. REGIME: VIX no longer in panic (< 35)
     We're buying recovery, not catching falling knife

Position structure:
  - SPY ATM 45 DTE call (broad-market recovery; not single-stock)
  - 15% NAV per trade — high conviction; this fires rarely
  - max_hold 60 days — recovery rallies need time to develop
  - Trailing-SL ladder captures asymmetric upside

Universe = ["SPY"] only — this is an index-level mechanical signal,
NOT a single-stock thesis. Per-name VIX9D would be needed for
single-stock version.

Need to track VIX9D/VIX history. We use a small ring buffer inside
the strategy (no need to expose in StrategyContext since only this
strategy needs it).

Expected fire rate:
  ~3-8 per year (one per crisis-recovery cycle). Asymmetric R/R
  expected: 30-60% WR; +50-300% wins; -30 to -50% losses.
"""
from __future__ import annotations

from collections import deque
from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class VixTermRecovery:
    id = "vix_term_recovery"
    enabled = True
    universe = ["SPY"]

    STRESS_THRESHOLD = 1.10        # past 5d max ratio must exceed this
    RECOVERY_THRESHOLD = 0.95      # today must be below this
    MAX_VIX = 35.0                 # don't trade in active panic
    HISTORY_LOOKBACK_DAYS = 5      # how far back to look for prior stress
    COOLDOWN_DAYS = 30

    def __init__(self):
        self._last_fired: dict[str, object] = {}
        # Ring buffer of (date, ratio) for past N days
        self._ratio_history: deque = deque(maxlen=self.HISTORY_LOOKBACK_DAYS + 1)

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Update history (whether we fire or not)
        r = ctx.vix9d_vix_ratio
        if r is not None:
            self._ratio_history.append((today, r))

        # Need enough history + current ratio to evaluate
        if r is None or len(self._ratio_history) < self.HISTORY_LOOKBACK_DAYS:
            return []
        if ctx.vix is None or ctx.vix >= self.MAX_VIX:
            return []

        # Cooldown
        last = self._last_fired.get("SPY")
        if last is not None and (today - last) < timedelta(days=self.COOLDOWN_DAYS):
            return []

        # Prior stress: was VIX9D/VIX > STRESS_THRESHOLD in last 5 days?
        # Build list defensively — at warmup edges, the strict-prior filter
        # (d < today) can be empty even when history has entries; max([]) raises.
        prior_ratios = [
            ratio for d, ratio in self._ratio_history
            if d < today and (today - d) <= timedelta(days=self.HISTORY_LOOKBACK_DAYS)
        ]
        if not prior_ratios:
            return []
        prior_max = max(prior_ratios)
        if prior_max < self.STRESS_THRESHOLD:
            return []

        # Resolution: today's ratio dropped below RECOVERY_THRESHOLD
        if r >= self.RECOVERY_THRESHOLD:
            return []

        self._last_fired["SPY"] = today
        return [StrategySignal(
            strategy_id=self.id,
            underlying="SPY",
            side="call",
            target_dte=45,
            edge_score=0.85,
            target_otm_pct=0.0,         # ATM
            take_profit_pct=None,
            stop_loss_pct=-0.50,
            max_hold_days=60,
            max_per_trade_pct_nav=0.15,  # high conviction rare-fire trade
            notes=f"VIX_TERM_RECOV prior_max={prior_max:.3f} today={r:.3f}",
        )]
