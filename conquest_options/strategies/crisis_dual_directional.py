"""B5 — Crisis dual-directional pre-positioning.

User directive 2026-05-25: build a separate crisis strategy that
pre-positions PUTS proactively (BEFORE crisis is confirmed, via early-warning
signal confluence) and transitions to CALLS on rebound confirmation.

Differs from existing strategies:
  - spy_crisis_put (B1): reactive only — fires AFTER crisis_state ∈ {warning, crash}
  - crisis_rebound_basket: reactive rebound on basket (failed 3/3 windows on chain)
  - tepper_vbottom_leaps (D2): LEAPS rebound on SPY (sharp-V only; covid20 win)
  - **THIS strategy (B5)**: PROACTIVE on early-warning + REACTIVE on rebound,
    single strategy handles both legs of the crisis cycle on SPY (deep chain).

Empirical basis:
  - VIX9D/VIX > 1.0 (term-stress) precedes major drawdowns by ~5-15 days
    (Whaley 2009; CBOE term structure studies)
  - Backwardation in VIX term structure = options market pricing imminent
    stress (Johnson 2017)
  - cstability 4-vote ensemble (existing in Conquest) aggregates HY-IEF,
    regime, term-structure signals — vote_count >= 1 = early warning
  - Crisis_state machine (already wired in main.py): normal → warning →
    crash → capitulation → rebound → recovery. We pre-position BEFORE the
    state machine confirms "warning", reducing entry-IV cost.

PUTS phase — PROACTIVE (fires BEFORE crisis_state is confirmed):
  Confluence trigger (need ≥2 of):
    - cstability_vote_count >= 1 (rising risk vote)
    - vix9d_vix_ratio > 1.0 (acute near-term stress in term structure)
    - term_regime == "backwardation" (forward-vol pricing stress)
    - vix > 20 (modestly elevated — but NOT panic yet)
  Cooldown: 60 days between PUT entries (one fire per warning cycle)
  Strike: SPY 7% OTM put (high gamma, reasonable cost)
  DTE: 45-60 (long enough for thesis to play out)

CALLS phase — REACTIVE (fires AFTER trough confirmation):
  Trigger (need all):
    - prior crisis_state was "crash" or "capitulation" in last 90 days
    - current crisis_state in ("rebound", "recovery")
    - vix < 25 (fear subsiding)
  Cooldown: 90 days between CALL entries (one fire per recovery cycle)
  Strike: SPY 5% OTM call
  DTE: 45-90

Exit (both legs):
  TP: trailing SL after +50% gain (no fixed TP cap — let winners run)
  SL: -40% (capped loss; relies on v13+ exit-feed fix to actually fire)
  max_hold: 90 days (longer than per-trade defaults; crisis cycles take time)

Sizing: 12% NAV per leg (event-driven, high-conviction; pairs with
  existing strategies for portfolio-level diversification).
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class CrisisDualDirectional:
    id = "crisis_dual_directional"
    enabled = True
    universe = ["SPY"]

    def __init__(self):
        self._last_put_fired = None
        self._last_call_fired = None
        self._put_cooldown_days = 60
        self._call_cooldown_days = 90
        # Track crisis_state history to identify recent crash/capitulation
        # for the CALL trigger. dict[date, crisis_state_str]
        self._state_history: dict[object, str] = {}

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # Record state history for CALL-phase lookback
        if ctx.crisis_state is not None:
            self._state_history[today] = ctx.crisis_state
            # Keep last 180 days only (avoid unbounded growth)
            cutoff = today - timedelta(days=180)
            self._state_history = {
                d: s for d, s in self._state_history.items() if d >= cutoff
            }

        signals: list[StrategySignal] = []

        # PUTS PHASE — PROACTIVE early-warning trigger
        signals.extend(self._evaluate_puts(ctx, today))
        # CALLS PHASE — REACTIVE rebound trigger
        signals.extend(self._evaluate_calls(ctx, today))

        return signals

    def _evaluate_puts(self, ctx: StrategyContext, today) -> list[StrategySignal]:
        # Per-cycle cooldown
        if (self._last_put_fired is not None and
                (today - self._last_put_fired) < timedelta(days=self._put_cooldown_days)):
            return []

        # Don't fire puts during confirmed crash (too late — IV peaked)
        # or rebound (wrong side)
        if ctx.crisis_state in ("crash", "capitulation", "rebound", "recovery"):
            return []

        # Confluence of early-warning signals (need ≥2)
        warnings = []
        if ctx.cstability_vote_count is not None and ctx.cstability_vote_count >= 1:
            warnings.append("vote")
        if ctx.vix9d_vix_ratio is not None and ctx.vix9d_vix_ratio > 1.0:
            warnings.append("vix9d")
        if ctx.term_regime == "backwardation":
            warnings.append("backwardation")
        if ctx.vix is not None and ctx.vix > 20:
            warnings.append("vix_elevated")

        if len(warnings) < 2:
            return []

        # Avoid firing if vol is already in panic mode (IV crush risk)
        if ctx.vix is not None and ctx.vix > 40:
            return []

        edge = min(1.0, 0.5 + 0.15 * len(warnings))   # 2 warnings → 0.8, 4 → 1.1 → cap 1.0
        self._last_put_fired = today
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="put",
                target_dte=50,
                edge_score=edge,
                target_otm_pct=0.07,        # 7% OTM (gamma sweet spot)
                take_profit_pct=None,        # no fixed TP — trailing exit
                stop_loss_pct=-0.4,         # -40% SL (relies on v13+ fix)
                max_hold_days=90,
                max_per_trade_pct_nav=0.12,
                notes=f"early_warning warns={','.join(warnings)} vix={ctx.vix}",
            )
        ]

    def _evaluate_calls(self, ctx: StrategyContext, today) -> list[StrategySignal]:
        # Per-cycle cooldown
        if (self._last_call_fired is not None and
                (today - self._last_call_fired) < timedelta(days=self._call_cooldown_days)):
            return []

        # Must be in rebound/recovery now
        if ctx.crisis_state not in ("rebound", "recovery"):
            return []

        # Must have seen crash/capitulation in the last 90 days (confirms cycle)
        saw_crash = any(
            s in ("crash", "capitulation")
            for s in self._state_history.values()
        )
        if not saw_crash:
            return []

        # Fear must be subsiding (VIX < 25)
        if ctx.vix is None or ctx.vix >= 25:
            return []

        edge = 0.85   # high-conviction — proven rebound thesis (tepper covid20 pattern)
        self._last_call_fired = today
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="call",
                target_dte=60,
                edge_score=edge,
                target_otm_pct=0.05,         # 5% OTM call
                take_profit_pct=None,         # trailing exit
                stop_loss_pct=-0.4,
                max_hold_days=90,
                max_per_trade_pct_nav=0.12,
                notes=f"rebound state={ctx.crisis_state} vix={ctx.vix} prior_crash=True",
            )
        ]
