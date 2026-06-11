"""Crisis state machine — detects market crashes and signals AGGRESSIVE
deploy mode during the V-bottom recovery phase.

Thesis: 1-3× per decade, markets crash and present uniquely asymmetric
long-options opportunities. Detecting these and deploying aggressively
(higher sizing, multi-vehicle basket) captures the recovery — Tepper's
documented playbook.

States:
  normal       — VIX < 20, no drawdown. Regular operations.
  warning      — VIX > 25 rising, momentum failing, modest drawdown. Defensive.
  crash        — VIX > 35, SPY -10%+, term backwardation. Stop calls; deploy puts if available.
  capitulation — VIX > 50, SPY -20%+, max panic. PREP — build cash for rebound.
  rebound      — VIX falling 30%+ from peak, SPY 5MA up-cross, drawdown still >10%.
                 → FIRE Crisis Rebound Basket aggressively. Once-per-crisis lock.
  recovery     — VIX < 25, SPY momentum positive, drawdown < 5%. Resume normal ops.

Per-crisis fire lock (180d) prevents re-firing during the same event.
"""
from __future__ import annotations

from collections import deque


class CrisisDetector:
    """State machine tracking crisis phase from per-tick context."""

    def __init__(self):
        self._state: str = "normal"
        self._vix_history: deque[float] = deque(maxlen=60)
        self._vix_peak: float = 0.0
        self._term_backward_days: int = 0
        self._last_state_change_date = None
        self._rebound_fired_date = None  # 180d lock after rebound fires
        # v22 fix: guard against per-minute calls at MINUTE resolution.
        # The state machine assumes day-by-day advancement: `_vix_history`
        # is a 60-DAY deque; `_term_backward_days` is a DAY counter that
        # increments at most once per calendar day. Without this guard,
        # 390 minute-bar calls per day would (a) blow out the 60-element
        # vix deque in 1 hour and (b) increment the backward-day counter
        # 390× per day, tripping "5 backwardation days" in 5 minutes.
        self._last_update_date = None

    def update(self, *, today, vix, vix_term_ratio, term_regime,
               spy_drawdown_from_252d_high, spy_5ma_above_20ma) -> str:
        """Compute new state from current context. Returns the state string.

        Idempotent within a calendar day: repeat calls on the same `today`
        return the cached state without advancing history/counters.
        """
        if today == self._last_update_date:
            return self._state  # already updated today
        self._last_update_date = today

        if vix is not None:
            self._vix_history.append(vix)
            self._vix_peak = max(self._vix_peak, vix)

        if term_regime == "backwardation":
            self._term_backward_days += 1
        else:
            self._term_backward_days = max(0, self._term_backward_days - 1)

        vix_30d_avg = (sum(self._vix_history) / len(self._vix_history)
                       if self._vix_history else None)
        dd = spy_drawdown_from_252d_high or 0.0
        vix_now = vix or 0.0
        vix_above_avg = (vix_now / vix_30d_avg if vix_30d_avg and vix_30d_avg > 0 else 1.0)
        vix_dropped_from_peak = ((self._vix_peak - vix_now) / self._vix_peak
                                  if self._vix_peak > 0 else 0.0)

        # 180-day lock — don't re-fire rebound for same crisis
        in_rebound_lock = (
            self._rebound_fired_date is not None and
            (today - self._rebound_fired_date).days < 180
        )

        # State decision tree — most-severe-first
        new_state = "normal"
        if vix_now > 60 and dd > 0.20:
            new_state = "capitulation"
        elif vix_now > 35 and dd > 0.10 and self._term_backward_days >= 5:
            new_state = "crash"
        elif vix_now > 25 and vix_above_avg > 1.3 and dd > 0.05:
            new_state = "warning"
        elif (not in_rebound_lock and
              dd > 0.10 and
              spy_5ma_above_20ma and
              vix_dropped_from_peak > 0.30 and
              self._vix_peak > 35):
            # Crash happened (vix peaked > 35), now V-recovery starting
            new_state = "rebound"
        elif self._state in ("crash", "capitulation", "rebound") and \
             vix_now < 25 and dd < 0.05:
            # Normalization after a crisis
            new_state = "recovery"
            # Reset vix_peak so next crisis starts clean
            self._vix_peak = vix_now or 0.0

        # Update state
        if new_state != self._state:
            self._last_state_change_date = today
        if new_state == "rebound" and self._state != "rebound":
            self._rebound_fired_date = today
        self._state = new_state
        return self._state

    @property
    def state(self) -> str:
        return self._state

    @property
    def vix_peak(self) -> float:
        return self._vix_peak
