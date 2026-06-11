"""D2 — Tepper V-bottom LEAPS on SPY.

Edge thesis (David Tepper's documented playbook): buy LEAPS calls on the
index after a significant drawdown when V-recovery is confirming. Phase
5.13 BT at $1M seed: 12.28% CAGR / 86% DD / 100% win on 3 historical
V-bottoms (2009 / 2020 / 2022). Brief-mandated re-test at $10k seed.

Trigger:
  SPY drawdown from 252d high > 10% AND SPY 5MA cross > 20MA
  (combination = "we've sold off AND recovery is starting")

Contract:
  180 DTE ATM SPY call (semi-LEAPS for $10k affordability; true 365-730
  LEAPS need ~$3-5k per contract, > 10% NAV cap)

Sizing/Risk:
  Default sizer rules. With ATM 180-DTE premium maybe $20/share = $2000
  per contract, we'll need NAV ≥ $20k for portfolio_cap_pct_nav=0.10 to
  allow even one. EXPECTED OUTCOME: fires only if NAV has grown sufficiently
  OR if SPY drops enough to make ATM cheaper.

Exit:
  TP+150% (large catch on V-recovery), SL-50%, max_hold_days=90
"""
from __future__ import annotations

from datetime import timedelta

from strategies.base import StrategyContext, StrategySignal


class TepperVbottomLeaps:
    id = "tepper_vbottom_leaps"
    enabled = True
    universe = ["SPY"]

    def __init__(self):
        self._last_fired = None
        self._cooldown_days = 90  # one V-bottom entry per quarter at most
        # v22 DIAG: print gate state once per day so cloud BT logs reveal
        # exactly why the strategy isn't firing.
        self._diag_last_logged: object = None
        self._diag_log_fn = None  # set externally for tests; None = best-effort print

    def _diag(self, msg: str) -> None:
        """Emit a diagnostic line. In QC runtime, prefer `algo.debug(msg)` —
        but we don't have algo here. Best-effort: print() lands in QC log."""
        if self._diag_log_fn is not None:
            self._diag_log_fn(msg)
        else:
            print(msg)

    def on_data(self, ctx: StrategyContext) -> list[StrategySignal]:
        ts = ctx.timestamp
        if ts is None:
            return []
        today = ts.date() if hasattr(ts, "date") else ts

        # v22 DIAG: log gate state once per day. Captures whether each gate
        # passes/fails and the underlying values, so the cloud BT log reveals
        # the actual cause of 0-trade outcomes.
        is_new_day = self._diag_last_logged != today
        if is_new_day:
            self._diag_last_logged = today
            dd_val = ctx.underlying_drawdown_from_252d_high.get("SPY")
            fivema = ctx.underlying_5ma_above_20ma.get("SPY", None)
            cd = self._last_fired
            cd_days = (today - cd).days if cd is not None else 999
            self._diag(
                f"[D2-DIAG] {today} "
                f"dd={dd_val if dd_val is None else f'{dd_val:.3f}'} "
                f"5>20={fivema} "
                f"vix={ctx.vix if ctx.vix is None else f'{ctx.vix:.1f}'} "
                f"term={ctx.term_regime} "
                f"cd_days={cd_days} "
                f"gate_dd={'PASS' if dd_val is not None and dd_val >= 0.10 else 'FAIL'} "
                f"gate_5ma={'PASS' if fivema else 'FAIL'}"
            )

        if self._last_fired is not None and (today - self._last_fired) < timedelta(days=self._cooldown_days):
            return []

        # Drawdown gate (the "V" left side)
        dd = ctx.underlying_drawdown_from_252d_high.get("SPY")
        if dd is None or dd < 0.10:
            return []

        # Recovery gate (the "V" right side starting)
        if not ctx.underlying_5ma_above_20ma.get("SPY", False):
            return []

        # Confluences
        confluences = 2  # drawdown + 5MA cross
        if ctx.vix is not None and ctx.vix > 25:
            confluences += 1   # elevated vol = real crash + opportunity
        if ctx.term_regime == "backwardation":
            confluences += 1   # backwardation usually peaks at bottoms
        if dd > 0.15:
            confluences += 1   # deeper drawdown = better entry
        edge = min(1.0, confluences / 5.0)

        self._last_fired = today
        return [
            StrategySignal(
                strategy_id=self.id,
                underlying="SPY",
                side="leaps_call",
                target_dte=180,
                edge_score=edge,
                target_delta=0.50,         # ATM (cheaper than DITM)
                take_profit_pct=1.5,
                stop_loss_pct=-0.5,
                max_hold_days=90,
                # v22: V-bottom is rare (1-3 fires/decade) + 3:1 reward/risk
                # asymmetry justifies concentration. 25% cap = half-Kelly at
                # ~80% win-rate prior. Brief documented 80% NAV worked at $1M.
                max_per_trade_pct_nav=0.25,
                notes=f"tepper V, dd={dd:.2f}, vix={ctx.vix or 0:.1f}, conf={confluences}",
            )
        ]
