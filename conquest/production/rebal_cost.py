"""Track the cost of every rebalance event — NAV drift, fees, n_orders, etc.

Designed for paper-IB / live deployments so each redeploy's FORCE INITIAL
REBALANCE cost (and every subsequent monthly/signal rebalance cost) lands
in an Object Store CSV for permanent auditing.

The "cost" we track is intentionally simple: NAV before the rebalance call
vs NAV at ~90 min after market close on the same trading day. That captures
slippage, fees, and any imperfect fills together as a single $ figure, which
matches what the user actually loses to execution friction. Distinguishing
slippage from market move requires a counterfactual hold-NAV computation;
the user is welcome to add that, but for now we keep it simple and honest.

Object Store key: ``conquest/redeploy_log/rebalances.csv``

Schema (header included on first write):
    ts_start          ISO 8601 of when _begin_rebal_capture() was called
    trigger           FORCE_INITIAL | MONTHLY | SIGNAL_CHANGE
    nav_before        USD — total_portfolio_value right before set_holdings
    nav_after_close   USD — total_portfolio_value at after_market_close + 90min
    nav_delta         USD — nav_after_close - nav_before
    nav_delta_pct     percentage — delta / nav_before * 100
    fees_delta        USD — total IB fees incurred between begin/end snapshots
    notes             free-text — "auto_settled" for normal flow, or warnings

Wiring (project main.py):
    from conquest.production import harden
    harden(self)                              # registers schedule + helpers
    ...
    def _monthly_rebalance(self, _scheduled=True):
        trigger = ("FORCE_INITIAL" if self.last_rebalance_at is None
                   else "MONTHLY" if _scheduled else "SIGNAL_CHANGE")
        self._begin_rebal_capture(trigger)    # before set_holdings calls
        ... self.set_holdings(...) ...
        # _end_rebal_capture fires automatically 90min after market close
"""
from __future__ import annotations

_STORE_KEY = "conquest/redeploy_log/rebalances.csv"
_CSV_HEADER = (
    "ts_start,trigger,nav_before,nav_after_close,nav_delta,"
    "nav_delta_pct,fees_delta,notes\n"
)


def attach_rebal_cost_tracker(algo) -> None:
    """Wire algo._begin_rebal_capture(trigger) + algo._end_rebal_capture()
    plus the daily after-market-close settle callback. Idempotent."""
    if hasattr(algo, "_begin_rebal_capture"):
        return

    algo._rebal_pending = None  # dict or None

    def _begin_rebal_capture(trigger: str) -> None:
        """Snapshot NAV + fees BEFORE the rebalance fires its orders."""
        try:
            nav = float(algo.portfolio.total_portfolio_value)
        except Exception as e:
            algo.log(f"[rebal_cost] BEGIN nav read failed: {e}")
            return
        try:
            fees = float(algo.portfolio.total_fees)
        except Exception:
            fees = 0.0
        algo._rebal_pending = {
            "ts_start":    algo.time.isoformat(),
            "trigger":     trigger,
            "nav_before":  nav,
            "fees_before": fees,
        }
        algo.log(
            f"[rebal_cost] BEGIN {trigger} at {algo.time}: "
            f"nav_before=${nav:,.2f} cumulative_fees_so_far=${fees:,.2f}"
        )

    def _end_rebal_capture(notes: str = "") -> None:
        """Read post-rebalance NAV, log delta, append row to Object Store CSV.
        Idempotent — if nothing is pending, no-op. Called automatically by
        the daily after-close schedule, OR can be invoked manually."""
        rec = algo._rebal_pending
        if rec is None:
            return
        try:
            nav_after = float(algo.portfolio.total_portfolio_value)
            fees_after = float(algo.portfolio.total_fees)
        except Exception as e:
            algo.log(f"[rebal_cost] END read failed: {e}")
            return
        delta = nav_after - rec["nav_before"]
        pct = (delta / rec["nav_before"] * 100.0) if rec["nav_before"] > 0 else 0.0
        fees_delta = fees_after - rec["fees_before"]

        # Headline log — visible in live log even without pulling the CSV.
        algo.log(
            f"[rebal_cost] END {rec['trigger']} (started {rec['ts_start']}): "
            f"nav_after=${nav_after:,.2f} delta=${delta:,.2f} "
            f"({pct:+.3f}%) fees_incurred=${fees_delta:,.2f}"
        )

        # Append to Object Store CSV. Read-modify-write — defensive against
        # missing key + empty content + corrupted header.
        try:
            existing = ""
            if algo.object_store.contains_key(_STORE_KEY):
                existing = algo.object_store.read(_STORE_KEY) or ""
            if not existing.strip().startswith("ts_start,"):
                existing = _CSV_HEADER
            row = (
                f"{rec['ts_start']},{rec['trigger']},"
                f"{rec['nav_before']:.2f},{nav_after:.2f},"
                f"{delta:.2f},{pct:.4f},{fees_delta:.2f},"
                f"{notes}\n"
            )
            algo.object_store.save(_STORE_KEY, existing + row)
            algo.log(f"[rebal_cost] persisted to Object Store {_STORE_KEY}")
        except Exception as e:
            algo.log(f"[rebal_cost] persist failed: {e}")

        algo._rebal_pending = None

    algo._begin_rebal_capture = _begin_rebal_capture
    algo._end_rebal_capture = _end_rebal_capture

    # Schedule daily settle 90 min after SPY market close. Fires every day,
    # but is a no-op unless there's a pending capture. 90 min ensures the
    # day's fills have settled and total_fees has been updated.
    def _daily_settle():
        if getattr(algo, "is_warming_up", False):
            return
        if algo._rebal_pending is None:
            return
        algo._end_rebal_capture(notes="auto_settled")

    algo.schedule.on(
        algo.date_rules.every_day("SPY"),
        algo.time_rules.after_market_close("SPY", 90),
        _daily_settle,
    )
    algo.log(
        "[rebal_cost] tracker wired — every rebalance produces a row in "
        f"{_STORE_KEY}, settled 90min after each market close"
    )
