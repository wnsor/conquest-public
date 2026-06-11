"""Per-sleeve NAV attribution for combined-fund algorithms (regime,
cstag_voltgt_combined, etc.).

Lean's portfolio object only knows total NAV — there's no built-in concept
of "this position belongs to sleeve A vs sleeve B". For combined funds
that run two strategies inside one IB account, you can't tell whether a
bad month is voltgt blowing up or cstag underperforming without per-sleeve
attribution.

This module reconstructs the split. At each rebalance, the algorithm
already knows the target dollar amount each sleeve wants in each symbol
(``w_m1 * m1_targets[sym]``, etc.). We persist those per-symbol shares
to memory; then daily we compute ``m1_nav = sum(holding_value × m1_share)``
and ``m2_nav = sum(holding_value × m2_share)``, plus the cash sleeve.

Limitations (worth knowing for analysis):
  - Cash dividends land in the algo's cash bucket and are attributed to
    sleeves proportionally via the cash-split ratio at last rebalance.
  - Fees + slippage are borne by both sleeves proportionally, not by the
    sleeve that triggered the trade. (Lean doesn't expose per-order
    fee → strategy attribution.)
  - Mid-period rebalances reset the per-symbol shares to the new split,
    so intra-period attribution is approximate.

Object Store key: ``conquest/sleeve_attribution/daily.csv``

Schema (header on first write):
    date          ISO 8601 of snapshot
    m1_nav        USD attributed to sleeve M1 (cstag)
    m2_nav        USD attributed to sleeve M2 (voltgt)
    cash_alloc    USD currently in cash (not attributed)
    total_nav     USD total portfolio value
    m1_pct        m1_nav / total_nav * 100
    m2_pct        m2_nav / total_nav * 100

Wiring (project main.py):
    from conquest.production import harden
    harden(self)              # registers schedule + helpers

    def _monthly_rebalance(self, _scheduled=True):
        ...
        # After computing combined[sym] from m1+m2 targets and BEFORE
        # set_holdings calls, register the split:
        self._record_sleeve_split(
            m1_dollar_per_sym={sym: w_m1 * w * total_nav
                               for sym, w in m1_targets.items()},
            m2_dollar_per_sym={sym: w_m2 * w * total_nav
                               for sym, w in m2_targets.items()},
            cash_m1=w_m1 * w_cash * total_nav,  # 0 in static 50/50
            cash_m2=w_m2 * w_cash * total_nav,  # 0 in static 50/50
            w_m1=w_m1,
            w_m2=w_m2,
        )
"""
from __future__ import annotations

_STORE_KEY = "conquest/sleeve_attribution/daily.csv"
_CSV_HEADER = "date,m1_nav,m2_nav,cash_alloc,total_nav,m1_pct,m2_pct\n"
_STATE_KEY = "conquest/sleeve_attribution/state.json"


def attach_sleeve_attribution(algo, *,
                              m1_label: str = "cstag",
                              m2_label: str = "voltgt") -> None:
    """Wire algo._record_sleeve_split(...) + the daily before-market-close
    snapshot. Idempotent.

    Args:
        m1_label / m2_label: sleeve names used for chart series and log lines
        (default cstag/voltgt for regime; override to "cstag"/"v17" for
        combined_v3 which pairs cstag with the v17 crypto sleeve instead of
        voltgt). CSV schema is unchanged — m1_nav/m2_nav columns are generic.
    """
    if hasattr(algo, "_record_sleeve_split"):
        return

    algo._sleeve_m1_label = m1_label
    algo._sleeve_m2_label = m2_label

    # algo._sleeve_split[sym] -> (m1_share, m2_share) where m1+m2 == 1.
    # Restored from Object Store on init so we don't lose attribution after
    # a restart between rebalances.
    algo._sleeve_split = {}
    algo._sleeve_cash_split_m1 = 0.5  # default 50/50 cash split
    algo._sleeve_w_m1 = 0.5
    algo._sleeve_w_m2 = 0.5

    # Try to restore prior split state — important for the period between
    # restart and next rebalance, where we'd otherwise default to 50/50.
    try:
        if algo.object_store.contains_key(_STATE_KEY):
            import json
            payload = json.loads(algo.object_store.read(_STATE_KEY) or "{}")
            algo._sleeve_split = {k: tuple(v) for k, v in payload.get("split", {}).items()}
            algo._sleeve_cash_split_m1 = float(payload.get("cash_m1_share", 0.5))
            algo._sleeve_w_m1 = float(payload.get("w_m1", 0.5))
            algo._sleeve_w_m2 = float(payload.get("w_m2", 0.5))
            algo.log(
                f"[sleeve_attr] restored split state ({len(algo._sleeve_split)} symbols, "
                f"cash_m1_share={algo._sleeve_cash_split_m1:.3f})"
            )
    except Exception as e:
        algo.log(f"[sleeve_attr] state restore skipped: {e}")

    def _record_sleeve_split(
        m1_dollar_per_sym: dict,
        m2_dollar_per_sym: dict,
        cash_m1: float = 0.0,
        cash_m2: float = 0.0,
        w_m1: float = 0.5,
        w_m2: float = 0.5,
    ) -> None:
        """Call from _monthly_rebalance AFTER computing m1/m2 dollar targets
        and BEFORE set_holdings(). Stores the per-symbol attribution shares
        so the daily snapshot can compute m1_nav vs m2_nav."""
        split = {}
        all_syms = set(m1_dollar_per_sym) | set(m2_dollar_per_sym)
        for sym in all_syms:
            m1 = float(m1_dollar_per_sym.get(sym, 0.0))
            m2 = float(m2_dollar_per_sym.get(sym, 0.0))
            total = m1 + m2
            if total > 0:
                split[str(sym)] = (m1 / total, m2 / total)
            else:
                split[str(sym)] = (0.5, 0.5)  # both want zero -> conventional 50/50
        algo._sleeve_split = split

        cash_total = cash_m1 + cash_m2
        if cash_total > 0:
            algo._sleeve_cash_split_m1 = cash_m1 / cash_total
        else:
            # Fall back to the global w_m1 / w_m2 weights (e.g., 50/50)
            algo._sleeve_cash_split_m1 = w_m1 / (w_m1 + w_m2) if (w_m1 + w_m2) > 0 else 0.5
        algo._sleeve_w_m1 = float(w_m1)
        algo._sleeve_w_m2 = float(w_m2)

        # Persist so a restart between rebalances doesn't lose attribution
        try:
            import json
            algo.object_store.save(_STATE_KEY, json.dumps({
                "split": {sym: list(v) for sym, v in split.items()},
                "cash_m1_share": algo._sleeve_cash_split_m1,
                "w_m1": algo._sleeve_w_m1,
                "w_m2": algo._sleeve_w_m2,
            }))
        except Exception as e:
            algo.log(f"[sleeve_attr] state persist failed: {e}")

        algo.log(
            f"[sleeve_attr] split recorded: {len(split)} symbols, "
            f"w_m1={w_m1:.2%} w_m2={w_m2:.2%} cash_m1_share={algo._sleeve_cash_split_m1:.2%}"
        )

    def _snapshot_sleeve_attribution() -> None:
        """Compute current m1_nav / m2_nav using the last-rebalance split,
        log + plot + append to Object Store CSV. Idempotent on errors."""
        if getattr(algo, "is_warming_up", False):
            return
        try:
            total_nav = float(algo.portfolio.total_portfolio_value)
            cash = float(algo.portfolio.cash)
        except Exception as e:
            algo.log(f"[sleeve_attr] portfolio read failed: {e}")
            return
        if total_nav <= 0:
            return

        m1_nav = 0.0
        m2_nav = 0.0
        # Iterate over invested positions
        for sym, holding in algo.portfolio.items():
            if not holding.invested:
                continue
            try:
                val = float(holding.holdings_value)
            except Exception:
                continue
            sym_str = str(sym)
            m1_share, m2_share = algo._sleeve_split.get(sym_str, (0.5, 0.5))
            m1_nav += val * m1_share
            m2_nav += val * m2_share

        # Cash attributed per last-rebal cash split
        m1_cash_share = algo._sleeve_cash_split_m1
        m1_nav += cash * m1_cash_share
        m2_nav += cash * (1.0 - m1_cash_share)

        m1_pct = (m1_nav / total_nav * 100.0) if total_nav > 0 else 0.0
        m2_pct = (m2_nav / total_nav * 100.0) if total_nav > 0 else 0.0

        algo.log(
            f"[sleeve_attr] {algo.time:%Y-%m-%d}: "
            f"M1({algo._sleeve_m1_label})=${m1_nav:,.2f} ({m1_pct:.1f}%) "
            f"M2({algo._sleeve_m2_label})=${m2_nav:,.2f} ({m2_pct:.1f}%) "
            f"total=${total_nav:,.2f}"
        )

        # Append to Object Store CSV
        try:
            existing = ""
            if algo.object_store.contains_key(_STORE_KEY):
                existing = algo.object_store.read(_STORE_KEY) or ""
            if not existing.strip().startswith("date,"):
                existing = _CSV_HEADER
            row = (
                f"{algo.time.isoformat()},{m1_nav:.2f},{m2_nav:.2f},"
                f"{cash:.2f},{total_nav:.2f},{m1_pct:.4f},{m2_pct:.4f}\n"
            )
            algo.object_store.save(_STORE_KEY, existing + row)
        except Exception as e:
            algo.log(f"[sleeve_attr] CSV persist failed: {e}")

        # Live chart for the QC live UI (separate chart so it doesn't clutter
        # the default Strategy Equity series).
        try:
            algo.plot("Sleeve NAV", f"M1 {algo._sleeve_m1_label}",  m1_nav)
            algo.plot("Sleeve NAV", f"M2 {algo._sleeve_m2_label}", m2_nav)
            algo.plot("Sleeve Allocation", f"M1 {algo._sleeve_m1_label} %",  m1_pct)
            algo.plot("Sleeve Allocation", f"M2 {algo._sleeve_m2_label} %", m2_pct)
        except Exception:
            pass

    algo._record_sleeve_split = _record_sleeve_split
    algo._snapshot_sleeve_attribution = _snapshot_sleeve_attribution

    # Schedule daily snapshot 30min BEFORE market close so prices are still
    # active and the snapshot captures the end-of-day state.
    def _daily_snapshot():
        if not getattr(algo, "is_warming_up", False):
            try:
                _snapshot_sleeve_attribution()
            except Exception as e:
                algo.log(f"[sleeve_attr] snapshot raised: {e}")

    algo.schedule.on(
        algo.date_rules.every_day("SPY"),
        algo.time_rules.before_market_close("SPY", 30),
        _daily_snapshot,
    )
    algo.log(
        "[sleeve_attr] tracker wired — daily snapshot 30min before market close, "
        f"persists to {_STORE_KEY}, live charts: 'Sleeve NAV' + 'Sleeve Allocation'"
    )
