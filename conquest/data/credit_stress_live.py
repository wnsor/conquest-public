"""In-algorithm HYG/IEF credit-stress computation (Option 4 of the
2026-05-21 credit-signal review).

Replaces the legacy yfinance-sourced ``conquest/credit/hyg_ief_spread.csv``
Object Store CSV with a direct in-algo computation using QC's authoritative
ETF price subscription. Same mathematical formula as the offline CSV
producer (``conquest.data.vix_term.credit_stress_proxy``):

    spread = log(HYG_t / HYG_{t-60}) - log(IEF_t / IEF_{t-60})

Sign convention: NEGATIVE values = HYG underperforming IEF = credit stress.
Vote fires when spread < threshold (default -0.05).

Why this exists in its own module:
  - Lean's per-file char limit (64,000) — keeps the CstabilitySleeve in
    regime/sleeves.py and combined_v3/sleeves.py under the limit
  - Reusable across any project that wants in-algo credit-stress signals
  - Single source of truth for the credit-stress formula

Wiring (project sleeves.py):
    from conquest.data.credit_stress_live import (
        subscribe_credit_symbols, compute_credit_stress,
    )

    def initialize_store(self):
        ...
        subscribe_credit_symbols(self.algo)

    def _vote_credit_stress(self):
        s = compute_credit_stress(self.algo, self.CREDIT_LOOKBACK_DAYS)
        return s is not None and s < self.CREDIT_STRESS_THRESHOLD
"""
from __future__ import annotations


HYG_TICKER = "HYG"
IEF_TICKER = "IEF"
HISTORY_BUFFER = 5  # request lookback + buffer for non-trading days


def subscribe_credit_symbols(algo) -> None:
    """Idempotently subscribe HYG + IEF for the credit-stress vote.

    Sets algo.hyg_sym / algo.ief_sym (or None on failure). Safe to call
    multiple times — re-running initialize_store() (e.g., after a daily
    FRED refresh reload) doesn't duplicate subscriptions.
    """
    if getattr(algo, "_credit_symbols_added", False):
        return
    algo._credit_symbols_added = True
    try:
        from AlgorithmImports import Resolution
        algo.hyg_sym = algo.add_equity(HYG_TICKER, Resolution.DAILY).symbol
        algo.ief_sym = algo.add_equity(IEF_TICKER, Resolution.DAILY).symbol
        algo.log(
            f"[cstab] subscribed {HYG_TICKER} + {IEF_TICKER} for in-algo "
            "credit-stress vote (60d log spread)"
        )
    except Exception as e:
        algo.log(
            f"[cstab] HYG+IEF subscription failed: {e}; credit vote will return False"
        )
        algo.hyg_sym = None
        algo.ief_sym = None


def compute_credit_stress(algo, lookback_days: int = 60) -> float | None:
    """Compute the current HYG/IEF 60-day log-return spread.

    Returns:
        - float (negative = credit stress, positive = risk-on, ~0 = neutral)
        - None on any failure (missing subscription, insufficient history,
          parse error, division-by-zero)

    Fail-safe by design — callers should treat None as "vote off". Avoids
    spurious defensive votes from transient infrastructure glitches.
    """
    hyg_sym = getattr(algo, "hyg_sym", None)
    ief_sym = getattr(algo, "ief_sym", None)
    if hyg_sym is None or ief_sym is None:
        return None
    try:
        from AlgorithmImports import Resolution
        bars = lookback_days + HISTORY_BUFFER
        h = algo.history(hyg_sym, bars, Resolution.DAILY)
        i = algo.history(ief_sym, bars, Resolution.DAILY)
    except Exception as e:
        algo.log(f"[cstab] credit history fetch failed: {e}; vote off")
        return None
    if h is None or i is None:
        return None
    try:
        h_close = h["close"] if "close" in h.columns else h.iloc[:, 0]
        i_close = i["close"] if "close" in i.columns else i.iloc[:, 0]
    except Exception:
        return None
    if len(h_close) < lookback_days + 1 or len(i_close) < lookback_days + 1:
        return None  # insufficient history (probably warmup)
    try:
        import math
        return (
            math.log(float(h_close.iloc[-1]) / float(h_close.iloc[-(lookback_days + 1)]))
            - math.log(float(i_close.iloc[-1]) / float(i_close.iloc[-(lookback_days + 1)]))
        )
    except (ValueError, ZeroDivisionError) as e:
        algo.log(f"[cstab] credit spread compute failed: {e}; vote off")
        return None
