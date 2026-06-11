"""Halt-on-stale guard.

Refuses to fire rebalances if we haven't successfully talked to FRED in
SIGNAL_HALT_DAYS (default 100) days. Existing positions are held; no
liquidation. The algorithm continues running so the daily refresh schedule
can retry — once a refresh succeeds, the next _daily_check unlocks normal
trading automatically.

The "freshness" metric is intentionally NOT the CSV's last_date row. That
would conflate two very different failure modes:

  (a) we've lost FRED access (network down, account banned, API key
      revoked) — TRULY dangerous, regime classifier is running on truly
      stale signals → HALT.

  (b) FRED's series itself hasn't released new data yet (e.g., GDP is
      quarterly, hasn't dropped this quarter's number) — NOT dangerous,
      we've confirmed FRED's current state via successful HTTP calls,
      we're just waiting for the BEA to publish → CONTINUE.

The right metric is ``algo.last_refresh_success_at`` — the timestamp of
the most recent ``refresh_macro_signals() == True`` call. macro_refresh.py
stamps this on every successful refresh. state.py persists it across
restarts. If FRED is reachable, daily refreshes keep this current even
during quarterly GDP gaps; if FRED is unreachable, this stops advancing
and eventually crosses the halt threshold.

Alerts:
  - Warning at (SIGNAL_HALT_DAYS - SIGNAL_HALT_WARN_DAYS) days: "data is
    getting stale, will halt at X days" — dedup_key=stale_warn so it
    fires once per stale episode, not daily.
  - Critical at SIGNAL_HALT_DAYS days: "HALTED — no rebalances until
    refresh succeeds" — dedup_key=stale_halt.
  - Info on recovery: "refresh succeeded — trading resumed" — only fires
    if we'd previously been halted (tracked via algo._was_halted).
"""
from __future__ import annotations
from datetime import datetime


def attach_halt_on_stale(algo) -> None:
    """Wire algo._signals_too_stale_to_trade() — call from the project's
    _daily_check at the top to short-circuit before any rebalance."""
    if hasattr(algo, "_signals_too_stale_to_trade"):
        return

    if not hasattr(algo, "last_refresh_success_at"):
        algo.last_refresh_success_at = None
    if not hasattr(algo, "_was_halted_on_stale"):
        algo._was_halted_on_stale = False

    try:
        halt_days = int(algo.get_parameter("SIGNAL_HALT_DAYS") or "100")
    except (ValueError, TypeError):
        halt_days = 100
    try:
        warn_days = int(algo.get_parameter("SIGNAL_HALT_WARN_DAYS") or "3")
    except (ValueError, TypeError):
        warn_days = 3
    algo._halt_days = halt_days
    algo._halt_warn_days = warn_days

    def _safe_alert(subject, body, *, severity="info", dedup_key=None):
        if not hasattr(algo, "_alert"):
            algo.log(f"[halt_on_stale] {subject}: {body}")
            return
        try:
            algo._alert(subject, body, dedup_key=dedup_key, severity=severity)
        except TypeError:
            try:
                algo._alert(subject, body)
            except Exception as e:
                algo.log(f"[halt_on_stale] alert call failed: {e}")
        except Exception as e:
            algo.log(f"[halt_on_stale] alert call failed: {e}")

    def _signals_too_stale_to_trade() -> bool:
        """Return True if the algorithm should refuse to rebalance because
        our FRED data is stale. Logs + alerts; recovery alerts fire too.

        Returns False (i.e., safe to trade) in these cases:
          - SIGNAL_HALT_DAYS <= 0: guard disabled
          - last_refresh_success_at is None AND we just started: grace period
            (waiting for first successful refresh; existing cache is assumed OK)
          - days_since_refresh < halt_days: not stale enough to halt
        """
        if halt_days <= 0:
            return False

        last_ok = getattr(algo, "last_refresh_success_at", None)

        # Normalize: state.py may restore as ISO string if datetime detection
        # didn't kick in. Try to parse.
        if isinstance(last_ok, str):
            try:
                last_ok = datetime.fromisoformat(last_ok)
                algo.last_refresh_success_at = last_ok
            except Exception:
                last_ok = None

        if last_ok is None:
            # Either truly first deploy (no successful refresh yet) OR the
            # field wasn't persisted in a prior version. Don't halt — the
            # startup refresh in attach_in_qc_refresh will fix this on the
            # next call, and the daily schedule retries every 24h.
            return False

        try:
            days_since = (algo.time - last_ok).days
        except Exception:
            return False

        if days_since >= halt_days:
            # HALT — refuse to rebalance
            if not algo._was_halted_on_stale:
                _safe_alert(
                    f"regime HALTED: FRED unreachable for {days_since}d",
                    f"Algorithm has not successfully refreshed FRED data in "
                    f"{days_since} days (threshold = {halt_days}d). All "
                    f"rebalances suspended. Existing positions held untouched. "
                    f"Daily refresh schedule continues retrying every market_close + 30min. "
                    f"Trading auto-resumes when refresh succeeds.\n\n"
                    f"To diagnose: check whether the FRED API key is still valid "
                    f"(https://fred.stlouisfed.org/docs/api/api_key.html), and "
                    f"whether QC's outbound network can reach api.stlouisfed.org.",
                    severity="critical",
                    dedup_key="stale_halt",
                )
                algo._was_halted_on_stale = True
            algo.log(
                f"[halt_on_stale] HALT: last_refresh_success_at={last_ok} "
                f"({days_since}d ago >= {halt_days}d threshold) — skipping rebalance"
            )
            return True

        # Warning band — fire one alert N days before halt
        if days_since >= (halt_days - warn_days):
            _safe_alert(
                f"regime data getting stale ({days_since}d / halt at {halt_days}d)",
                f"FRED has not been successfully refreshed in {days_since} days. "
                f"Trading will HALT at {halt_days}d (in {halt_days - days_since} days) "
                f"unless a refresh succeeds. The daily schedule keeps retrying "
                f"every market_close + 30min.",
                severity="warning",
                dedup_key="stale_warn",
            )
            algo.log(
                f"[halt_on_stale] WARN: last_refresh={last_ok} "
                f"({days_since}d ago, halt at {halt_days}d) — still trading"
            )
            return False

        # Healthy: if we were previously halted, send recovery alert
        if algo._was_halted_on_stale:
            _safe_alert(
                "regime trading RESUMED",
                f"FRED refresh has succeeded again. Last successful refresh: "
                f"{last_ok} ({days_since}d ago). Rebalances will fire normally "
                f"on the next _daily_check.",
                severity="info",
                dedup_key="stale_recovered",
            )
            algo._was_halted_on_stale = False

        return False

    algo._signals_too_stale_to_trade = _signals_too_stale_to_trade
    algo.log(
        f"[halt_on_stale] guard wired: halt_days={halt_days}, "
        f"warn_days={warn_days} (warning fires at {halt_days - warn_days}d)"
    )
