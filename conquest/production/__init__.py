"""Shared production-hardening utilities for Conquest Lean algorithms.

Every production-deployable Lean project (cstag_voltgt_combined,
voltgt_standalone, cstag, cstability, cgrowth, cf) should call
production.harden(algo) early in initialize() to get:

  - Email alerts via algo._alert(subject, body) (reads ALERT_EMAIL parameter)
  - Slippage model setup (reads SLIPPAGE_MODEL parameter)
  - State persistence helpers algo._persist_state() / algo._restore_state()
  - Object Store data freshness check + alert hook
  - In-QC FRED data refresh (daily scheduled task)
  - Standard runtime statistics

Each piece is independently togglable via parameters so a project can opt
into only what it needs. Defaults are safe-on for the things that matter
most operationally.
"""
from .alerts   import attach_alert
from .slippage import attach_slippage
from .state    import attach_state_persistence
from .freshness import attach_freshness_check
from .macro_refresh import attach_in_qc_refresh
from .rebal_cost import attach_rebal_cost_tracker
from .sleeve_attribution import attach_sleeve_attribution
from .halt_on_stale import attach_halt_on_stale
from .weekly_health import attach_weekly_health_email


def harden(algo, *, state_key: str | None = None,
           enable_in_qc_refresh: bool = True,
           freshness_keys: list[str] | None = None,
           sleeve_m1_label: str = "cstag",
           sleeve_m2_label: str = "voltgt") -> None:
    """One-call hardening for a Conquest live algorithm.

    Call from initialize() AFTER sleeves are set up but BEFORE any other
    schedule.on() so the production schedules run before strategy logic.

    Parameters:
      state_key: Object Store key for persisted runtime state. Default is
        derived from algo class name. **For combined_v3 pass explicitly**
        (e.g. ``state_key="combined_v3/runtime_state.json"``) since its
        RegimeRotator class name collides with regime's.
      enable_in_qc_refresh: If True (default), schedules daily FRED refresh.
        Disabled automatically if FRED_API_KEY parameter is empty.
      freshness_keys: list of Object Store keys whose CSV last-row dates to
        check on warmup-finish. Default checks the cstability + voltgt set.
      sleeve_m1_label / sleeve_m2_label: labels used in sleeve_attribution
        log lines and live charts. Default cstag/voltgt for regime;
        combined_v3 should pass "cstag" + "v17" since it pairs cstag with
        the v17 crypto sleeve instead of voltgt.
    """
    attach_alert(algo)
    attach_slippage(algo)
    attach_state_persistence(algo, state_key=state_key)
    attach_freshness_check(algo, keys=freshness_keys)
    if enable_in_qc_refresh:
        attach_in_qc_refresh(algo)
    attach_rebal_cost_tracker(algo)
    attach_sleeve_attribution(algo, m1_label=sleeve_m1_label, m2_label=sleeve_m2_label)
    attach_halt_on_stale(algo)
    attach_weekly_health_email(algo)
    _smoke_test(algo)
    algo.log("[production] hardened — alerts, slippage, state, freshness, refresh, rebal_cost, sleeve_attr, halt_on_stale, weekly_health wired")


def _smoke_test(algo) -> None:
    """Runs at deploy time to catch integration bugs before scheduled tasks
    can hit them. Each check is wrapped so a failure logs a warning but
    does NOT crash initialize() — surfaces the bug at init (recoverable)
    instead of inside a scheduled callback (where it kills the live algo).

    Currently checks:
      - _alert() accepts dedup_key kwarg (the exact bug that took down
        a live deploy on 2026-05-12)
      - _alert() accepts severity kwarg (v5: log-only by default, email
        only on severity="critical")
      - _persist_state() can write a payload without raising
      - _check_data_freshness() can be called without raising
    """
    # Check 1a: _alert signature compatibility with conquest.production helpers
    try:
        if hasattr(algo, "_alert"):
            algo._alert("__harden_smoke",
                        "init-time signature check — safe to ignore in logs",
                        dedup_key="__harden_smoke")
            algo.log("[production] ✓ smoke: _alert(dedup_key=) signature OK")
    except TypeError as e:
        algo.log(f"[production] ⚠ SMOKE FAIL: _alert signature mismatch — {e}")
        algo.log("[production]    → conquest.production helpers WILL crash at runtime if they call _alert(dedup_key=...)")
        algo.log("[production]    → fix: update the project's _alert to accept '**_' catch-all or 'dedup_key' kwarg")
    except Exception as e:
        algo.log(f"[production] ⚠ smoke: _alert raised non-TypeError exception: {e}")

    # Check 1b: _alert accepts severity kwarg (v5 alert refactor)
    try:
        if hasattr(algo, "_alert"):
            algo._alert("__harden_smoke_severity",
                        "init-time severity-kwarg check — log-only",
                        dedup_key="__harden_smoke_severity",
                        severity="info")
            algo.log("[production] ✓ smoke: _alert(severity=) signature OK")
    except TypeError as e:
        algo.log(f"[production] ⚠ SMOKE FAIL: _alert severity kwarg missing — {e}")
        algo.log("[production]    → fix: update the project's _alert to accept 'severity' kwarg")
    except Exception as e:
        algo.log(f"[production] ⚠ smoke: _alert(severity=) raised: {e}")

    # Check 2: state persistence round-trip works (write only, no read)
    try:
        if hasattr(algo, "_persist_state"):
            ok = algo._persist_state()
            algo.log(f"[production] ✓ smoke: _persist_state() returned {ok}")
    except Exception as e:
        algo.log(f"[production] ⚠ SMOKE FAIL: _persist_state raised — {e}")

    # Check 3: freshness check is callable
    try:
        if hasattr(algo, "_check_data_freshness"):
            stale = algo._check_data_freshness()
            algo.log(f"[production] ✓ smoke: _check_data_freshness() returned {len(stale)} stale key(s)")
    except Exception as e:
        algo.log(f"[production] ⚠ SMOKE FAIL: _check_data_freshness raised — {e}")
