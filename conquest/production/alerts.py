"""Email + log alert helper for live algorithms.

Attaches algo._alert(subject, body, *, severity='info'|'critical') which:
  - Always writes a [ALERT] log line (visible in QC logs)
  - Fires an email ONLY when severity='critical' AND ALERT_EMAIL is set
  - Catches email-send exceptions so a flaky notification provider can't
    crash the live algo

Also attaches algo._critical_alert(subject, body) as a convenience wrapper.

The Sunday-night watchdog email rolls up severity='info' alerts from the
QC live logs so the operator sees them weekly without per-event spam.

Behavior note (v5 refactor, 2026-05-12): the default severity is "info"
(log-only). Any algorithm using this shared factory will NOT send email
unless it explicitly calls _critical_alert() or passes severity="critical".
Today only cstag_voltgt_combined has a critical-alert path (its 4 DD
alerts). Other Conquest projects (cstability, cgrowth, cf, voltgt_standalone)
are log-only by default — if any of them is deployed live and needs
critical-channel events of its own (DD-20%, RuntimeError handler, etc.),
add explicit _critical_alert() call sites to that project's main.py.
"""
from __future__ import annotations


def attach_alert(algo) -> None:
    """Wire algo._alert() / algo._critical_alert() and algo._alert_state for
    spurious-dedup tracking. Safe to call multiple times — idempotent."""
    if hasattr(algo, "_alert"):
        return  # already attached by project-specific code

    algo._alert_email = (algo.get_parameter("ALERT_EMAIL") or "").strip()
    algo._alert_state = {}  # key → last-fired-value, for spurious-dedup

    def _alert(subject: str, body: str, *,
               dedup_key: str | None = None,
               severity: str = "info") -> None:
        # If dedup_key is given, only fire when the body changes vs last fire
        if dedup_key is not None:
            prev = algo._alert_state.get(dedup_key)
            if prev == body:
                return
            algo._alert_state[dedup_key] = body
        prefixed = f"[ALERT] {subject}: {body}"
        algo.log(prefixed)
        if severity == "critical" and algo._alert_email:
            try:
                algo.notify.email(algo._alert_email, f"[conquest] {subject}", body)
            except Exception as e:
                algo.log(f"[alerts] email send failed: {e}")

    def _critical_alert(subject: str, body: str, *, dedup_key: str | None = None) -> None:
        _alert(subject, body, dedup_key=dedup_key, severity="critical")

    algo._alert = _alert
    algo._critical_alert = _critical_alert
    if algo._alert_email:
        algo.log(f"[alerts] critical-only email alerts to {algo._alert_email}")
    else:
        algo.log("[alerts] no ALERT_EMAIL configured — log-only alerts")
