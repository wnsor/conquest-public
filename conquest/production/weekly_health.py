"""Weekly Sunday health-summary email.

Sends one email every Sunday at 12:00 UTC summarizing FRED refresh
reachability over the past 7 days. Specifically calls out days where
T10Y2Y (the daily series) couldn't be fetched — these are the unambiguous
"we lost FRED access" signal the user cares about.

The email body lists, for each day in the past 7 days, the status of
each FRED series (GDP, CPI, UNRATE, T10Y2Y) plus a per-series success
rate and a recommendation banner.

Reads from: ``conquest/refresh_log/attempts.csv`` (written by
macro_refresh.py at every refresh cycle, success or fail).

Sends via algo._alert(...) at severity="info" so it shows up as a regular
weekly status email, not a critical alert. dedup_key includes the ISO
week so each week's email is distinct.
"""
from __future__ import annotations
from datetime import datetime, timedelta


REFRESH_LOG_KEY = "conquest/refresh_log/attempts.csv"


def attach_weekly_health_email(algo) -> None:
    """Schedule a Sunday 12:00 UTC email summarizing the past week's FRED
    reachability. Idempotent."""
    if hasattr(algo, "_send_weekly_health_email"):
        return

    def _read_refresh_log() -> list[dict]:
        """Parse the refresh-log CSV and return a list of attempt dicts.
        Returns [] on any read or parse error (the email body will note
        the empty log)."""
        try:
            if not algo.object_store.contains_key(REFRESH_LOG_KEY):
                return []
            txt = algo.object_store.read(REFRESH_LOG_KEY) or ""
        except Exception as e:
            algo.log(f"[weekly_health] log read failed: {e}")
            return []
        lines = txt.strip().split("\n")
        if len(lines) < 2:
            return []
        header = lines[0].split(",")
        out = []
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            try:
                row["_dt"] = datetime.fromisoformat(row["attempt_time"])
            except Exception:
                continue
            out.append(row)
        return out

    def _send_weekly_health_email() -> None:
        """Compose + send the weekly summary. No-op if algorithm is warming
        up or if today is not Sunday (the scheduler fires every day; we
        gate inside the callback)."""
        if getattr(algo, "is_warming_up", False):
            return
        # algo.time.weekday(): Monday=0, Sunday=6
        if algo.time.weekday() != 6:
            return

        now = algo.time
        week_start = now - timedelta(days=7)
        attempts = [a for a in _read_refresh_log() if a["_dt"] >= week_start]

        # Group by date
        by_day: dict[str, list[dict]] = {}
        for a in attempts:
            day = a["_dt"].strftime("%Y-%m-%d (%a)")
            by_day.setdefault(day, []).append(a)

        # T10Y2Y is the canonical "did we reach FRED today" signal —
        # released daily, so any business day with failed status is a
        # real miss the user wants to see.
        t10y2y_misses = []
        per_series_ok = {"gdp": 0, "cpi": 0, "unrate": 0, "t10y2y": 0}
        per_series_total = 0
        for a in attempts:
            per_series_total += 1
            for col, key in (
                ("gdp_status", "gdp"),
                ("cpi_status", "cpi"),
                ("unrate_status", "unrate"),
                ("t10y2y_status", "t10y2y"),
            ):
                if a.get(col, "") == "success":
                    per_series_ok[key] += 1
            if a.get("t10y2y_status", "") != "success":
                t10y2y_misses.append({
                    "day": a["_dt"].strftime("%Y-%m-%d (%a)"),
                    "notes": a.get("error_notes", "") or "(no error notes)",
                })

        # Banner: ALL_CLEAR / DEGRADED / HALTED
        last_ok = getattr(algo, "last_refresh_success_at", None)
        if isinstance(last_ok, str):
            try:
                last_ok = datetime.fromisoformat(last_ok)
            except Exception:
                last_ok = None
        days_since_ok = (now - last_ok).days if last_ok else None
        halt_days = getattr(algo, "_halt_days", 100)
        if days_since_ok is not None and days_since_ok >= halt_days:
            banner = f"⚠️  HALTED ({days_since_ok}d since last refresh, threshold {halt_days}d)"
        elif t10y2y_misses:
            banner = f"⚠️  DEGRADED ({len(t10y2y_misses)} T10Y2Y miss(es) this week)"
        elif not attempts:
            banner = "⚠️  NO REFRESH ATTEMPTS LOGGED THIS WEEK"
        else:
            banner = "✓  ALL CLEAR"

        # Compose body
        body_lines = [
            f"regime weekly health summary — week ending {now.strftime('%Y-%m-%d')}",
            "",
            f"Status: {banner}",
            "",
            f"Refresh attempts this week: {len(attempts)} cycles across {len(by_day)} days",
            f"Last successful FRED refresh: {last_ok.strftime('%Y-%m-%d %H:%M UTC') if last_ok else '(never recorded)'}"
            + (f" ({days_since_ok}d ago)" if days_since_ok is not None else ""),
            f"Halt threshold: {halt_days} days",
            "",
            "─── T10Y2Y daily misses ───",
        ]
        if t10y2y_misses:
            body_lines.append(f"{len(t10y2y_misses)} day(s) where T10Y2Y could not be fetched:")
            for m in t10y2y_misses:
                body_lines.append(f"  • {m['day']}: {m['notes']}")
            body_lines.append("")
            body_lines.append(
                "Action: T10Y2Y is FRED's daily 10Y-2Y Treasury spread series. "
                "A missed fetch usually means FRED rate-limited our request "
                "(transient — next day's retry should succeed) OR FRED's API "
                "endpoint had a brief outage. If misses are clustered, check "
                "your FRED key validity and QC's outbound network."
            )
        else:
            body_lines.append("No T10Y2Y misses this week — FRED reachable every refresh cycle. ✓")
        body_lines += [
            "",
            "─── Per-series success rate (week) ───",
        ]
        for label, key in (
            ("GDPC1",    "gdp"),
            ("CPIAUCSL", "cpi"),
            ("UNRATE",   "unrate"),
            ("T10Y2Y",   "t10y2y"),
        ):
            ok = per_series_ok[key]
            tot = per_series_total
            pct = (ok / tot * 100.0) if tot > 0 else 0.0
            body_lines.append(f"  {label:9s}: {ok}/{tot} ({pct:.0f}%)")

        # Per-day detail
        body_lines += ["", "─── Per-day detail ───"]
        if by_day:
            for day, day_attempts in sorted(by_day.items()):
                # Most days have one attempt (the scheduled 16:30 ET fire) +
                # possibly a startup-refresh from a restart
                statuses = []
                for a in day_attempts:
                    s = (
                        f"GDP={a.get('gdp_status', '?')[:3]} "
                        f"CPI={a.get('cpi_status', '?')[:3]} "
                        f"UNR={a.get('unrate_status', '?')[:3]} "
                        f"T10={a.get('t10y2y_status', '?')[:3]}"
                    )
                    statuses.append(s)
                body_lines.append(f"  {day}: " + " | ".join(statuses))
        else:
            body_lines.append(
                "  (no refresh attempts logged this week — possible scheduler "
                "issue or the algorithm was stopped for most of the week)"
            )

        body_lines += [
            "",
            "─── Notes ───",
            "• A 'failed' GDP/CPI is OK if FRED itself hasn't released new data yet "
            "(GDP is quarterly, CPI is monthly). The algorithm continues trading "
            "in that case because we've confirmed FRED's current state. The halt-on-stale "
            "guard fires only when we haven't SUCCESSFULLY talked to FRED for "
            f"{halt_days}+ days.",
            "• T10Y2Y misses are the canonical 'we lost FRED access' signal since "
            "T10Y2Y is daily — a miss means we couldn't reach FRED on that day, "
            "not that FRED was silent.",
            "",
            f"Algorithm live URL: https://www.quantconnect.com/project/{getattr(algo, '_qc_project_id', '')}/live",
        ]
        body = "\n".join(body_lines)

        subject = f"regime weekly health — {now.strftime('%Y-%m-%d')} ({banner.split('(')[0].strip()})"
        iso_year, iso_week, _ = now.isocalendar()
        dedup = f"weekly_health_{iso_year}W{iso_week:02d}"

        try:
            if hasattr(algo, "_alert"):
                algo._alert(subject, body, dedup_key=dedup, severity="info")
            else:
                algo.log(f"[weekly_health] {subject}\n{body}")
        except TypeError:
            # Fall back to legacy _alert signature
            try:
                algo._alert(subject, body)
            except Exception as e:
                algo.log(f"[weekly_health] alert call failed: {e}")
        except Exception as e:
            algo.log(f"[weekly_health] alert call failed: {e}")

        algo.log(
            f"[weekly_health] sent weekly summary email: "
            f"{len(attempts)} attempts, {len(t10y2y_misses)} T10Y2Y misses, banner='{banner}'"
        )

    algo._send_weekly_health_email = _send_weekly_health_email

    # Calendar-day schedule (no symbol) so it fires on Sundays even though
    # SPY is closed. 12:00 UTC = 8 AM ET (Sunday morning, user gets it with
    # coffee).
    algo.schedule.on(
        algo.date_rules.every_day(),
        algo.time_rules.at(12, 0),
        _send_weekly_health_email,
    )
    algo.log("[weekly_health] wired — Sundays at 12:00 UTC")
