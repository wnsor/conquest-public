"""Render data_refresh_report.json as a GitHub-friendly markdown freshness table.

Used by the "Summarize results" step of refresh_data.yml / refresh_data_weekly.yml
so the workflow-run Summary in GitHub answers, explicitly and at a glance:
  - WHEN did we last attempt a pull?            → "Last pull attempt" (run timestamp)
  - WHAT is the last data date we actually HAVE? → "Last data date" per source
  - did THIS attempt refresh each source?        → "This run" status column

Critically, "Last data date" is the date of the data ON DISK (the orchestrator's
`data_date`, read post-run regardless of success) — NOT just what this run wrote.
So a source whose daily pull FAILED still shows the date of the data we already
have. This matters because FRED series (GDP quarterly, CPI/UNRATE monthly) don't
change daily: a failed daily pull is fine as long as the data we have is current
for that series' cadence (its "This run" will be ❌ but its "Last data date" the
real, still-usable date).

Two source shapes have no dated CSV and show their shape instead of a date:
  - cache : FRED/BLS parquet cache (fred_macro) — judge by "This run" status.
  - list  : membership snapshots (sp500, leveraged_etfs, acwx_top).

Usage:  python scripts/data/render_refresh_summary.py [report.json]
Writes markdown to stdout. Falls back gracefully if the report is unreadable.
"""
from __future__ import annotations

import json
import sys

EMOJI = {"ok": "✅", "stale": "⚠️", "failed": "❌", "skipped": "⏭️"}
CACHE_SOURCES = {"fred_macro"}                       # parquet cache, no dated CSV
LIST_SOURCES = {"sp500", "leveraged_etfs", "acwx_top"}  # membership snapshots

# Approximate cadence at which each source's UPSTREAM API publishes new data —
# so a "Last data date" can be judged (a monthly series dated ~3 weeks ago is
# current, not stale). Derived sources update daily but inherit their slowest
# input's effective cadence (noted in parens).
CADENCE = {
    "fred_macro":      "FRED — mo/qtr",
    "vix":             "daily",
    "t10y2y":          "daily (FRED)",
    "unrate":          "monthly (BLS)",
    "regime":          "daily (macro: mo/qtr)",
    "credit_vix_term": "daily (CPI: monthly)",
    "votes":           "daily",
    "sp500":           "~quarterly",
    "leveraged_etfs":  "infrequent",
    "acwx_top":        "infrequent",
    "form4":           "daily (SEC)",
    "gdelt":           "daily",
    "earnings":        "daily/event",
    "finra_si":        "biweekly (FINRA)",
    "options_screen":  "daily",
}


def _last_data_date(r: dict) -> str:
    """The last data date we HAVE on disk for this source (set even when this
    run failed). Falls back to cache/list labels or — for a dated source we
    don't have any data for yet."""
    dd = r.get("data_date") or r.get("max_date")
    if dd:
        return str(dd)[:10]
    s = r.get("source")
    if s in CACHE_SOURCES:
        return "cache"
    if s in LIST_SOURCES:
        return "list"
    return "—"


def _rows_cell(r: dict) -> str:
    v = r.get("data_rows")
    if v is None:
        v = r.get("rows")
    return str(v) if v else "—"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data_refresh_report.json"
    try:
        rep = json.loads(open(path).read())
    except Exception as e:
        print(f"_(could not read {path}: {e})_")
        return 0

    # Drop the internal "push" pseudo-source — it's the Object Store upload, not
    # a data feed.
    results = [r for r in rep.get("results", []) if r.get("source") != "push"]
    out: list[str] = []

    ts = rep.get("run_timestamp", "")
    if ts:
        out.append(f"**Last pull attempt:** {ts}")

    # One-line verdict, by THIS-run status.
    failed = [r.get("source") for r in results if r.get("status") == "failed"]
    stale = [r.get("source") for r in results if r.get("status") == "stale"]
    ok_n = sum(1 for r in results if r.get("status") == "ok")
    parts = []
    if failed:
        parts.append(f"❌ {len(failed)} failed ({', '.join(failed)})")
    if stale:
        parts.append(f"⚠️ {len(stale)} stale ({', '.join(stale)})")
    parts.append(f"✅ {ok_n} fresh")
    out.append("**This run:** " + " · ".join(parts))
    if failed:
        out.append("> A ❌ source did NOT refresh this run — its **Last data date** "
                   "below is the data we already have (unchanged, not lost). FRED "
                   "series are monthly/quarterly, so that data is usually still current.")
    out.append("")

    # Per-source table: the data we HAVE + whether this run updated it.
    out.append("| Source | Last data date | Updates | Rows | This run |")
    out.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: str(r.get("source", ""))):
        status = r.get("status", "")
        src = r.get("source", "?")
        out.append(
            f"| {src} "
            f"| {_last_data_date(r)} "
            f"| {CADENCE.get(src, '—')} "
            f"| {_rows_cell(r)} "
            f"| {EMOJI.get(status, '')} {status} |"
        )
    out.append("")
    out.append("_`Last data date` = newest date in the data on disk (shown even if this "
               "run failed) · `Updates` = how often the upstream API publishes new data "
               "(so judge the date against this) · `This run` = did today's pull refresh it._")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
