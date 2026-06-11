"""Daily data-refresh orchestrator — covers EVERY external data source.

Runs every signal-data ingester and derivation script in dependency order,
then verifies the resulting CSVs cover the requested date range. Designed
to be called from .github/workflows/refresh_data.yml — but works locally too.

Architectural principles (per user directive 2026-05-26 "no gaps ever"):

  1. EVERY external data source referenced by any algorithm has automation here.
     If a new key is added to an algorithm, it MUST also be added below.
  2. Dependency order matters — derived signals (regime, votes) refresh AFTER
     their raw inputs (FRED, VIX) succeed.
  3. One failure doesn't block others — each source runs independently and
     reports its own status; final exit code is nonzero only on hard failures.
  4. The orchestrator reports `stale` (warning) when an output's max_date is
     too far behind today — surfaces silent-freeze bugs early.

Dependency graph:

    [Tier 1 — external raw]
        FRED/BLS macro ────┐
        Yahoo VIX/HYG/IEF ─┼─→ [Tier 2 — derived]
        SEC EDGAR Form 4   │       regime/daily.csv ─┐
        GDELT GKG          │       vix/daily.csv     ├─→ votes/cstability_4vote_daily.csv
        EDGAR 8-K + Yahoo  │       credit/hyg_ief    │
        FINRA SI biweekly  │       vix/term_ratio    │
        Wikipedia SP500    │       regime/probability┘
        yfinance Lev-ETFs ─┘       yield_curve/t10y2y
                                   macro/unrate

Tier 1 and the derived Tier 2 outputs are all pushed to QC Object Store at
the end, so algorithms running on QC Cloud see the same state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent.parent
STORAGE = WORKSPACE / "storage" / "conquest"


@dataclass
class RefreshResult:
    source: str
    status: str                # "ok" | "stale" | "failed" | "skipped" (THIS run)
    rows: int = 0
    max_date: str | None = None
    duration_sec: float = 0.0
    error: str | None = None
    # Set post-run by the data-inventory pass: the last data date / row count we
    # currently HAVE on disk, regardless of whether this run's pull succeeded.
    data_date: str | None = None
    data_rows: int | None = None


def _run(cmd: list[str], cwd: Path | None = None,
         env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
    """Run cmd, return (exit_code, combined_output). Streams last lines."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=cwd or WORKSPACE,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    last = "\n".join(out.splitlines()[-30:])
    print(last, flush=True)
    return proc.returncode, out


def _csv_max_date(path: Path, date_col_candidates: list[str]) -> str | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) == 0:
        return None
    for c in date_col_candidates:
        if c in df.columns:
            try:
                return str(pd.to_datetime(df[c], errors="coerce").max().date())
            except Exception:
                continue
    try:
        first = df.columns[0]
        return str(pd.to_datetime(df[first], errors="coerce").max().date())
    except Exception:
        return None


def _rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def _stale(max_date: str | None, today: date, max_lag_days: int) -> bool:
    if not max_date:
        return False
    try:
        d = pd.to_datetime(max_date).date()
        return (today - d) > timedelta(days=max_lag_days)
    except Exception:
        return False


def _result(name: str, out_path: Path, t0: float,
            date_cols: list[str], max_lag_days: int,
            today: date, rc: int = 0, error: str | None = None) -> RefreshResult:
    """Build the RefreshResult for one source.

    2026-05-27: extended with full schema + content validation via
    `_validators.validate_by_name`. Previously a subprocess exit 0 + any
    output file was treated as `ok` regardless of whether columns were
    present, the file was empty, or dates were unparseable. Multiple
    leading-indicator strategies fired 0 trades because of silent
    validation gaps like this. Now: validator runs after every successful
    ingester and the worse of (status, validator.status) is reported.
    """
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult(name, "failed", duration_sec=duration,
                             error=error or f"exit code {rc}")

    # Run the new content + schema validator (no-op for sources without a spec).
    # Use absolute import path because refresh_all.py is invoked as a script,
    # not a package module.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from _validators import validate_by_name, ValidationResult  # type: ignore
    except ImportError as e:
        print(f"  [warn] validators module not importable: {e}", flush=True)
        return RefreshResult(source=name, status="ok", duration_sec=duration,
                             rows=_rows(out_path),
                             max_date=_csv_max_date(out_path, date_cols))
    try:
        val: ValidationResult = validate_by_name(name, today)
    except Exception as e:
        # Don't let validator bugs mask the underlying refresh result
        print(f"  [warn] validator raised for {name}: {e}", flush=True)
        val = ValidationResult(name=name, status="ok", issues=[f"validator error: {e}"])

    # Legacy max_lag check is still applied (lets the orchestrator override
    # validator's max_lag_days for sources that don't have a spec yet).
    max_date = _csv_max_date(out_path, date_cols)
    legacy_stale = _stale(max_date, today, max_lag_days)

    # Combine statuses — worst wins
    status_severity = {
        "ok": 0, "missing_spec": 0,   # informational only
        "stale": 1, "empty": 2, "schema_mismatch": 3, "invalid": 4,
        "failed": 5,
    }
    legacy_status = "stale" if legacy_stale else "ok"
    final_status = max(
        (legacy_status, val.status),
        key=lambda s: status_severity.get(s, 0),
    )

    # If validator caught issues, surface them in the error field so the
    # workflow report tells us WHY a previously-silent failure is now failing.
    error_msg = None
    if val.issues:
        error_msg = "; ".join(val.issues[:3])
        print(f"  [validator] {name} status={val.status} issues={val.issues}", flush=True)

    return RefreshResult(
        source=name,
        status=final_status,
        rows=val.rows or _rows(out_path),
        max_date=val.max_date or max_date,
        duration_sec=duration,
        error=error_msg,
    )


# ─── Tier 1: external raw ────────────────────────────────────────────────

def refresh_fred_macro(today: date) -> RefreshResult:
    """Refresh FRED+BLS macro parquet cache (GDP/CPI/UNRATE/HY/etc)."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/refresh_data.py", "--refresh"],
                 timeout=1800)
    # No single CSV output; success = exit 0 and parquet cache present
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("fred_macro", "failed", duration_sec=duration,
                             error=f"exit code {rc}")
    return RefreshResult("fred_macro", "ok", duration_sec=duration)


def refresh_vix(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/export_vix.py"], timeout=600)
    return _result("vix", STORAGE / "vix" / "daily.csv",
                   t0, ["date"], max_lag_days=4, today=today, rc=rc)


def refresh_t10y2y(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/export_t10y2y.py"], timeout=600)
    return _result("t10y2y", STORAGE / "yield_curve" / "t10y2y.csv",
                   t0, ["date"], max_lag_days=5, today=today, rc=rc)


def refresh_unrate(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/export_unrate.py"], timeout=600)
    # UNRATE is monthly — release lag ~30 days
    return _result("unrate", STORAGE / "macro" / "unrate.csv",
                   t0, ["date"], max_lag_days=40, today=today, rc=rc)


def refresh_credit_vix_term_probs(today: date) -> RefreshResult:
    """Single script writes 3 files: credit/hyg_ief, vix/term_ratio, regime/probability."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/export_v10_signals.py"], timeout=900)
    # Verify the most-lagged of the 3 outputs (regime/probability requires FRED CPI)
    out_path = STORAGE / "credit" / "hyg_ief_spread.csv"
    return _result("credit_vix_term", out_path,
                   t0, ["date"], max_lag_days=4, today=today, rc=rc)


def refresh_form4(today: date) -> RefreshResult:
    t0 = time.time()
    email = os.environ.get("SEC_EDGAR_EMAIL", "conquest-research@example.com")
    # 150-min budget — leaves ~200min for GDELT + everything else within the
    # 350-min job cap. Parquet cache makes subsequent runs incremental.
    rc, _ = _run([
        sys.executable, "scripts/data/form4.py",
        "--incremental", "--email", email,
        "--start", "2018", "--end", str(today.year),
    ], timeout=9000)
    return _result("form4", STORAGE / "insider" / "form4_opportunistic_buys_daily.csv",
                   t0, ["filing_date"], max_lag_days=10, today=today, rc=rc)


def refresh_gdelt(today: date) -> RefreshResult:
    t0 = time.time()
    # 120-min budget — GDELT is network-bound (one GKG zip per day from
    # data.gdeltproject.org); this is the slow tail.
    rc, _ = _run([
        sys.executable, "scripts/ingest_gdelt_sentiment.py",
        "--start", "2018-01-01", "--end", today.strftime("%Y-%m-%d"),
    ], timeout=7200)
    return _result("gdelt", STORAGE / "sentiment" / "gdelt_daily.csv",
                   t0, ["date"], max_lag_days=3, today=today, rc=rc)


def refresh_earnings(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([
        sys.executable, "scripts/data/earnings_calendar.py",
        "--universe", "sp500",
    ], timeout=1200)
    # Forward-looking: extends INTO the future, so "max_date < today" is the
    # warning condition (calendar isn't extending forward enough)
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("earnings", "failed", duration_sec=duration,
                             error=f"exit code {rc}")
    out_path = STORAGE / "options" / "earnings_calendar.csv"
    max_date = _csv_max_date(out_path, ["earnings_date"])
    stale = False
    if max_date:
        d = pd.to_datetime(max_date).date()
        stale = (d - today) < timedelta(days=14)   # should extend ≥14d forward
    return RefreshResult("earnings",
                         "stale" if stale else "ok",
                         rows=_rows(out_path), max_date=max_date,
                         duration_sec=duration)


def refresh_finra_si(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([
        sys.executable, "scripts/data/finra_short_interest.py",
        "--start", "2018-01-01", "--end", today.strftime("%Y-%m-%d"),
    ], timeout=1800)
    return _result("finra_si", STORAGE / "options" / "finra_si_biweekly.csv",
                   t0, ["settlement_date"], max_lag_days=20, today=today, rc=rc)


# ─── Minimum-signal-set additions (2026-05-26) ────────────────────────────

def refresh_cboe_indices(today: date) -> RefreshResult:
    """VVIX + SKEW via yfinance — tail-hedge regime + vol-of-vol signals."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/data/cboe_indices.py"], timeout=300)
    return _result("cboe_indices", STORAGE / "options" / "cboe_indices_daily.csv",
                   t0, ["date"], max_lag_days=4, today=today, rc=rc)


def refresh_nfci(today: date) -> RefreshResult:
    """Chicago Fed NFCI — single best macro leading indicator."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/data/nfci.py"], timeout=300)
    return _result("nfci", STORAGE / "macro" / "nfci_weekly.csv",
                   t0, ["date"], max_lag_days=10, today=today, rc=rc)


def refresh_earnings_revisions(today: date) -> RefreshResult:
    """yfinance EPS estimate snapshots — daily append-only history."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/data/earnings_revisions.py"], timeout=600)
    return _result("earnings_revisions", STORAGE / "options" / "earnings_revisions_daily.csv",
                   t0, ["date"], max_lag_days=3, today=today, rc=rc)


def refresh_cftc_cot(today: date) -> RefreshResult:
    """CFTC Commitments of Traders — weekly institutional positioning."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/data/cftc_cot.py"], timeout=900)
    return _result("cftc_cot", STORAGE / "macro" / "cftc_cot_weekly.csv",
                   t0, ["date"], max_lag_days=10, today=today, rc=rc)


def refresh_edgar_13d(today: date) -> RefreshResult:
    """SEC 13D/G activist filings — incremental, last 90 days."""
    t0 = time.time()
    import os
    email = os.environ.get("SEC_EDGAR_EMAIL", "conquest-research@example.com")
    rc, _ = _run([sys.executable, "scripts/data/edgar_13d.py",
                  "--incremental", "--email", email,
                  "--days-back", "90"], timeout=1800)
    return _result("edgar_13d", STORAGE / "insider" / "edgar_13d_filings.csv",
                   t0, ["filing_date"], max_lag_days=14, today=today, rc=rc)


def refresh_edgar_8k_count(today: date) -> RefreshResult:
    """SEC 8-K material-disclosure rolling 14-day count per ticker."""
    t0 = time.time()
    import os
    email = os.environ.get("SEC_EDGAR_EMAIL", "conquest-research@example.com")
    rc, _ = _run([sys.executable, "scripts/data/edgar_8k_count.py",
                  "--incremental", "--email", email,
                  "--days-back", "90"], timeout=1800)
    return _result("edgar_8k_count", STORAGE / "insider" / "edgar_8k_count_daily.csv",
                   t0, ["date"], max_lag_days=7, today=today, rc=rc)


def refresh_google_trends(today: date) -> RefreshResult:
    """Google Trends per-ticker search-volume via pytrends. Best-effort —
    Google sometimes returns 429/CAPTCHA on the runner, accept partial."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/data/google_trends.py",
                  "--timeframe", "today 3-m"], timeout=900)
    # Lenient — Google Trends failures don't break the whole refresh
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("google_trends", "stale", duration_sec=duration,
                             error=f"pytrends exit {rc} (Google rate-limit common)")
    return _result("google_trends", STORAGE / "sentiment" / "google_trends_daily.csv",
                   t0, ["date"], max_lag_days=7, today=today, rc=0)


def refresh_reddit_wsb(today: date) -> RefreshResult:
    """r/wallstreetbets mention counts via PRAW. Requires REDDIT_CLIENT_ID +
    REDDIT_CLIENT_SECRET secrets to be set in the workflow."""
    t0 = time.time()
    import os
    if not (os.environ.get("REDDIT_CLIENT_ID") and os.environ.get("REDDIT_CLIENT_SECRET")):
        return RefreshResult("reddit_wsb", "skipped",
                             error="REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET env not set")
    rc, _ = _run([sys.executable, "scripts/data/reddit_wsb_mentions.py",
                  "--days-back", "1"], timeout=600)
    return _result("reddit_wsb", STORAGE / "sentiment" / "reddit_wsb_daily.csv",
                   t0, ["date"], max_lag_days=2, today=today, rc=rc)


def refresh_ipo_lockup(today: date) -> RefreshResult:
    """IPO lockup-expiry calendar via EDGAR 424B prospectus scrape."""
    t0 = time.time()
    import os
    email = os.environ.get("SEC_EDGAR_EMAIL", "conquest-research@example.com")
    rc, _ = _run([sys.executable, "scripts/data/ipo_lockup_calendar.py",
                  "--email", email, "--days-back", "365"], timeout=900)
    # Output schema doesn't have a single "date" column — verify by row count
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("ipo_lockup", "failed", duration_sec=duration,
                             error=f"exit {rc}")
    out_path = STORAGE / "options" / "ipo_lockup_calendar.csv"
    return RefreshResult("ipo_lockup", "ok" if out_path.exists() else "stale",
                         rows=_rows(out_path), duration_sec=duration)


# ─── Tier 2: derived (run AFTER Tier 1 succeeds) ─────────────────────────

def refresh_regime(today: date) -> RefreshResult:
    """Regime classifier daily output (depends on FRED macro + VIX)."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/classify_regime.py"], timeout=600)
    return _result("regime", STORAGE / "regime" / "daily.csv",
                   t0, ["date"], max_lag_days=5, today=today, rc=rc)


def refresh_votes(today: date) -> RefreshResult:
    """cstability 4-vote signal (depends on regime + credit + vix_term)."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/compute_4vote_signal.py"], timeout=600)
    return _result("votes", STORAGE / "votes" / "cstability_4vote_daily.csv",
                   t0, ["date"], max_lag_days=4, today=today, rc=rc)


# ─── Universes (changes slowly; refresh weekly but daily-safe) ───────────

def refresh_sp500(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/export_sp500_universe.py"], timeout=600)
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("sp500", "failed", duration_sec=duration,
                             error=f"exit code {rc}")
    out_path = STORAGE / "universe" / "sp500.csv"
    return RefreshResult("sp500", "ok",
                         rows=_rows(out_path), duration_sec=duration)


def refresh_leveraged_etfs(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/refresh_leveraged_etf_universe.py"],
                 timeout=600)
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("leveraged_etfs", "failed", duration_sec=duration,
                             error=f"exit code {rc}")
    out_path = STORAGE / "universe" / "leveraged_etfs.csv"
    return RefreshResult("leveraged_etfs", "ok",
                         rows=_rows(out_path), duration_sec=duration)


def refresh_acwx_top(today: date) -> RefreshResult:
    """Curated ADR list — script is idempotent (validates curated list)."""
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/build_acwx_top_universe.py"],
                 timeout=300)
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("acwx_top", "failed", duration_sec=duration,
                             error=f"exit code {rc}")
    out_path = STORAGE / "universe" / "acwx_top.csv"
    return RefreshResult("acwx_top", "ok",
                         rows=_rows(out_path), duration_sec=duration)


def refresh_options_screen(today: date) -> RefreshResult:
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/screen_options_universe.py"], timeout=900)
    duration = time.time() - t0
    if rc != 0:
        return RefreshResult("options_screen", "failed", duration_sec=duration,
                             error=f"exit code {rc}")
    out_path = STORAGE / "universe" / "options_screen_daily.csv"
    return _result("options_screen", out_path, t0,
                   ["date"], max_lag_days=4, today=today, rc=rc)


def refresh_breadth(today: date) -> RefreshResult:
    """S&P-500 breadth (% members > 200DMA) → the BULL options gate signal.

    Extends storage/conquest/breadth/sp500_above_200d_ma.csv and writes the
    algo-facing storage/conquest/leading/confidence.csv (date,leading_confidence).
    The conquest_options BULL book (PREMIUM_CAP_MODE=gate) reads confidence.csv
    from the Object Store and deploys ONLY when leading_confidence > 0.50. A
    frozen signal here makes the gate decorative — hence this must stay automated.
    Fails CLOSED (export_breadth exits non-zero if too few members download), so
    a bad fetch flags stale rather than writing a biased partial-universe breadth.
    """
    t0 = time.time()
    rc, _ = _run([sys.executable, "scripts/export_breadth.py"], timeout=600)
    out_path = STORAGE / "leading" / "confidence.csv"
    # max_lag 8d: weekly cadence; one missed Sunday run is still "fresh enough"
    # for a slow 200DMA gate (the algo's fail-safe carries forward within tolerance).
    return _result("breadth", out_path, t0,
                   ["date"], max_lag_days=8, today=today, rc=rc)


def push_to_object_store() -> RefreshResult:
    """Push all refreshed CSVs to QC Object Store via object_store_push.py."""
    t0 = time.time()
    manifest = WORKSPACE / "scripts" / "data" / "refresh_manifest.txt"
    args = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        key, rel = parts[0], parts[1]
        path = WORKSPACE / rel
        if not path.exists():
            print(f"  skip {key}: local file missing")
            continue
        args.extend([key, str(path)])
    if not args:
        return RefreshResult("push", "skipped", error="no local files to push")
    rc, _ = _run([sys.executable, "scripts/object_store_push.py", *args], timeout=900)
    duration = time.time() - t0
    return RefreshResult(
        "push", "ok" if rc == 0 else "failed",
        duration_sec=duration,
        error=None if rc == 0 else f"push exit code {rc}",
    )


# ─── Source registry ─────────────────────────────────────────────────────
# Order matters: Tier 1 first, then Tier 2 derived, then universes, then push.
HANDLERS = {
    # Tier 1a: pure-external raw (no dependencies)
    "fred_macro":      refresh_fred_macro,
    "vix":             refresh_vix,
    "t10y2y":          refresh_t10y2y,
    "unrate":          refresh_unrate,
    # Tier 1b: REGIME must run BEFORE credit_vix_term — credit_vix_term reads
    # regime/daily.csv as input. 2026-05-27 fix: was running after credit
    # which caused FileNotFoundError on the regime/daily.csv lookup.
    "regime":          refresh_regime,
    "credit_vix_term": refresh_credit_vix_term_probs,
    # Options-model sources (conquest_options reads these). Refreshed WEEKLY,
    # not daily — see .github/workflows/refresh_data_weekly.yml. The handlers
    # stay registered so the weekly job can request them via --sources.
    # (options_screen is also weekly-options but kept at the end below, after
    # the universes, since it builds off the sp500 universe.)
    "form4":           refresh_form4,
    "gdelt":           refresh_gdelt,
    "earnings":        refresh_earnings,
    "finra_si":        refresh_finra_si,
    # SHELVED 2026-06-01 — the 2026-05-26 "minimum-signal-set" additions feed
    # standby strategies (tail_hedge, retail_attention, vvix_divergence, …) that
    # were never built. No live/promoted OR options model reads them, so they're
    # removed from the refresh registry. The refresh_* functions remain defined
    # below; re-add an entry here to un-shelve. (Per user directive 2026-06-01.)
    #   "cboe_indices":       refresh_cboe_indices,
    #   "nfci":               refresh_nfci,
    #   "earnings_revisions": refresh_earnings_revisions,
    #   "cftc_cot":           refresh_cftc_cot,
    #   "edgar_13d":          refresh_edgar_13d,
    #   "edgar_8k_count":     refresh_edgar_8k_count,
    #   "google_trends":      refresh_google_trends,
    #   "reddit_wsb":         refresh_reddit_wsb,
    #   "ipo_lockup":         refresh_ipo_lockup,
    # Tier 2: derived signals (need regime + credit + vix_term to be fresh)
    "votes":           refresh_votes,
    # Universes
    "sp500":           refresh_sp500,
    "leveraged_etfs":  refresh_leveraged_etfs,
    "acwx_top":        refresh_acwx_top,
    # Options-model universe (weekly; after sp500 since it builds off it)
    "options_screen":  refresh_options_screen,
    # BULL options gate signal: S&P breadth → confidence.csv (weekly; reads the
    # pulled sp500.csv membership). Must stay automated — a frozen gate signal
    # makes the live bull book unable to tell bull from bear.
    "breadth":         refresh_breadth,
}

# Default daily set — everything. Subsets useful for testing.
DEFAULT_SOURCES = list(HANDLERS.keys())

# source → (CSV path relative to STORAGE, date column). Read post-run to record
# the last data date we HAVE on disk (existing-pulled OR freshly-refreshed),
# independent of whether THIS run's pull succeeded — so a source that failed to
# fetch today still reports the date of the data we already have. Sources absent
# here have no dated CSV: fred_macro (parquet cache) or membership lists
# (sp500, leveraged_etfs, acwx_top).
PRIMARY_CSV = {
    "vix":             ("vix/daily.csv", "date"),
    "t10y2y":          ("yield_curve/t10y2y.csv", "date"),
    "unrate":          ("macro/unrate.csv", "date"),
    "credit_vix_term": ("regime/probability.csv", "date"),  # most-lagged FRED-derived output
    "regime":          ("regime/daily.csv", "date"),
    "votes":           ("votes/cstability_4vote_daily.csv", "date"),
    "form4":           ("insider/form4_opportunistic_buys_daily.csv", "date"),
    "gdelt":           ("sentiment/gdelt_daily.csv", "date"),
    "earnings":        ("options/earnings_calendar.csv", "earnings_date"),
    "finra_si":        ("options/finra_si_biweekly.csv", "date"),
    "options_screen":  ("universe/options_screen_daily.csv", "date"),
    "breadth":         ("leading/confidence.csv", "date"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help=f"Comma-separated list. Available: {','.join(DEFAULT_SOURCES)}")
    ap.add_argument("--skip-push", action="store_true",
                    help="Don't push to Object Store after refresh (testing)")
    ap.add_argument("--report", default="data_refresh_report.json",
                    help="Write JSON report to this path")
    args = ap.parse_args()

    sources_requested = [s.strip() for s in args.sources.split(",") if s.strip()]
    today = date.today()

    print(f"=== Daily data refresh — {today} ===")
    print(f"sources: {sources_requested}")
    print(f"workspace: {WORKSPACE}")

    results: list[RefreshResult] = []
    for src in sources_requested:
        if src not in HANDLERS:
            print(f"  unknown source: {src}; skipping")
            continue
        print(f"\n--- {src} ---")
        try:
            r = HANDLERS[src](today)
        except Exception as e:
            r = RefreshResult(src, "failed", error=str(e)[:200])
        results.append(r)
        print(f"  {src}: status={r.status} rows={r.rows} "
              f"max_date={r.max_date} duration={r.duration_sec:.0f}s")

    if not args.skip_push:
        print(f"\n--- pushing to QC Object Store ---")
        try:
            r = push_to_object_store()
        except Exception as e:
            r = RefreshResult("push", "failed", error=str(e)[:200])
        results.append(r)
        print(f"  push: status={r.status} duration={r.duration_sec:.0f}s")

    # Data inventory: record the last data date we HAVE on disk per source,
    # regardless of whether this run's pull succeeded. FRED series are monthly/
    # quarterly and don't change daily, so a failed daily pull still leaves the
    # data we have current for that series' cadence — this surfaces that.
    for r in results:
        spec = PRIMARY_CSV.get(r.source)
        if spec:
            rel, col = spec
            p = STORAGE / rel
            r.data_date = _csv_max_date(p, [col])
            r.data_rows = _rows(p)

    report = {
        "run_date": str(today),
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "results": [asdict(r) for r in results],
        "summary": {
            "ok":      sum(1 for r in results if r.status == "ok"),
            "stale":   sum(1 for r in results if r.status == "stale"),
            "failed":  sum(1 for r in results if r.status == "failed"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
        },
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n=== Report ===\n{json.dumps(report['summary'], indent=2)}")

    # Exit nonzero on any failure → GitHub Actions email alert fires
    failed = [r for r in results if r.status == "failed"]
    if failed:
        print(f"\nFAILED: {[r.source for r in failed]}")
        return 1
    # Don't fail on `stale` — it's a warning condition, not a hard failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
