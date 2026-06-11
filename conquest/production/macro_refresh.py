"""Shared in-QC macro-data refresh — fetches FRED daily, runs inside QC,
no local cron dependency.

Promoted from cstag_voltgt_combined/in_qc_refresh.py to conquest/production/
so every Conquest project can opt into the same self-refresh behavior.

Usage from a project's main.py:

    from conquest.production import harden
    harden(self)   # registers the daily refresh schedule automatically

…provided FRED_API_KEY is set in config.json. If empty, refresh is skipped
and the algorithm falls back to whatever the Object Store already has.
"""
from __future__ import annotations
import json


FRED_SERIES = {
    "GDPC1":    "gdp_real",
    "CPIAUCSL": "cpi_headline",
    "UNRATE":   "unemployment_rate",
    "T10Y2Y":   "yield_curve_spread",
}
OBJECT_STORE_KEYS = {
    "regime":  "conquest/regime/daily.csv",
    "unrate":  "conquest/macro/unrate.csv",
    "t10y2y":  "conquest/yield_curve/t10y2y.csv",
}

# One row per refresh CYCLE (per scheduled fire), used by the weekly Sunday
# health email to enumerate FRED reachability over the past 7 days. T10Y2Y
# is daily so its status column is the canonical "did we reach FRED today?"
# signal the user cares about.
REFRESH_LOG_KEY = "conquest/refresh_log/attempts.csv"
REFRESH_LOG_HEADER = (
    "attempt_time,gdp_status,cpi_status,unrate_status,t10y2y_status,"
    "t10y2y_latest_obs,t10y2y_age_days,regime_csv_updated,error_notes\n"
)
# Object Store key for FRED API key. Used as a CLI-friendly fallback when
# the FRED_API_KEY config-parameter is empty (e.g. the QC UI is unreachable
# and the user can't set the project parameter via the website). To
# populate from your laptop:
#     echo -n "<your_fred_api_key>" > /tmp/fred.key
#     lean cloud object-store set --key conquest/secrets/fred_api_key --path /tmp/fred.key
#     rm /tmp/fred.key
# Object Store is project-scoped and encrypted at rest in QC. The key never
# touches git or the public log.
FRED_API_KEY_STORE_KEY = "conquest/secrets/fred_api_key"


def _read_fred_api_key(algo) -> str:
    """Read the FRED API key, trying parameter first, Object Store fallback
    second. Empty string means no key is available."""
    param_key = (algo.get_parameter("FRED_API_KEY") or "").strip()
    if param_key:
        return param_key
    # Fall back to Object Store — lets the user set the secret via lean CLI
    # when the QC web UI isn't accessible.
    try:
        if algo.object_store.contains_key(FRED_API_KEY_STORE_KEY):
            store_key = (algo.object_store.read(FRED_API_KEY_STORE_KEY) or "").strip()
            if store_key:
                return store_key
    except Exception as e:
        algo.log(f"[refresh] FRED key Object Store read failed: {e}")
    return ""


def _series_age_days(algo, idx) -> int | None:
    """Return age (in days) of the most recent observation in a pandas index,
    relative to the algorithm's current time. Returns None on failure."""
    import pandas as pd
    try:
        today = pd.Timestamp(algo.time.date())
        return int((today - pd.Timestamp(idx[-1])).days)
    except Exception:
        return None


# Retry config for FRED. FRED's free tier is 120 req/min/key but the limiter
# is bursty — back-to-back calls can return empty bodies until a brief cooldown.
# A small retry handles the bursty case without making us wait too long inside
# a Lean scheduled callback (which blocks the algo thread).
_FRED_MAX_RETRIES = 2          # try original + 2 retries = 3 attempts total
_FRED_RETRY_SLEEP_SEC = 5      # backoff between attempts


def _fred_fetch(algo, series_id: str, api_key: str):
    """Fetch a FRED series with retry/backoff. Returns a sorted pandas Series
    or None on permanent failure. Each transient failure (empty body, parse
    error, network timeout) is retried up to _FRED_MAX_RETRIES times with a
    fixed 5s backoff before giving up."""
    import time
    import pandas as pd
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json"
    )
    headers = {"User-Agent": "Conquest/1.0 (Lean live algo; )"}

    response_text = None
    payload = None
    for attempt in range(_FRED_MAX_RETRIES + 1):
        # 1: download
        try:
            response_text = algo.download(url, headers)
        except Exception as e:
            algo.log(f"[refresh] FRED {series_id} download failed (attempt {attempt+1}/{_FRED_MAX_RETRIES+1}): {e}")
            if attempt < _FRED_MAX_RETRIES:
                time.sleep(_FRED_RETRY_SLEEP_SEC)
                continue
            return None

        # 2: empty-body check (FRED rate-limiter returns 200 + empty)
        if not response_text or not response_text.strip():
            algo.log(f"[refresh] FRED {series_id}: empty response body (attempt {attempt+1}/{_FRED_MAX_RETRIES+1}, rate limited?)")
            if attempt < _FRED_MAX_RETRIES:
                time.sleep(_FRED_RETRY_SLEEP_SEC)
                continue
            return None

        # 3: JSON parse
        try:
            payload = json.loads(response_text)
            break  # success — exit retry loop
        except Exception as e:
            sample = response_text[:120].replace("\n", "\\n")
            algo.log(f"[refresh] FRED {series_id} parse failed (attempt {attempt+1}/{_FRED_MAX_RETRIES+1}): {e}; len={len(response_text)} first_chars={sample!r}")
            if attempt < _FRED_MAX_RETRIES:
                time.sleep(_FRED_RETRY_SLEEP_SEC)
                continue
            return None

    if payload is None:
        return None

    obs = payload.get("observations", [])
    if not obs:
        if "error_code" in payload or "error_message" in payload:
            algo.log(f"[refresh] FRED API error for {series_id}: {payload.get('error_message', payload)}")
        return None
    rows = []
    for o in obs:
        v = o.get("value")
        d = o.get("date")
        if v and v != "." and d:
            try:
                rows.append((pd.Timestamp(d), float(v)))
            except Exception:
                continue
    if not rows:
        return None
    idx, vals = zip(*rows)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name=series_id).sort_index()
    # Surface freshness of the FRED-side data — useful for diagnosing FRED
    # lag vs Object Store lag (different problems, different fixes).
    age = _series_age_days(algo, s.index)
    if age is not None:
        algo.log(f"[refresh] FRED {series_id}: {len(s)} obs, latest={s.index[-1].date()}, age={age}d")
    return s


def _yoy(s, periods: int):
    return (s / s.shift(periods) - 1.0) * 100.0


def _build_regime_csv(gdp_series, cpi_series) -> tuple[str, "pd.DataFrame"] | tuple[None, None]:
    """Rebuild the daily-regime CSV from FRED GDP + CPI series. Returns the
    CSV text plus the underlying DataFrame so the caller can log freshness
    (last_date, row count, etc.).

    The daily index is forward-filled to today's date because GDP is
    quarterly (lag 1-3 months) and CPI is monthly (lag 2 weeks). Without
    forward-fill, the CSV's last_date trails today by 30-90 days and the
    freshness check fires a stale-data alert every day until the next
    quarterly GDP release. Regime classification is a slow-moving signal
    (transitions take months), so today's regime label is the latest known
    label — which is what we want the algorithm to read.
    """
    import pandas as pd
    try:
        from conquest.regime.classifier import RegimeClassifier
    except Exception:
        return None, None
    gdp_yoy = _yoy(gdp_series, periods=4)
    cpi_yoy = _yoy(cpi_series, periods=12)
    classifier = RegimeClassifier(lookback_months=120, min_dwell_months=2, hysteresis=0.5)
    daily = classifier.classify_to_daily(gdp_yoy.dropna(), cpi_yoy.dropna())
    daily = daily.loc["2008-01-01":]
    daily.index.name = "date"

    # Forward-fill the daily index to today's date so freshness checks see
    # current data. Use a business-day range matching the rest of the daily
    # CSVs (vix/daily.csv, credit/hyg_ief_spread.csv also use bdate_range).
    today = pd.Timestamp.today().normalize()
    if daily.index.max() < today:
        extended_idx = pd.bdate_range(start=daily.index.min(), end=today)
        # Pre-extension columns (e.g. regime, scores) get ffill; release_date
        # is recomputed from the new index below so it stays consistent.
        cols_to_ffill = [c for c in daily.columns if c != "release_date"]
        daily = daily.reindex(extended_idx)[cols_to_ffill].ffill()
        daily.index.name = "date"

    daily["release_date"] = daily.index + pd.Timedelta(days=30)
    return daily.to_csv(), daily


def _series_to_daily_csv(s, value_col_name: str) -> str:
    import pandas as pd
    df = s.to_frame(name=value_col_name)
    df.index.name = "date"
    bdi = pd.bdate_range(start=df.index.min(), end=pd.Timestamp.today())
    daily = df.reindex(bdi).ffill()
    daily.index.name = "date"
    return daily.to_csv()


def _log_refresh_cycle(algo, gdp_ok, cpi_ok, unrate_ok, t10y2y_series, regime_updated, error_notes=""):
    """Append one row to the refresh log CSV. Called at the END of every
    refresh_macro_signals() cycle (success OR partial failure). The weekly
    Sunday email reads this log to enumerate FRED reachability per day."""
    try:
        t10y2y_latest = ""
        t10y2y_age = ""
        t10y2y_status = "failed"
        if t10y2y_series is not None and len(t10y2y_series) > 0:
            t10y2y_status = "success"
            t10y2y_latest = str(t10y2y_series.index[-1].date())
            age = _series_age_days(algo, t10y2y_series.index)
            t10y2y_age = str(age) if age is not None else ""

        row = (
            f"{algo.time.isoformat()},"
            f"{'success' if gdp_ok else 'failed'},"
            f"{'success' if cpi_ok else 'failed'},"
            f"{'success' if unrate_ok else 'failed'},"
            f"{t10y2y_status},"
            f"{t10y2y_latest},"
            f"{t10y2y_age},"
            f"{'yes' if regime_updated else 'no'},"
            f"{error_notes.replace(',', ';')}\n"
        )

        existing = ""
        if algo.object_store.contains_key(REFRESH_LOG_KEY):
            existing = algo.object_store.read(REFRESH_LOG_KEY) or ""
        if not existing.strip().startswith("attempt_time,"):
            existing = REFRESH_LOG_HEADER
        # Cap log at ~60 days so it doesn't grow unbounded
        lines = (existing + row).splitlines(keepends=True)
        if len(lines) > 62:  # header + 60 days
            lines = [lines[0]] + lines[-60:]
        algo.object_store.save(REFRESH_LOG_KEY, "".join(lines))
    except Exception as e:
        algo.log(f"[refresh] refresh-log write failed: {e}")


def refresh_macro_signals(algo) -> bool:
    """Top-level entry: fetch FRED, rebuild regime + supporting CSVs, save to
    Object Store, trigger sleeves to re-read. Returns True/False.

    Always writes one row to conquest/refresh_log/attempts.csv before
    returning, even on partial failure — so the Sunday weekly email can
    enumerate FRED reachability per day."""
    api_key = _read_fred_api_key(algo)
    if not api_key:
        algo.log("[refresh] FRED_API_KEY not set (parameter empty, no Object Store fallback); skipping")
        _log_refresh_cycle(algo, False, False, False, None, False,
                           error_notes="FRED_API_KEY not set")
        return False

    try:
        gdp    = _fred_fetch(algo, "GDPC1", api_key)
        cpi    = _fred_fetch(algo, "CPIAUCSL", api_key)
        unrate = _fred_fetch(algo, "UNRATE", api_key)
        t10y2y = _fred_fetch(algo, "T10Y2Y", api_key)

        if gdp is None or cpi is None:
            algo.log("[refresh] GDP or CPI missing; cannot rebuild regime CSV")
            _log_refresh_cycle(
                algo,
                gdp_ok=(gdp is not None),
                cpi_ok=(cpi is not None),
                unrate_ok=(unrate is not None),
                t10y2y_series=t10y2y,
                regime_updated=False,
                error_notes="GDP or CPI fetch failed; regime CSV NOT rebuilt",
            )
            return False

        regime_csv, regime_df = _build_regime_csv(gdp, cpi)
        if regime_csv:
            algo.object_store.save(OBJECT_STORE_KEYS["regime"], regime_csv)
            last_regime_date = regime_df.index[-1].date()
            age = _series_age_days(algo, regime_df.index)
            algo.log(
                f"[refresh] OBJECT STORE WRITE {OBJECT_STORE_KEYS['regime']} "
                f"at {algo.time} ({len(regime_csv):,} chars, rows={len(regime_df)}, "
                f"last_date={last_regime_date}, age={age}d)"
            )

        if unrate is not None:
            algo.object_store.save(OBJECT_STORE_KEYS["unrate"],
                                   _series_to_daily_csv(unrate, "unrate"))
            age = _series_age_days(algo, unrate.index)
            algo.log(
                f"[refresh] OBJECT STORE WRITE {OBJECT_STORE_KEYS['unrate']} "
                f"at {algo.time} (latest={unrate.index[-1].date()}, age={age}d)"
            )
        if t10y2y is not None:
            algo.object_store.save(OBJECT_STORE_KEYS["t10y2y"],
                                   _series_to_daily_csv(t10y2y, "t10y2y"))
            age = _series_age_days(algo, t10y2y.index)
            algo.log(
                f"[refresh] OBJECT STORE WRITE {OBJECT_STORE_KEYS['t10y2y']} "
                f"at {algo.time} (latest={t10y2y.index[-1].date()}, age={age}d)"
            )

        # Trigger sleeves to re-read (each project's sleeves expose
        # initialize_store(); we try each one we know about, ignore missing).
        for sleeve_attr in ("cstag_sleeve", "voltgt_sleeve",
                            "cstability_sleeve", "cgrowth_sleeve",
                            "cf_sleeve"):
            sleeve = getattr(algo, sleeve_attr, None)
            if sleeve is None:
                continue
            for path in ("initialize_store", "cstab.initialize_store"):
                target = sleeve
                ok = True
                for part in path.split("."):
                    target = getattr(target, part, None)
                    if target is None:
                        ok = False
                        break
                if ok and callable(target):
                    try:
                        target()
                        algo.log(f"[refresh] reloaded {sleeve_attr}.{path}")
                    except Exception as e:
                        algo.log(f"[refresh] reload {sleeve_attr}.{path} failed: {e}")

        # Stamp the success timestamp. The halt-on-stale guard reads this to
        # distinguish "we lost FRED access" (halt) from "FRED's GDP just hasn't
        # released yet" (continue). last_refresh_success_at is persisted via
        # the state module so a restart inherits the prior success time.
        try:
            algo.last_refresh_success_at = algo.time
            if hasattr(algo, "_persist_state"):
                algo._persist_state()
        except Exception as e:
            algo.log(f"[refresh] success timestamp persist failed: {e}")

        # Log this cycle's per-series outcomes — feeds the weekly Sunday email
        _log_refresh_cycle(
            algo,
            gdp_ok=(gdp is not None),
            cpi_ok=(cpi is not None),
            unrate_ok=(unrate is not None),
            t10y2y_series=t10y2y,
            regime_updated=bool(regime_csv),
        )

        algo.log("[refresh] in-QC macro refresh COMPLETE")
        return True
    except Exception as e:
        algo.error(f"[refresh] in-QC macro refresh FAILED: {e}")
        _log_refresh_cycle(algo, False, False, False, None, False,
                           error_notes=f"refresh exception: {e}")
        return False


def attach_in_qc_refresh(algo) -> None:
    """Schedule the daily FRED refresh inside QC. Skipped if FRED_API_KEY is
    neither set as a project parameter nor stored in the Object Store."""
    fred_key = _read_fred_api_key(algo)
    if not fred_key:
        algo.log(
            f"[refresh] FRED_API_KEY empty — in-QC refresh DISABLED, using Object Store cache. "
            f"To enable: either set the FRED_API_KEY parameter via QC project settings, "
            f"OR upload it to Object Store key '{FRED_API_KEY_STORE_KEY}' via "
            f"`lean cloud object-store set --key {FRED_API_KEY_STORE_KEY} --path <local_file>`"
        )
        return

    def _safe_alert(subject, body, dedup_key=None):
        """Alert call that NEVER raises — defensive wrapper so a broken _alert
        method (e.g. signature mismatch) cannot crash the live algorithm.
        Falls back to plain log() if anything in the alert chain throws."""
        if not hasattr(algo, "_alert"):
            algo.log(f"[refresh] {subject}: {body}")
            return
        # Try with dedup_key, fall back to without it (legacy _alert signatures)
        try:
            algo._alert(subject, body, dedup_key=dedup_key)
        except TypeError:
            try:
                algo._alert(subject, body)
            except Exception as e:
                algo.log(f"[refresh] alert call failed: {e}; original: {subject}: {body}")
        except Exception as e:
            algo.log(f"[refresh] alert call failed: {e}; original: {subject}: {body}")

    def _run():
        # Don't fire during warmup — it's a Lean replay of historical market
        # days, and the schedule.on() callback fires on each one. With 270+
        # warmup days × 4 FRED calls per refresh, we burn through FRED's
        # rate limit (~120/min/key) and get empty responses for the rest of
        # warmup. The refresh is only meaningful when live anyway.
        if getattr(algo, "is_warming_up", False):
            return
        algo.log(f"[refresh] scheduled in-QC refresh START at {algo.time}")
        try:
            ok = refresh_macro_signals(algo)
            if ok:
                algo.log(f"[refresh] scheduled in-QC refresh END OK at {algo.time}")
                # Re-run freshness check so the next log line confirms the
                # Object Store is now current. This is the explicit
                # "data is X days old after refresh" confirmation.
                if hasattr(algo, "_check_data_freshness"):
                    try:
                        algo._check_data_freshness()
                    except Exception as e:
                        algo.log(f"[refresh] post-refresh freshness check failed: {e}")
            else:
                _safe_alert("in-QC refresh returned False",
                            "FRED fetch or regime rebuild failed; using cached data.",
                            dedup_key="refresh_failed")
        except Exception as e:
            _safe_alert("in-QC refresh exception", str(e),
                        dedup_key="refresh_exception")

    # Schedule daily after market close (SPY-anchored)
    algo.schedule.on(
        algo.date_rules.every_day("SPY"),
        algo.time_rules.after_market_close("SPY", 30),
        _run,
    )
    # Mask all but last 4 chars of the key — confirms it's set without
    # leaking the secret into the publicly-viewable QC live log.
    masked = ("*" * max(0, len(fred_key) - 4)) + fred_key[-4:] if len(fred_key) >= 4 else "****"
    # Identify which source the key came from so the user can audit
    param_source = (algo.get_parameter("FRED_API_KEY") or "").strip()
    src = "project_parameter" if param_source else f"object_store:{FRED_API_KEY_STORE_KEY}"
    algo.log(
        f"[refresh] in-QC daily macro refresh ENABLED (FRED_API_KEY={masked} from {src}, "
        f"schedule=after_market_close + 30min, SPY-anchored)"
    )

    # Startup refresh: fire ONE refresh immediately during init so the algorithm
    # starts with a freshly-rebuilt regime CSV instead of whatever stale state
    # the Object Store had from a previous deploy (e.g. 140d old after a long
    # gap). Without this, every restart forced the algo to wait until the next
    # market_close + 30min for its first refresh — meaning all the warmup +
    # initial-rebalance decisions ran against stale signals.
    #
    # This is a synchronous network call inside initialize(). It can add a few
    # seconds to deploy time but eliminates the stale-data mismatch the user
    # saw repeatedly across restarts. Each call has built-in retry/backoff so
    # transient FRED rate limits don't kill the launch.
    try:
        algo.log("[refresh] startup refresh START — aligning Object Store with current FRED")
        ok = refresh_macro_signals(algo)
        if ok:
            algo.log("[refresh] startup refresh END OK — Object Store is now current")
            # Re-run the freshness check so the post-startup log line shows
            # the new ages (should all be 0-1d now, not 140d).
            if hasattr(algo, "_check_data_freshness"):
                try:
                    algo._check_data_freshness()
                except Exception as e:
                    algo.log(f"[refresh] post-startup freshness check failed: {e}")
        else:
            algo.log(
                "[refresh] startup refresh returned False — algorithm will run on the "
                "Object Store cache until the next scheduled refresh fires"
            )
    except Exception as e:
        # Never let startup-refresh failures crash initialize. The daily
        # schedule will retry in a few hours.
        algo.log(f"[refresh] startup refresh exception (non-fatal): {e}")
