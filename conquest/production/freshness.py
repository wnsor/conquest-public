"""Object Store data freshness check — fires an alert if any signal CSV is
older than DATA_FRESHNESS_DAYS days behind the algorithm's current time.

Useful for catching:
  - Local refresh-and-push cron jobs that silently failed
  - QC Object Store propagation issues
  - FRED API outages that left stale data through deploy

Should be called from on_warmup_finished() — once at startup is enough to
catch deploy-time data issues. Re-call after each refresh for live tracking.
"""
from __future__ import annotations
import pandas as pd


DEFAULT_KEYS = [
    "conquest/regime/daily.csv",
    "conquest/vix/daily.csv",
    "conquest/vix/term_ratio.csv",
    # conquest/credit/hyg_ief_spread.csv RETIRED — regime/v3 now compute
    # the HYG/IEF 60d log spread in-algo from QC's price subscription. Older
    # projects (cstability, cf) that still read it from Object Store can pass
    # explicit freshness_keys=[...] including the credit CSV to opt back in.
    "conquest/votes/cstability_4vote_daily.csv",
]


def attach_freshness_check(algo, *, keys: list[str] | None = None) -> None:
    """Wire algo._check_data_freshness() which scans Object Store CSVs."""
    if hasattr(algo, "_check_data_freshness"):
        return
    algo._freshness_keys = keys if keys is not None else DEFAULT_KEYS
    algo._freshness_threshold_days = int(algo.get_parameter("DATA_FRESHNESS_DAYS") or "35")

    def _last_row_date(key: str) -> pd.Timestamp | None:
        try:
            if not algo.object_store.contains_key(key):
                return None
            txt = algo.object_store.read(key)
        except Exception:
            return None
        # CSV with header row, date in first column
        lines = txt.strip().split("\n")
        if len(lines) < 2:
            return None
        last_date = lines[-1].split(",")[0]
        try:
            return pd.Timestamp(last_date)
        except Exception:
            return None

    def _check_data_freshness() -> dict:
        """Scan every signal CSV in the Object Store and log its age (days
        behind algo.time). Always emits a one-line summary of EVERY key —
        fresh and stale — so the live log makes the current data state
        visible without having to manually inspect the Object Store.

        Returns {key: days_behind} for keys exceeding threshold (or -1 if
        the key is missing entirely). Fires _alert() only on stale/missing.
        """
        if not hasattr(algo, "_alert"):
            from .alerts import attach_alert
            attach_alert(algo)

        today = pd.Timestamp(algo.time.date())
        ages: dict[str, int] = {}      # all keys: age in days, -1 if missing
        stale: dict[str, int] = {}     # only keys exceeding threshold
        for key in algo._freshness_keys:
            last = _last_row_date(key)
            if last is None:
                ages[key] = -1
                stale[key] = -1
                continue
            days = (today - last).days
            ages[key] = days
            if days > algo._freshness_threshold_days:
                stale[key] = days

        # Always log the full freshness picture, so the user sees
        # "regime/daily.csv=0d, vix/daily.csv=1d, ..." after each refresh.
        # The lone existing alert path only fired when stale, which made
        # the "everything is current" state invisible in logs.
        short = [f"{k.split('/')[-1]}={v}d" if v >= 0 else f"{k.split('/')[-1]}=MISSING"
                 for k, v in ages.items()]
        algo.log(
            f"[freshness] {algo.time}: " + ", ".join(short)
            + f" (threshold={algo._freshness_threshold_days}d)"
        )

        if stale:
            parts = [f"{k.split('/')[-1]}={v}d" if v >= 0 else f"{k}=MISSING"
                     for k, v in stale.items()]
            body = (
                f"feeds older than {algo._freshness_threshold_days} days: "
                + ", ".join(parts)
                + ". Strategy will continue with last-known signal — refresh sources."
            )
            algo._alert("Stale Object Store signals", body, dedup_key="freshness")
        return stale

    algo._check_data_freshness = _check_data_freshness
