"""Fetch a QC cloud backtest result (equity curve + benchmark) via REST API.

The `lean` CLI doesn't expose a 'read backtest' subcommand, so we hit the QC
REST API directly using credentials from ~/.lean/credentials.

Modes:
  Default (single fund):
    python scripts/fetch_cloud_backtest.py <project_id> <backtest_id> [<fund_label>]
    Output:
      storage/conquest/lean/<fund_label>_lean_equity.json  (Strategy Equity)
      storage/conquest/lean/<fund_label>_spy_benchmark.json (Benchmark / SPY)

  Buy-hold benchmarks (4 series from one Lean project):
    python scripts/fetch_cloud_backtest.py <project_id> <backtest_id> buyhold
    Extracts QQQ / IWM / EFA / GLD series from the "Buy-Hold Benchmarks"
    custom chart emitted by benchmarks_buy_hold/main.py.
    Output (one JSON per ticker):
      storage/conquest/lean/qqq_buyhold_lean.json
      storage/conquest/lean/iwm_buyhold_lean.json
      storage/conquest/lean/efa_buyhold_lean.json
      storage/conquest/lean/gld_buyhold_lean.json

Schema (matches the existing exporter):
    {
        "fund": "cstability" | "cgrowth" | "cf" | "QQQ" | ...,
        "version_label": "v11 LIVE",
        "engine": "lean cloud (REST)",
        "source_backtest": "<backtest_id>",
        "period": {"start": "2008-01-01", "end": "2026-05-04"},
        "start_equity": 25000,
        "values": [{"date": "2008-01-02", "equity": 25000.0}, ...]
    }
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "storage" / "conquest" / "lean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QC_BASE = "https://www.quantconnect.com/api/v2"


def load_credentials() -> tuple[str, str]:
    cred_path = Path(os.path.expanduser("~/.lean/credentials"))
    if not cred_path.exists():
        raise SystemExit(f"missing {cred_path}; run `lean login` first")
    data = json.loads(cred_path.read_text())
    return str(data["user-id"]), str(data["api-token"])


def auth_headers(user_id: str, api_token: str) -> dict:
    """QC REST API auth: Authorization: Basic <user-id>:<sha256(token:timestamp)>,
    plus a Timestamp header."""
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{api_token}:{ts}".encode()).hexdigest()
    raw = f"{user_id}:{digest}".encode()
    return {
        "Authorization": "Basic " + base64.b64encode(raw).decode(),
        "Timestamp": ts,
    }


def fetch_backtest_summary(project_id: str, backtest_id: str,
                           user_id: str, api_token: str) -> dict:
    """Pull the backtest's full summary (statistics + runtimeStatistics + state)
    via /backtests/read. Used to extract QC's official metrics so the webapp can
    display the same numbers users see in the QC cloud UI rather than re-deriving
    Sharpe/DD from end-of-day equity samples (which differ from QC's intra-day
    methodology)."""
    url = f"{QC_BASE}/backtests/read"
    headers = auth_headers(user_id, api_token)
    headers["Content-Type"] = "application/json"
    body = {"projectId": int(project_id), "backtestId": backtest_id}
    r = requests.post(url, headers=headers, json=body, timeout=180)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        return {}
    bt = data.get("backtest") or {}
    return {
        "statistics": bt.get("statistics") or {},
        "runtimeStatistics": bt.get("runtimeStatistics") or {},
        "state": bt.get("state") or {},
        "name": bt.get("name"),
        "created": bt.get("created"),
        "completed": bt.get("completed"),
    }


def write_stats(out_path: Path, *, fund: str, backtest_id: str, summary: dict) -> None:
    """Save QC's authoritative statistics block as a sidecar JSON. The webapp
    prefers these values over re-derived ones whenever this sidecar exists."""
    if not summary or not summary.get("statistics"):
        print(f"[skip] {out_path.name}: no statistics in backtest summary")
        return
    payload = {
        "fund": fund,
        "engine": "lean cloud (REST) — official statistics block",
        "source_backtest": backtest_id,
        "backtest_name": summary.get("name"),
        "created": summary.get("created"),
        "completed": summary.get("completed"),
        "statistics": summary.get("statistics", {}),
        "runtimeStatistics": summary.get("runtimeStatistics", {}),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    n = len(summary.get("statistics", {}))
    print(f"[ok] {out_path.relative_to(ROOT)}  ({n} statistics keys)")


def fetch_chart(project_id: str, backtest_id: str, chart_name: str, series_name: str,
                user_id: str, api_token: str, max_points: int = 10000,
                retries: int = 4, backoff: float = 3.0) -> list[dict]:
    """Pull a single chart's daily values via /backtests/chart/read.

    The QC API's `count` parameter caps the number of returned points; setting
    it large gets the full daily resolution. Returns one record per UTC day
    [{"date": "...", "equity": ...}, ...].

    QC generates chart data lazily: the FIRST request to a never-rendered chart
    often returns success=False (or empty values) while it builds server-side,
    then serves it on a subsequent call. We retry with backoff so a cold chart
    doesn't silently come back as "[skip] no values".
    """
    url = f"{QC_BASE}/backtests/chart/read"
    headers = auth_headers(user_id, api_token)
    headers["Content-Type"] = "application/json"
    body = {
        "projectId": int(project_id),
        "backtestId": backtest_id,
        "name": chart_name,
        "count": max_points,
        "start": 0,
    }
    raw = []
    for attempt in range(retries):
        r = requests.post(url, headers=headers, json=body, timeout=180)
        r.raise_for_status()
        data = r.json()
        if data.get("success"):
            chart = data.get("chart") or {}
            series = chart.get("series", {}).get(series_name) or {}
            raw = series.get("values") or []
            if raw:
                break
        if attempt < retries - 1:
            print(f"  [cold] {chart_name!r}/{series_name!r} not ready "
                  f"(attempt {attempt + 1}/{retries}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
    by_day: dict[str, float] = {}
    for entry in raw:
        if not entry or not isinstance(entry, (list, tuple)):
            continue
        ts = entry[0]
        # Lean candlestick rows are [ts, open, high, low, close]; line series are [ts, value].
        if len(entry) >= 5:
            value = float(entry[4])
        elif len(entry) >= 2:
            value = float(entry[1])
        else:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[d] = value
    return [{"date": d, "equity": by_day[d]} for d in sorted(by_day.keys())]


def write_curve(out_path: Path, *, fund: str, version_label: str, backtest_id: str, values: list[dict]) -> None:
    if not values:
        print(f"[skip] {out_path.name}: no values")
        return
    payload = {
        "fund": fund,
        "version_label": version_label,
        "engine": "lean cloud (REST)",
        "source_backtest": backtest_id,
        "period": {"start": values[0]["date"], "end": values[-1]["date"]},
        "start_equity": values[0]["equity"],
        "values": values,
    }
    out_path.write_text(json.dumps(payload) + "\n")
    print(f"[ok] {out_path.relative_to(ROOT)}  ({len(values)} pts, {values[0]['date']} → {values[-1]['date']}, end ${values[-1]['equity']:,.0f})")


def normalize_benchmark(values: list[dict], target_start: float) -> list[dict]:
    """Lean's Benchmark series uses normalized index values starting at ~1.0,
    not dollar amounts. We rescale to a $25k seed so it overlays cleanly with
    the strategy curve."""
    if not values:
        return values
    start = values[0]["equity"]
    if start <= 0:
        return values
    return [
        {"date": v["date"], "equity": v["equity"] / start * target_start}
        for v in values
    ]


def fetch_buyhold(project_id: str, backtest_id: str, user_id: str, api_token: str) -> None:
    """Extract the 4 buy-hold tracker series from the benchmarks_buy_hold project.

    Each series ($25k seed × cumulative daily returns for QQQ / IWM / EFA / GLD)
    is saved to its own JSON so the webapp can toggle them independently. The
    chart name and ticker list match benchmarks_buy_hold/main.py:CHART_NAME and
    TICKERS.
    """
    chart_name = "Buy-Hold Benchmarks"
    # Must match benchmarks_buy_hold/main.py TICKERS — the curated Surge/ctactical set.
    tickers = (
        "QQQ", "IWM", "GLD", "TLT", "EFA",
        "TQQQ",  # 3x Nasdaq — Surge's flagship leveraged peer
        "UVXY",  # 1.5x VIX-futures ETP — Surge's overlay instrument
    )
    print(f"Fetching buy-hold benchmarks (project {project_id}, backtest {backtest_id}) ...")
    for t in tickers:
        values = fetch_chart(project_id, backtest_id, chart_name, t, user_id, api_token)
        print(f"  {t}: {len(values)} daily points")
        write_curve(
            OUT_DIR / f"{t.lower()}_buyhold_lean.json",
            fund=t,
            version_label=f"Buy-and-Hold {t} (Lean cloud, $25k seed)",
            backtest_id=backtest_id,
            values=values,
        )


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    project_id = sys.argv[1]
    backtest_id = sys.argv[2]
    fund = sys.argv[3] if len(sys.argv) > 3 else "fund"

    user_id, api_token = load_credentials()

    if fund == "buyhold":
        fetch_buyhold(project_id, backtest_id, user_id, api_token)
        return

    print(f"Fetching backtest {backtest_id} (project {project_id}) ...")
    eq_values = fetch_chart(project_id, backtest_id, "Strategy Equity", "Equity", user_id, api_token)
    print(f"  Strategy Equity: {len(eq_values)} daily points")
    bm_values = fetch_chart(project_id, backtest_id, "Benchmark", "Benchmark", user_id, api_token)
    print(f"  Benchmark: {len(bm_values)} daily points")

    # Pull the official QC statistics block — webapp prefers these authoritative
    # values (intra-day MaxDD, Sharpe with non-zero risk-free rate, etc.) over
    # re-deriving from end-of-day samples.
    summary = fetch_backtest_summary(project_id, backtest_id, user_id, api_token)
    print(f"  Statistics: {len(summary.get('statistics') or {})} keys, runtime: {len(summary.get('runtimeStatistics') or {})} keys")

    write_curve(
        OUT_DIR / f"{fund}_lean_equity.json",
        fund=fund,
        version_label="v11 LIVE",
        backtest_id=backtest_id,
        values=eq_values,
    )
    write_stats(
        OUT_DIR / f"{fund}_lean_stats.json",
        fund=fund,
        backtest_id=backtest_id,
        summary=summary,
    )

    if bm_values:
        target_start = eq_values[0]["equity"] if eq_values else 25_000.0
        bm_dollar = normalize_benchmark(bm_values, target_start)
        write_curve(
            OUT_DIR / f"{fund}_spy_benchmark.json",
            fund="SPY",
            version_label="Buy-and-Hold SPY (Lean cloud benchmark)",
            backtest_id=backtest_id,
            values=bm_dollar,
        )
    else:
        print(f"  [warn] no Benchmark chart found for {fund}; SPY trace will remain pending")


if __name__ == "__main__":
    main()
