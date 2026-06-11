"""Extract daily equity curves from the latest Lean local backtests of each fund.

Lean writes a backtest folder per run (timestamped) with a results JSON containing
the full equity time series in `charts['Strategy Equity']['series']['Equity']`.
The path is dynamic; the webapp can't list directories, so we extract the latest
curves to stable paths under `storage/conquest/lean/`:

    storage/conquest/lean/cstability_lean_equity.json
    storage/conquest/lean/cgrowth_lean_equity.json

Schema (small, compact):
    {
        "fund": "cstability" | "cgrowth",
        "version_label": "v8 LIVE" | "v8.5 LIVE",
        "engine": "lean 1.0.225",
        "period": {"start": "2018-01-01", "end": "2024-12-31"},
        "start_equity": 100000,
        "values": [{"date": "2018-01-02", "equity": 100000.0}, ...]   # daily, last value per UTC day
    }

Re-run after each new Lean backtest; webapp picks up the latest automatically.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCES = [
    {
        "fund": "cstability",
        "version_label": "v10 LIVE",
        "backtest_dir": ROOT / "cstability" / "backtests",
    },
    {
        "fund": "cgrowth",
        "version_label": "v9 LIVE",
        "backtest_dir": ROOT / "cgrowth" / "backtests",
    },
]

OUT_DIR = ROOT / "storage" / "conquest" / "lean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def latest_backtest(folder: Path):
    candidates = sorted(p for p in folder.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def extract(src):
    bt_dir = latest_backtest(src["backtest_dir"])
    if bt_dir is None:
        print(f"[skip] no backtests found in {src['backtest_dir']}")
        return
    # Find the *.json file (not the *-summary.json, not *-log.txt)
    main_json = next(
        (p for p in bt_dir.iterdir()
         if p.suffix == ".json"
         and not p.name.endswith("-summary.json")
         and not p.name.endswith("-order-events.json")
         and not p.name.startswith("data-monitor")),
        None,
    )
    if main_json is None:
        print(f"[skip] no main JSON in {bt_dir}")
        return

    data = json.loads(main_json.read_text())
    equity_series = data["charts"]["Strategy Equity"]["series"]["Equity"]
    raw_values = equity_series["values"]  # [unix, open, high, low, close]

    # Take the LAST value per UTC date (daily close) so the chart series is one point/day.
    by_day = {}
    for entry in raw_values:
        ts, _o, _h, _l, close = entry[0], entry[1], entry[2], entry[3], entry[4]
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[date_str] = close  # later entries overwrite -> final close of that UTC day

    values = [{"date": d, "equity": by_day[d]} for d in sorted(by_day.keys())]

    out = {
        "fund": src["fund"],
        "version_label": src["version_label"],
        "engine": "lean 1.0.225",
        "source_backtest": bt_dir.name,
        "period": {"start": values[0]["date"], "end": values[-1]["date"]},
        "start_equity": values[0]["equity"],
        "values": values,
    }
    out_path = OUT_DIR / f"{src['fund']}_lean_equity.json"
    out_path.write_text(json.dumps(out) + "\n")
    print(f"[ok] {src['fund']:11s} {src['version_label']:10s} -> {out_path.relative_to(ROOT)}  ({len(values)} daily points, "
          f"{values[0]['date']} -> {values[-1]['date']}, start ${out['start_equity']:.0f})")


if __name__ == "__main__":
    for s in SOURCES:
        extract(s)
