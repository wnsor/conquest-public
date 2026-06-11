"""Reconstruct a per-trade journal from a cloud backtest's ORDERS via the QC API.

Why: QC Object Store *export* is institutional-only, so we can't pull
`conquest/options/trade_journal.json` out of the cloud. But the orders endpoint
(`/backtests/orders/read`) IS accessible, and for a long-only options strategy
every contract has exactly one buy (entry) and one sell (exit) — so we can
reconstruct exact per-trade pnl_pct = exit_price/entry_price - 1 by pairing fills
per contract symbol. Output is the {"trades":[{strategy_id,pnl_pct,...}]} shape
that scripts/options_dsr_haircut.py --journal consumes.

Strategy attribution: orders carry no strategy_id tag, but the DYNAMIC_PIT BTs run
exactly one strategy, so all trades are attributed to --strategy-id (default
dynamic_pit_momentum_calls). Mixed-strategy BTs would need tag-based attribution.

Usage:
    python scripts/reconstruct_trades_from_orders.py --backtest-id <id> \
        --out /tmp/journal.json
    python scripts/options_dsr_haircut.py --journal /tmp/journal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "https://www.quantconnect.com/api/v2"


def _auth():
    c = json.loads((Path.home() / ".lean" / "credentials").read_text())
    u = str(c["user-id"]); tok = c["api-token"]; ts = str(int(time.time()))
    return (u, hashlib.sha256(f"{tok}:{ts}".encode()).hexdigest()), {"Timestamp": ts}


def fetch_all_orders(project_id: int, backtest_id: str, page: int = 100) -> list[dict]:
    out: list[dict] = []
    start = 0
    while True:
        auth, h = _auth()
        r = requests.post(f"{BASE}/backtests/orders/read", headers=h, auth=auth,
                          json={"projectId": project_id, "backtestId": backtest_id,
                                "start": start, "end": start + page}, timeout=60)
        r.raise_for_status()
        j = r.json()
        if not j.get("success"):
            raise RuntimeError(f"orders/read failed: {j.get('errors')}")
        orders = j.get("orders", [])
        out.extend(orders)
        if len(orders) < page:
            break
        start += page
    return out


def reconstruct(orders: list[dict], strategy_id: str) -> list[dict]:
    """Pair each contract's fills in time order into buy→sell round-trips."""
    FILLED = 3
    by_sym: dict[str, list[dict]] = {}
    for o in orders:
        if o.get("status") != FILLED:
            continue
        sym = (o.get("symbol") or {}).get("value") or str(o.get("symbol"))
        by_sym.setdefault(sym, []).append(o)

    trades: list[dict] = []
    for sym, fills in by_sym.items():
        fills.sort(key=lambda x: x.get("time") or x.get("lastFillTime") or "")
        open_px = None
        for f in fills:
            qty = float(f.get("quantity") or 0)
            px = float(f.get("price") or 0)
            if px <= 0:
                continue
            if qty > 0:                      # buy = entry (long call/put)
                open_px = px
            elif qty < 0 and open_px:        # sell = exit → close the round-trip
                trades.append({
                    "strategy_id": strategy_id,
                    "symbol": sym,
                    "entry_px": open_px,
                    "exit_px": px,
                    "pnl_pct": px / open_px - 1.0,
                })
                open_px = None
    return trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", type=int, default=32043797)
    ap.add_argument("--backtest-id", required=True)
    ap.add_argument("--strategy-id", default="dynamic_pit_momentum_calls")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    orders = fetch_all_orders(args.project_id, args.backtest_id)
    trades = reconstruct(orders, args.strategy_id)
    Path(args.out).write_text(json.dumps({"trades": trades}, indent=2))

    if trades:
        rets = [t["pnl_pct"] for t in trades]
        n = len(rets); mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
        std = var ** 0.5
        wins = sum(1 for r in rets if r > 0)
        print(f"reconstructed {n} round-trips from {len(orders)} orders")
        print(f"  mean pnl/trade = {mean*100:+.1f}%   std = {std*100:.1f}%   "
              f"win rate = {wins/n*100:.1f}%   t-stat = {mean/(std/n**0.5):.3f}" if std > 0
              else f"  mean={mean*100:+.1f}% (std=0)")
    else:
        print(f"no round-trips reconstructed from {len(orders)} orders")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
