"""Per-trade logger.

Each opened position becomes a TradeRecord on entry. On exit (TP / SL /
time / expiry / signal), the record is finalized with exit_date, exit_price,
realized PnL, R-multiple, and exit_reason.

At end of backtest the algorithm dumps the full journal (one row per closed
trade) to the Object Store as JSON. The post-processor in
scripts/aggregate_per_trade_metrics.py reads that JSON and computes the
per-strategy promotion-gate metrics (Expectancy, Profit Factor, Win Rate,
R-mean, Sortino, max losing streak, time-in-market).

Object Store key: `conquest/options/trade_journal.json`
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, date

from strategies.base import ExitReason


@dataclass
class TradeRecord:
    # Identifying
    trade_id: int
    strategy_id: str
    underlying: str
    contract_symbol: str
    side: str                          # "call" | "put" | etc.
    edge_score: float

    # Entry
    entry_date: str                    # YYYY-MM-DD
    entry_premium_per_share: float
    contracts: int
    entry_underlying_price: float
    strike: float
    expiry: str                        # YYYY-MM-DD
    dte_at_open: int

    # Risk basis (R-multiple reference)
    risk_at_entry_dollars: float       # initial cost (= premium * 100 * contracts) for long-only

    # Exit (filled in later)
    exit_date: str | None = None
    exit_premium_per_share: float | None = None
    exit_underlying_price: float | None = None
    exit_reason: ExitReason | None = None
    pnl_dollars: float | None = None
    pnl_pct: float | None = None       # 1.0 = +100%
    r_multiple: float | None = None    # pnl_dollars / risk_at_entry_dollars
    commissions_dollars: float = 0.0
    dte_at_close: int | None = None


class TradeLogger:
    def __init__(self):
        self._next_id = 1
        self._trades: dict[str, TradeRecord] = {}  # keyed by contract_symbol while open
        self._closed: list[TradeRecord] = []

    def log_entry(
        self,
        *,
        strategy_id: str,
        underlying: str,
        contract_symbol: str,
        side: str,
        edge_score: float,
        entry_time: datetime,
        entry_premium_per_share: float,
        contracts: int,
        entry_underlying_price: float,
        strike: float,
        expiry: date,
    ) -> TradeRecord:
        rec = TradeRecord(
            trade_id=self._next_id,
            strategy_id=strategy_id,
            underlying=underlying,
            contract_symbol=contract_symbol,
            side=side,
            edge_score=edge_score,
            entry_date=entry_time.date().isoformat(),
            entry_premium_per_share=entry_premium_per_share,
            contracts=contracts,
            entry_underlying_price=entry_underlying_price,
            strike=strike,
            expiry=expiry.isoformat(),
            dte_at_open=(expiry - entry_time.date()).days,
            risk_at_entry_dollars=entry_premium_per_share * 100.0 * contracts,
        )
        self._next_id += 1
        self._trades[contract_symbol] = rec
        return rec

    def log_exit(
        self,
        *,
        contract_symbol: str,
        exit_time: datetime,
        exit_premium_per_share: float,
        exit_underlying_price: float,
        exit_reason: ExitReason,
        commissions_dollars: float = 0.0,
    ) -> TradeRecord | None:
        rec = self._trades.pop(contract_symbol, None)
        if rec is None:
            return None
        rec.exit_date = exit_time.date().isoformat()
        rec.exit_premium_per_share = exit_premium_per_share
        rec.exit_underlying_price = exit_underlying_price
        rec.exit_reason = exit_reason
        # Long-only PnL: (exit - entry) per share * 100 * contracts - commissions
        rec.pnl_dollars = (
            (exit_premium_per_share - rec.entry_premium_per_share) * 100.0 * rec.contracts
            - commissions_dollars
        )
        if rec.risk_at_entry_dollars > 0:
            rec.pnl_pct = rec.pnl_dollars / rec.risk_at_entry_dollars
            rec.r_multiple = rec.pnl_dollars / rec.risk_at_entry_dollars
        rec.commissions_dollars = commissions_dollars
        expiry_date = date.fromisoformat(rec.expiry)
        rec.dte_at_close = (expiry_date - exit_time.date()).days
        self._closed.append(rec)
        return rec

    def open_count(self) -> int:
        return len(self._trades)

    def closed_records(self) -> list[TradeRecord]:
        return list(self._closed)

    def to_json(self) -> str:
        """JSON dump of every CLOSED trade (open positions excluded — they're
        reported separately by the algorithm at end-of-backtest)."""
        payload = {
            "schema_version": 1,
            "trades": [asdict(t) for t in self._closed],
        }
        return json.dumps(payload, indent=2, default=str)
