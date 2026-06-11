"""Per-contract options cost model — IB tiered + half-spread slippage.

Pessimistic-by-design (~30% above realized for a sophisticated trader).

Per-leg cost (USD):
    commission = $0.85/contract  (IB tiered + OCC + ORF baked in)
    slippage   = $5.00/contract  (~$0.05 half-spread on $100-multiplier)
    per_leg    = $5.85

Per-roll = entry leg + exit leg = contracts × $11.70.

Why not bps-of-turnover?
------------------------
Equity costs scale with dollar turnover. Options costs scale with *number of
contracts*, which decouples from notional (a 1-lot SPY put costs the same in
commissions whether SPX is 4000 or 6000). Modeling as bps of dollar notional
massively understates costs in bull markets.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptionsCostModel:
    commission_per_contract: float = 0.85
    slippage_per_contract: float = 5.00

    @property
    def per_leg_cost(self) -> float:
        return self.commission_per_contract + self.slippage_per_contract

    def roll_cost_usd(self, contracts_open: float, contracts_close: float = None) -> float:
        """Total round-trip cost in USD for a roll.

        If `contracts_close` is None, assume same-size roll (close existing,
        open same-size new). For partial rolls, pass the close size separately.
        """
        if contracts_close is None:
            contracts_close = contracts_open
        return abs(contracts_open) * self.per_leg_cost + abs(contracts_close) * self.per_leg_cost
