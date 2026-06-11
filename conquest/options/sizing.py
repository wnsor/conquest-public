"""Sizing modes for the put-roll overlay.

Three modes (sweep all three in the bake-off):
  - NotionalSizer(fraction=1.0)   — 100% of equity NAV, beta-adjusted to SPY.
  - NotionalSizer(fraction=0.5)   — half-hedge.
  - DeltaTargetSizer(target=0.7)  — size puts so portfolio net delta = target.

Beta adjustment (NotionalSizer): when the equity sleeve has β > 1 vs SPY (e.g.
top-5 momentum names in 2020 were concentrated growth, β ≈ 1.3), hedging 100%
of *equity NAV* with SPY puts under-hedges the actual market exposure. We
multiply contracts by the rolling 60d β to compensate. Use β = 1.0 when
insufficient history (caller's responsibility to feed it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Sizer(Protocol):
    """Compute the contract count to hold given current state."""

    def contracts(
        self,
        equity_nav: float,
        spx_price: float,
        equity_beta: float,
        put_delta: float,
    ) -> float:
        ...


@dataclass
class NotionalSizer:
    """Notional-targeted sizer: contracts = (NAV * fraction * beta) / (S * 100).

    `fraction=1.0` hedges 100% of equity-NAV-beta-equivalent SPY exposure.
    `fraction=0.5` hedges half. SPY contracts are 100 shares each.
    """
    fraction: float = 1.0

    def contracts(
        self,
        equity_nav: float,
        spx_price: float,
        equity_beta: float,
        put_delta: float,
    ) -> float:
        if spx_price <= 0:
            return 0.0
        notional = equity_nav * self.fraction * equity_beta
        return notional / (spx_price * 100.0)


@dataclass
class DeltaTargetSizer:
    """Delta-targeted sizer: contracts so portfolio_net_delta ≈ target_net_delta.

    portfolio_delta = equity_beta * NAV / NAV          # equity sleeve δ in NAV terms
                    + n_contracts * put_delta * 100 * S / NAV
    Solve for n_contracts:
        n = (target - equity_beta) * NAV / (put_delta * 100 * S)
    Note `put_delta` is negative for puts; (target - equity_beta) typically
    negative (we want to *reduce* delta from β toward target<β); the two
    negatives produce a positive contract count. Clamps to 0 if beta < target
    (no hedge needed).
    """
    target_net_delta: float = 0.7

    def contracts(
        self,
        equity_nav: float,
        spx_price: float,
        equity_beta: float,
        put_delta: float,
    ) -> float:
        if spx_price <= 0 or put_delta == 0:
            return 0.0
        if equity_beta <= self.target_net_delta:
            return 0.0
        n = (self.target_net_delta - equity_beta) * equity_nav / (put_delta * 100.0 * spx_price)
        return max(n, 0.0)
