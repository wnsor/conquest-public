"""Per-strategy position sizer.

Two modes:

  flat    A fixed % of NAV per trade, scaled linearly by edge_score.
          Phase 1 default — sane baseline before any per-strategy edge
          estimate exists. Trade size = base_pct * (0.5 + 0.5*edge_score)
          so edge_score=0 → 0.5×base, edge_score=1 → 1×base.

  kelly   Kelly-fractional sizing using per-strategy historical
          (win_rate, win_loss_ratio). Caps at kelly_cap_fraction of
          full Kelly (default 0.25) to dampen Kelly's volatility.
          Phase 2+ — only usable once a strategy has ≥50 trades of
          history to estimate (p, b).

In either mode, the result is converted into an integer contract count
respecting:
  - per-strategy max_concurrent_per_underlying
  - absolute floor (1 contract per signal — we either size or skip)
  - portfolio-level cap (no single trade > 10% of NAV after entry)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from strategies.base import StrategySignal


SizerMode = Literal["flat", "kelly"]


@dataclass
class SizerConfig:
    mode: SizerMode = "flat"
    # v9: bumped to 5% base / 20% cap based on Phase 4 diagnostic — at $10k seed
    # × 1.5% = $150 target, but SPY/QQQ option premiums are $500-1500/contract,
    # so 0 contracts always fit. Fish_ trader runs 1-2 contracts = $200-400 per
    # play on cheaper names; for $10k seed we need 5-10% targets to fit any single
    # option contract. Concentration risk vs zero-fire trade-off.
    # v10c: 10% base / 30% cap — SPY 5%-OTM 35-DTE premium is $5-8/share = $500-800/contract.
    # At 5%/$500 target we get 0-1 contracts (still borderline). 10%/$1000 reliably
    # fits 1+ contracts. Concentration risk accepted as $10k-seed structural reality.
    base_pct_nav: float = 0.10
    portfolio_cap_pct_nav: float = 0.30
    kelly_cap_fraction: float = 0.25         # 25% of full Kelly
    min_premium_dollars: float = 50.0        # don't open positions smaller than this
    # v5: global + per-strategy + drawdown caps. Main algorithm enforces.
    global_open_cap_pct_nav: float = 0.50    # total open premium across all positions (v9: up from 30%)
    per_strategy_cap_pct_nav: float = 0.30   # max NAV any single strategy can hold open (v9: up from 20%)
    drawdown_derisk_threshold: float = 0.80  # if NAV < 80% starting, halve base sizing
    drawdown_derisk_factor: float = 0.5


@dataclass(frozen=True)
class SizingResult:
    contracts: int
    capital_committed_dollars: float
    skipped_reason: str | None = None


def size_position(
    signal: StrategySignal,
    *,
    contract_mid_price: float,           # mid premium per contract, dollars
    nav: float,                          # current portfolio NAV
    config: SizerConfig,
    strategy_history: dict | None = None,  # for kelly mode: {win_rate, win_loss_ratio}
    n_legs: int = 1,                     # v5: # signals emitted by same strategy this tick (straddle=2)
    starting_nav: float | None = None,   # v5: for drawdown-aware de-risk
) -> SizingResult:
    """Return integer contract count for a single signal.

    Mid price is per-share; multiplier=100 baked into capital calc.
    Returns SizingResult.contracts=0 + reason when the trade is skipped.

    v5 additions:
      n_legs: when a strategy emits a multi-leg trade (straddle = 2 signals,
              call + put), divide the base sizing across legs so the PAIR is
              sized at base_pct_nav, not each leg.
      starting_nav: for drawdown de-risk. When current nav < starting × threshold,
                    base sizing scales by drawdown_derisk_factor (default 0.5).
    """
    if contract_mid_price <= 0:
        return SizingResult(0, 0.0, "non-positive premium")

    contract_cost = contract_mid_price * 100.0  # equity options multiplier

    # v22: per-strategy cap override. When signal.max_per_trade_pct_nav is set,
    # use that instead of config.base_pct_nav as the base allocation.
    # Lets crisis-window strategies (D2 Tepper, CrisisRebound, D1 LEAPS)
    # take 25% positions while keeping per-trade-gate strategies at 10%.
    effective_base_pct = (signal.max_per_trade_pct_nav
                         if signal.max_per_trade_pct_nav is not None
                         else config.base_pct_nav)

    if config.mode == "flat":
        scaled_pct = effective_base_pct * (0.5 + 0.5 * max(0.0, min(1.0, signal.edge_score)))
        # v5: multi-leg awareness — divide across legs so 2-leg straddle pair = 1× base
        if n_legs > 1:
            scaled_pct /= n_legs
        # v5: drawdown de-risk
        if starting_nav and nav > 0 and nav < starting_nav * config.drawdown_derisk_threshold:
            scaled_pct *= config.drawdown_derisk_factor
    elif config.mode == "kelly":
        if not strategy_history or "win_rate" not in strategy_history or "win_loss_ratio" not in strategy_history:
            return SizingResult(0, 0.0, "kelly mode requires strategy_history with win_rate/win_loss_ratio")
        p = float(strategy_history["win_rate"])
        b = float(strategy_history["win_loss_ratio"])
        # Kelly fraction f* = (bp - (1-p)) / b. Clamp to [0, cap].
        full_kelly = (b * p - (1 - p)) / b if b > 0 else 0.0
        scaled_pct = max(0.0, min(config.portfolio_cap_pct_nav, full_kelly * config.kelly_cap_fraction))
        # Edge score still scales within Kelly recommendation.
        scaled_pct *= 0.5 + 0.5 * max(0.0, min(1.0, signal.edge_score))
    else:
        return SizingResult(0, 0.0, f"unknown sizer mode {config.mode}")

    # Cap at portfolio limit — but allow signal.max_per_trade_pct_nav to push
    # above config.portfolio_cap_pct_nav if explicitly opted in by the strategy.
    portfolio_cap = max(config.portfolio_cap_pct_nav,
                       signal.max_per_trade_pct_nav or 0.0)
    scaled_pct = min(scaled_pct, portfolio_cap)

    target_capital = nav * scaled_pct
    if target_capital < config.min_premium_dollars:
        return SizingResult(0, 0.0, f"target capital ${target_capital:.0f} < floor ${config.min_premium_dollars:.0f}")

    contracts = int(target_capital // contract_cost)
    if contracts < 1:
        return SizingResult(0, 0.0, f"premium ${contract_cost:.0f} > target ${target_capital:.0f}")

    capital = contracts * contract_cost
    return SizingResult(contracts=contracts, capital_committed_dollars=capital, skipped_reason=None)
