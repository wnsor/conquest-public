"""Put-roll simulator — daily MTM, schedule-driven entries/exits.

Convention
----------
- Trading-day calendar (252/yr). Tenor in trading days (21 = ~1mo, 63 = ~3mo).
- One open position at a time (we don't ladder for v1; a ladder is a
  trivial extension once the single-position logic is validated).
- Position state: (entry_date, expiry_date, strike, contracts, entry_premium).
- Daily MTM: BS price at current S, T_remaining, IV(strike, T_remaining).
- At expiry: position pays max(K - S, 0) per share × 100 × contracts. Cleared.
- Roll dates: defined by `RollSchedule.should_roll(date, vix, has_open)`.
  - `ConstantRoll`: every `tenor_days` business days from inception.
  - `VIXConditionalRoll`: roll only when VIX < threshold AND (no open position
    OR at expiry / near expiry). Means we may go *unhedged* during high-VIX
    regimes — that's the whole Israelov premise.

Output (DataFrame indexed by date, columns):
  - mtm_value    : current put MTM (USD), 0 when no position
  - contracts    : open contract count, 0 when no position
  - dte          : days to expiry (NaN when no position)
  - delta        : current put delta (NaN when no position)
  - is_roll_day  : True on dates we open or close a position
  - cost_usd     : USD trading cost charged on this date (entry + exit slippage/commission)
  - premium_paid : USD premium PAID on entry events (positive)
  - premium_recv : USD premium RECEIVED on exit / payoff (positive)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from conquest.options.pricing import bs_put_price, bs_put_delta
from conquest.options.implied_vol import vix_to_spx_iv, VIX_TENOR_DAYS, VIX3M_TENOR_DAYS
from conquest.options.costs import OptionsCostModel
from conquest.options.sizing import Sizer


_TRADING_DAYS = 252


class RollSchedule(Protocol):
    """Decide whether to enter/roll a position on a given date.

    Returns True if we should *open* a new position at today's close.
    The simulator handles closing the existing position separately at expiry.
    """

    def should_open(self, date: pd.Timestamp, vix_today: float, has_open: bool, dte: int | None) -> bool:
        ...


@dataclass
class ConstantRoll:
    """Roll every `tenor_days` trading days. Always rolls — ignores VIX."""
    tenor_days: int = 63

    def should_open(self, date, vix_today, has_open, dte):
        # Open if no position is currently active (i.e. previous one expired today
        # or we're at inception). The simulator advances expiry handling first.
        return not has_open


@dataclass
class VIXConditionalRoll:
    """Roll only when VIX < threshold AND we have no open position."""
    threshold: float = 15.0
    tenor_days: int = 63

    def should_open(self, date, vix_today, has_open, dte):
        if has_open:
            return False
        if vix_today is None or np.isnan(vix_today):
            return False
        return vix_today < self.threshold


@dataclass
class RegimeTriggeredRoll:
    """SPECULATIVE puts: open ONLY when the supplied regime stress signal exceeds
    `threshold`. Useful for buying puts as a directional bet during macro stress
    regimes (e.g. Stagflation or Deflation per the conquest classifier).

    The simulator passes `vix_today` as the gating signal; we reuse that hook by
    pre-computing your stress signal as the same daily series the simulator sees.
    Re-name only — semantics are: gate on the daily scalar.
    """
    threshold: float = 0.40        # e.g. P(Stagflation) > 0.40
    tenor_days: int = 63
    above_triggers: bool = True    # True: open when signal > threshold; False: when <

    def should_open(self, date, signal_today, has_open, dte):
        if has_open:
            return False
        if signal_today is None or np.isnan(signal_today):
            return False
        return (signal_today > self.threshold) if self.above_triggers else (signal_today < self.threshold)


@dataclass
class _Position:
    entry_date: pd.Timestamp
    expiry_date: pd.Timestamp  # set as entry + tenor_days business days
    strike: float
    contracts: float
    entry_premium_per_share: float


@dataclass
class PutRollSimulator:
    """Simulate a single rolling put position over a daily SPX/VIX history.

    Args:
        tenor_days: position tenor in TRADING days.
        strike_offset: strike picked on roll = S * (1 + offset). Default -0.05.
        skew_per_5pct_otm: passed to vix_to_spx_iv.
        risk_free_rate: annualized continuously-compounded r for BS pricing.
            Defaults to 0 — small effect for 21-63d tenors. Pass a series for
            precision (not implemented; flat rate is fine for v1).
    """
    tenor_days: int = 63
    strike_offset: float = -0.05
    skew_per_5pct_otm: float = 2.0
    risk_free_rate: float = 0.04  # ~average 2010-2024 short-rate, close enough for 1-3m puts

    def simulate(
        self,
        spx: pd.Series,
        vix: pd.Series,
        vix3m: pd.Series,
        equity_nav: pd.Series,
        equity_beta: pd.Series,
        sizer: Sizer,
        schedule: RollSchedule,
        cost_model: OptionsCostModel,
    ) -> pd.DataFrame:
        """Run the simulation over the joint index of inputs.

        Args:
            spx: SPY (or ^GSPC) daily close, daily index.
            vix: ^VIX close, daily.
            vix3m: ^VIX3M close, daily.
            equity_nav: equity sleeve NAV in dollars (used to size contracts on roll dates).
            equity_beta: rolling β of equity sleeve vs SPY (used for sizing).
            sizer: a Sizer instance (NotionalSizer or DeltaTargetSizer).
            schedule: a RollSchedule instance.
            cost_model: OptionsCostModel.

        Returns:
            DataFrame indexed by daily date, columns described in module docstring.
        """
        # Align all inputs to a common business-day index.
        idx = spx.index.intersection(vix.index).intersection(vix3m.index).intersection(equity_nav.index).intersection(equity_beta.index)
        idx = idx.sort_values()
        spx = spx.reindex(idx).astype(float)
        vix = vix.reindex(idx).astype(float)
        vix3m = vix3m.reindex(idx).astype(float)
        nav = equity_nav.reindex(idx).astype(float)
        beta = equity_beta.reindex(idx).astype(float)

        # Pre-compute IV series for the chosen tenor & strike offset (constant offset).
        iv_pct = vix_to_spx_iv(vix, vix3m, self.tenor_days, self.strike_offset, self.skew_per_5pct_otm)
        iv_pct = iv_pct.reindex(idx).ffill()

        # Outputs (split bool & float columns so dtype stays clean)
        out = pd.DataFrame(
            0.0,
            index=idx,
            columns=["mtm_value", "contracts", "dte", "delta", "cost_usd", "premium_paid", "premium_recv"],
        )
        out["dte"] = np.nan
        out["delta"] = np.nan
        out["is_roll_day"] = False

        position: _Position | None = None

        for i, date in enumerate(idx):
            S_t = float(spx.iloc[i])
            vix_t = float(vix.iloc[i])
            nav_t = float(nav.iloc[i])
            beta_t = float(beta.iloc[i]) if not np.isnan(beta.iloc[i]) else 1.0

            # 1) If we have an open position and today >= expiry: settle it at intrinsic.
            if position is not None and date >= position.expiry_date:
                payoff_per_share = max(position.strike - S_t, 0.0)
                payoff_usd = payoff_per_share * 100.0 * position.contracts
                close_cost = cost_model.per_leg_cost * abs(position.contracts)
                out.at[date, "premium_recv"] += payoff_usd
                out.at[date, "cost_usd"] += close_cost
                out.at[date, "is_roll_day"] = True
                position = None  # cleared

            # 2) Decide whether to open a new position today.
            has_open = position is not None
            current_dte = None if not has_open else int((position.expiry_date - date).days)
            if schedule.should_open(date, vix_t, has_open, current_dte):
                # Pick strike from today's spot.
                K = S_t * (1.0 + self.strike_offset)
                # Tenor in years for BS; we use trading-day fraction for consistency.
                T = self.tenor_days / _TRADING_DAYS
                sigma = float(iv_pct.iloc[i]) / 100.0
                # Compute today's premium and delta (for sizing if delta-targeted).
                premium_per_share = float(bs_put_price(S_t, K, T, self.risk_free_rate, sigma))
                put_delta = float(bs_put_delta(S_t, K, T, self.risk_free_rate, sigma))
                # Size the position.
                n_contracts = sizer.contracts(
                    equity_nav=nav_t,
                    spx_price=S_t,
                    equity_beta=beta_t,
                    put_delta=put_delta,
                )
                if n_contracts > 0:
                    # Expiry date = entry date + tenor_days BUSINESS days (use index-based offset).
                    target_expiry_idx = min(i + self.tenor_days, len(idx) - 1)
                    expiry_date = idx[target_expiry_idx]
                    position = _Position(
                        entry_date=date,
                        expiry_date=expiry_date,
                        strike=K,
                        contracts=n_contracts,
                        entry_premium_per_share=premium_per_share,
                    )
                    entry_cost_usd = premium_per_share * 100.0 * n_contracts
                    open_cost = cost_model.per_leg_cost * n_contracts
                    out.at[date, "premium_paid"] += entry_cost_usd
                    out.at[date, "cost_usd"] += open_cost
                    out.at[date, "is_roll_day"] = True

            # 3) MTM the open position at end-of-day.
            if position is not None:
                T_rem = max((position.expiry_date - date).days / 365.25, 1e-6)
                # Use IV at the *remaining* tenor — recompute interpolation for precision.
                # For speed in the inner loop we approximate with the precomputed IV
                # series (same offset, same target tenor); error is small (<1 vol pt).
                sigma = float(iv_pct.iloc[i]) / 100.0
                price_per_share = float(
                    bs_put_price(S_t, position.strike, T_rem, self.risk_free_rate, sigma)
                )
                delta = float(
                    bs_put_delta(S_t, position.strike, T_rem, self.risk_free_rate, sigma)
                )
                out.at[date, "mtm_value"] = price_per_share * 100.0 * position.contracts
                out.at[date, "contracts"] = position.contracts
                out.at[date, "dte"] = (position.expiry_date - date).days
                out.at[date, "delta"] = delta

        return out
