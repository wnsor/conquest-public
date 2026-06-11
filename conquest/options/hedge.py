"""Aggregate the put-roll daily MTM into a daily return contribution series.

The hedge "return" on day t is the change in MTM relative to the equity NAV
that was sized on. Specifically:

    hedge_pnl_usd_t = (mtm_t + premium_recv_t)
                    - (mtm_{t-1} + premium_paid_t)
                    - cost_usd_t

(Premium paid is treated as a debit on the entry day; premium received credits
on the payoff day. MTM moves daily once the position is open.)

Hedge return contribution (fraction of NAV):
    hedge_ret_t = hedge_pnl_usd_t / nav_{t-1}

This composes additively with the equity-sleeve return:
    combined_t = equity_ret_t + hedge_ret_t
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class HedgePnL:
    @staticmethod
    def from_roll(roll_df: pd.DataFrame, equity_nav: pd.Series) -> pd.Series:
        """Convert roll simulator output into a daily fractional-return series.

        Args:
            roll_df: output of `PutRollSimulator.simulate(...)`.
            equity_nav: same equity NAV series passed to the simulator (used as
                the denominator for converting USD PnL to a return).

        Returns:
            Series indexed like roll_df, daily fractional return contribution.
        """
        idx = roll_df.index
        nav = equity_nav.reindex(idx).astype(float).ffill()

        mtm = roll_df["mtm_value"].astype(float)
        premium_paid = roll_df["premium_paid"].astype(float)
        premium_recv = roll_df["premium_recv"].astype(float)
        cost_usd = roll_df["cost_usd"].astype(float)

        # Day-over-day change in mark, plus realized cash flows, minus trading costs.
        d_mtm = mtm.diff().fillna(mtm.iloc[0])  # day 0 establishes the position; treat as Δmtm = 0 if no entry
        # Actually: on entry day, mtm_t = premium_paid (we just paid for it). So
        # the *value* of the position is mtm_t, but cash outflow is premium_paid.
        # Net PnL on entry day = mtm_t - premium_paid_t = 0 (clean).
        # Subsequent days: PnL = Δmtm. On expiry day, mtm_t == 0 (position cleared),
        # but we recv premium_recv (intrinsic payoff). Net = -mtm_{t-1} + premium_recv.
        # The diff-based formulation handles this correctly:
        pnl_usd = d_mtm + premium_recv - premium_paid - cost_usd

        # Denominator: previous-day NAV. On day 0, use today's NAV (avoids div-by-zero).
        nav_prev = nav.shift(1).fillna(nav.iloc[0])
        # Avoid division by zero if NAV is ever 0 (shouldn't happen in practice).
        nav_prev = nav_prev.replace(0, np.nan)
        hedge_ret = (pnl_usd / nav_prev).fillna(0.0)
        hedge_ret.name = "hedge_return"
        return hedge_ret
