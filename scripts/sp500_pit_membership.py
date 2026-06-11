"""Reconstruct point-in-time S&P 500 membership from the changes log.

Reads `storage/conquest/universe/sp1500_changes.csv` (a long history of
add/remove events for sp500 / sp400 / sp600) and `sp1500_current.csv`
(today's snapshot) and walks the change log backwards to derive the
constituent set as of any historical date.

Why: cgrowth's bake-off currently uses the CURRENT S&P 500 list applied
historically, which overstates returns by ~50-100 bps/yr per the academic
literature on survivorship bias (Brown/Goetzmann/Ross 1992 etc.) — names
that DROPPED OUT of the index (typically because they fell) are excluded
from the historical universe, so the backtest never gets to lose money
on them.

Run:
    python scripts/sp500_pit_membership.py --as-of 2014-12-31
    python scripts/sp500_pit_membership.py --quantify   # bias size by year
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent
UNIVERSE_DIR = WORKSPACE / "storage" / "conquest" / "universe"
SHARED_UNIVERSE_DIR = Path("~/Documents/workspace/storage/conquest/universe")


def _resolve_universe_dir() -> Path:
    """Worktree may shadow the symlinked storage dir locally."""
    for d in (UNIVERSE_DIR, SHARED_UNIVERSE_DIR):
        if (d / "sp1500_changes.csv").exists():
            return d
    raise FileNotFoundError(
        f"sp1500_changes.csv not found in {UNIVERSE_DIR} or {SHARED_UNIVERSE_DIR}"
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    udir = _resolve_universe_dir()
    current = pd.read_csv(udir / "sp1500_current.csv")
    changes = pd.read_csv(udir / "sp1500_changes.csv", parse_dates=["date"])
    return current, changes


def members_as_of(target: str, current: pd.DataFrame, changes: pd.DataFrame,
                  index_filter: str = "sp500") -> set[str]:
    """Return the set of tickers in `index_filter` (default sp500) on `target` date.

    Algorithm: today's set is `current`. Walk the changes log backwards from
    today to `target` date, undoing each event:
      - 'added'   on date D means ticker was NOT in the index on D-1 → remove it
      - 'removed' on date D means ticker WAS in the index on D-1   → add it back
    """
    target_ts = pd.Timestamp(target)
    members = set(current.loc[current["primary_index"] == index_filter, "ticker"].astype(str))

    # Walk backwards through events that happened AFTER target_ts
    after = changes[(changes["date"] > target_ts) & (changes["index"] == index_filter)]
    after = after.sort_values("date", ascending=False)

    for _, row in after.iterrows():
        ticker = str(row["ticker"])
        action = row["action"]
        if action == "added":
            members.discard(ticker)
        elif action == "removed":
            members.add(ticker)
    return members


def quantify_bias(current: pd.DataFrame, changes: pd.DataFrame,
                  start: str = "2014-01-01", end: str | None = None):
    # END default = today's year so the audit table stays current.
    # Pre-2026-05-24 was hardcoded "2024-12-31".
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    """Print year-end membership delta vs current to estimate bias size."""
    today_set = set(current.loc[current["primary_index"] == "sp500", "ticker"].astype(str))
    print(f"Current sp500 membership: {len(today_set)} names")
    print()
    print(f"{'Year-end':<12} {'PIT size':<10} {'In current':<12} {'Not in current':<14} {'Bias indicator'}")
    print("-" * 80)
    for year in range(int(start[:4]), int(end[:4]) + 1):
        d = f"{year}-12-31"
        pit = members_as_of(d, current, changes)
        in_cur = pit & today_set
        not_in_cur = pit - today_set
        # "Not in current" tickers represent names that were in the index then
        # but have since left. A backtest that uses only today's list misses
        # these names entirely — i.e., never trades them.
        bias_pct = 100.0 * len(not_in_cur) / max(len(pit), 1)
        print(f"{d:<12} {len(pit):<10} {len(in_cur):<12} {len(not_in_cur):<14} "
              f"{bias_pct:.1f}% of PIT universe absent from today's list")


def main() -> int:
    ap = argparse.ArgumentParser(description="Point-in-time S&P 500 membership")
    ap.add_argument("--as-of", default=None,
                    help="Date (YYYY-MM-DD) to report membership for")
    ap.add_argument("--quantify", action="store_true",
                    help="Print year-end bias indicator 2014-2024")
    args = ap.parse_args()

    current, changes = load_data()

    if args.quantify:
        quantify_bias(current, changes)
        return 0

    if args.as_of:
        members = members_as_of(args.as_of, current, changes)
        print(f"sp500 membership on {args.as_of}: {len(members)} tickers")
        # Print a few examples
        for t in sorted(members)[:20]:
            print(f"  {t}")
        if len(members) > 20:
            print(f"  ... and {len(members) - 20} more")
        return 0

    print("Specify --as-of YYYY-MM-DD or --quantify")
    return 1


if __name__ == "__main__":
    sys.exit(main())
