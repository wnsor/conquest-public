"""Pre-compute the cstability 4-vote ensemble signal as a daily time series.

Mirrors cstability/main.py:307-388 vote logic exactly:
    vote_regime = (regime == "Stagflation")
    vote_prob   = (p_stagflation > 0.30)
    vote_credit = (hy_stress_proxy < -0.05)
    vote_vix    = (vix_term_ratio > 1.05)
    vote_count  = sum of the four booleans (0..4)

Each daily row uses an at-or-before lookup matching cstability's `_latest_value()`
semantics (cstability/main.py:307-316) — same-day inclusion, no T+1 leakage.

Output: storage/conquest/votes/cstability_4vote_daily.csv (date, vote_count).

This is published to QC Object Store so cgrowth and cgrowth_options can read
the same vote that cstability computes live, enabling vote-gated put entry
(cgrowth_options sub-fix b) and crisis-conditional equity gating (cgrowth
sub-fix c) for the GFC drawdown-fix sweep.

Pre-2011-06-30 the p_stagflation feed is unavailable, so vote_count is over a
3-input ensemble (regime + credit + VIX term). Pre-2008-03-31 the credit
feed is unavailable (HYG inception 2007 + 60d rolling lookback). The
3-of-4 / 2-of-3 degradation is documented in the audit JSON.

Usage:
    python scripts/compute_4vote_signal.py
    python scripts/compute_4vote_signal.py --cutoff 2018-12-31  # no-look-ahead check
    python scripts/compute_4vote_signal.py --validate           # spot-check vs known dates
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Thresholds — must match cstability/main.py:125-127
PROB_STAG_THRESHOLD = 0.30
CREDIT_STRESS_THRESHOLD = -0.05
VIX_TERM_THRESHOLD = 1.05

# Output series spans the full PIT-extended window plus modern era.
START = "2008-01-01"
# END dynamically tracks today so daily cron extends the vote series.
# Pre-2026-05-24 hardcoded "2025-12-31" — froze 4-vote signal at YE2025
# despite refresh runs, breaking LIVE risk-off triggers in 2026.
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def _load(path: Path, label: str, cutoff: pd.Timestamp | None = None) -> pd.DataFrame | None:
    """Load a signal CSV. If cutoff is set, drop rows after cutoff (forward-leak check)."""
    if not path.exists():
        print(f"  WARN: {path.relative_to(ROOT)} not found; {label} vote will return False")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if cutoff is not None:
        df = df.loc[df.index <= cutoff]
    print(f"  {label}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def _at_or_before(df: pd.DataFrame | None, column: str, ts: pd.Timestamp) -> float | None:
    """Replicate cstability/main.py:307-316 _latest_value() semantics."""
    if df is None or column not in df.columns:
        return None
    valid = df.index <= ts
    if not valid.any():
        return None
    return df.loc[valid, column].iloc[-1]


def compute_vote_series(
    regime_df: pd.DataFrame | None,
    probability_df: pd.DataFrame | None,
    credit_df: pd.DataFrame | None,
    vix_term_df: pd.DataFrame | None,
    start: str = START,
    end: str = END,
    include_prob: bool = True,
) -> pd.DataFrame:
    """Daily vote_count + per-vote breakdown for [start, end] business days.

    `include_prob=True` (default, v11 LIVE behavior) sums all 4 votes.
    `include_prob=False` (v11.2 LIVE — INCLUDE_PROB_VOTE=0 ablation) drops the
    P(Stagflation) GBM forecast vote so vote_count maxes at 3 — matches what
    cstability v11.2 computes internally, removing the cgrowth-vs-cstability
    drift that arises from the 4-vote cached CSV being read by cgrowth-standalone
    while CF / cstability live use the 3-vote ensemble.
    """
    daily_idx = pd.date_range(start, end, freq="B")
    rows = []
    for ts in daily_idx:
        # Mirror cstability/main.py:_update_votes vote rules
        regime = _at_or_before(regime_df, "regime", ts)
        v_regime = (regime == "Stagflation") if regime is not None else False

        p = _at_or_before(probability_df, "p_stagflation", ts)
        v_prob = (p is not None) and (float(p) > PROB_STAG_THRESHOLD)

        c = _at_or_before(credit_df, "hy_stress_proxy", ts)
        v_credit = (c is not None) and (float(c) < CREDIT_STRESS_THRESHOLD)

        v = _at_or_before(vix_term_df, "vix_term_ratio", ts)
        v_vix = (v is not None) and (float(v) > VIX_TERM_THRESHOLD)

        prob_contribution = int(v_prob) if include_prob else 0
        rows.append({
            "date": ts,
            "vote_regime": int(v_regime),
            "vote_prob": int(v_prob),
            "vote_credit": int(v_credit),
            "vote_vix": int(v_vix),
            "vote_count": int(v_regime) + prob_contribution + int(v_credit) + int(v_vix),
        })
    return pd.DataFrame(rows).set_index("date")


def write_output(votes: pd.DataFrame, out_path: Path) -> None:
    """Persist `vote_count` only — diagnostics columns kept out of the Object-Store CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    votes[["vote_count"]].to_csv(out_path)
    print(f"  -> {out_path.relative_to(ROOT)} ({len(votes)} rows)")


def cmd_compute(cutoff: str | None, include_prob: bool = True) -> pd.DataFrame:
    print(f"Loading input signals (cutoff={cutoff or 'none'}, include_prob={include_prob})...")
    cutoff_ts = pd.Timestamp(cutoff) if cutoff else None
    regime_df = _load(ROOT / "storage/conquest/regime/daily.csv", "regime", cutoff_ts)
    probability_df = _load(ROOT / "storage/conquest/regime/probability.csv", "probability", cutoff_ts) if include_prob else None
    credit_df = _load(ROOT / "storage/conquest/credit/hyg_ief_spread.csv", "credit", cutoff_ts)
    vix_term_df = _load(ROOT / "storage/conquest/vix/term_ratio.csv", "vix term", cutoff_ts)

    end_for_compute = cutoff if cutoff else END
    mode_label = "4-vote (regime+prob+credit+vix)" if include_prob else "3-vote v11.2 LIVE (regime+credit+vix)"
    print(f"\nComputing daily vote_count over {START} -> {end_for_compute} — {mode_label}...")
    votes = compute_vote_series(
        regime_df, probability_df, credit_df, vix_term_df,
        start=START, end=end_for_compute, include_prob=include_prob,
    )

    # Distribution summary
    print(f"\nvote_count distribution (full window):")
    print(votes["vote_count"].value_counts().sort_index().to_string())

    # GFC peak window spot-check (Sep 2008 - Mar 2009)
    gfc = votes.loc["2008-09-15":"2009-03-09"]
    print(f"\nGFC peak window (2008-09-15 -> 2009-03-09): {len(gfc)} rows")
    print(f"  vote_count distribution: {dict(gfc['vote_count'].value_counts().sort_index())}")
    print(f"  days with vote_count >= 2 (gate fires): {(gfc['vote_count'] >= 2).sum()}")
    print(f"  days with vote_count == 4 (all fire):    {(gfc['vote_count'] == 4).sum()}")

    return votes


def cmd_validate() -> None:
    """Spot-check 2024-Q2 against expected behavior + verify no-look-ahead."""
    print("=== Sanity validation: 2024-Q2 sample ===")
    votes_full = cmd_compute(cutoff=None)

    # Spot check: April 2024 was a normal market period; vote should be low (0-1)
    sample = votes_full.loc["2024-04-15"]
    print(f"\n2024-04-15 votes: regime={sample['vote_regime']} prob={sample['vote_prob']} "
          f"credit={sample['vote_credit']} vix={sample['vote_vix']} -> count={sample['vote_count']}")

    # No-look-ahead check: re-run with cutoff=2018-12-31 and confirm 2018 rows match
    print("\n=== No-look-ahead check: cutoff=2018-12-31 ===")
    votes_2018 = cmd_compute(cutoff="2018-12-31")

    # Compare 2018 rows in both runs — if they differ, something downstream depends on
    # future data (forward leak in regime_probabilities() GBM is the prime suspect).
    common = votes_2018.index.intersection(votes_full.index)
    same = votes_full.loc[common, "vote_count"].equals(votes_2018.loc[common, "vote_count"])
    print(f"\nNo-look-ahead verdict: full-sample 2018 rows {'MATCH' if same else 'DIFFER'} cutoff=2018 rows.")
    if not same:
        diff = (votes_full.loc[common, "vote_count"] != votes_2018.loc[common, "vote_count"]).sum()
        print(f"  {diff} differing rows out of {len(common)} — INVESTIGATE before trusting the signal.")
    return votes_full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", type=str, default=None, help="YYYY-MM-DD: drop input data after this date")
    parser.add_argument("--validate", action="store_true", help="run sanity + no-look-ahead checks (no write)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default depends on --vote-mode)")
    parser.add_argument("--vote-mode", type=str, default="4vote", choices=["4vote", "3vote"],
                        help="4vote: regime+prob+credit+vix (v11 LIVE); "
                             "3vote: regime+credit+vix (v11.2 LIVE — drops the ablated SJSU vote)")
    args = parser.parse_args()

    if args.validate:
        cmd_validate()
        return

    include_prob = (args.vote_mode == "4vote")
    default_out = (
        "storage/conquest/votes/cstability_4vote_daily.csv" if include_prob
        else "storage/conquest/votes/cstability_3vote_daily.csv"
    )
    out_path = args.out if args.out else default_out
    object_store_key = (
        "conquest/votes/cstability_4vote_daily.csv" if include_prob
        else "conquest/votes/cstability_3vote_daily.csv"
    )

    votes = cmd_compute(cutoff=args.cutoff, include_prob=include_prob)
    if args.cutoff is None:
        out = ROOT / out_path
        write_output(votes, out)
        print(f"\nDone. Push to Object Store with:")
        print(f"  lean cloud object-store set {object_store_key} {out_path}")


if __name__ == "__main__":
    main()
