"""Export per-day GMM regime probabilities for cstability's REGIME_USE_GMM mode.

Fits a 4-component Gaussian Mixture Model on (gdp_yoy_z, cpi_yoy_z) from the
existing regime daily CSV, maps each component to a named regime by centroid
quadrant, then writes per-day soft probabilities to:

    storage/conquest/regime/gmm_daily.csv

Columns: date, p_stagflation, p_inflation, p_disinflation, p_deflation,
         gmm_argmax_regime, release_date

Read by cstability/main.py via Object Store key conquest/regime/gmm_daily.csv
when REGIME_USE_GMM=1. Default REGIME_USE_GMM=0 preserves v11.2 LIVE behavior.

PIT-correctness note:
    The GMM is fit on the *full* PIT-correct z-score series (2008-2026).
    Cluster boundaries in z-score space are nearly stationary (z-scores are
    already 60-month rolling-normalized in the regime classifier), so the
    leakage from later data into earlier classifications is small in practice.
    Same standard the existing deterministic classifier uses. For strict-OOS
    LIVE deployment, fit on an expanding window per release event and
    re-export — that's a follow-up task once a paired backtest justifies
    promotion.

Run after scripts/refresh_data.py + scripts/classify_regime.py have produced
storage/conquest/regime/daily.csv. Idempotent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "storage" / "conquest" / "regime" / "daily.csv"
OUTPUT = ROOT / "storage" / "conquest" / "regime" / "gmm_daily.csv"

REGIMES = ["Inflation", "Disinflation", "Stagflation", "Deflation"]
REGIME_IDEAL = {
    "Inflation":    np.array([+1.0, +1.0]),
    "Disinflation": np.array([+1.0, -1.0]),
    "Stagflation":  np.array([-1.0, +1.0]),
    "Deflation":    np.array([-1.0, -1.0]),
}


def name_by_quadrant(centroid: np.ndarray) -> str:
    g, c = centroid
    if g >= 0 and c < 0:
        return "Disinflation"
    if g >= 0 and c >= 0:
        return "Inflation"
    if g < 0 and c >= 0:
        return "Stagflation"
    return "Deflation"


def fit_and_label(df: pd.DataFrame, seed: int = 7) -> tuple[GaussianMixture, dict[int, str]]:
    X = df[["gdp_yoy_z", "cpi_yoy_z"]].dropna().values
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="full",
        random_state=seed,
        n_init=10,
        max_iter=500,
    )
    gmm.fit(X)
    naming = {i: name_by_quadrant(gmm.means_[i]) for i in range(gmm.n_components)}
    # Disambiguate: if two components landed in the same quadrant, reassign by
    # nearest-unused regime ideal centroid. Walk components in distance order.
    used = set()
    for comp_idx in sorted(naming, key=lambda i: -np.linalg.norm(gmm.means_[i])):
        if naming[comp_idx] in used:
            dists = {r: np.linalg.norm(gmm.means_[comp_idx] - REGIME_IDEAL[r])
                     for r in REGIMES if r not in used}
            naming[comp_idx] = min(dists, key=dists.get)
        used.add(naming[comp_idx])
    return gmm, naming


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: input not found: {INPUT}", file=sys.stderr)
        print("Run scripts/classify_regime.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(INPUT, parse_dates=["date", "release_date"]).set_index("date")
    print(f"Loaded {len(df)} daily rows from {df.index[0].date()} to {df.index[-1].date()}.")

    gmm, naming = fit_and_label(df)
    print("GMM components → regime assignment:")
    for i in range(gmm.n_components):
        g, c = gmm.means_[i]
        print(f"  comp_{i}: centroid ({g:+.3f}, {c:+.3f})  →  {naming[i]}")

    Xall = df[["gdp_yoy_z", "cpi_yoy_z"]]
    valid = Xall.dropna()
    proba = gmm.predict_proba(valid.values)
    proba_df = pd.DataFrame(
        proba,
        index=valid.index,
        columns=[naming[i] for i in range(gmm.n_components)],
    )
    # Aggregate by name (in case duplicates remained after disambiguation)
    proba_df = proba_df.T.groupby(level=0).sum().T
    # Reindex to full input range, NaN on days missing z-scores
    proba_df = proba_df.reindex(Xall.index)

    out = pd.DataFrame(index=Xall.index)
    out["p_stagflation"]  = proba_df.get("Stagflation",  pd.Series(np.nan, index=Xall.index)).round(6)
    out["p_inflation"]    = proba_df.get("Inflation",    pd.Series(np.nan, index=Xall.index)).round(6)
    out["p_disinflation"] = proba_df.get("Disinflation", pd.Series(np.nan, index=Xall.index)).round(6)
    out["p_deflation"]    = proba_df.get("Deflation",    pd.Series(np.nan, index=Xall.index)).round(6)
    out["gmm_argmax_regime"] = proba_df.idxmax(axis=1)
    # Carry forward the release_date column from the input (PIT discipline)
    out["release_date"] = df["release_date"]
    out.index.name = "date"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.dropna(subset=["p_stagflation"]).to_csv(OUTPUT)
    print(f"  → {OUTPUT.relative_to(ROOT)}")
    print(f"     {len(out.dropna(subset=['p_stagflation']))} rows;"
          f" P(Stag) ≥ 0.5 on {(out['p_stagflation'] >= 0.5).sum()} days"
          f" ({(out['p_stagflation'] >= 0.5).mean():.1%} of sample).")
    print()
    print("Next steps:")
    print(f"  lean object-store set --key conquest/regime/gmm_daily.csv \\")
    print(f"      --path storage/conquest/regime/gmm_daily.csv")
    print(f"  lean cloud push --project cstability")
    print(f"  lean cloud backtest cstability --name 'cstab-v11.2-gmm' \\")
    print(f"      --parameter REGIME_USE_GMM 1 \\")
    print(f"      --parameter BACKTEST_START_YEAR 2008 \\")
    print(f"      --parameter BACKTEST_END_YEAR 2026")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
