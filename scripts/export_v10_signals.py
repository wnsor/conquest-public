"""Export the 3 daily signal CSVs needed by cstability v10's 4-vote ensemble.

v10 uses MultiSignalVote with 4 boolean stress signals:
    1. regime classifier flags Stagflation       (already in storage/conquest/regime/daily.csv)
    2. P(Stagflation) > 0.30                     (this script writes P(Stag) daily)
    3. HY-IEF 60d log spread < -0.05             (this script writes hy_stress_proxy daily)
    4. VIX/VIX3M ratio > 1.05                    (this script writes vix_term_ratio daily)

The Lean cstability/main.py reads these CSVs at Initialize() via Object Store,
then on each daily check computes the vote count and uses it to blend the base
allocation (top-3 momentum + Stagflation rotation) with the defensive basket
(GLD/TIP/TLT) per blend_weights = [0, 0.15, 0.40, 0.75, 1.0].

Output CSVs (all gitignored, regenerated as data refreshes):
    storage/conquest/credit/hyg_ief_spread.csv          # date, hy_stress
    storage/conquest/vix/term_ratio.csv                 # date, vix_term_ratio
    storage/conquest/regime/probability.csv             # date, p_stagflation (+ other p_*)

Run after `scripts/refresh_data.py` (which refreshes regime CSV with monthly
GDP/CPI YoY) and yfinance HYG/IEF/VIX/VIX3M caches. Idempotent.
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conquest.data.vix_term import (
    fetch_vix_term, credit_stress_proxy, vix_term_inversion,
)
from conquest.regime.probability import regime_probabilities, probability_to_daily

START = "2008-01-01"
# END dynamically tracks today so daily cron runs extend the series.
# Pre-2026-05-24 this was hardcoded "2025-12-31" — that froze HYG-IEF + VIX-term
# signals at YE2025 despite daily refresh runs, causing 5-month-stale LIVE inputs.
END = pd.Timestamp.today().strftime("%Y-%m-%d")


def export_credit_stress():
    """Daily HY-IEF 60d log spread proxy. Negative values = credit stress."""
    out_path = ROOT / "storage" / "conquest" / "credit" / "hyg_ief_spread.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching HYG and IEF from yfinance...")
    hyg = yf.download("HYG", start=START, end=END, auto_adjust=False, progress=False)["Close"]
    ief = yf.download("IEF", start=START, end=END, auto_adjust=False, progress=False)["Close"]
    if isinstance(hyg, pd.DataFrame): hyg = hyg.iloc[:, 0]  # flatten if MultiIndex
    if isinstance(ief, pd.DataFrame): ief = ief.iloc[:, 0]

    proxy = credit_stress_proxy(hyg, ief, lookback_days=60).dropna()
    proxy.index.name = "date"
    proxy.to_frame().to_csv(out_path)
    print(f"  → {out_path.relative_to(ROOT)}")
    print(f"     {len(proxy)} daily rows; range "
          f"{proxy.index[0].date()} → {proxy.index[-1].date()}; "
          f"latest value = {proxy.iloc[-1]:.4f}")


def export_vix_term():
    """Daily VIX/VIX3M ratio. > 1.05 = backwardation = stress regime."""
    out_path = ROOT / "storage" / "conquest" / "vix" / "term_ratio.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching VIX, VIX3M from yfinance via fetch_vix_term()...")
    vt = fetch_vix_term(start=START, end=END)
    ratio = vix_term_inversion(vt["vix"], vt["vix3m"]).dropna()
    ratio.index.name = "date"
    ratio.to_frame().to_csv(out_path)
    print(f"  → {out_path.relative_to(ROOT)}")
    print(f"     {len(ratio)} daily rows; range "
          f"{ratio.index[0].date()} → {ratio.index[-1].date()}; "
          f"latest value = {ratio.iloc[-1]:.4f}")


def export_regime_probability():
    """Daily P(each regime), forward-projected from monthly GDP/CPI YoY."""
    out_path = ROOT / "storage" / "conquest" / "regime" / "probability.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regime_csv = ROOT / "storage" / "conquest" / "regime" / "daily.csv"
    if not regime_csv.exists():
        raise FileNotFoundError(
            f"{regime_csv} not found. Run `python scripts/classify_regime.py` first."
        )

    print(f"Loading {regime_csv.relative_to(ROOT)} for monthly GDP/CPI YoY...")
    r_df = pd.read_csv(regime_csv, index_col=0, parse_dates=True)
    monthly = r_df[["gdp_yoy", "cpi_yoy"]].resample("ME").last().dropna()
    print(f"  monthly history: {len(monthly)} rows ({monthly.index[0].date()} → {monthly.index[-1].date()})")

    prob_df = regime_probabilities(monthly["gdp_yoy"], monthly["cpi_yoy"])
    daily_idx = pd.date_range(monthly.index[0], END, freq="B")
    prob_daily = probability_to_daily(prob_df, daily_index=daily_idx).dropna()
    prob_daily.index.name = "date"
    prob_daily.to_csv(out_path)
    print(f"  → {out_path.relative_to(ROOT)}")
    print(f"     {len(prob_daily)} daily rows; range "
          f"{prob_daily.index[0].date()} → {prob_daily.index[-1].date()}; "
          f"latest P(Stag) = {prob_daily['p_stagflation'].iloc[-1]:.3f}")


if __name__ == "__main__":
    export_credit_stress()
    export_vix_term()
    export_regime_probability()
    print("\nAll v10 signals exported. Lean cstability/main.py can now be refactored to use them.")
