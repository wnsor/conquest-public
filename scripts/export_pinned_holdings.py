"""Export per-month holdings of the v8 pinned models for the webapp hover panel.

Output: ``storage/conquest/models/pinned_holdings.csv``
Columns: date, cstability_holdings, cstability_regime, cgrowth_holdings, cgrowth_vix_state

Holdings are encoded as a pipe-separated ticker list (e.g., "AAPL|MSFT|NVDA|GOOG|META").
The webapp loads this CSV and shows the holdings for the date the user is hovering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from conquest.backtest import IBCostModel, backtest
from conquest.data.sp500 import sector_map as sp500_sector_map
from conquest.models import (
    DualMomentum, RegimeRotator, VixGated, SectorCapped, ThreeLayerStacked,
    MultiSignalVote,
)
from conquest.data.vix_term import (
    fetch_vix_term, credit_stress_proxy, vix_term_inversion,
)
from conquest.regime.probability import regime_probabilities, probability_to_daily


ETF_PRICES = WORKSPACE / "data" / "alternative" / "conquest" / "raw" / "prices" / "etf_basket_daily.parquet"
STOCK_PRICES = WORKSPACE / "data" / "alternative" / "conquest" / "raw" / "prices" / "sp500_daily.parquet"
VIX_CACHE = WORKSPACE / "data" / "alternative" / "conquest" / "raw" / "prices" / "vix_daily.parquet"
REGIME_CSV = WORKSPACE / "storage" / "conquest" / "regime" / "daily.csv"
OUTPUT = WORKSPACE / "storage" / "conquest" / "models" / "pinned_holdings.csv"


def main() -> int:
    etf_prices = pd.read_parquet(ETF_PRICES).dropna(how="all", axis=1)
    stock_prices = pd.read_parquet(STOCK_PRICES).dropna(how="all", axis=1)
    regime = pd.read_csv(REGIME_CSV, index_col=0, parse_dates=True)["regime"]
    vix = pd.read_parquet(VIX_CACHE).iloc[:, 0]
    smap = sp500_sector_map()

    period_start, period_end = "2014-01-01", "2024-12-31"
    etf_p = etf_prices.loc[period_start:period_end]
    stock_p = stock_prices.loc[period_start:period_end]

    # cstability v8: dual_momentum_top3_regime_rotated
    cs_model = RegimeRotator(
        DualMomentum(top_n=3, lookback=252),
        regime_series=regime,
        regime_baskets={"Stagflation": ("GLD", "TIP", "TLT")},
    )
    cs_signals = cs_model.signal(etf_p)
    cs_result = backtest(etf_p, cs_signals, IBCostModel(bps_per_turnover=2.0),
                         initial_capital=25000, rebalance_freq="ME")

    # v9.2 three-layer stack — same base as v8 + credit + VIX-term voting
    vt_panel = fetch_vix_term(start=str(etf_p.index[0].date()),
                              end=str(etf_p.index[-1].date()))
    vix_ratio_v9 = vix_term_inversion(vt_panel["vix"], vt_panel["vix3m"])
    credit_proxy_v9 = credit_stress_proxy(etf_p["HYG"], etf_p["IEF"], lookback_days=60)
    v9_model = ThreeLayerStacked(
        RegimeRotator(DualMomentum(top_n=3, lookback=252),
                      regime_series=regime,
                      regime_baskets={"Stagflation": ("GLD", "TIP", "TLT")}),
        regime_series=regime,
        credit_stress_series=credit_proxy_v9,
        vix_term_series=vix_ratio_v9,
        defensive_basket=("GLD", "TIP", "TLT"),
    )
    v9_signals = v9_model.signal(etf_p)
    v9_result = backtest(etf_p, v9_signals, IBCostModel(bps_per_turnover=2.0),
                         initial_capital=25000, rebalance_freq="ME")

    # v10 four-layer stack — adds P(Stagflation) probability forecast as 4th vote
    monthly_macro = (
        pd.read_csv(REGIME_CSV, index_col=0, parse_dates=True)[["gdp_yoy", "cpi_yoy"]]
        .resample("ME").last().dropna()
    )
    prob_df = regime_probabilities(monthly_macro["gdp_yoy"], monthly_macro["cpi_yoy"])
    prob_daily = probability_to_daily(prob_df, daily_index=etf_p.index)
    v10_votes = [
        regime.isin(["Stagflation"]),
        prob_daily["p_stagflation"] > 0.30,
        credit_proxy_v9 < -0.05,
        vix_ratio_v9 > 1.05,
    ]
    v10_model = MultiSignalVote(
        RegimeRotator(DualMomentum(top_n=3, lookback=252),
                      regime_series=regime,
                      regime_baskets={"Stagflation": ("GLD", "TIP", "TLT")}),
        vote_signals=v10_votes,
        defensive_basket=("GLD", "TIP", "TLT"),
        blend_weights=[0.0, 0.15, 0.40, 0.75, 1.0],
        name_suffix="four_layer",
    )
    v10_signals = v10_model.signal(etf_p)
    v10_result = backtest(etf_p, v10_signals, IBCostModel(bps_per_turnover=2.0),
                          initial_capital=25000, rebalance_freq="ME")

    # cgrowth v8: dual_momentum_vix_gated_sector_capped_top5
    cg_model = SectorCapped(
        VixGated(DualMomentum(top_n=5, lookback=252), vix_series=vix),
        sector_map=smap,
        max_per_sector=0.30,
    )
    cg_signals = cg_model.signal(stock_p)
    cg_result = backtest(stock_p, cg_signals, IBCostModel(bps_per_turnover=5.0),
                         initial_capital=25000, rebalance_freq="ME")

    def holdings_strings(weights_df: pd.DataFrame) -> pd.Series:
        """For each row, return pipe-separated tickers with non-zero weight."""
        def fmt(row):
            held = row[row.abs() > 1e-6]
            if held.empty:
                return ""
            return "|".join(f"{t}:{w:.3f}" for t, w in held.sort_values(ascending=False).items())
        return weights_df.apply(fmt, axis=1)

    cs_holdings = holdings_strings(cs_result.weights)
    cg_holdings = holdings_strings(cg_result.weights)
    v9_holdings = holdings_strings(v9_result.weights)
    v10_holdings = holdings_strings(v10_result.weights)

    # Align all to the ETF index (master), forward-fill where needed
    idx = cs_result.weights.index
    regime_aligned = regime.reindex(idx, method="ffill")
    v9_holdings = v9_holdings.reindex(idx, method="ffill")
    v10_holdings = v10_holdings.reindex(idx, method="ffill")

    # VIX state: simple risk-off flag based on >25/<15 hysteresis
    vix_aligned = vix.reindex(idx, method="ffill")
    vix_state = pd.Series("normal", index=idx)
    risk_off = False
    for i, v in enumerate(vix_aligned):
        if pd.isna(v):
            continue
        if not risk_off and v > 25.0:
            risk_off = True
        elif risk_off and v < 15.0:
            risk_off = False
        vix_state.iloc[i] = "risk_off" if risk_off else "normal"

    cg_holdings_aligned = cg_holdings.reindex(idx, method="ffill")

    out = pd.DataFrame({
        "cstability_holdings": cs_holdings,
        "cstability_regime": regime_aligned,
        "cgrowth_holdings": cg_holdings_aligned,
        "cgrowth_vix_state": vix_state,
        "v9_three_layer_holdings": v9_holdings,
        "v10_four_layer_holdings": v10_holdings,
    })
    out.index.name = "date"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT)
    print(f"Wrote {len(out)} rows to {OUTPUT}")
    print(f"  cstability holdings non-empty:  {(out['cstability_holdings'] != '').sum()}")
    print(f"  cgrowth holdings non-empty:     {(out['cgrowth_holdings'] != '').sum()}")
    print(f"  v9 3-layer holdings non-empty:  {(out['v9_three_layer_holdings'] != '').sum()}")
    print(f"  v10 4-layer holdings non-empty: {(out['v10_four_layer_holdings'] != '').sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
