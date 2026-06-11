"""cspec composite score — z-summed (60d ROC + 20d $-vol spike + 252d breakout proximity).

Cross-sectional z-scores per rebalance date. The composite is the score used
to rank survivors of cspec's directional/activity filter (5d ROC > 0 AND
$-vol spike > 1.5x) — top-3 by composite become the speculative basket.

The score weights all three axes equally (z-sum, no per-axis multiplier). v1
intentionally avoids tunable axis weights to keep the parameter surface
narrow and reduce overfitting risk against the GME/NVDA/AMD case studies.
A v2 candidate could parameterize axis weights once a v1 LIVE pin exists.
"""
from __future__ import annotations

import pandas as pd

from conquest.signals.breakout_proximity import breakout_proximity
from conquest.signals.volume_spike import dollar_volume_spike


def cspec_composite_score(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    momentum_lookback: int = 60,
    vol_lookback: int = 20,
    breakout_lookback: int = 252,
) -> pd.DataFrame:
    """Compute the cspec composite score for a panel of symbols.

    Args:
        closes: wide-form DataFrame of daily closes, indexed by date,
            columns are tickers/symbols.
        volumes: wide-form DataFrame of daily share volumes, same index/columns.
        momentum_lookback: bars for the ROC term (default 60d).
        vol_lookback: bars for the dollar-volume spike (default 20d).
        breakout_lookback: bars for the breakout-proximity high (default 252d).

    Returns:
        long-form DataFrame indexed by (date, symbol) with columns
        `roc`, `vol_spike`, `breakout_prox`, `score` (z-sum of the three).
        Rows where any component is NaN are dropped (typically the first
        `breakout_lookback` bars per symbol).
    """
    if not closes.columns.equals(volumes.columns):
        raise ValueError("closes and volumes must have identical columns")

    roc = closes.pct_change(periods=momentum_lookback)
    spike = closes.apply(
        lambda c: dollar_volume_spike(c, volumes[c.name], lookback=vol_lookback)
    )
    prox = closes.apply(lambda c: breakout_proximity(c, lookback=breakout_lookback))

    # Cross-sectional z per date (mean/std across symbols, not over time).
    def _cs_z(panel: pd.DataFrame) -> pd.DataFrame:
        mu = panel.mean(axis=1)
        sd = panel.std(axis=1).replace(0, pd.NA)
        return panel.sub(mu, axis=0).div(sd, axis=0)

    z_roc = _cs_z(roc)
    z_spk = _cs_z(spike)
    z_prx = _cs_z(prox)
    score = z_roc + z_spk + z_prx

    out = pd.concat(
        {
            "roc": roc.stack(),
            "vol_spike": spike.stack(),
            "breakout_prox": prox.stack(),
            "score": score.stack(),
        },
        axis=1,
    ).dropna()
    out.index.names = ["date", "symbol"]
    return out
