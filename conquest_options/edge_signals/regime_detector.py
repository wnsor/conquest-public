"""Phase 8 — single-label regime classifier for strategies to gate on.

DESIGN (per CLAUDE.md + project_regime_rotation_rejected):
  NOT a router that rotates strategies in/out. Per prior empirical work,
  strategy rotation by regime HURTS — false-positive switches and
  re-entry frictions dominate the alpha.

  Instead: a single regime label exposed via `ctx.regime` that strategies
  inspect like any other gate. Each strategy decides whether to fire
  given the regime — and an already-open position is NEVER closed by
  regime change. Only the entry decision is regime-conditioned.

REGIME LABELS:
  bull_low_vol   VIX<15, vote=0, GEX long_gamma, term contango.
                 Best regime for: LEAPS (D1), GEX SPY calls (A_*), straddles
                 on calm names (less likely IV expansion = less premium pay).

  bull_high_vol  VIX 15-20, vote=0. Bullish but vol elevated.
                 Best for: pre-earnings strangles (C2; IV crush less impactful),
                 PEAD (A5*; earnings vol is the driver).

  neutral        VIX 20-25, vote 0-1. Mixed market.
                 Most strategies fire normally; sizing baseline.

  warning        VIX 25-30, OR vote>=1, OR term backwardation.
                 Stop most call entries. B1 SPY crisis put fires here.

  crisis         VIX>30, OR vote>=2.
                 No new call entries. B1 already fired (180d cooldown).
                 Wait for rebound.

  rebound        Inherited from ctx.crisis_state when CrisisDetector
                 transitions out of crisis. Triggers CrisisReboundBasket +
                 D2 Tepper aggressively (180d-locked).

  recovery       Inherited from ctx.crisis_state. Normal calls resume;
                 D1 LEAPS re-arm.
"""
from __future__ import annotations


REGIME_LABELS = [
    "bull_low_vol",
    "bull_high_vol",
    "neutral",
    "warning",
    "crisis",
    "rebound",
    "recovery",
]


def classify_regime(
    *,
    vix: float | None,
    vote_count: int | None,
    gex_regime: str | None = None,
    term_regime: str | None = None,
    crisis_state: str | None = None,
) -> str:
    """Single-label regime classifier. Inputs are all per-day context fields
    already in StrategyContext. Returns one of REGIME_LABELS.

    Precedence (most-severe-first; crisis_state takes priority over VIX/vote):
      1. crisis_state in ("rebound", "recovery", "capitulation", "crash") →
         pass through (rebound/recovery) or map to "crisis"
      2. VIX > 30 or vote >= 2 → "crisis"
      3. VIX 25-30 or vote >= 1 or term backwardation → "warning"
      4. VIX 20-25 → "neutral"
      5. VIX 15-20 → "bull_high_vol"
      6. VIX < 15 → "bull_low_vol"
    """
    # Crisis-state passthrough (takes priority — CrisisDetector has more
    # context than VIX alone, including peak tracking).
    if crisis_state in ("rebound", "recovery"):
        return crisis_state
    if crisis_state in ("capitulation", "crash"):
        return "crisis"

    if vix is None:
        return "neutral"

    # Hard regime thresholds (VIX-based with vote/term modulators)
    if vix >= 30 or (vote_count is not None and vote_count >= 2):
        return "crisis"
    if (vix >= 25 or
            (vote_count is not None and vote_count >= 1) or
            term_regime == "backwardation"):
        return "warning"
    if vix >= 20:
        return "neutral"
    if vix >= 15:
        return "bull_high_vol"
    return "bull_low_vol"


# Default per-strategy regime allow-list. Strategies can override.
# Empty list means "fire in any regime" (default).
STRATEGY_REGIME_DEFAULTS: dict[str, list[str]] = {
    # Bullish call strategies — only fire in bull regimes
    "gex_spy_selective":       ["bull_low_vol", "bull_high_vol", "neutral", "recovery"],
    "gex_spy_baseline":   ["bull_low_vol", "bull_high_vol", "neutral"],
    "momentum_otm_calls":      ["bull_low_vol", "bull_high_vol"],
    "pead_megacap":     ["bull_low_vol", "bull_high_vol", "neutral"],
    "pead_midcap":      ["bull_low_vol", "bull_high_vol"],
    "insider_buy_calls":      ["bull_low_vol", "bull_high_vol", "neutral", "recovery"],
    "uoa_following_calls":          ["bull_low_vol", "bull_high_vol", "neutral"],
    # Volatility-sellers via straddles (long both legs — IV-crush risk)
    "earnings_straddle":  ["bull_low_vol", "bull_high_vol", "neutral"],
    "earnings_strangle":  ["bull_low_vol", "bull_high_vol", "neutral"],
    # LEAPS — only buy in low-vol bull markets
    "cgrowth_leaps":     ["bull_low_vol", "recovery"],
    # Crisis-window strategies — fire ONLY in their specific regime
    "tepper_vbottom_leaps":  ["rebound"],
    "crisis_rebound_basket":     ["rebound"],
    "spy_crisis_put":         ["warning", "crisis"],
    # Single-stock momentum-failure put — fires in mid-cycle, not bull or crisis
    "momentum_failure_put":  ["bull_high_vol", "neutral", "warning"],
}


def is_strategy_allowed_in_regime(strategy_id: str, regime: str) -> bool:
    """Check if `strategy_id` is allowed to fire in `regime` per the default
    regime allow-list. Returns True if the strategy has no allow-list or
    the regime is in its allow-list."""
    allow = STRATEGY_REGIME_DEFAULTS.get(strategy_id)
    if not allow:
        return True  # no list = always allowed
    return regime in allow
