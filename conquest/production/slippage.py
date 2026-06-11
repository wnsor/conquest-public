"""Slippage model setup — reads SLIPPAGE_MODEL parameter to pick a model.

Options (from config.json SLIPPAGE_MODEL parameter):
  - "volume_share"  → VolumeShareSlippageModel(0.025, price_impact) [recommended for live]
                      Default price_impact=0.1; scale via SLIPPAGE_PRICE_IMPACT param
                      for stress tests (e.g. 0.15 = 1.5× slippage).
  - "constant"      → ConstantSlippageModel(slippage_bps / 10000)
                      Default 1bp; scale via SLIPPAGE_CONSTANT_BPS param.
  - "none" / ""     → no slippage (Lean default — matches original BTs)

Stress-test usage:
  SLIPPAGE_MODEL=volume_share + SLIPPAGE_PRICE_IMPACT=0.15  → 1.5× of default impact
  SLIPPAGE_MODEL=volume_share + SLIPPAGE_PRICE_IMPACT=0.2   → 2.0× of default impact

Applied via set_security_initializer so every newly-added security inherits
the right model. Run AFTER add_equity() calls but BEFORE add_universe().
"""
from __future__ import annotations


def attach_slippage(algo) -> None:
    model = (algo.get_parameter("SLIPPAGE_MODEL") or "").strip().lower()

    if not model or model == "none":
        algo.log("[slippage] disabled (Lean default — no slippage)")
        return

    if model == "volume_share":
        try:
            from QuantConnect.Orders.Slippage import VolumeShareSlippageModel
            # Allow stress-test scaling via SLIPPAGE_PRICE_IMPACT (default 0.1).
            # Cap at 1.0 to prevent absurd values from typos.
            impact_str = (algo.get_parameter("SLIPPAGE_PRICE_IMPACT") or "0.1").strip()
            try:
                impact = float(impact_str)
                impact = max(0.0, min(impact, 1.0))
            except (ValueError, TypeError):
                impact = 0.1
            slip = VolumeShareSlippageModel(0.025, impact)
            label = f"VolumeShareSlippageModel(0.025, {impact:.3f})"
        except Exception as e:
            algo.log(f"[slippage] VolumeShare unavailable, falling back to constant: {e}")
            model = "constant"

    if model == "constant":
        try:
            from QuantConnect.Orders.Slippage import ConstantSlippageModel
            # Allow stress-test scaling via SLIPPAGE_CONSTANT_BPS (default 1bp).
            bps_str = (algo.get_parameter("SLIPPAGE_CONSTANT_BPS") or "1.0").strip()
            try:
                bps = float(bps_str)
                bps = max(0.0, min(bps, 1000.0))  # cap at 10% to prevent typos
            except (ValueError, TypeError):
                bps = 1.0
            slip = ConstantSlippageModel(bps / 10000.0)
            label = f"ConstantSlippageModel({bps}bp)"
        except Exception as e:
            algo.log(f"[slippage] ConstantSlippage unavailable: {e}")
            return

    if model not in ("volume_share", "constant"):
        algo.log(f"[slippage] unknown SLIPPAGE_MODEL={model!r}; using Lean default")
        return

    def _init_security(security):
        try:
            security.set_slippage_model(slip)
        except Exception:
            pass

    algo.set_security_initializer(_init_security)

    # BUG FIX (2026-05-13): set_security_initializer only affects securities
    # added AFTER this call. If harden() is called after add_equity (which is
    # the case in cstag/main.py + cstag_voltgt_combined/main.py + cresearch
    # because Object Store load needs to come first), the slippage model never
    # applies to anything. Defensive fix: iterate existing Securities and apply
    # directly. Phase 7/7b empirically confirmed this bug — all SLIPPAGE_*
    # parameter variants produced byte-identical results because no model was
    # being attached.
    try:
        for sym, sec in list(algo.Securities.items()):
            try:
                sec.set_slippage_model(slip)
            except Exception:
                pass
    except Exception as e:
        algo.log(f"[slippage] could not iterate existing Securities: {e}")

    algo.log(f"[slippage] using {label}")
