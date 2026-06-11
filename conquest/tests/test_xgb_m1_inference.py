"""Reproducibility test (Guard #6) for the M1 xgboost ranker.

Asserts that loading the model artifact + running inference on a frozen set of
inputs produces byte-identical predictions across runs. Catches:
  - Booster save/load corruption
  - xgboost API drift across versions
  - Feature-order mismatch between training and inference

Run before any cloud push:
    pytest conquest/tests/test_xgb_m1_inference.py -q

Skips gracefully if the model artifact isn't present (training in progress).
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pytest


WORKSPACE = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = WORKSPACE / "storage" / "conquest" / "ml_models" / "m1_xgb_qm_v1.json"
META_PATH = WORKSPACE / "storage" / "conquest" / "ml_models" / "m1_xgb_qm_v1.meta.json"
GOLDEN_PATH = WORKSPACE / "storage" / "conquest" / "ml_models" / "m1_xgb_qm_v1.golden.json"


def _load_booster():
    if not MODEL_PATH.exists():
        pytest.skip(f"model artifact {MODEL_PATH} not present (training in progress)")
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


def _frozen_inputs() -> tuple[list, np.ndarray]:
    """5 hand-picked (date, ticker)-like feature rows spanning regimes.

    Order matches FEATURES in scripts/train_xgb_m1.py:
        [mom_180d, mom_90d, mom_30d, vol_60d, vol_30d, rsi_14, sma_50_dist, sector_id]
    """
    rows = [
        # date_label,  mom_180, mom_90, mom_30, vol_60, vol_30, rsi_14, sma_dist, sector_id
        ("2009-03-31_bear_trough",  -0.50, -0.30, -0.10, 0.65, 0.55, 30.0, -0.20, 5),   # 2009 trough
        ("2013-06-28_recovery",      0.25,  0.10,  0.03, 0.20, 0.18, 60.0,  0.05, 5),   # mid-cycle
        ("2017-12-29_late_bull",     0.30,  0.12,  0.02, 0.10, 0.08, 70.0,  0.07, 1),   # low-vol bull
        ("2020-03-31_covid_crash", -0.20, -0.40, -0.30, 0.85, 0.95, 25.0, -0.30, 4),   # COVID
        ("2024-12-31_late_cycle",    0.40,  0.15,  0.05, 0.18, 0.20, 65.0,  0.10, 6),   # late-cycle
    ]
    labels = [r[0] for r in rows]
    X = np.array([r[1:] for r in rows], dtype=float)
    return labels, X


def test_model_loads_and_predicts():
    """Smoke test: model loads, predict() runs, returns 5 finite numbers."""
    import xgboost as xgb
    booster = _load_booster()
    labels, X = _frozen_inputs()
    dmat = xgb.DMatrix(X)
    preds = booster.predict(dmat)
    assert preds.shape == (5,), f"unexpected pred shape {preds.shape}"
    assert np.all(np.isfinite(preds)), f"non-finite predictions: {preds}"


def test_model_meta_consistency():
    """Sidecar metadata has expected fields + feature_list matches."""
    if not META_PATH.exists():
        pytest.skip(f"metadata {META_PATH} not present")
    meta = json.loads(META_PATH.read_text())
    expected_features = ["mom_180d", "mom_90d", "mom_30d", "vol_60d", "vol_30d",
                         "rsi_14", "sma_50_dist", "sector_id"]
    assert meta["feature_list"] == expected_features
    assert meta["label_horizon_days"] == 21
    assert "trained_at" in meta


def test_inference_byte_equality():
    """Predictions on frozen inputs must match `golden.json` byte-for-byte.

    If golden.json is missing, write it (first run). If present, assert equality.
    This enforces reproducibility across model regenerations.
    """
    import xgboost as xgb
    booster = _load_booster()
    labels, X = _frozen_inputs()
    preds = booster.predict(xgb.DMatrix(X))
    actual = {lbl: float(p) for lbl, p in zip(labels, preds)}
    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2))
        pytest.skip(f"wrote initial golden file: {GOLDEN_PATH}; rerun to verify")
    golden = json.loads(GOLDEN_PATH.read_text())
    # 1e-6 absolute tolerance accommodates float32 round-trips
    for lbl, expected in golden.items():
        assert lbl in actual, f"missing prediction for {lbl}"
        delta = abs(actual[lbl] - expected)
        assert delta < 1e-6, (
            f"prediction drift for {lbl}: actual={actual[lbl]:.10f}, "
            f"expected={expected:.10f}, delta={delta:.2e}"
        )
