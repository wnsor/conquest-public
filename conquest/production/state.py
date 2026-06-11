"""State persistence — survives QC restarts (auto-restart, IB session refresh,
node maintenance) by saving runtime state to Object Store.

Without persistence, every restart wipes in-memory Python attributes and the
algorithm forgets its DD watermarks, last rebalance time, voltgt overlay
position, recent alert state, etc. With persistence, the algorithm restores
its prior state at on_warmup_finished and continues seamlessly.

Pattern:
  algo._state_key      = "<project>/runtime_state.json"
  algo._persist_state()  → write current state to Object Store
  algo._restore_state()  → read prior state from Object Store; set defaults if absent

What's persisted (project decides via algo._state_fields list):
  - peak_combined_nav, peak_voltgt_nav (DD reference points)
  - last_rebalance_at (force-initial-rebalance gate; None on fresh deploy)
  - last_alert_state (deduped alert tracking)
  - voltgt overlay state (UVXY position tracking)
  - last_daily_check_at (watchdog freshness signal)
"""
from __future__ import annotations
import json
from datetime import datetime, timezone


def attach_state_persistence(algo, *, state_key: str | None = None) -> None:
    """Wire algo._persist_state() / algo._restore_state() helpers.

    The project's main.py should declare what to persist by either:
      a) setting algo._state_fields = [...] before initialize() returns, or
      b) overriding _state_payload() / _state_apply() if state needs custom serialization
    """
    if hasattr(algo, "_persist_state"):
        return  # project already has its own

    if state_key is None:
        # Default to <classname>/runtime_state.json
        cls = algo.__class__.__name__
        # Map common class names to project directories
        project = {
            "CstagVoltgtCombined":      "cstag_voltgt_combined",
            "CstagVoltgt67_33":         "cstag_voltgt_combined",  # legacy name
            "VoltgtStandalone":         "voltgt_standalone",
            "CstagAlgorithm":           "cstag",
            "CstabilityAlgorithm":      "cstability",
            "CgrowthAlgorithm":         "cgrowth",
            "ConquestFund":             "chybrid",
            "RegimeRotator":            "regime",
        }.get(cls, cls.lower())
        state_key = f"{project}/runtime_state.json"

    algo._state_key = state_key
    if not hasattr(algo, "_state_fields"):
        # Sensible defaults — projects can override
        algo._state_fields = [
            "peak_combined_nav",
            "peak_voltgt_nav",
            "last_rebalance_at",
            "last_daily_check_at",
            "combined_in_dd_lockout",
            "voltgt_overlay_active",
            "voltgt_overlay_entry_date",
            # Tracks the last time refresh_macro_signals() returned True.
            # The halt-on-stale guard uses this (not the CSV's last_date) to
            # distinguish "we lost FRED access" from "FRED data is legitimately
            # this old". Survives restarts so a long outage isn't reset just
            # by redeploying.
            "last_refresh_success_at",
        ]

    def _state_payload() -> dict:
        """Build the payload dict from algo attributes listed in _state_fields."""
        payload = {"saved_at": datetime.now(timezone.utc).isoformat()}
        for f in algo._state_fields:
            v = getattr(algo, f, None)
            # ISO-format datetimes for JSON
            if isinstance(v, datetime):
                v = v.isoformat()
            payload[f] = v
        return payload

    def _state_apply(payload: dict) -> None:
        """Restore algo attributes from a state payload dict.

        Datetime values are serialized as ISO strings. We try to restore them
        as datetimes when either (a) the existing attribute is already a
        datetime, or (b) the value LOOKS LIKE an ISO datetime string. The
        latter case matters for fields like ``last_rebalance_at`` that
        initialize to None on a fresh deploy — without ISO sniffing, the
        restored value would stay a string and break ``self.time -
        last_rebalance_at`` arithmetic.
        """
        for f in algo._state_fields:
            if f not in payload:
                continue
            v = payload[f]
            cur = getattr(algo, f, None)
            # (a) attribute is datetime-typed → parse
            # (b) attribute is None / unknown but value sniffs as ISO → parse
            if isinstance(v, str) and (
                isinstance(cur, datetime)
                or (cur is None and len(v) >= 10 and v[4] == "-" and v[7] == "-")
            ):
                try:
                    v = datetime.fromisoformat(v)
                except Exception:
                    pass
            try:
                setattr(algo, f, v)
            except Exception:
                pass

    def _persist_state() -> bool:
        try:
            payload = _state_payload()
            algo.object_store.save(algo._state_key, json.dumps(payload))
            return True
        except Exception as e:
            algo.log(f"[state] persist failed: {e}")
            return False

    def _restore_state() -> bool:
        try:
            if not algo.object_store.contains_key(algo._state_key):
                algo.log(f"[state] no prior state at {algo._state_key} — fresh deploy")
                return False
            txt = algo.object_store.read(algo._state_key)
            payload = json.loads(txt)
            _state_apply(payload)
            algo.log(f"[state] restored from {algo._state_key} (saved_at={payload.get('saved_at', '?')})")
            return True
        except Exception as e:
            algo.log(f"[state] restore failed: {e} — continuing with defaults")
            return False

    algo._state_payload = _state_payload
    algo._state_apply = _state_apply
    algo._persist_state = _persist_state
    algo._restore_state = _restore_state
