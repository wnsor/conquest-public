"""Loader for the workspace-root secret.yaml.

`secret.yaml` is gitignored. It holds API keys for FRED + BLS, plus an IB block
that stays empty until Phase 4 (live trading). Missing keys return empty strings
rather than raising — callers should validate before using.
"""
from __future__ import annotations
from pathlib import Path
import yaml


_SECRET_FILE = Path(__file__).resolve().parent.parent / "secret.yaml"


def _load() -> dict:
    if not _SECRET_FILE.exists():
        return {}
    with _SECRET_FILE.open() as f:
        return yaml.safe_load(f) or {}


def get(key: str, default: str = "") -> str:
    return _load().get(key, default) or default


def fred_api_key() -> str:
    return get("fred_api_key")


def bls_api_key() -> str:
    return get("bls_api_key")


def ib_credentials() -> dict:
    """Return the IB block. Empty strings mean 'not configured' (Phase 4 gate)."""
    block = _load().get("ib") or {}
    return {
        "account":      block.get("account", ""),
        "user":         block.get("user", ""),
        "password":     block.get("password", ""),
        "trading_mode": block.get("trading_mode", "paper"),
    }
