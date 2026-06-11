"""Pull a signal CSV from QC Cloud Object Store via direct REST API.

Counterpart to scripts/object_store_push.py. Used by the daily data-refresh
GitHub Action to fetch existing data state before computing incremental
updates.

Authentication priority (so this works in both local dev and CI):
  1. Env vars QC_USER_ID + QC_API_TOKEN + QC_ORG_ID (preferred in CI)
  2. ~/.lean/credentials + lean.json (local dev fallback)

Usage:
  # Pull a single key
  python scripts/data/object_store_pull.py \\
      --key conquest/insider/form4_opportunistic_buys_daily.csv \\
      --out storage/conquest/insider/form4_opportunistic_buys_daily.csv

  # Pull from a manifest file (one "key path" per line)
  python scripts/data/object_store_pull.py --manifest scripts/data/refresh_manifest.txt
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

WORKSPACE = Path(__file__).resolve().parent.parent.parent
QC_BASE = "https://www.quantconnect.com/api/v2"


def load_credentials_and_org() -> tuple[str, str, str]:
    """Resolve (user_id, api_token, org_id) from env vars (CI) or local files."""
    user_id = os.environ.get("QC_USER_ID")
    api_token = os.environ.get("QC_API_TOKEN")
    org_id = os.environ.get("QC_ORG_ID")
    if user_id and api_token and org_id:
        return str(user_id), str(api_token), str(org_id)
    # Fall back to local files (developer-laptop path)
    cred_path = Path(os.path.expanduser("~/.lean/credentials"))
    if not cred_path.exists():
        raise SystemExit(
            "Set env vars QC_USER_ID + QC_API_TOKEN + QC_ORG_ID for CI, "
            f"OR run `lean login` to create {cred_path}.")
    data = json.loads(cred_path.read_text())
    user_id = str(data["user-id"])
    api_token = str(data["api-token"])
    lean_json = WORKSPACE / "lean.json"
    if lean_json.exists():
        org_id = json.loads(lean_json.read_text()).get("organization-id")
    if not org_id:
        raise SystemExit("organization-id not found in env or lean.json")
    return user_id, api_token, str(org_id)


def auth_headers(user_id: str, api_token: str) -> dict:
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{api_token}:{ts}".encode()).hexdigest()
    raw = f"{user_id}:{digest}".encode()
    return {
        "Authorization": "Basic " + base64.b64encode(raw).decode(),
        "Timestamp": ts,
    }


def fetch_one(user_id: str, api_token: str, org_id: str, key: str,
              out_path: Path) -> bool:
    """Download `key` from Object Store into `out_path`. Returns True on success.

    Returns False (and prints a note) if the key doesn't exist in cloud — this
    is a normal first-run state, not a hard error."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # /object/get returns a presigned URL we then GET
    r = requests.post(
        f"{QC_BASE}/object/get",
        headers=auth_headers(user_id, api_token),
        json={"organizationId": org_id, "keys": [key]},
        timeout=60,
    )
    try:
        resp = r.json()
    except Exception:
        print(f"  FAILED {key:55s} HTTP {r.status_code}: {r.text[:200]}")
        return False
    if not resp.get("success"):
        errs = resp.get("errors", [])
        # Missing key returns a specific error message
        if any("not found" in str(e).lower() or "does not exist" in str(e).lower()
               for e in errs):
            print(f"  miss   {key:55s} (key not in cloud yet — bootstrap state)")
            return False
        print(f"  FAILED {key:55s} {errs}")
        return False
    # The response contains the URL inside objects[].url
    urls = resp.get("objects", [])
    if not urls:
        print(f"  miss   {key:55s} (no objects returned)")
        return False
    url = urls[0].get("url")
    if not url:
        print(f"  miss   {key:55s} (no presigned URL)")
        return False
    # Stream the actual content
    r2 = requests.get(url, timeout=300, stream=True)
    if r2.status_code != 200:
        print(f"  FAILED {key:55s} download HTTP {r2.status_code}")
        return False
    with open(out_path, "wb") as f:
        for chunk in r2.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
    sz = out_path.stat().st_size
    print(f"  ok     {key:55s} {sz:>10d} bytes -> {out_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull QC Object Store keys to local files")
    ap.add_argument("--key", help="Single key to pull")
    ap.add_argument("--out", help="Output file path for --key mode")
    ap.add_argument("--manifest", help="Manifest file (one 'key relative_path' per line)")
    args = ap.parse_args()

    user_id, api_token, org_id = load_credentials_and_org()

    pairs: list[tuple[str, Path]] = []
    if args.key:
        if not args.out:
            ap.error("--out is required when --key is used")
        pairs.append((args.key, Path(args.out)))
    elif args.manifest:
        for line in Path(args.manifest).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key, rel = parts[0], parts[1]
            pairs.append((key, WORKSPACE / rel))
    else:
        ap.error("must pass --key/--out OR --manifest")

    print(f"Pulling {len(pairs)} key(s) from Object Store...")
    n_ok, n_miss = 0, 0
    for key, out_path in pairs:
        if fetch_one(user_id, api_token, org_id, key, out_path):
            n_ok += 1
        else:
            n_miss += 1
    print(f"\nPull summary: ok={n_ok}, missing={n_miss}, total={len(pairs)}")
    # A missing key on first run is not an error; failures-only would be a
    # different state. Always exit 0 — the orchestrator handles missing-data.
    return 0


if __name__ == "__main__":
    sys.exit(main())
