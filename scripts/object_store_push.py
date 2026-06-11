"""Push signal CSVs to QC Cloud Object Store via direct REST API.

Workaround for an observed Lean CLI bug where `lean cloud object-store set`
returns success=true but silently fails to upload the multipart body — leaving
cloud copies stale even though the CLI reports success. This shows up as the
live algorithm reporting "feeds older than 35 days" despite a fresh
`object-store set` run.

This script bypasses the CLI and posts directly to /api/v2/object/set with
proper multipart/form-data, which works reliably.

Usage:
  python scripts/object_store_push.py [<key> <path>]...

  If no args: pushes the canonical signal-CSV set used by cstag_voltgt_combined.
  With args: pushes only the specified (key, path) pairs.

Verify after upload by comparing local file size with
  `lean cloud object-store list conquest/<folder>/`
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

WORKSPACE = Path(__file__).resolve().parent.parent
QC_BASE = "https://www.quantconnect.com/api/v2"

# Default signal-CSV set the live algorithm reads at init. Update if new
# signals are added to the algorithm's Object Store reads.
DEFAULT_KEYS = [
    ("conquest/regime/daily.csv",                  "storage/conquest/regime/daily.csv"),
    ("conquest/regime/probability.csv",            "storage/conquest/regime/probability.csv"),
    ("conquest/vix/daily.csv",                     "storage/conquest/vix/daily.csv"),
    ("conquest/vix/term_ratio.csv",                "storage/conquest/vix/term_ratio.csv"),
    ("conquest/credit/hyg_ief_spread.csv",         "storage/conquest/credit/hyg_ief_spread.csv"),
    ("conquest/yield_curve/t10y2y.csv",            "storage/conquest/yield_curve/t10y2y.csv"),
    ("conquest/votes/cstability_4vote_daily.csv",  "storage/conquest/votes/cstability_4vote_daily.csv"),
    ("conquest/macro/unrate.csv",                  "storage/conquest/macro/unrate.csv"),
]


def load_credentials() -> tuple[str, str]:
    # CI path — env vars
    user_id = os.environ.get("QC_USER_ID")
    api_token = os.environ.get("QC_API_TOKEN")
    if user_id and api_token:
        return str(user_id), str(api_token)
    # Local dev path — ~/.lean/credentials
    cred_path = Path(os.path.expanduser("~/.lean/credentials"))
    if not cred_path.exists():
        raise SystemExit(
            "Set QC_USER_ID + QC_API_TOKEN env vars (CI) "
            f"OR run `lean login` to create {cred_path}.")
    data = json.loads(cred_path.read_text())
    return str(data["user-id"]), str(data["api-token"])


def load_org_id() -> str:
    # CI path
    org = os.environ.get("QC_ORG_ID")
    if org:
        return str(org)
    # Local dev path
    lean_json = WORKSPACE / "lean.json"
    if not lean_json.exists():
        raise SystemExit(
            "Set QC_ORG_ID env var (CI) OR have lean.json in the workspace root.")
    data = json.loads(lean_json.read_text())
    org = data.get("organization-id") or data.get("job-organization-id")
    if not org:
        raise SystemExit("organization-id not found in env or lean.json")
    return str(org)


def auth_headers(user_id: str, api_token: str) -> dict:
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{api_token}:{ts}".encode()).hexdigest()
    raw = f"{user_id}:{digest}".encode()
    return {
        "Authorization": "Basic " + base64.b64encode(raw).decode(),
        "Timestamp": ts,
    }


def _do_upload(user_id: str, api_token: str, org_id: str, key: str, path: Path) -> dict:
    """Single multipart POST attempt. Returns parsed JSON response."""
    with open(path, "rb") as f:
        files = {"objectData": (path.name, f, "application/octet-stream")}
        data = {"organizationId": org_id, "key": key}
        r = requests.post(
            f"{QC_BASE}/object/set",
            headers=auth_headers(user_id, api_token),
            data=data,
            files=files,
            timeout=120,
        )
    try:
        return r.json()
    except Exception:
        return {"success": False, "raw": r.text[:200]}


def _fetch_cloud_meta(user_id: str, api_token: str, org_id: str, key: str) -> tuple[int | None, float | None]:
    """Query /object/list for (size_bytes, modified_epoch) of `key`. Returns
    (None, None) if not found / API error. Uses METADATA only — allowed without
    the institutional export entitlement (unlike reading the object body)."""
    folder = "/".join(key.split("/")[:-1]) + "/"
    filename = key.split("/")[-1]
    r = requests.post(
        f"{QC_BASE}/object/list",
        headers=auth_headers(user_id, api_token),
        json={"organizationId": org_id, "path": folder},
        timeout=30,
    )
    try:
        resp = r.json()
    except Exception:
        return None, None
    if not resp.get("success"):
        return None, None
    for obj in resp.get("objects", []):
        if obj.get("key", "").endswith("/" + filename) or obj.get("key") == key:
            size = obj.get("size") or obj.get("bytes")
            mod_ts = None
            mod = obj.get("modified")
            if mod:
                try:
                    mod_ts = datetime.fromisoformat(str(mod).replace("Z", "+00:00")).timestamp()
                except Exception:
                    mod_ts = None
            return size, mod_ts
    return None, None


def _validate_csv_before_push(path: Path, key: str) -> bool:
    """Hard guard: refuse to push files that fail basic content validation.

    Catches the case where a previous ingester run wrote an empty CSV
    (header-only or zero rows) — pushing it overwrites the prior good
    copy in QC Object Store with garbage, silently breaking algorithms.

    Returns True if push should proceed, False to skip.

    Only validates .csv files (other formats — .json, .parquet — passed
    through unchecked since the validator framework is CSV-specific).
    """
    if not path.suffix.lower() == ".csv":
        return True
    try:
        # Cheap structural check: file > minimal-header size + has any data row
        if path.stat().st_size < 50:
            print(f"  REFUSE {key}: file size {path.stat().st_size}B — likely empty/header-only")
            return False
        with open(path) as f:
            header = f.readline()
            first_row = f.readline().strip()
            if not header.strip() or not first_row:
                print(f"  REFUSE {key}: header-only CSV (no data rows)")
                return False
    except Exception as e:
        print(f"  REFUSE {key}: pre-push validation error: {e}")
        return False
    return True


def push_one(user_id: str, api_token: str, org_id: str, key: str, path: Path,
             max_attempts: int = 3) -> bool:
    """Push a single (key, path) pair with retry + size verification.

    Returns True if cloud size matches local size after upload, False otherwise.
    Retries up to `max_attempts` times if the cloud copy doesn't match local
    after a push — this defends against the known Lean CLI bug where /object/set
    returns success=true but silently drops the file body.

    2026-05-27: added _validate_csv_before_push() guard — refuses to upload
    empty/header-only CSVs that would overwrite known-good cloud data.
    """
    if not path.exists():
        print(f"  SKIP {key}: local file missing ({path})")
        return False
    if not _validate_csv_before_push(path, key):
        return False
    local_size = path.stat().st_size

    for attempt in range(1, max_attempts + 1):
        push_start = time.time()
        resp = _do_upload(user_id, api_token, org_id, key, path)
        api_ok = bool(resp.get("success"))
        if not api_ok:
            errs = resp.get("errors", resp)
            print(f"  RETRY  {key:55s} attempt {attempt}/{max_attempts} — API error: {errs}")
            time.sleep(min(2 ** attempt, 10))
            continue
        # Verify by reading back cloud metadata (size + modified time).
        cloud_size, cloud_mod = _fetch_cloud_meta(user_id, api_token, org_id, key)
        if cloud_size is None:
            print(f"  WARN   {key:55s} attempt {attempt}/{max_attempts} — could not verify cloud meta; retrying")
            time.sleep(min(2 ** attempt, 10))
            continue
        if cloud_size == local_size:
            print(f"  ok     {key:55s} local={local_size:>7d}  cloud={cloud_size:>7d}  (attempt {attempt})")
            return True
        # Size differs — but QC's /object/list size is eventually-consistent and
        # lags a fresh write by seconds. Treat it as lag (success) ONLY when both:
        #   (a) the object was just (re)written (modified >= our push start), and
        #   (b) the size is CLOSE to local (a stale list read is at most ~1
        #       version / a few rows behind).
        # A wildly different cloud size (e.g. an old 4x-larger file the push is
        # NOT replacing) is a REAL mismatch and still fails even if modified
        # looks recent — so a genuine silent drop / wrong-content push is never
        # masked. (2026-06-07)
        size_close = abs((cloud_size or 0) - local_size) <= max(256, int(0.01 * local_size))
        if size_close and cloud_mod is not None and cloud_mod >= push_start - 5:
            print(f"  ok*    {key:55s} local={local_size:>7d}  cloud-list={cloud_size:>7d} "
                  f"(size-list lag; modified just now → push landed) (attempt {attempt})")
            return True
        print(f"  RETRY  {key:55s} attempt {attempt}/{max_attempts} — size mismatch: "
              f"local={local_size} cloud={cloud_size} (modified stale → not rewritten)")
        time.sleep(min(2 ** attempt, 10))

    print(f"  FAILED {key:55s} local={local_size:>7d}  cloud size never matched / not rewritten after {max_attempts} attempts")
    return False


def main() -> int:
    user_id, api_token = load_credentials()
    org_id = load_org_id()

    args = sys.argv[1:]
    if args:
        if len(args) % 2 != 0:
            raise SystemExit("usage: object_store_push.py [<key> <path>] ...")
        pairs = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    else:
        pairs = DEFAULT_KEYS

    print(f"Pushing {len(pairs)} files to QC Cloud Object Store (org={org_id}) ...")
    failures = 0
    for key, path_str in pairs:
        path = WORKSPACE / path_str if not os.path.isabs(path_str) else Path(path_str)
        if not push_one(user_id, api_token, org_id, key, path):
            failures += 1

    if failures:
        print(f"\n{failures} failure(s) — investigate.")
        return 1
    print(f"\nAll {len(pairs)} files pushed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
