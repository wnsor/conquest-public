"""Copy conquest/ into each Lean project's local conquest/ subdirectory.

QC cloud push (``lean cloud push``) only ships files inside the Lean project
directory. To use the conquest package in cloud backtests or live trading,
each project needs a self-contained copy of conquest/. This script regenerates
those copies on demand.

    workspace/conquest/  →  cstability/conquest/   (gitignored copy)
    workspace/conquest/  →  cgrowth/conquest/      (gitignored copy)

The in-project copies are listed in ``.gitignore``; never edit them directly.
After running this script, run ``lean cloud push --project cstability`` (or
cgrowth) to ship the bundle.

Usage
-----
    python scripts/sync_to_lean.py
    python scripts/sync_to_lean.py --target cstability    # only one project
    python scripts/sync_to_lean.py --dry-run              # preview, no copy
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent
SOURCE = WORKSPACE / "conquest"
TARGETS = ["cstability", "cgrowth", "chybrid"]
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache",
    # secrets.py opens a YAML file — QC sandbox forbids open(); not needed at runtime
    # in Lean Algorithms (FRED/BLS/IB keys are research-only).
    "secrets.py",
)


def sync_one(target: str, dry_run: bool = False) -> int:
    dst = WORKSPACE / target / "conquest"

    if not SOURCE.exists():
        print(f"ERROR: source {SOURCE} doesn't exist", file=sys.stderr)
        return 1
    if not (WORKSPACE / target).exists():
        # Silently skip targets that aren't present in this worktree.
        # TARGETS lists every Lean project that COULD be synced; research clones
        # (combined_v2_research, cstability_research) only exist in specific
        # worktrees. Returning 0 (skip) lets `--target all` work across all
        # worktrees regardless of which research clones happen to be local.
        return 0

    if dry_run:
        n_files = sum(
            1 for p in SOURCE.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
        print(f"  [DRY-RUN] {SOURCE} → {dst}  ({n_files} files would be copied)")
        return 0

    if dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(SOURCE, dst, ignore=IGNORE)
    n_files = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"  ✓ {target:12s} → {dst.relative_to(WORKSPACE)}  ({n_files} files)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync conquest/ into Lean projects pre-cloud-push")
    ap.add_argument("--target", choices=TARGETS + ["all"], default="all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = TARGETS if args.target == "all" else [args.target]
    print(f"Syncing conquest/ → {len(targets)} Lean project(s):")
    for t in targets:
        rc = sync_one(t, dry_run=args.dry_run)
        if rc != 0:
            return rc

    if not args.dry_run:
        print(
            "\nDone. To push to QC cloud:\n"
            "    lean object-store set --key conquest/regime/daily.csv --path storage/conquest/regime/daily.csv\n"
            "    lean object-store set --key conquest/vix/daily.csv    --path storage/conquest/vix/daily.csv\n"
            "    lean object-store set --key conquest/universe/sp500.csv --path storage/conquest/universe/sp500.csv\n"
            "    lean cloud push --project cstability\n"
            "    lean cloud push --project cgrowth\n"
            "    lean cloud push --project cgrowth_options    # v11 candidate (puts overlay)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
