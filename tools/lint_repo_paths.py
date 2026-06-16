#!/usr/bin/env python3
"""Lint tracked repository paths for cross-platform safety.

Guards against two recurring problems:

1. Windows MAX_PATH (260 chars). Long sweep filenames under
   ``experiments/results`` once reached 266 chars and broke ``git reset --hard``
   and clean clones on Windows. We cap relative path length and basename length
   well under the limit.
2. Regenerable data artifacts (csv/pkl/pos/npz/parquet) getting re-tracked under
   ``experiments/results``. These are reproducible via the CLIs recorded in
   ``internal_docs/plan.md`` and belong outside git (or in object storage),
   except for the curated ``paper_assets`` set.

Run from the repo root. Exits non-zero (and prints every offending path) when a
violation is found, so it can gate CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

# Directories/extensions that must not be tracked as generated artifacts.
FORBIDDEN_ARTIFACT_PREFIXES = ("experiments/results/",)
FORBIDDEN_ARTIFACT_SUFFIXES = {".csv", ".pkl", ".pos", ".npz", ".parquet"}
# Curated exceptions that are intentionally versioned.
ARTIFACT_ALLOW_PREFIXES = ("experiments/results/paper_assets/",)


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], text=True)
    return [p for p in out.split("\0") if p]


def is_forbidden_artifact(path: str) -> bool:
    if not path.startswith(FORBIDDEN_ARTIFACT_PREFIXES):
        return False
    if path.startswith(ARTIFACT_ALLOW_PREFIXES):
        return False
    return PurePosixPath(path).suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-path", type=int, default=180,
                    help="max tracked relative path length in chars (default 180)")
    ap.add_argument("--max-name", type=int, default=96,
                    help="max basename length in chars (default 96)")
    ap.add_argument("--max-depth", type=int, default=10,
                    help="max directory depth (default 10)")
    args = ap.parse_args()

    failures: list[str] = []
    for path in tracked_files():
        name = PurePosixPath(path).name
        if len(path) > args.max_path:
            failures.append(f"PATH TOO LONG ({len(path)} > {args.max_path}): {path}")
        if len(name) > args.max_name:
            failures.append(f"BASENAME TOO LONG ({len(name)} > {args.max_name}): {path}")
        depth = path.count("/")
        if depth > args.max_depth:
            failures.append(f"PATH TOO DEEP ({depth} > {args.max_depth}): {path}")
        if is_forbidden_artifact(path):
            failures.append(f"GENERATED ARTIFACT TRACKED: {path}")

    if failures:
        print(f"repo path lint: {len(failures)} violation(s)\n", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        print(
            "\nGenerated artifacts under experiments/results are regenerable and "
            "should not be tracked (see .gitignore). Long paths break Windows "
            "clones (MAX_PATH 260). Use short run-ids + metadata sidecars instead.",
            file=sys.stderr,
        )
        return 1

    print("repo path lint: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
