#!/usr/bin/env python3
"""Lint tracked repository paths for cross-platform safety.

Guards the Windows ``MAX_PATH`` (260 chars) hazard: long sweep filenames under
``experiments/results`` once reached 266 chars and broke ``git reset --hard`` and
clean clones on Windows. We cap tracked relative path length, basename length,
and directory depth well under that limit.

This intentionally does *not* police whether result CSVs are tracked at all --
the repo deliberately versions a curated set of result deliverables (the site /
paper build chain reads them straight from ``experiments/results``), so an
"artifact must not be tracked" rule would produce false positives. The portable
invariant we enforce is purely path length.

Run from the repo root. Exits non-zero (and prints every offending path) when a
violation is found, so it can gate CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import PurePosixPath


def tracked_files() -> list[str]:
    out = subprocess.check_output(["git", "ls-files", "-z"], text=True)
    return [p for p in out.split("\0") if p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-path", type=int, default=180,
                    help="max tracked relative path length in chars (default 180)")
    ap.add_argument("--max-name", type=int, default=128,
                    help="max basename length in chars (default 128)")
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

    if failures:
        print(f"repo path lint: {len(failures)} violation(s)\n", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        print(
            "\nLong paths break Windows clones (MAX_PATH 260). Keep tracked paths "
            "short -- for sweep outputs use short run-ids + metadata sidecars and "
            "leave the bulk untracked (see .gitignore).",
            file=sys.stderr,
        )
        return 1

    print("repo path lint: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
