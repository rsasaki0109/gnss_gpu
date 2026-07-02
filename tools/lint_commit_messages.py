#!/usr/bin/env python3
"""Reject AI tool Co-authored-by trailers in commit messages.

Scans commits reachable from HEAD but not from ``--base`` (default
``origin/main``). Used in CI so the rule is enforced even when local hooks are
not installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

FORBIDDEN = re.compile(
    r"^co-authored-by:\s*.*("
    r"cursor|claude|anthropic|openai|copilot|github copilot|chatgpt|gemini|codex"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def commits_since(base: str) -> list[tuple[str, str]]:
    out = subprocess.check_output(
        ["git", "log", f"{base}..HEAD", "--format=%H%x00%B%x00"],
        text=True,
    )
    commits: list[tuple[str, str]] = []
    for chunk in out.split("\0\0"):
        chunk = chunk.strip("\0")
        if not chunk:
            continue
        sha, _, body = chunk.partition("\0")
        commits.append((sha, body))
    return commits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Exclude commits already on this ref (default: origin/main)",
    )
    args = parser.parse_args()

    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", args.base],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        print(f"lint_commit_messages: base ref not found: {args.base}", file=sys.stderr)
        return 1

    violations: list[tuple[str, str]] = []
    for sha, body in commits_since(args.base):
        if FORBIDDEN.search(body):
            violations.append((sha[:12], body))

    if not violations:
        return 0

    print("AI Co-authored-by trailers are not allowed:", file=sys.stderr)
    for sha, body in violations:
        print(f"\n--- {sha} ---", file=sys.stderr)
        for line in body.splitlines():
            if FORBIDDEN.search(line):
                print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
