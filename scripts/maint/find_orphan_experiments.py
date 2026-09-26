#!/usr/bin/env python3
"""List unreferenced top-level experiments using all Git-tracked text files.

A whole-word mention in any other tracked file retains a module. Only
__init__.py is excluded; private script names follow the same rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


def find_orphans(root: Path) -> list[str]:
    candidates = {
        path.stem: path.relative_to(root).as_posix()
        for path in (root / "experiments").glob("*.py")
        if path.name != "__init__.py"
    }
    referenced = set()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    # Scan each file once, retaining only tokens matching candidate stems.
    # Decode with surrogateescape so non-UTF-8 text is still searched.
    for raw_path in tracked.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = root / relative
        if not path.is_file():
            continue  # Includes unstaged moves and submodule directories.
        data = path.read_bytes()
        if b"\0" in data:
            continue  # Git's usual binary/text distinction.
        tokens = set(re.findall(r"\w+", data.decode("utf-8", errors="surrogateescape")))
        for stem in candidates.keys() & tokens:
            if relative != candidates[stem]:
                referenced.add(stem)
    return sorted(path for stem, path in candidates.items() if stem not in referenced)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="output a JSON array")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = find_orphans(root)
    if args.json:
        print(json.dumps(paths, indent=2))
    else:
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
