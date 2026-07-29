#!/usr/bin/env python3
"""Evaluate a locked Phase 5 cross-domain result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_gpu.cross_domain_validation import evaluate_cross_domain
from gnss_gpu.evaluation_contract import write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate_cross_domain(payload, args.repo_root.resolve())
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
