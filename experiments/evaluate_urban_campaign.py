#!/usr/bin/env python3
"""Run the common fail-closed urban-navigation campaign evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnss_gpu.evaluation_contract import (
    build_reproducibility_manifest,
    default_command,
    evaluate_campaign,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="normalized campaign JSON")
    parser.add_argument("--output", type=Path, required=True, help="evaluation result JSON")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    input_path = args.input.resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    config = payload.pop("manifest_config", {})
    payload["reproducibility_manifest"] = build_reproducibility_manifest(
        repo_root=repo_root,
        input_paths=[input_path],
        config=config,
        command=default_command(),
    )
    result = evaluate_campaign(payload, repo_root)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
