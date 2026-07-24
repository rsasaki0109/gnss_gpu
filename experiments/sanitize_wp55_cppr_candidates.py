#!/usr/bin/env python3
"""Remove post-selection truth audits from a WP55 CP/PR candidate source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sanitize(source_bytes: bytes) -> dict[str, Any]:
    source = json.loads(source_bytes)
    if bool(source.get("production_input_truth", True)):
        raise ValueError("candidate generation itself was truth-seeded")
    hypotheses = source.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("candidate source has no hypotheses")
    output = dict(source)
    output.pop("truth_seeded_oracle_diagnostic", None)
    output.pop("osm_road_local_supply_audit", None)
    output["hypotheses"] = [
        {key: value for key, value in row.items() if not key.startswith("audit_")}
        for row in hypotheses
    ]
    output["schema"] = "wp55_cppr_candidates_truthfree_v1"
    output["truth_usage"] = "none"
    output["production_input_truth"] = False
    output["development_source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    encoded = json.dumps(output, sort_keys=True)
    if "audit_" in encoded or "truth_seeded_oracle" in encoded:
        raise AssertionError("truth audit survived sanitization")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = sanitize(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
