#!/usr/bin/env python3
"""Audit that the WP27 online diagnostic arm is operationally neutral."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


_OPERATIONAL_FIELDS = (
    "tow",
    "ecef_x",
    "ecef_y",
    "ecef_z",
    "fix",
    "gamma_fixed",
    "map_assignment_id",
    "gamma",
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-diagnostics", type=Path, required=True)
    parser.add_argument("--integrity-diagnostics", type=Path, required=True)
    parser.add_argument("--control-trajectory", type=Path, required=True)
    parser.add_argument("--integrity-trajectory", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args(argv)

    control = _rows(args.control_diagnostics)
    integrity = _rows(args.integrity_diagnostics)
    if len(control) != len(integrity):
        raise RuntimeError("control and integrity epoch counts differ")
    field_mismatches = {
        field: sum(left[field] != right[field] for left, right in zip(control, integrity))
        for field in _OPERATIONAL_FIELDS
    }
    position_delta = np.asarray(
        [
            [
                float(right[field]) - float(left[field])
                for field in ("ecef_x", "ecef_y", "ecef_z")
            ]
            for left, right in zip(control, integrity)
        ],
        dtype=np.float64,
    )
    control_hash = _sha256(args.control_trajectory)
    integrity_hash = _sha256(args.integrity_trajectory)
    summary = {
        "epochs": len(control),
        "operational_fields": list(_OPERATIONAL_FIELDS),
        "operational_field_mismatches": field_mismatches,
        "operational_mismatches": int(sum(field_mismatches.values())),
        "maximum_position_delta_m": float(
            np.max(np.linalg.norm(position_delta, axis=1))
        ),
        "control_trajectory_sha256": control_hash,
        "integrity_trajectory_sha256": integrity_hash,
        "trajectory_bit_identical": control_hash == integrity_hash,
        "integrity_anchor_epochs": sum(
            row["integrity_anchor_available"] == "1" for row in integrity
        ),
        "integrity_tdcp_intervals": sum(
            row["integrity_tdcp_available"] == "1" for row in integrity
        ),
        "integrity_map_sub50cm_epochs": sum(
            float(row["integrity_map_error_m"]) < 0.5 for row in integrity
        ),
        "truth_used_by_integrity_filter": False,
    }
    if summary["operational_mismatches"] or not summary["trajectory_bit_identical"]:
        raise RuntimeError("WP27 diagnostic arm changed operational output")
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
