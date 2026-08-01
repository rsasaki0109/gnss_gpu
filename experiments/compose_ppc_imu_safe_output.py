#!/usr/bin/env python3
"""Materialize an authoritative truth-free PPC IMU PF/FGO decision stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    from experiments.run_multisd_fgo_ppc_cv import read_solutions
except ModuleNotFoundError:
    from run_multisd_fgo_ppc_cv import read_solutions  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tracker(path: Path) -> dict[float, dict[str, str]]:
    output: dict[float, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                tow = round(float(row["tow"]), 3)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid tracker TOW on line {line_number}") from exc
            if tow in output:
                raise ValueError(f"duplicate tracker TOW {tow}")
            output[tow] = row
    return output


def compose_safe_output(baseline_pos: Path, tracker_csv: Path) -> list[dict[str, object]]:
    """Use the safe tracker as the sole FIX authority; never inherit status 4."""

    baseline = read_solutions(baseline_pos)
    tracker = _read_tracker(tracker_csv)
    rows: list[dict[str, object]] = []
    for index, tow in enumerate(sorted(set(baseline) | set(tracker))):
        tracker_row = tracker.get(tow)
        baseline_row = baseline.get(tow)
        tracker_fixed = (
            tracker_row is not None and tracker_row.get("shadow_fixed") == "1"
        )
        if tracker_fixed:
            source = "imu_pf_fgo_fixed"
            position = tuple(float(tracker_row[axis]) for axis in "xyz")
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"non-finite fixed tracker position at TOW {tow}")
            status = 4
        elif baseline_row is not None:
            source = "baseline_float"
            position = tuple(float(baseline_row[axis]) for axis in "xyz")
            status = 3
        elif tracker_row is not None:
            source = "imu_pf_fgo_float"
            position = tuple(float(tracker_row[axis]) for axis in "xyz")
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"non-finite tracker position at TOW {tow}")
            status = 3
        else:  # pragma: no cover - set union makes this unreachable
            continue
        rows.append(
            {
                "epoch_index": index,
                "tow": tow,
                "shadow_fixed": int(status == 4),
                "status": status,
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "source": source,
                "baseline_status": (
                    int(baseline_row["status"]) if baseline_row is not None else ""
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pos", type=Path, required=True)
    parser.add_argument("--tracker-csv", type=Path, required=True)
    parser.add_argument("--tracker-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    tracker_summary = json.loads(args.tracker_summary.read_text(encoding="utf-8"))
    if (
        tracker_summary.get("schema") != "gnss_gpu_ppc_basin_fgo_tracker_v1"
        or tracker_summary.get("production_input_truth") is not False
        or tracker_summary.get("truth_usage") != "none"
        or tracker_summary.get("output_sha256") != _sha256(args.tracker_csv)
    ):
        parser.error("tracker summary integrity check failed")
    rows = compose_safe_output(args.baseline_pos, args.tracker_csv)
    if not rows:
        parser.error("safe output is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "gnss_gpu_ppc_imu_safe_output_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "fix_authority": "imu_pf_fgo_tracker_only",
        "legacy_fixed_status_inherited": False,
        "epochs": len(rows),
        "fixed_epochs": sum(int(row["shadow_fixed"]) for row in rows),
        "source_counts": {
            source: sum(row["source"] == source for row in rows)
            for source in (
                "imu_pf_fgo_fixed",
                "baseline_float",
                "imu_pf_fgo_float",
            )
        },
        "input_sha256": {
            "baseline_pos": _sha256(args.baseline_pos),
            "tracker_csv": _sha256(args.tracker_csv),
            "tracker_summary": _sha256(args.tracker_summary),
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
