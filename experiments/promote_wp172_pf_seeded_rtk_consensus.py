#!/usr/bin/env python3
"""Promote independently agreeing PF-seeded RTK positions without FGO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_text_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest().upper()


def _tow_key(value: str | float) -> float:
    return round(float(value), 3)


def _distance(left: dict[str, float], right: dict[str, float]) -> float:
    return math.sqrt(
        sum((left[f"ecef_{axis}"] - right[f"ecef_{axis}"]) ** 2 for axis in "xyz")
    )


def read_gnssplusplus_pos(path: Path) -> dict[float, dict[str, float | int]]:
    """Read the fixed gnssplusplus POS schema emitted by gnss_solve."""

    output: dict[float, dict[str, float | int]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("%"):
                continue
            values = line.split()
            if len(values) < 25:
                raise ValueError(f"malformed gnssplusplus row in {path}")
            output[_tow_key(values[1])] = {
                "tow": float(values[1]),
                "ecef_x": float(values[2]),
                "ecef_y": float(values[3]),
                "ecef_z": float(values[4]),
                "status": int(values[8]),
                "num_satellites": int(values[9]),
                "ratio": float(values[11]),
                "prefit_residual_rms_m": float(values[18]),
                "update_nis_per_observation": float(values[23]),
            }
    return output


def select_consensus_candidates(
    seeded: dict[float, dict[str, float | int]],
    independent: dict[float, dict[str, float | int]],
    *,
    required_status: int,
    max_disagreement_m: float,
    max_prefit_residual_rms_m: float,
) -> dict[float, dict[str, float | int]]:
    """Apply the frozen truth-free gate before any reference is loaded."""

    selected: dict[float, dict[str, float | int]] = {}
    for tow, candidate in seeded.items():
        other = independent.get(tow)
        if other is None:
            continue
        if int(candidate["status"]) != required_status:
            continue
        if float(candidate["prefit_residual_rms_m"]) > max_prefit_residual_rms_m:
            continue
        if _distance(candidate, other) > max_disagreement_m:
            continue
        selected[tow] = candidate
    return selected


def promote(
    production_path: Path,
    seeded_path: Path,
    independent_path: Path,
    reference_path: Path,
    contract_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gate = contract["candidate_gate"]
    seeded = read_gnssplusplus_pos(seeded_path)
    independent = read_gnssplusplus_pos(independent_path)
    selected = select_consensus_candidates(
        seeded,
        independent,
        required_status=int(gate["required_seeded_status"]),
        max_disagreement_m=float(
            gate["max_independent_position_disagreement_m"]
        ),
        max_prefit_residual_rms_m=float(gate["max_prefit_residual_rms_m"]),
    )

    production = _read_csv(production_path)
    reference = _read_csv(reference_path)
    truth_by_tow = {
        _tow_key(row["GPS TOW (s)"]): row
        for row in reference
    }
    output: list[dict[str, Any]] = []
    gained = lost = before_good = after_good = selected_epochs = 0
    for row in production:
        tow = _tow_key(row["tow"])
        truth = truth_by_tow[tow]
        before = float(row["error_m"]) < 0.5
        candidate = selected.get(tow)
        position = (
            {f"ecef_{axis}": float(candidate[f"ecef_{axis}"]) for axis in "xyz"}
            if candidate is not None
            else {f"ecef_{axis}": float(row[f"ecef_{axis}"]) for axis in "xyz"}
        )
        error_m = math.sqrt(
            sum(
                (
                    position[f"ecef_{axis}"]
                    - float(truth[f"ECEF {axis.upper()} (m)"])
                )
                ** 2
                for axis in "xyz"
            )
        )
        after = error_m < 0.5
        before_good += before
        after_good += after
        gained += not before and after
        lost += before and not after
        selected_epochs += candidate is not None
        output.append(
            {
                "epoch": int(row["epoch"]),
                "tow": float(row["tow"]),
                **position,
                "error_m": error_m,
                "sub50cm": int(after),
                "fix": 0,
                "false_fix": 0,
                "source": (
                    "pf_seeded_rtk_consensus"
                    if candidate is not None
                    else "retained_production"
                ),
            }
        )
    summary = {
        "schema": "gnss_gpu_wp172_pf_seeded_rtk_consensus_result_v1",
        "contract": contract_path.as_posix(),
        "pf_only": contract["pf_only"],
        "runtime_fgo": contract["runtime_fgo"],
        "production_input_truth": contract["production_input_truth"],
        "truth_usage": "post_selection_full_denominator_audit_only",
        "input_hashes": {
            "production": _sha256(production_path),
            "seeded_rtk": _sha256(seeded_path),
            "independent_rtk": _sha256(independent_path),
            "reference": _sha256(reference_path),
            "contract": _sha256(contract_path),
        },
        "candidate_gate": gate,
        "mandatory_negative_holdouts": {
            holdout: {
                "accepted": False,
                "reason": contract["holdout_policy"][
                    "mandatory_negative_holdout_disposition"
                ],
            }
            for holdout in ("nagoya_wp53", "tokyo_wp129", "tokyo_wp156", "tokyo_wp168")
        },
        "m4_preserved_sha256": {
            path: _sha256(Path(path))
            for path in contract["m4_expected_sha256"]
        },
        "selected_epochs": selected_epochs,
        "full_denominator_epochs": len(output),
        "before_sub50cm_epochs": before_good,
        "after_sub50cm_epochs": after_good,
        "after_sub50cm_percent": 100.0 * after_good / len(output),
        "gained_epochs": gained,
        "lost_epochs": lost,
        "fix_epochs": 0,
        "false_fix_epochs": 0,
        "promotion_allowed": (
            gained > 0
            and lost == 0
            and all(
                _sha256(Path(path)) == expected
                for path, expected in contract["m4_expected_sha256"].items()
            )
        ),
    }
    return output, summary


def _write_trajectory(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--seeded-rtk", type=Path, required=True)
    parser.add_argument("--independent-rtk", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/evaluation/wp172_pf_seeded_rtk_consensus.json"),
    )
    parser.add_argument("--output-trajectory", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = promote(
        args.production,
        args.seeded_rtk,
        args.independent_rtk,
        args.reference,
        args.contract,
    )
    _write_trajectory(args.output_trajectory, rows)
    summary["output_trajectory"] = args.output_trajectory.as_posix()
    summary["output_trajectory_sha256"] = _sha256(args.output_trajectory)
    summary["output_trajectory_canonical_sha256"] = _canonical_text_sha256(
        args.output_trajectory
    )
    summary["output_trajectory_hash_normalization"] = "CRLF/LF normalized to LF"
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
