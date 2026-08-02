#!/usr/bin/env python3
"""Select a causal FLOAT trajectory without changing the safe FIX authority."""

from __future__ import annotations

import argparse
from collections import deque
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


def _read_safe_output(path: Path) -> dict[float, dict[str, str]]:
    output: dict[float, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), start=2):
            try:
                tow = round(float(row["tow"]), 3)
                int(row["status"])
                position = tuple(float(row[axis]) for axis in "xyz")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid safe-output row {line_number}") from exc
            if tow in output:
                raise ValueError(f"duplicate safe-output TOW {tow}")
            if not all(math.isfinite(value) for value in position):
                raise ValueError(f"non-finite safe-output position at TOW {tow}")
            output[tow] = row
    return output


def _validate_inputs(
    safe_output: Path,
    safe_summary: Path,
    float_candidate: Path,
    candidate_manifest: Path,
    expected_route: str | None = None,
) -> None:
    safe_metadata = json.loads(safe_summary.read_text(encoding="utf-8"))
    candidate_metadata = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    if (
        safe_metadata.get("schema") != "gnss_gpu_ppc_imu_safe_output_v1"
        or safe_metadata.get("production_input_truth") is not False
        or safe_metadata.get("truth_usage") != "none"
        or safe_metadata.get("output_sha256") != _sha256(safe_output)
    ):
        raise ValueError("safe-output summary integrity check failed")
    candidate_hashes = candidate_metadata.get("output_sha256", {})
    if (
        candidate_metadata.get("schema") != "gnss_gpu_ppc_float_candidate_run_v1"
        or candidate_metadata.get("production_input_truth") is not False
        or candidate_metadata.get("truth_usage") != "none"
        or candidate_hashes.get("position") != _sha256(float_candidate)
        or (
            expected_route is not None
            and candidate_metadata.get("route") != expected_route
        )
    ):
        raise ValueError("float-candidate manifest integrity check failed")


def compose_causal_float_selector(
    safe_output_csv: Path,
    float_candidate_pos: Path,
    *,
    health_window_epochs: int = 500,
    health_fixed_fraction: float = 0.9,
    maximum_gap_s: float = 1.0,
) -> list[dict[str, object]]:
    """Compose positions causally while preserving the safe stream's FIX rows.

    Candidate status 4 is only a trajectory-health observation. Candidate rows
    selected at non-safe-FIX epochs are always emitted as FLOAT.
    """

    if health_window_epochs < 1:
        raise ValueError("health_window_epochs must be positive")
    if not 0.0 <= health_fixed_fraction <= 1.0:
        raise ValueError("health_fixed_fraction must be in [0, 1]")
    if maximum_gap_s <= 0.0:
        raise ValueError("maximum_gap_s must be positive")

    safe = _read_safe_output(safe_output_csv)
    candidate = read_solutions(float_candidate_pos)
    history: deque[int] = deque(maxlen=health_window_epochs)
    previous_candidate_tow: float | None = None
    rows: list[dict[str, object]] = []

    for index, tow in enumerate(sorted(set(safe) | set(candidate))):
        safe_row = safe.get(tow)
        candidate_row = candidate.get(tow)
        if candidate_row is not None:
            if (
                previous_candidate_tow is not None
                and tow - previous_candidate_tow > maximum_gap_s
            ):
                history.clear()
            history.append(int(candidate_row["status"] == 4))
            previous_candidate_tow = tow

        health_ready = len(history) == health_window_epochs
        health_fraction = sum(history) / len(history) if history else 0.0
        safe_fixed = safe_row is not None and int(safe_row["status"]) == 4
        candidate_fixed_observation = (
            candidate_row is not None and int(candidate_row["status"]) == 4
        )
        candidate_healthy = (
            candidate_row is not None
            and health_ready
            and health_fraction >= health_fixed_fraction
        )

        if safe_fixed:
            selected = safe_row
            source = "safe_fixed"
            status = 4
        elif candidate_fixed_observation or candidate_healthy:
            selected = candidate_row
            source = (
                "float_candidate_fixed_observation"
                if candidate_fixed_observation
                else "float_candidate_healthy"
            )
            status = 3
        elif safe_row is not None:
            selected = safe_row
            source = "safe_primary_float"
            status = 3
        elif candidate_row is not None:
            selected = candidate_row
            source = "float_candidate_only"
            status = 3
        else:  # pragma: no cover - set union makes this unreachable
            continue

        position = tuple(float(selected[axis]) for axis in "xyz")
        if not all(math.isfinite(value) for value in position):
            raise ValueError(f"non-finite selected position at TOW {tow}")
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
                "safe_status": int(safe_row["status"]) if safe_row else "",
                "candidate_status": (
                    int(candidate_row["status"]) if candidate_row else ""
                ),
                "candidate_health_ready": int(health_ready),
                "candidate_health_fixed_fraction": health_fraction,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        required=True,
        choices=(
            "tokyo_run1",
            "tokyo_run2",
            "tokyo_run3",
            "nagoya_run1",
            "nagoya_run2",
            "nagoya_run3",
        ),
    )
    parser.add_argument("--safe-output", type=Path, required=True)
    parser.add_argument("--safe-summary", type=Path, required=True)
    parser.add_argument("--float-candidate-pos", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--health-window-epochs", type=int, default=500)
    parser.add_argument("--health-fixed-fraction", type=float, default=0.9)
    parser.add_argument("--maximum-gap-s", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        _validate_inputs(
            args.safe_output,
            args.safe_summary,
            args.float_candidate_pos,
            args.candidate_manifest,
            args.route,
        )
        rows = compose_causal_float_selector(
            args.safe_output,
            args.float_candidate_pos,
            health_window_epochs=args.health_window_epochs,
            health_fixed_fraction=args.health_fixed_fraction,
            maximum_gap_s=args.maximum_gap_s,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("selected output is empty")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_counts = {
        source: sum(row["source"] == source for row in rows)
        for source in (
            "safe_fixed",
            "float_candidate_fixed_observation",
            "float_candidate_healthy",
            "float_candidate_only",
            "safe_primary_float",
        )
    }
    summary = {
        "schema": "gnss_gpu_ppc_causal_float_selector_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "forward_only": True,
        "route": args.route,
        "fix_authority": "safe_input_only",
        "candidate_fixed_status_inherited": False,
        "policy": {
            "health_window_epochs": args.health_window_epochs,
            "health_fixed_fraction": args.health_fixed_fraction,
            "maximum_gap_s": args.maximum_gap_s,
        },
        "epochs": len(rows),
        "fixed_epochs": source_counts["safe_fixed"],
        "source_counts": source_counts,
        "input_sha256": {
            "safe_output": _sha256(args.safe_output),
            "safe_summary": _sha256(args.safe_summary),
            "float_candidate_pos": _sha256(args.float_candidate_pos),
            "candidate_manifest": _sha256(args.candidate_manifest),
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
