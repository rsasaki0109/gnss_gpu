#!/usr/bin/env python3
"""Compose a truth-free integrity-guarded library/PF-FGO FIX stream."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    from experiments.evaluate_ppc_official_score import evaluate_route
    from experiments.run_multisd_fgo_ppc_cv import read_solutions
except ModuleNotFoundError:
    from evaluate_ppc_official_score import evaluate_route  # type: ignore[no-redef]
    from run_multisd_fgo_ppc_cv import read_solutions  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv_by_tow(path: Path) -> dict[float, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "tow" not in (reader.fieldnames or []):
            raise ValueError(f"missing tow column in {path}")
        output: dict[float, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                tow = round(float(row["tow"]), 3)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid TOW in {path} line {line_number}") from exc
            if not math.isfinite(tow):
                raise ValueError(f"non-finite TOW in {path} line {line_number}")
            if tow in output:
                raise ValueError(f"duplicate TOW {tow} in {path} line {line_number}")
            output[tow] = row
        return output


def compose_safe_union(
    monitor_pos: Path,
    active_pos: Path,
    active_integrity_csv: Path,
    tracker_csv: Path,
    *,
    motion_innovation_limit_m: float = 0.25,
    maximum_causal_arc_resets: int = 2,
    promotion_streak_epochs: int = 2,
) -> list[dict[str, object]]:
    """Create decisions using estimator artifacts only; no reference is accepted."""

    if not math.isfinite(motion_innovation_limit_m) or motion_innovation_limit_m < 0:
        raise ValueError("motion innovation limit must be finite and non-negative")
    if maximum_causal_arc_resets < 0 or promotion_streak_epochs < 2:
        raise ValueError("reset limit must be non-negative and streak must be at least two")
    monitor = read_solutions(monitor_pos)
    active = read_solutions(active_pos)
    integrity = _read_csv_by_tow(active_integrity_csv)
    tracker = _read_csv_by_tow(tracker_csv)
    tows = sorted(active)
    if not tows:
        raise ValueError("active solution stream is empty")
    output: list[dict[str, object]] = []
    promotion_streak = 0
    for index, tow in enumerate(tows):
        active_row = active[tow]
        monitor_row = monitor.get(tow)
        telemetry = integrity.get(tow)
        if telemetry is None:
            raise ValueError(f"missing integrity telemetry at TOW {tow}")
        monitor_fixed = monitor_row is not None and int(monitor_row["status"]) == 4
        active_fixed = int(active_row["status"]) == 4
        promoted = active_fixed and not monitor_fixed
        promotion_streak = promotion_streak + 1 if promoted else 0

        strict_surplus = telemetry.get("satellite_par_surplus_passed") == "1"
        motion_innovation_m = math.inf
        motion_pass = False
        if index >= 2 and int(telemetry.get("causal_arc_resets") or 0) <= int(
            maximum_causal_arc_resets
        ):
            current = tuple(float(active_row[axis]) for axis in "xyz")
            previous = tuple(float(active[tows[index - 1]][axis]) for axis in "xyz")
            previous2 = tuple(float(active[tows[index - 2]][axis]) for axis in "xyz")
            predicted = tuple(2.0 * previous[axis] - previous2[axis] for axis in range(3))
            motion_innovation_m = math.dist(current, predicted)
            motion_pass = motion_innovation_m <= float(motion_innovation_limit_m)
        guarded_promotion = promoted and (
            strict_surplus
            or promotion_streak >= int(promotion_streak_epochs)
            or motion_pass
        )

        tracker_row = tracker.get(tow)
        tracker_fixed = (
            tracker_row is not None
            and int(tracker_row.get("shadow_fixed", "0")) == 1
        )
        if monitor_fixed:
            source = "library_monitor"
            position_row = monitor_row
        elif guarded_promotion:
            source = "library_guarded_promotion"
            position_row = active_row
        elif tracker_fixed:
            source = "pf_fgo_rescue"
            position_row = tracker_row
        else:
            source = "abstain"
            position_row = active_row
        position = tuple(float(position_row[axis]) for axis in "xyz")
        if source != "abstain" and not all(math.isfinite(value) for value in position):
            raise ValueError(f"non-finite FIX candidate at TOW {tow} from {source}")
        output.append(
            {
                "epoch_index": index,
                "tow": tow,
                "shadow_fixed": int(source != "abstain"),
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "source": source,
                "monitor_fixed": int(monitor_fixed),
                "active_promoted": int(promoted),
                "strict_surplus": int(strict_surplus),
                "promotion_streak": promotion_streak,
                "motion_innovation_m": motion_innovation_m,
                "motion_pass": int(motion_pass),
                "tracker_fixed": int(tracker_fixed),
            }
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-pos", type=Path, required=True)
    parser.add_argument("--active-pos", type=Path, required=True)
    parser.add_argument("--active-integrity", type=Path, required=True)
    parser.add_argument("--tracker-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="optional reference used only after composition for official scoring",
    )
    parser.add_argument(
        "--official-audit",
        type=Path,
        help="write official score/FIX audit; requires --reference",
    )
    parser.add_argument("--motion-innovation-limit", type=float, default=0.25)
    parser.add_argument("--maximum-causal-arc-resets", type=int, default=2)
    parser.add_argument("--promotion-streak", type=int, default=2)
    args = parser.parse_args(argv)
    if (args.reference is None) != (args.official_audit is None):
        parser.error("--reference and --official-audit must be supplied together")
    rows = compose_safe_union(
        args.monitor_pos,
        args.active_pos,
        args.active_integrity,
        args.tracker_csv,
        motion_innovation_limit_m=args.motion_innovation_limit,
        maximum_causal_arc_resets=args.maximum_causal_arc_resets,
        promotion_streak_epochs=args.promotion_streak,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "gnss_gpu_ppc_safe_basin_union_v1",
        "truth_usage": "none",
        "production_input_truth": False,
        "config": {
            "motion_innovation_limit_m": args.motion_innovation_limit,
            "maximum_causal_arc_resets": args.maximum_causal_arc_resets,
            "promotion_streak_epochs": args.promotion_streak,
        },
        "epochs": len(rows),
        "fixed_epochs": sum(int(row["shadow_fixed"]) for row in rows),
        "source_counts": {
            source: sum(row["source"] == source for row in rows)
            for source in (
                "library_monitor",
                "library_guarded_promotion",
                "pf_fgo_rescue",
                "abstain",
            )
        },
        "input_sha256": {
            "monitor_pos": _sha256(args.monitor_pos),
            "active_pos": _sha256(args.active_pos),
            "active_integrity": _sha256(args.active_integrity),
            "tracker_csv": _sha256(args.tracker_csv),
        },
        "output_sha256": _sha256(args.output),
    }
    if args.reference is not None:
        # Composition above is complete before truth is opened.  The evaluator
        # receives only the frozen output and cannot affect any FIX decision.
        audit = evaluate_route(args.output, args.reference)
        args.official_audit.parent.mkdir(parents=True, exist_ok=True)
        args.official_audit.write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        summary["official_audit"] = {
            "path": str(args.official_audit),
            "sha256": _sha256(args.official_audit),
            "ppc_score_pct": audit["ppc_score_pct"],
            "fixed_epochs": audit["fixed_epochs"],
            "false_fix_epochs": audit["false_fix_epochs"],
            "false_fix_above_1m_epochs": audit["false_fix_above_1m_epochs"],
            "truth_usage": "post_estimator_scoring_only",
        }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
