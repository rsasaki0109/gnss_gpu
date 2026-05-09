#!/usr/bin/env python3
"""Build single-trip and leave-one-out submission candidates from a target delta."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.build_gsdc2023_pre_submit_manifest import (
    DELTA_CHANGED_THRESHOLD_M,
    REQUIRED_COLUMNS,
    sha256_file,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def _assert_same_keys(reference: pd.DataFrame, target: pd.DataFrame) -> None:
    if len(reference) != len(target):
        raise SystemExit(f"row count mismatch: {len(reference)} != {len(target)}")
    for column in ("tripId", "UnixTimeMillis"):
        if not reference[column].equals(target[column]):
            raise SystemExit(f"{column} mismatch")


def _slug_trip(trip_id: str) -> str:
    slug = trip_id.replace("/", "__")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", slug)
    return slug.strip("_").lower()


def _format_alpha(alpha: float) -> str:
    text = f"{alpha:.8g}".replace("-", "m").replace(".", "p")
    return text


def _format_trip_list(trips: tuple[str, ...] | list[str]) -> str:
    return "|".join(trips)


def _row_deltas(reference: pd.DataFrame, candidate: pd.DataFrame):
    return haversine_m(
        reference["LatitudeDegrees"].to_numpy(),
        reference["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )


def _delta_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float | int]:
    deltas = _row_deltas(reference, candidate)
    score = gsdc_score_m(deltas)
    return {
        "changed_rows": int((deltas > DELTA_CHANGED_THRESHOLD_M).sum()),
        "score_m": float(score["score_m"]),
        "p50_m": float(score["p50_m"]),
        "p95_m": float(score["p95_m"]),
        "max_m": float(score["max_m"]),
    }


def _changed_trips(reference: pd.DataFrame, target: pd.DataFrame) -> list[str]:
    changed = _row_deltas(reference, target) > DELTA_CHANGED_THRESHOLD_M
    trips = reference.loc[changed, "tripId"].drop_duplicates().tolist()
    return [str(trip) for trip in trips]


def _blend(reference: pd.DataFrame, target: pd.DataFrame, *, alpha: float, active_trips: set[str]) -> pd.DataFrame:
    candidate = reference.copy()
    active = candidate["tripId"].isin(active_trips)
    for column in ("LatitudeDegrees", "LongitudeDegrees"):
        candidate.loc[active, column] = (
            reference.loc[active, column].to_numpy()
            + alpha * (target.loc[active, column].to_numpy() - reference.loc[active, column].to_numpy())
        )
    return candidate


def _write_candidate(
    *,
    candidate: pd.DataFrame,
    output_dir: Path,
    tag: str,
    mode: str,
    trip_id: str,
    alpha: float,
) -> Path:
    filename = (
        f"submission_trip_weight_{mode}_{_slug_trip(trip_id)}_a{_format_alpha(alpha)}_{tag}.csv"
    )
    output_path = output_dir / filename
    candidate.to_csv(output_path, index=False)
    return output_path


def build_trip_weight_ablation_candidates(
    *,
    reference_path: Path,
    target_path: Path,
    output_dir: Path,
    tag: str,
    alpha: float = 1.0,
    modes: tuple[str, ...] = ("single", "leave_one_out"),
    trips: tuple[str, ...] | None = None,
    group_held_trips: tuple[str, ...] = (),
    group_extra_count: int = 1,
) -> list[dict[str, Any]]:
    reference = _read_submission(reference_path.expanduser().resolve())
    target = _read_submission(target_path.expanduser().resolve())
    _assert_same_keys(reference, target)

    changed_trips = _changed_trips(reference, target)
    selected_trips = list(trips) if trips else changed_trips
    missing = [trip for trip in [*selected_trips, *group_held_trips] if trip not in set(changed_trips)]
    if missing:
        raise SystemExit(f"selected trip(s) do not move in target: {', '.join(missing)}")
    if group_extra_count < 0:
        raise SystemExit("group_extra_count must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    changed_trip_set = set(changed_trips)
    rows: list[dict[str, Any]] = []
    group_base = tuple(dict.fromkeys(group_held_trips))
    group_pool = [trip for trip in selected_trips if trip not in set(group_base)]
    group_specs = [
        tuple([*group_base, *extra])
        for extra in itertools.combinations(group_pool, group_extra_count)
    ]

    for mode in modes:
        if mode in {"single", "leave_one_out"}:
            candidate_specs = [(trip_id,) for trip_id in selected_trips]
        elif mode == "leave_group_out":
            candidate_specs = group_specs
        else:
            raise SystemExit(f"unknown mode: {mode}")

        for spec in candidate_specs:
            if not spec:
                raise SystemExit("leave_group_out requires at least one held trip")
            if mode == "single":
                active_trips = {spec[0]}
                held_trips: tuple[str, ...] = ()
                active_trip = spec[0]
                filename_trip_id = spec[0]
            elif mode == "leave_one_out":
                active_trips = changed_trip_set - {spec[0]}
                held_trips = (spec[0],)
                active_trip = ""
                filename_trip_id = spec[0]
            elif mode == "leave_group_out":
                active_trips = changed_trip_set - set(spec)
                held_trips = spec
                active_trip = ""
                filename_trip_id = _format_trip_list(spec)
            else:
                raise SystemExit(f"unknown mode: {mode}")

            candidate = _blend(reference, target, alpha=alpha, active_trips=active_trips)
            output_path = _write_candidate(
                candidate=candidate,
                output_dir=output_dir,
                tag=tag,
                mode=mode,
                trip_id=filename_trip_id,
                alpha=alpha,
            )
            summary = _delta_summary(reference, candidate)
            rows.append(
                {
                    "mode": mode,
                    "held_trip": _format_trip_list(held_trips),
                    "active_trip": active_trip,
                    "held_trip_count": len(held_trips),
                    "alpha": alpha,
                    "active_trip_count": len(active_trips),
                    "output": str(output_path),
                    "output_sha256": sha256_file(output_path),
                    **summary,
                },
            )

    manifest_path = output_dir / f"trip_weight_ablation_manifest_{tag}.csv"
    fieldnames = list(rows[0]) if rows else [
        "mode",
        "held_trip",
        "active_trip",
        "held_trip_count",
        "alpha",
        "active_trip_count",
        "output",
        "output_sha256",
        "changed_rows",
        "score_m",
        "p50_m",
        "p95_m",
        "max_m",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / f"trip_weight_ablation_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(
            {
                "reference": str(reference_path),
                "target": str(target_path),
                "output_dir": str(output_dir),
                "tag": tag,
                "alpha": alpha,
                "modes": list(modes),
                "group_extra_count": group_extra_count,
                "group_held_trips": list(group_held_trips),
                "changed_trip_count": len(changed_trips),
                "selected_trip_count": len(selected_trips),
                "candidate_count": len(rows),
                "changed_trips": changed_trips,
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {manifest_path}")
    print(f"saved: {summary_path}")
    print(f"prepared: {len(rows)} candidate(s)")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--mode", action="append", choices=("single", "leave_one_out", "leave_group_out"))
    parser.add_argument("--trip", action="append", dest="trips")
    parser.add_argument(
        "--group-held-trip",
        action="append",
        default=[],
        help="Trip to hold in every leave_group_out candidate; repeat for fixed multi-trip holds.",
    )
    parser.add_argument(
        "--group-extra-count",
        type=int,
        default=1,
        help="Number of selected trips to add to --group-held-trip for each leave_group_out candidate.",
    )
    args = parser.parse_args(argv)

    build_trip_weight_ablation_candidates(
        reference_path=args.reference,
        target_path=args.target,
        output_dir=args.output_dir,
        tag=args.tag,
        alpha=args.alpha,
        modes=tuple(args.mode) if args.mode else ("single", "leave_one_out"),
        trips=tuple(args.trips) if args.trips else None,
        group_held_trips=tuple(args.group_held_trip),
        group_extra_count=args.group_extra_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
