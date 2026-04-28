"""Reproduce the current best GSDC2023 submission artifact.

This is intentionally narrow: it rebuilds the scored 2026-04-24 artifact from
the recorded source submission, the reset-safe smoother, and the one accepted
row-level cap1000 repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.smooth_gsdc2023_submission import (
    SmoothConfig,
    haversine_m,
    smooth_dataframe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_0555_SUBMISSION = (
    REPO_ROOT
    / "../ref/gsdc2023/results/test_parallel/20260421_0555/submission_20260421_0555.csv"
).resolve()
CURRENT_1450_SUBMISSION = (
    REPO_ROOT
    / "../ref/gsdc2023/results/test_parallel/20260423_1450/submission_20260423_1450.csv"
).resolve()
EXPECTED_SOURCE_CANDIDATE = (
    REPO_ROOT
    / "experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/"
    / "submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv"
)
EXPECTED_FINAL = (
    REPO_ROOT
    / "experiments/results/post_smooth_cap_policy_20260424/"
    / "submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/reproduce_best_submission_20260424"
PIXEL4XL_BRIDGE_POSITIONS = DEFAULT_OUTPUT_DIR / "regenerate_patch_trips/pixel4xl/bridge_positions.csv"
SM_A505U_BRIDGE_POSITIONS = DEFAULT_OUTPUT_DIR / "regenerate_patch_trips/sm_a505u/bridge_positions.csv"
SM_A505U_EXCEPTION_ROWS = (
    DEFAULT_OUTPUT_DIR / "regenerate_patch_trips/sm_a505u_1450_exception_rows_188_410.csv"
)
FINAL_TRIP_ID = "2020-12-11-19-30-us-ca-mtv-e/pixel4xl"
FINAL_UNIX_TIME_MS = 1607716162442
SOURCE_PATCH_TRIPS = (
    "2020-12-11-19-30-us-ca-mtv-e/pixel4xl",
    "2023-05-09-23-10-us-ca-sjc-r/sm-a505u",
)
EXPECTED_BASE_0555_SHA256 = "b60b1dcbe188540b8d35c56f487e580b8f5aee7f0138c8685e9745d02cef61c8"
EXPECTED_CURRENT_1450_SHA256 = "2ff02b916c642956285e0421f7a8dab171f9a88ca30dc42317ea404e9685029c"
EXPECTED_PIXEL4XL_BRIDGE_SHA256 = "43f97d81a4e660fbf8862158445fad04deda4b6b4727c35cfe5534ac21724bd5"
EXPECTED_SM_A505U_BRIDGE_SHA256 = "dd5211c4427286ed000617f909cde0c5831c3d82b94c3cf5e682c3f6aba51419"
EXPECTED_SM_A505U_EXCEPTION_SHA256 = "359f4960ba5d281305965eb71eb7e14c4ba0af68307c61946bb4ef54ad137d5c"
PATCH_SOURCE_CHOICES = ("historical-submission", "bridge-exception")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_key_columns(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        len(left) == len(right)
        and left["tripId"].equals(right["tripId"])
        and left["UnixTimeMillis"].equals(right["UnixTimeMillis"])
    )


def parse_trip_position_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected TRIP=path for --patch-trip-positions, got {value!r}")
        trip, path = value.split("=", 1)
        if not trip or not path:
            raise ValueError(f"expected non-empty TRIP=path for --patch-trip-positions, got {value!r}")
        overrides[trip] = Path(path)
    return overrides


def _trip_path_arg(trip: str, path: Path) -> str:
    return f"{trip}={path}"


def reduced_patch_trip_position_args() -> list[str]:
    return [
        _trip_path_arg("2020-12-11-19-30-us-ca-mtv-e/pixel4xl", PIXEL4XL_BRIDGE_POSITIONS),
        _trip_path_arg("2023-05-09-23-10-us-ca-sjc-r/sm-a505u", SM_A505U_BRIDGE_POSITIONS),
    ]


def reduced_patch_row_position_args() -> list[str]:
    return [
        _trip_path_arg("2023-05-09-23-10-us-ca-sjc-r/sm-a505u", SM_A505U_EXCEPTION_ROWS),
    ]


def apply_patch_source_preset(
    patch_source: str,
    *,
    base_submission: Path,
    patch_submission: Path,
    patch_trip_positions: list[str],
    patch_row_positions: list[str],
) -> tuple[Path, list[str], list[str]]:
    if patch_source not in PATCH_SOURCE_CHOICES:
        raise ValueError(f"unsupported patch source: {patch_source}")
    effective_patch_submission = patch_submission
    effective_trip_positions = list(patch_trip_positions)
    effective_row_positions = list(patch_row_positions)
    if patch_source == "bridge-exception":
        if patch_submission.resolve() == CURRENT_1450_SUBMISSION:
            effective_patch_submission = base_submission
        effective_trip_positions.extend(reduced_patch_trip_position_args())
        effective_row_positions.extend(reduced_patch_row_position_args())
    return effective_patch_submission, effective_trip_positions, effective_row_positions


def _trip_override_coordinates(
    path: Path,
    base_trip: pd.DataFrame,
    trip: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(path)
    if "tripId" in frame.columns:
        frame = frame[frame["tripId"] == trip]
    required = {"UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame["UnixTimeMillis"].duplicated().any():
        raise ValueError(f"{path} contains duplicate UnixTimeMillis values")
    keyed = frame.set_index("UnixTimeMillis")
    times = base_trip["UnixTimeMillis"].to_numpy()
    missing_times = sorted(set(times).difference(keyed.index))
    extra_times = sorted(set(keyed.index).difference(times))
    if missing_times or extra_times:
        raise ValueError(
            f"{path} keys differ for {trip}: "
            f"missing={len(missing_times)} extra={len(extra_times)}",
        )
    coords = keyed.loc[times, ["LatitudeDegrees", "LongitudeDegrees"]]
    if coords.isna().any(axis=None):
        raise ValueError(f"{path} contains NaN coordinates for {trip}")
    return coords.to_numpy(), {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(coords)),
    }


def replace_trip_coordinates(
    base: pd.DataFrame,
    patch: pd.DataFrame,
    trips: tuple[str, ...],
    trip_position_overrides: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not same_key_columns(base, patch):
        raise ValueError("base and patch submissions have different key columns")
    out = base.copy()
    overrides = trip_position_overrides or {}
    rows_by_trip: dict[str, int] = {}
    source_by_trip: dict[str, str] = {}
    override_summaries: dict[str, Any] = {}
    rows_replaced = 0
    for trip in trips:
        mask = out["tripId"] == trip
        row_count = int(mask.sum())
        if row_count == 0:
            raise ValueError(f"no rows matched requested patch trip: {trip}")
        if trip in overrides:
            coords, override_summary = _trip_override_coordinates(overrides[trip], out.loc[mask], trip)
            source_by_trip[trip] = "bridge_positions"
            override_summaries[trip] = override_summary
        else:
            coords = patch.loc[mask, ["LatitudeDegrees", "LongitudeDegrees"]].to_numpy()
            source_by_trip[trip] = "patch_submission"
        out.loc[mask, ["LatitudeDegrees", "LongitudeDegrees"]] = coords
        rows_by_trip[trip] = row_count
        rows_replaced += row_count
    return out, {
        "trips": list(trips),
        "rows_replaced": rows_replaced,
        "rows_by_trip": rows_by_trip,
        "source_by_trip": source_by_trip,
        "trip_position_overrides": override_summaries,
    }


def apply_row_coordinate_overrides(
    frame: pd.DataFrame,
    row_position_overrides: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    summaries: dict[str, Any] = {}
    total_rows = 0
    for trip, path in row_position_overrides.items():
        override = pd.read_csv(path)
        if "tripId" in override.columns:
            override = override[override["tripId"] == trip]
        required = {"UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
        missing = sorted(required.difference(override.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if override["UnixTimeMillis"].duplicated().any():
            raise ValueError(f"{path} contains duplicate UnixTimeMillis values for {trip}")
        if override.empty:
            raise ValueError(f"{path} has no override rows for {trip}")

        trip_mask = out["tripId"] == trip
        trip_times = set(out.loc[trip_mask, "UnixTimeMillis"])
        override_times = set(override["UnixTimeMillis"])
        missing_times = sorted(override_times.difference(trip_times))
        if missing_times:
            raise ValueError(f"{path} contains {len(missing_times)} rows absent from {trip}")

        row_mask = trip_mask & out["UnixTimeMillis"].isin(override_times)
        row_indices = np.flatnonzero(row_mask.to_numpy())
        if int(row_mask.sum()) != len(override):
            raise ValueError(f"{path} does not map one-to-one onto {trip}")
        keyed = override.set_index("UnixTimeMillis")
        coords = keyed.loc[
            out.loc[row_mask, "UnixTimeMillis"].to_numpy(),
            ["LatitudeDegrees", "LongitudeDegrees"],
        ]
        if coords.isna().any(axis=None):
            raise ValueError(f"{path} contains NaN coordinates for {trip}")
        out.loc[row_mask, ["LatitudeDegrees", "LongitudeDegrees"]] = coords.to_numpy()

        rows = int(row_mask.sum())
        total_rows += rows
        summaries[trip] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": rows,
            "row_index_min": int(row_indices.min()),
            "row_index_max": int(row_indices.max()),
        }
    return out, {
        "rows_replaced": total_rows,
        "trips": summaries,
    }


def smooth_submission(source: pd.DataFrame, cap_m: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = SmoothConfig(
        median_window=5,
        smooth_window=5,
        blend=1.0,
        max_correction_m=float(cap_m),
    )
    smoothed, stats = smooth_dataframe(source, config)
    return smoothed, {
        "config": {
            "median_window": config.median_window,
            "smooth_window": config.smooth_window,
            "blend": config.blend,
            "max_correction_m": config.max_correction_m,
            "smooth_kernel": config.smooth_kernel,
            "gaussian_sigma": config.gaussian_sigma,
            "hampel_sigma": config.hampel_sigma,
            "hampel_min_m": config.hampel_min_m,
            "segment_gap_ms": config.segment_gap_ms,
            "segment_step_m": config.segment_step_m,
            "min_segment_points": config.min_segment_points,
        },
        "stats": {
            "groups": stats.groups,
            "rows": stats.rows,
            "segments": stats.segments,
            "corrected_rows": stats.corrected_rows,
            "hampel_rows": stats.hampel_rows,
            "max_correction_m": stats.max_correction_m,
            "mean_correction_m": stats.mean_correction_m,
            "p95_correction_m": stats.p95_correction_m,
        },
    }


def verify_optional(output: Path, expected: Path) -> dict[str, Any]:
    if expected.is_file():
        return verify_against_expected(output, expected)
    return {
        "expected": str(expected),
        "expected_exists": False,
        "byte_identical": None,
        "output_sha256": sha256_file(output),
    }


def promote_final_row(cap100: pd.DataFrame, cap1000: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not same_key_columns(cap100, cap1000):
        raise ValueError("cap100 and cap1000 outputs have different key columns")

    row_mask = (cap100["tripId"] == FINAL_TRIP_ID) & (cap100["UnixTimeMillis"] == FINAL_UNIX_TIME_MS)
    if int(row_mask.sum()) != 1:
        raise ValueError(f"expected exactly one promoted row, got {int(row_mask.sum())}")

    final = cap100.copy()
    final.loc[row_mask, ["LatitudeDegrees", "LongitudeDegrees"]] = cap1000.loc[
        row_mask,
        ["LatitudeDegrees", "LongitudeDegrees"],
    ].to_numpy()
    promoted_distance_m = float(
        haversine_m(
            cap100.loc[row_mask, "LatitudeDegrees"].to_numpy(),
            cap100.loc[row_mask, "LongitudeDegrees"].to_numpy(),
            cap1000.loc[row_mask, "LatitudeDegrees"].to_numpy(),
            cap1000.loc[row_mask, "LongitudeDegrees"].to_numpy(),
        )[0],
    )
    return final, {
        "tripId": FINAL_TRIP_ID,
        "UnixTimeMillis": FINAL_UNIX_TIME_MS,
        "row_index": int(np.flatnonzero(row_mask.to_numpy())[0]),
        "cap100_to_cap1000_m": promoted_distance_m,
    }


def build_final(source: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    cap100, cap100_summary = smooth_submission(source, 100.0)
    cap1000, cap1000_summary = smooth_submission(source, 1000.0)

    cap100_path = output_dir / "submission_best_all_cap100p0_20260424_reproduced.csv"
    cap1000_path = output_dir / "submission_best_all_cap1000p0_20260424_reproduced.csv"
    cap100.to_csv(cap100_path, index=False)
    cap1000.to_csv(cap1000_path, index=False)

    # The scored artifact was produced from written intermediate CSVs.  Read
    # them back before the row promotion so pandas' float serialization state is
    # identical to the original artifact.
    cap100_roundtrip = pd.read_csv(cap100_path)
    cap1000_roundtrip = pd.read_csv(cap1000_path)
    final, promoted_row = promote_final_row(cap100_roundtrip, cap1000_roundtrip)
    return final, {
        "cap100": cap100_summary,
        "cap1000": cap1000_summary,
        "intermediate_outputs": {
            "cap100": str(cap100_path),
            "cap1000": str(cap1000_path),
        },
        "promoted_row": promoted_row,
    }


def verify_against_expected(output: Path, expected: Path) -> dict[str, Any]:
    output_bytes = output.read_bytes()
    expected_bytes = expected.read_bytes()
    return {
        "expected": str(expected),
        "expected_exists": True,
        "byte_identical": output_bytes == expected_bytes,
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
        "output_bytes": len(output_bytes),
        "expected_bytes": len(expected_bytes),
    }


def input_artifact_summary(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "sha256_matches": None if expected_sha256 is None else actual_sha256 == expected_sha256,
    }


def expected_sha_for_default(path: Path, default_path: Path, expected_sha256: str) -> str | None:
    return expected_sha256 if path.resolve() == default_path else None


def expected_sha_for_known_reduced_artifact(path: Path) -> str | None:
    known = {
        PIXEL4XL_BRIDGE_POSITIONS.resolve(): EXPECTED_PIXEL4XL_BRIDGE_SHA256,
        SM_A505U_BRIDGE_POSITIONS.resolve(): EXPECTED_SM_A505U_BRIDGE_SHA256,
        SM_A505U_EXCEPTION_ROWS.resolve(): EXPECTED_SM_A505U_EXCEPTION_SHA256,
    }
    return known.get(path.resolve())


def override_artifact_summaries(overrides: dict[str, Path]) -> dict[str, Any]:
    return {
        trip: input_artifact_summary(path, expected_sha_for_known_reduced_artifact(path))
        for trip, path in overrides.items()
    }


def mismatched_input_artifacts(input_artifacts: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for name, artifact in input_artifacts.items():
        if isinstance(artifact, dict) and artifact.get("sha256_matches") is False:
            mismatches.append(name)
        elif isinstance(artifact, dict):
            for child_name, child in artifact.items():
                if isinstance(child, dict) and child.get("sha256_matches") is False:
                    mismatches.append(f"{name}.{child_name}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-submission", type=Path, default=BASE_0555_SUBMISSION)
    parser.add_argument("--patch-submission", type=Path, default=CURRENT_1450_SUBMISSION)
    parser.add_argument("--expected-source", type=Path, default=EXPECTED_SOURCE_CANDIDATE)
    parser.add_argument("--expected", type=Path, default=EXPECTED_FINAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-output", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--allow-mismatch", action="store_true")
    parser.add_argument(
        "--patch-source",
        choices=PATCH_SOURCE_CHOICES,
        default="historical-submission",
        help="historical-submission uses 20260423_1450; bridge-exception uses regenerated bridge CSVs plus a 223-row sm-a505u exception",
    )
    parser.add_argument(
        "--allow-input-mismatch",
        action="store_true",
        help="allow default historical input artifacts to have unexpected SHA256 hashes",
    )
    parser.add_argument(
        "--patch-trip-positions",
        action="append",
        default=[],
        metavar="TRIP=CSV",
        help="override one patch trip with a bridge_positions.csv/submission-style CSV",
    )
    parser.add_argument(
        "--patch-row-positions",
        action="append",
        default=[],
        metavar="TRIP=CSV",
        help="override selected rows after trip replacement with a partial CSV keyed by UnixTimeMillis",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    source_output = args.source_output or output_dir / "submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv"
    output = args.output or output_dir / "submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv"
    summary_path = args.summary or output_dir / "reproduction_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    source_output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    effective_patch_submission, patch_trip_values, patch_row_values = apply_patch_source_preset(
        args.patch_source,
        base_submission=args.base_submission,
        patch_submission=args.patch_submission,
        patch_trip_positions=args.patch_trip_positions,
        patch_row_positions=args.patch_row_positions,
    )
    patch_trip_overrides = parse_trip_position_overrides(patch_trip_values)
    patch_row_overrides = parse_trip_position_overrides(patch_row_values)
    input_artifacts = {
        "base_submission": input_artifact_summary(
            args.base_submission,
            expected_sha_for_default(args.base_submission, BASE_0555_SUBMISSION, EXPECTED_BASE_0555_SHA256),
        ),
        "patch_submission": input_artifact_summary(
            effective_patch_submission,
            expected_sha_for_default(effective_patch_submission, CURRENT_1450_SUBMISSION, EXPECTED_CURRENT_1450_SHA256),
        ),
        "patch_trip_positions": override_artifact_summaries(patch_trip_overrides),
        "patch_row_positions": override_artifact_summaries(patch_row_overrides),
    }
    input_mismatches = mismatched_input_artifacts(input_artifacts)
    if input_mismatches and not args.allow_input_mismatch:
        raise SystemExit(f"input artifact SHA256 mismatch: {', '.join(input_mismatches)}")

    base_frame = pd.read_csv(args.base_submission)
    patch_frame = pd.read_csv(effective_patch_submission)
    source_frame, source_build = replace_trip_coordinates(
        base_frame,
        patch_frame,
        SOURCE_PATCH_TRIPS,
        trip_position_overrides=patch_trip_overrides,
    )
    if patch_row_overrides:
        source_frame, row_override_summary = apply_row_coordinate_overrides(source_frame, patch_row_overrides)
    else:
        row_override_summary = {"rows_replaced": 0, "trips": {}}
    source_build["row_position_overrides"] = row_override_summary
    source_frame.to_csv(source_output, index=False)

    final, build_summary = build_final(source_frame, output_dir)
    final.to_csv(output, index=False)

    source_verification = verify_optional(source_output, args.expected_source)
    verification = verify_optional(output, args.expected)

    payload = {
        "base_submission": str(args.base_submission),
        "patch_submission": str(effective_patch_submission),
        "patch_source": args.patch_source,
        "input_artifacts": input_artifacts,
        "source": str(source_output),
        "output": str(output),
        "rows": int(len(final)),
        "nan_lat_lon_rows": int(final[["LatitudeDegrees", "LongitudeDegrees"]].isna().any(axis=1).sum()),
        "source_build": source_build,
        "source_verification": source_verification,
        "build": build_summary,
        "verification": verification,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if (
        (source_verification.get("byte_identical") is False or verification.get("byte_identical") is False)
        and not args.allow_mismatch
    ):
        raise SystemExit("reproduced output differs from expected artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
