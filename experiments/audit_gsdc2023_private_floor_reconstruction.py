#!/usr/bin/env python3
"""Audit whether a GSDC2023 private-floor submission can be reconstructed locally."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    DEFAULT_INPUT as DEFAULT_PRIVATE_FLOOR_BUILDER_INPUT,
    DEFAULT_PIXEL5_PATCH,
)
from experiments.build_gsdc2023_pre_submit_manifest import (
    DELTA_CHANGED_THRESHOLD_M,
    REQUIRED_COLUMNS,
    sha256_file,
)
from experiments.reproduce_gsdc2023_matlab_reference_final import (
    DEFAULT_BRIDGE_ROOT as DEFAULT_MATLAB_BRIDGE_ROOT,
    DEFAULT_CANDIDATE_SUBMISSION as DEFAULT_MATLAB_CANDIDATE_SUBMISSION,
    DEFAULT_REFERENCE_SUBMISSION as DEFAULT_MATLAB_REFERENCE_SUBMISSION,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


PRIVATE_FLOOR_PRIVATE_MAX = 4.713


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def _as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coordinate_sanity(frame: pd.DataFrame) -> dict[str, Any]:
    lat = frame["LatitudeDegrees"].to_numpy(dtype=float)
    lon = frame["LongitudeDegrees"].to_numpy(dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    in_bounds = (lat >= 30.0) & (lat <= 40.0) & (lon >= -130.0) & (lon <= -110.0)
    return {
        "rows": int(len(frame)),
        "coordinate_sanity_pass": bool(finite.all() and in_bounds.all()),
        "nonfinite_latlon_rows": int(np.count_nonzero(~finite)),
        "out_of_bounds_rows": int(np.count_nonzero(finite & ~in_bounds)),
        "latitude_min": float(np.nanmin(lat)),
        "latitude_max": float(np.nanmax(lat)),
        "longitude_min": float(np.nanmin(lon)),
        "longitude_max": float(np.nanmax(lon)),
    }


def _assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> None:
    if len(left) != len(right):
        raise SystemExit(f"{label}: row count mismatch {len(left)} != {len(right)}")
    for column in ("tripId", "UnixTimeMillis"):
        if not left[column].equals(right[column]):
            raise SystemExit(f"{label}: {column} mismatch")


def _delta_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    delta = haversine_m(
        left["LatitudeDegrees"].to_numpy(),
        left["LongitudeDegrees"].to_numpy(),
        right["LatitudeDegrees"].to_numpy(),
        right["LongitudeDegrees"].to_numpy(),
    )
    score = gsdc_score_m(delta)
    return {
        "changed_rows": int(np.count_nonzero(delta > DELTA_CHANGED_THRESHOLD_M)),
        "changed_rows_gt_0p01m": int(np.count_nonzero(delta > 0.01)),
        "score_m": float(score["score_m"]),
        "p50_m": float(score["p50_m"]),
        "p95_m": float(score["p95_m"]),
        "max_m": float(score["max_m"]),
    }


def _read_score_history(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _submission_family(filename: str) -> str:
    lowered = filename.lower()
    if "mtv700" in lowered:
        return "mtv700"
    if "private_floor" in lowered:
        return "private_floor"
    if "p3p25" in lowered:
        return "p3p25"
    if "pixel5" in lowered and "basecorr_posoffset" in lowered:
        return "pixel5_offset"
    if "reconstructed_matlab" in lowered or "20260501_0526" in lowered:
        return "matlab_reference"
    if "bridge" in lowered:
        return "bridge"
    return "other"


def _find_exact_filename(filename: str, roots: tuple[Path, ...]) -> list[Path]:
    matches: list[Path] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for path in root.rglob(filename):
            if path.is_file():
                matches.append(path.resolve())
    return sorted(set(matches), key=lambda path: str(path))


def _score_rows(
    *,
    score_history: list[dict[str, str]],
    search_roots: tuple[Path, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in score_history:
        filename = str(row.get("fileName", ""))
        if not filename:
            continue
        public_score = _as_float(row.get("publicScore"))
        private_score = _as_float(row.get("privateScore"))
        is_private_floor = private_score is not None and private_score <= PRIVATE_FLOOR_PRIVATE_MAX
        matches = _find_exact_filename(filename, search_roots) if is_private_floor else []
        rows.append(
            {
                "filename": filename,
                "date": row.get("date", ""),
                "description": row.get("description", ""),
                "public_score": public_score,
                "private_score": private_score,
                "family": _submission_family(filename),
                "is_private_floor_score": is_private_floor,
                "local_exact_match_count": len(matches),
                "local_exact_matches": "|".join(str(path) for path in matches),
            },
        )
    return rows


def _local_candidate_paths(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    return sorted(
        path
        for path in root.glob("gsdc2023_submission*.csv")
        if path.is_file() and "bridge" not in path.name
    )


def _local_candidate_rows(paths: list[Path], score_by_filename: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        frame = _read_submission(path)
        score = score_by_filename.get(path.name, {})
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "exact_score_history_match": bool(score),
                "public_score": score.get("public_score"),
                "private_score": score.get("private_score"),
                **_coordinate_sanity(frame),
            },
        )
    return rows


def _pairwise_rows(paths: list[Path]) -> list[dict[str, Any]]:
    frames = {path: _read_submission(path) for path in paths}
    rows: list[dict[str, Any]] = []
    for left, right in combinations(paths, 2):
        _assert_same_keys(frames[left], frames[right], label=f"{left.name} vs {right.name}")
        rows.append(
            {
                "left": left.name,
                "right": right.name,
                **_delta_summary(frames[left], frames[right]),
            },
        )
    return rows


def _prerequisite_rows() -> list[dict[str, Any]]:
    specs = [
        ("private_floor_builder_input", DEFAULT_PRIVATE_FLOOR_BUILDER_INPUT),
        ("private_floor_pixel5_patch", DEFAULT_PIXEL5_PATCH),
        ("matlab_reference_submission", DEFAULT_MATLAB_REFERENCE_SUBMISSION),
        ("matlab_candidate_submission", DEFAULT_MATLAB_CANDIDATE_SUBMISSION),
        ("matlab_bridge_root", DEFAULT_MATLAB_BRIDGE_ROOT),
    ]
    return [
        {
            "artifact": label,
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
        }
        for label, path in specs
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_private_floor_reconstruction(
    *,
    results_root: Path,
    output_dir: Path,
    score_history_csv: Path | None = None,
    search_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = search_roots or (results_root, Path.cwd())
    score_rows = _score_rows(score_history=_read_score_history(score_history_csv), search_roots=roots)
    score_by_filename = {str(row["filename"]): row for row in score_rows}
    candidate_paths = _local_candidate_paths(results_root)
    local_rows = _local_candidate_rows(candidate_paths, score_by_filename)
    pairwise_rows = _pairwise_rows(candidate_paths)
    prereq_rows = _prerequisite_rows()

    private_floor_rows = [row for row in score_rows if row["is_private_floor_score"]]
    exact_private_floor_matches = [
        row for row in private_floor_rows if int(row["local_exact_match_count"]) > 0
    ]
    missing_prereqs = [row for row in prereq_rows if not bool(row["exists"])]
    reconstructable = bool(exact_private_floor_matches) or not missing_prereqs

    _write_csv(output_dir / "private_floor_score_history_audit.csv", score_rows)
    _write_csv(output_dir / "local_submission_candidate_audit.csv", local_rows)
    _write_csv(output_dir / "local_submission_pairwise_delta.csv", pairwise_rows)
    _write_csv(output_dir / "reconstruction_prerequisites.csv", prereq_rows)

    payload = {
        "results_root": str(results_root),
        "score_history_csv": str(score_history_csv) if score_history_csv is not None else None,
        "private_floor_private_max": PRIVATE_FLOOR_PRIVATE_MAX,
        "private_floor_score_rows": len(private_floor_rows),
        "exact_private_floor_local_matches": len(exact_private_floor_matches),
        "local_candidate_count": len(local_rows),
        "missing_prerequisite_count": len(missing_prereqs),
        "missing_prerequisites": missing_prereqs,
        "private_floor_reconstructable_from_available_files": reconstructable,
        "read": _read_verdict(reconstructable, exact_private_floor_matches, missing_prereqs),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "private_floor_reconstruction_audit.md").write_text(
        _render_markdown(payload, private_floor_rows, local_rows, pairwise_rows),
        encoding="utf-8",
    )
    print(f"saved: {output_dir / 'summary.json'}")
    print(f"saved: {output_dir / 'private_floor_reconstruction_audit.md'}")
    return payload


def _read_verdict(
    reconstructable: bool,
    exact_matches: list[dict[str, Any]],
    missing_prereqs: list[dict[str, Any]],
) -> str:
    if exact_matches:
        return "A scored private-floor CSV is available locally by exact filename."
    if reconstructable:
        return "No exact scored private-floor CSV is local, but reconstruction prerequisites are present."
    missing = ", ".join(str(row["artifact"]) for row in missing_prereqs)
    return f"Not reconstructable from available files; missing prerequisites: {missing}."


def _render_markdown(
    payload: dict[str, Any],
    private_floor_rows: list[dict[str, Any]],
    local_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# GSDC2023 Private-Floor Reconstruction Audit",
        "",
        f"- Reconstructable from available files: `{payload['private_floor_reconstructable_from_available_files']}`",
        f"- Private-floor score rows: `{payload['private_floor_score_rows']}`",
        f"- Exact private-floor local matches: `{payload['exact_private_floor_local_matches']}`",
        f"- Local old candidate count: `{payload['local_candidate_count']}`",
        f"- Missing prerequisite count: `{payload['missing_prerequisite_count']}`",
        f"- Read: {payload['read']}",
        "",
        "## Top Private-Floor Score Rows",
        "",
        "| File | Public | Private | Family | Local exact matches |",
        "|---|---:|---:|---|---:|",
    ]
    for row in private_floor_rows[:20]:
        lines.append(
            "| `{filename}` | {public_score} | {private_score} | `{family}` | {local_exact_match_count} |".format(
                **row,
            ),
        )
    lines.extend(
        [
            "",
            "## Local Old Candidates",
            "",
            "| File | Coordinate sane | Exact score match | Public | Private |",
            "|---|---|---|---:|---:|",
        ],
    )
    for row in local_rows:
        lines.append(
            "| `{filename}` | `{coordinate_sanity_pass}` | `{exact_score_history_match}` | {public_score} | {private_score} |".format(
                **row,
            ),
        )
    lines.extend(
        [
            "",
            "## Closest Local Candidate Pairs",
            "",
            "| Left | Right | Changed rows | Score m | P95 m | Max m |",
            "|---|---|---:|---:|---:|---:|",
        ],
    )
    for row in sorted(pairwise_rows, key=lambda item: float(item["score_m"]))[:10]:
        lines.append(
            "| `{left}` | `{right}` | {changed_rows} | {score_m:.3f} | {p95_m:.3f} | {max_m:.3f} |".format(
                **row,
            ),
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--score-history-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    args = parser.parse_args(argv)

    audit_private_floor_reconstruction(
        results_root=args.results_root,
        score_history_csv=args.score_history_csv,
        output_dir=args.output_dir,
        search_roots=tuple(args.search_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
