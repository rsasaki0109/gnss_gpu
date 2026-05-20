#!/usr/bin/env python3
"""Select a local proxy base for dry-run GSDC2023 source-family ablations."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
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


def _local_candidate_paths(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    return sorted(
        path
        for path in root.glob("gsdc2023_submission*.csv")
        if path.is_file() and "bridge" not in path.name
    )


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


def _same_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return (
        len(left) == len(right)
        and left["tripId"].equals(right["tripId"])
        and left["UnixTimeMillis"].equals(right["UnixTimeMillis"])
    )


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


def _pairwise_rows(paths: list[Path], frames: dict[Path, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(paths, 2):
        key_compatible = _same_keys(frames[left], frames[right])
        row: dict[str, Any] = {
            "left": left.name,
            "right": right.name,
            "key_compatible": key_compatible,
        }
        if key_compatible:
            row.update(_delta_summary(frames[left], frames[right]))
        else:
            row.update(
                {
                    "changed_rows": None,
                    "changed_rows_gt_0p01m": None,
                    "score_m": None,
                    "p50_m": None,
                    "p95_m": None,
                    "max_m": None,
                },
            )
        rows.append(row)
    return rows


def _base_candidate_rows(
    *,
    paths: list[Path],
    frames: dict[Path, pd.DataFrame],
    pairwise_rows: list[dict[str, Any]],
    reject_filenames: set[str],
) -> list[dict[str, Any]]:
    sanity_by_file = {path.name: _coordinate_sanity(frames[path]) for path in paths}
    rejection_by_file: dict[str, list[str]] = {}
    for path in paths:
        rejection_reasons: list[str] = []
        if not bool(sanity_by_file[path.name]["coordinate_sanity_pass"]):
            rejection_reasons.append("coordinate_sanity_failed")
        if path.name in reject_filenames:
            rejection_reasons.append("explicit_reject_filename")
        rejection_by_file[path.name] = rejection_reasons

    scores_by_file: dict[str, list[tuple[str, float]]] = {path.name: [] for path in paths}
    for row in pairwise_rows:
        if not row["key_compatible"] or row["score_m"] is None:
            continue
        left = str(row["left"])
        right = str(row["right"])
        if rejection_by_file[left] or rejection_by_file[right]:
            continue
        score = float(row["score_m"])
        scores_by_file[left].append((right, score))
        scores_by_file[right].append((left, score))

    rows: list[dict[str, Any]] = []
    for path in paths:
        sanity = sanity_by_file[path.name]
        rejection_reasons = rejection_by_file[path.name]

        neighbor_scores = [score for _other, score in scores_by_file[path.name]]
        closest_neighbor = ""
        nearest_score: float | None = None
        if scores_by_file[path.name]:
            closest_neighbor, nearest_score = min(scores_by_file[path.name], key=lambda item: item[1])

        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "eligible_proxy_base": not rejection_reasons,
                "rejection_reason": ";".join(rejection_reasons),
                "pairwise_neighbor_count": len(neighbor_scores),
                "pairwise_score_sum_m": float(sum(neighbor_scores)) if neighbor_scores else 0.0,
                "pairwise_score_mean_m": float(np.mean(neighbor_scores)) if neighbor_scores else None,
                "pairwise_score_max_m": float(np.max(neighbor_scores)) if neighbor_scores else None,
                "nearest_neighbor": closest_neighbor,
                "nearest_neighbor_score_m": nearest_score,
                "selected_proxy_base": False,
                "selection_rank": None,
                **sanity,
            },
        )
    return rows


def _select_proxy_base(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if bool(row["eligible_proxy_base"])
        and row["pairwise_score_mean_m"] is not None
        and int(row["pairwise_neighbor_count"]) > 0
    ]
    eligible.sort(
        key=lambda row: (
            float(row["pairwise_score_mean_m"]),
            float(row["pairwise_score_max_m"]),
            str(row["filename"]),
        ),
    )
    for index, row in enumerate(eligible, start=1):
        row["selection_rank"] = index
    if not eligible:
        return None
    selected = eligible[0]
    selected["selected_proxy_base"] = True
    return selected


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# GSDC2023 Local Proxy Base Selection",
        "",
        f"- Selected proxy base: `{payload['selected_proxy_base']}`",
        f"- Selected path: `{payload['selected_proxy_base_path']}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        f"- Eligible count: `{payload['eligible_count']}`",
        f"- Submit allowed: `{payload['submit_allowed']}`",
        f"- Read: {payload['read']}",
        "",
        "## Candidate Ranking",
        "",
        "| Rank | File | Eligible | Mean pairwise m | Max pairwise m | Nearest | Reason |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            row["selection_rank"] is None,
            int(row["selection_rank"] or 9999),
            str(row["filename"]),
        ),
    )
    for row in ranked:
        mean_m = row["pairwise_score_mean_m"]
        max_m = row["pairwise_score_max_m"]
        lines.append(
            "| {rank} | `{filename}` | `{eligible}` | {mean} | {maxv} | `{nearest}` | {reason} |".format(
                rank=row["selection_rank"] or "",
                filename=row["filename"],
                eligible=row["eligible_proxy_base"],
                mean=f"{float(mean_m):.3f}" if mean_m is not None else "",
                maxv=f"{float(max_m):.3f}" if max_m is not None else "",
                nearest=row["nearest_neighbor"],
                reason=row["rejection_reason"],
            ),
        )
    lines.append("")
    return "\n".join(lines)


def select_local_proxy_base(
    *,
    results_root: Path,
    output_dir: Path,
    reject_filenames: set[str] | None = None,
) -> dict[str, Any]:
    reject_filenames = reject_filenames or set()
    paths = _local_candidate_paths(results_root)
    frames = {path: _read_submission(path) for path in paths}
    pairwise_rows = _pairwise_rows(paths, frames)
    candidate_rows = _base_candidate_rows(
        paths=paths,
        frames=frames,
        pairwise_rows=pairwise_rows,
        reject_filenames=reject_filenames,
    )
    selected = _select_proxy_base(candidate_rows)

    selected_path = str(selected["path"]) if selected is not None else ""
    payload = {
        "results_root": str(results_root),
        "candidate_count": len(candidate_rows),
        "eligible_count": sum(1 for row in candidate_rows if bool(row["eligible_proxy_base"])),
        "selected_proxy_base": str(selected["filename"]) if selected is not None else None,
        "selected_proxy_base_path": selected_path,
        "selected_proxy_base_sha256": str(selected["sha256"]) if selected is not None else None,
        "submit_allowed": False,
        "read": (
            "Use this only as a dry-run proxy base; it is a local medoid, not a score-backed private-floor body."
            if selected is not None
            else "No usable local proxy base was selected."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "local_proxy_base_candidates.csv", candidate_rows)
    _write_csv(output_dir / "local_proxy_base_pairwise_delta.csv", pairwise_rows)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "local_proxy_base_selection.md").write_text(
        _render_markdown(payload, candidate_rows),
        encoding="utf-8",
    )
    print(f"saved: {output_dir / 'summary.json'}")
    print(f"saved: {output_dir / 'local_proxy_base_selection.md'}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reject-filename", action="append", default=[])
    args = parser.parse_args(argv)

    select_local_proxy_base(
        results_root=args.results_root,
        output_dir=args.output_dir,
        reject_filenames=set(args.reject_filename),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
