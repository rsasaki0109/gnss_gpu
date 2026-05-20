#!/usr/bin/env python3
"""Audit score-backed basecorr/private-floor lineage against local builders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    CANDIDATES,
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PIXEL5_PATCH,
)


PRIVATE_FLOOR_PRIVATE_MAX = 4.713


def _as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _read_score_history(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: list[dict[str, Any]] = []
    for row in rows:
        public_score = _as_float(row.get("publicScore"))
        private_score = _as_float(row.get("privateScore"))
        out.append(
            {
                "filename": str(row.get("fileName", "")),
                "date": str(row.get("date", "")),
                "description": str(row.get("description", "")),
                "status": str(row.get("status", "")),
                "public_score": public_score,
                "private_score": private_score,
                "is_private_floor_score": (
                    private_score is not None and private_score <= PRIVATE_FLOOR_PRIVATE_MAX
                ),
            },
        )
    return out


def _filename_index(roots: tuple[Path, ...]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, []).append(path.resolve())
    return {
        filename: sorted(set(paths), key=lambda path: str(path))
        for filename, paths in index.items()
    }


def _filename_matches_candidate(filename: str, candidate_name: str) -> bool:
    text = Path(filename).name.lower()
    needle = candidate_name.lower()
    return any(
        token in text
        for token in (
            f"{needle}_plus_pixel5_patch",
            f"{needle}_plus_laxx_patch",
            f"{needle}.csv",
        )
    )


def _candidate_score_rows(candidate_name: str, score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in score_rows
        if _filename_matches_candidate(str(row.get("filename", "")), candidate_name)
    ]


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if row.get("private_score") is not None]
    if not scored:
        return None
    return min(
        scored,
        key=lambda row: (
            float(row["private_score"]),
            float(row["public_score"] if row.get("public_score") is not None else 999.0),
            str(row.get("date", "")),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_basecorr_private_floor_lineage(
    *,
    score_history_csv: Path,
    output_dir: Path,
    input_path: Path = DEFAULT_INPUT,
    pixel5_patch_path: Path = DEFAULT_PIXEL5_PATCH,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
    search_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    score_rows = _read_score_history(score_history_csv)
    roots = search_roots or (score_history_csv.parent, Path.cwd())
    filename_index = _filename_index(roots)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    matched_filenames: set[str] = set()
    for name, config in sorted(CANDIDATES.items()):
        matches = _candidate_score_rows(name, score_rows)
        matched_filenames.update(str(row["filename"]) for row in matches)
        private_matches = [row for row in matches if row["is_private_floor_score"]]
        best = _best_row(matches)
        local_exact_matches: list[Path] = []
        if best is not None:
            local_exact_matches = filename_index.get(str(best["filename"]), [])
        candidate_rows.append(
            {
                "candidate": name,
                "reference_public": config.kaggle_public,
                "reference_private": config.kaggle_private,
                "reference_is_private_floor": (
                    config.kaggle_private is not None
                    and config.kaggle_private <= PRIVATE_FLOOR_PRIVATE_MAX
                ),
                "score_history_match_count": len(matches),
                "score_backed_private_floor_match_count": len(private_matches),
                "score_backed_private_floor": bool(private_matches),
                "best_score_filename": best["filename"] if best else "",
                "best_score_date": best["date"] if best else "",
                "best_public_score": best["public_score"] if best else None,
                "best_private_score": best["private_score"] if best else None,
                "best_exact_local_match_count": len(local_exact_matches),
                "best_exact_local_matches": "|".join(str(path) for path in local_exact_matches),
            },
        )

    private_floor_rows = [row for row in score_rows if row["is_private_floor_score"]]
    unmatched_private_floor_rows = [
        {
            **row,
            "exact_local_match_count": len(filename_index.get(str(row["filename"]), [])),
            "exact_local_matches": "|".join(
                str(path) for path in filename_index.get(str(row["filename"]), [])
            ),
        }
        for row in private_floor_rows
        if str(row["filename"]) not in matched_filenames
    ]
    prereq_rows = [
        {
            "artifact": "basecorr_builder_input",
            "path": str(input_path),
            "exists": input_path.exists(),
            "is_file": input_path.is_file(),
        },
        {
            "artifact": "pixel5_patch",
            "path": str(pixel5_patch_path),
            "exists": pixel5_patch_path.exists(),
            "is_file": pixel5_patch_path.is_file(),
        },
        {
            "artifact": "default_output_dir",
            "path": str(default_output_dir),
            "exists": default_output_dir.exists(),
            "is_file": default_output_dir.is_file(),
        },
    ]
    missing_prereqs = [
        row
        for row in prereq_rows
        if row["artifact"] != "default_output_dir" and not row["is_file"]
    ]
    score_backed_candidates = [
        row for row in candidate_rows if row["score_backed_private_floor"]
    ]
    exact_local_matches = [
        row for row in candidate_rows if int(row["best_exact_local_match_count"]) > 0
    ]
    top_recovery_targets = sorted(
        score_backed_candidates,
        key=lambda row: (
            float(row["best_private_score"] if row["best_private_score"] is not None else 999.0),
            float(row["best_public_score"] if row["best_public_score"] is not None else 999.0),
            str(row["best_score_date"]),
        ),
    )[:12]
    summary = {
        "score_history_csv": str(score_history_csv),
        "candidate_count": len(candidate_rows),
        "score_matched_candidate_count": int(
            sum(1 for row in candidate_rows if int(row["score_history_match_count"]) > 0),
        ),
        "score_backed_private_floor_candidate_count": len(score_backed_candidates),
        "exact_local_private_floor_candidate_count": len(exact_local_matches),
        "private_floor_score_rows": len(private_floor_rows),
        "unmatched_private_floor_score_rows": len(unmatched_private_floor_rows),
        "build_prerequisite_missing_count": len(missing_prereqs),
        "build_possible_locally": len(missing_prereqs) == 0,
        "missing_prerequisites": missing_prereqs,
        "top_recovery_targets": top_recovery_targets,
    }

    candidate_fields = [
        "candidate",
        "reference_public",
        "reference_private",
        "reference_is_private_floor",
        "score_history_match_count",
        "score_backed_private_floor_match_count",
        "score_backed_private_floor",
        "best_score_filename",
        "best_score_date",
        "best_public_score",
        "best_private_score",
        "best_exact_local_match_count",
        "best_exact_local_matches",
    ]
    private_floor_fields = [
        "filename",
        "date",
        "description",
        "status",
        "public_score",
        "private_score",
        "is_private_floor_score",
        "exact_local_match_count",
        "exact_local_matches",
    ]
    prereq_fields = ["artifact", "path", "exists", "is_file"]
    _write_csv(output_dir / "basecorr_candidate_lineage.csv", candidate_rows, candidate_fields)
    _write_csv(
        output_dir / "unmatched_private_floor_score_rows.csv",
        unmatched_private_floor_rows,
        private_floor_fields,
    )
    _write_csv(output_dir / "basecorr_reconstruction_prerequisites.csv", prereq_rows, prereq_fields)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "basecorr_private_floor_lineage.md").write_text(
        "\n".join(
            [
                "# GSDC2023 Basecorr Private-Floor Lineage Audit",
                "",
                f"- Score history: `{score_history_csv}`",
                f"- Builder input exists: `{input_path.is_file()}` (`{input_path}`)",
                f"- Pixel5 patch exists: `{pixel5_patch_path.is_file()}` (`{pixel5_patch_path}`)",
                f"- Score-matched builder candidates: `{summary['score_matched_candidate_count']}` / `{summary['candidate_count']}`",
                f"- Score-backed private-floor builder candidates: `{summary['score_backed_private_floor_candidate_count']}`",
                f"- Exact local private-floor candidate bodies: `{summary['exact_local_private_floor_candidate_count']}`",
                f"- Unmatched private-floor score rows: `{summary['unmatched_private_floor_score_rows']}`",
                "",
                "## Top Recovery Targets",
                "",
                "| candidate | filename | public | private | local exact |",
                "|---|---:|---:|---:|---:|",
                *[
                    "| {candidate} | `{best_score_filename}` | {best_public_score} | {best_private_score} | {best_exact_local_match_count} |".format(
                        **row,
                    )
                    for row in top_recovery_targets
                ],
                "",
                "## Verdict",
                "",
                (
                    "The score history links several builder presets to private-floor scores, "
                    "but local reconstruction still requires the missing builder input and Pixel5 patch."
                    if missing_prereqs
                    else "The builder input and Pixel5 patch are present, so these presets are locally rebuildable."
                ),
                "",
            ],
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-history-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pixel5-patch", type=Path, default=DEFAULT_PIXEL5_PATCH)
    parser.add_argument("--default-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="root to search for exact scored CSV filenames; repeatable",
    )
    args = parser.parse_args(argv)
    summary = audit_basecorr_private_floor_lineage(
        score_history_csv=args.score_history_csv,
        output_dir=args.output_dir,
        input_path=args.input,
        pixel5_patch_path=args.pixel5_patch,
        default_output_dir=args.default_output_dir,
        search_roots=tuple(args.search_root),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
