#!/usr/bin/env python3
"""Audit provenance of local GSDC2023 submission CSVs against score history."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

from experiments.audit_gsdc2023_private_floor_reconstruction import PRIVATE_FLOOR_PRIVATE_MAX
from experiments.build_gsdc2023_pre_submit_manifest import sha256_file


def _as_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _read_score_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _local_candidate_paths(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    return sorted(
        path
        for path in root.glob("gsdc2023_submission*.csv")
        if path.is_file() and "bridge" not in path.name
    )


def _score_family(filename: str) -> str:
    lowered = filename.lower()
    if "mtv700" in lowered:
        return "mtv700"
    if "private_floor" in lowered:
        return "private_floor"
    if "p3p25" in lowered:
        return "p3p25"
    if "basecorr_posoffset" in lowered:
        return "basecorr_posoffset"
    if "reconstructed_matlab" in lowered or "20260501_0526" in lowered:
        return "matlab_reference"
    if "bridge" in lowered:
        return "bridge"
    return "other"


def _git_added_at(repo_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        relative = path
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "log",
                "--diff-filter=A",
                "--follow",
                "--format=%aI",
                "-1",
                "--",
                str(relative),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def _classify_local(filename: str, score: dict[str, Any] | None) -> str:
    if score is not None:
        private_score = _as_float(score.get("privateScore"))
        if private_score is not None and private_score <= PRIVATE_FLOOR_PRIVATE_MAX:
            return "score_backed_private_floor"
        return "score_backed_non_private_floor"
    if filename == "gsdc2023_submission.csv" or filename.startswith("gsdc2023_submission_v"):
        return "legacy_pf_local_unscored"
    return "unscored_local"


def _score_rows(score_history: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in score_history:
        filename = str(row.get("fileName", ""))
        if not filename:
            continue
        private_score = _as_float(row.get("privateScore"))
        public_score = _as_float(row.get("publicScore"))
        rows.append(
            {
                "filename": filename,
                "date": row.get("date", ""),
                "description": row.get("description", ""),
                "public_score": public_score,
                "private_score": private_score,
                "family": _score_family(filename),
                "is_private_floor": private_score is not None and private_score <= PRIVATE_FLOOR_PRIVATE_MAX,
            },
        )
    return rows


def _local_rows(
    *,
    paths: list[Path],
    repo_root: Path,
    score_by_filename: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        score = score_by_filename.get(path.name)
        private_score = _as_float(score.get("privateScore")) if score is not None else None
        public_score = _as_float(score.get("publicScore")) if score is not None else None
        score_backed_private_floor = private_score is not None and private_score <= PRIVATE_FLOOR_PRIVATE_MAX
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "git_added_at": _git_added_at(repo_root, path),
                "exact_score_history_match": score is not None,
                "public_score": public_score,
                "private_score": private_score,
                "score_family": _score_family(path.name),
                "provenance_class": _classify_local(path.name, score),
                "score_backed_private_floor": score_backed_private_floor,
                "usable_as_private_floor_base": score_backed_private_floor,
            },
        )
    return rows


def _family_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = sorted({str(row["family"]) for row in rows})
    out: list[dict[str, Any]] = []
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        private_rows = [row for row in family_rows if bool(row["is_private_floor"])]
        out.append(
            {
                "family": family,
                "score_rows": len(family_rows),
                "private_floor_rows": len(private_rows),
                "best_public": min(
                    (float(row["public_score"]) for row in family_rows if row["public_score"] is not None),
                    default=None,
                ),
                "best_private": min(
                    (float(row["private_score"]) for row in family_rows if row["private_score"] is not None),
                    default=None,
                ),
            },
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    payload: dict[str, Any],
    local_rows: list[dict[str, Any]],
    private_floor_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# GSDC2023 Local Submission Provenance Audit",
        "",
        f"- Local candidates: `{payload['local_candidate_count']}`",
        f"- Local score-backed private-floor candidates: `{payload['local_score_backed_private_floor_count']}`",
        f"- Private-floor score rows: `{payload['private_floor_score_rows']}`",
        f"- Earliest private-floor score date: `{payload['earliest_private_floor_score_date']}`",
        f"- Read: {payload['read']}",
        "",
        "## Local Candidates",
        "",
        "| File | Provenance | Exact score match | Private | Git added | Usable as base |",
        "|---|---|---|---:|---|---|",
    ]
    for row in local_rows:
        lines.append(
            "| `{filename}` | `{provenance_class}` | `{exact_score_history_match}` | {private} | `{git_added_at}` | `{usable}` |".format(
                filename=row["filename"],
                provenance_class=row["provenance_class"],
                exact_score_history_match=row["exact_score_history_match"],
                private="" if row["private_score"] is None else row["private_score"],
                git_added_at=row["git_added_at"],
                usable=row["usable_as_private_floor_base"],
            ),
        )
    lines.extend(
        [
            "",
            "## Earliest Private-Floor Score Rows",
            "",
            "| Date | File | Public | Private | Family |",
            "|---|---|---:|---:|---|",
        ],
    )
    for row in private_floor_rows[:12]:
        lines.append(
            "| {date} | `{filename}` | {public_score} | {private_score} | `{family}` |".format(**row),
        )
    lines.append("")
    return "\n".join(lines)


def audit_local_submission_provenance(
    *,
    results_root: Path,
    score_history_csv: Path,
    output_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    repo_root = (repo_root or Path.cwd()).expanduser().resolve()
    score_history = _read_score_history(score_history_csv)
    score_by_filename = {str(row.get("fileName", "")): row for row in score_history}
    scores = _score_rows(score_history)
    private_floor_rows = sorted(
        [row for row in scores if bool(row["is_private_floor"])],
        key=lambda row: (str(row["date"]), str(row["filename"])),
    )
    local_rows = _local_rows(
        paths=_local_candidate_paths(results_root),
        repo_root=repo_root,
        score_by_filename=score_by_filename,
    )
    family_rows = _family_counts(scores)
    local_private_count = sum(1 for row in local_rows if bool(row["score_backed_private_floor"]))

    payload = {
        "results_root": str(results_root),
        "score_history_csv": str(score_history_csv),
        "local_candidate_count": len(local_rows),
        "local_score_backed_private_floor_count": local_private_count,
        "private_floor_private_max": PRIVATE_FLOOR_PRIVATE_MAX,
        "private_floor_score_rows": len(private_floor_rows),
        "earliest_private_floor_score_date": private_floor_rows[0]["date"] if private_floor_rows else "",
        "earliest_private_floor_score_filename": private_floor_rows[0]["filename"] if private_floor_rows else "",
        "local_candidates_usable_as_private_floor_base": local_private_count,
        "read": (
            "No local gsdc2023_submission*.csv is a score-backed private-floor body; keep them as legacy PF artifacts only."
            if local_private_count == 0
            else "At least one local candidate is score-backed private-floor."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "local_submission_provenance.csv", local_rows)
    _write_csv(output_dir / "score_family_counts.csv", family_rows)
    _write_csv(output_dir / "private_floor_score_rows.csv", private_floor_rows)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "local_submission_provenance.md").write_text(
        _render_markdown(payload, local_rows, private_floor_rows),
        encoding="utf-8",
    )
    print(f"saved: {output_dir / 'summary.json'}")
    print(f"saved: {output_dir / 'local_submission_provenance.md'}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--score-history-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    audit_local_submission_provenance(
        results_root=args.results_root,
        score_history_csv=args.score_history_csv,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
