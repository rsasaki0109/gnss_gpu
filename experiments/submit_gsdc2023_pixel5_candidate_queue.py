#!/usr/bin/env python3
"""List or submit queued Pixel5 GSDC2023 Kaggle candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    CANDIDATES,
    DEFAULT_OUTPUT_DIR,
)
from experiments.build_gsdc2023_pre_submit_manifest import (
    DEFAULT_RISKY_TRIPS,
    build_pre_submit_manifest,
)


COMPETITION = "smartphone-decimeter-2023"
DEFAULT_TAG = "20260501"
PRE_SUBMIT_MANIFEST = "pre_submit_manifest.json"
PRE_SUBMIT_TRIP_CHECKS = "pre_submit_trip_delta_checks.csv"


@dataclass(frozen=True)
class QueueItem:
    candidate: str
    message: str
    priority_group: str


PENDING_QUEUE: tuple[QueueItem, ...] = (
    QueueItem(
        "pixel5phone_3p375_sjc_r0p84375",
        "20260501 pixel5 3.375 sjc r scale 0.84375",
        "sjc_r_scale_sweep",
    ),
    QueueItem(
        "pixel5phone_3p375_sjc_r1p6875",
        "20260501 pixel5 3.375 sjc r scale 1.6875",
        "sjc_r_scale_sweep",
    ),
    QueueItem(
        "pixel5phone_3p375_sjc_r2p53125",
        "20260501 pixel5 3.375 sjc r scale 2.53125",
        "sjc_r_scale_sweep",
    ),
    QueueItem(
        "pixel5phone_3p375_sjc_r0p84375_p6p0",
        "20260505 pixel5 3.375 sjc r scale 0.84375 p6p0 clean",
        "p6p0_clean_sjc_r_scale_sweep",
    ),
    QueueItem(
        "pixel5phone_3p375_sjc_r1p6875_p6p0",
        "20260505 pixel5 3.375 sjc r scale 1.6875 p6p0 clean",
        "p6p0_clean_sjc_r_scale_sweep",
    ),
    QueueItem(
        "pixel5phone_3p375_sjc_r2p53125_p6p0",
        "20260505 pixel5 3.375 sjc r scale 2.53125 p6p0 clean",
        "p6p0_clean_sjc_r_scale_sweep",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_mtv_de1_20230523",
        "20260501 pixel5 sjcr0 ablate mtv de1 20230523",
        "mtv_sjc_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_sjc_he2_20230606",
        "20260501 pixel5 sjcr0 ablate sjc he2 20230606",
        "mtv_sjc_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_sjc_be2_20230526",
        "20260501 pixel5 sjcr0 ablate sjc be2 20230526",
        "mtv_sjc_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_sjc_q_20230427",
        "20260501 pixel5 sjcr0 ablate sjc q 20230427",
        "mtv_sjc_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20230427",
        "20260501 pixel5 sjcr0 ablate mtv pe1 20230427",
        "mtv_sjc_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_mtv_pe1_20220322",
        "20260501 pixel5 sjcr0 ablate mtv pe1 20220322",
        "mtv_sjc_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_lax_p_20220224",
        "20260501 pixel5 sjcr0 ablate lax p 20220224",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_lax_i_20220224",
        "20260501 pixel5 sjcr0 ablate lax i 20220224",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_lax_m_20220223",
        "20260501 pixel5 sjcr0 ablate lax m 20220223",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_lax_n_20220223",
        "20260501 pixel5 sjcr0 ablate lax n 20220223",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_ebf_z_20220425",
        "20260501 pixel5 sjcr0 ablate ebf z 20220425",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_ebf_y_20220422",
        "20260501 pixel5 sjcr0 ablate ebf y 20220422",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_ebf_xx_20220427",
        "20260501 pixel5 sjcr0 ablate ebf xx 20220427",
        "lax_ebf_trip_ablation",
    ),
    QueueItem(
        "pixel5phone_3p375_sjcr0_ablate_ebf_zz_20220427",
        "20260501 pixel5 sjcr0 ablate ebf zz 20220427",
        "lax_ebf_trip_ablation",
    ),
)


def candidate_submission_path(candidate: str, output_dir: Path, tag: str) -> Path:
    if candidate not in CANDIDATES:
        raise KeyError(f"unknown candidate: {candidate}")
    return output_dir / candidate / f"submission_best_basecorr_posoffset_{candidate}_plus_pixel5_patch_{tag}.csv"


def selected_queue(groups: set[str] | None = None) -> list[QueueItem]:
    if not groups:
        return list(PENDING_QUEUE)
    return [item for item in PENDING_QUEUE if item.priority_group in groups]


def kaggle_submit_command(path: Path, message: str) -> list[str]:
    return [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(path),
        "-m",
        message,
    ]


def risk_report_payload(output_dir: Path) -> dict[str, object] | None:
    summary_path = output_dir / "build_summary.json"
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    report = payload.get("pr_proxy_risk_report")
    return report if isinstance(report, dict) else None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pre_submit_manifest_payload(output_dir: Path) -> dict[str, object] | None:
    manifest_path = output_dir / PRE_SUBMIT_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def p6p0_candidates(items: list[QueueItem]) -> list[str]:
    return [item.candidate for item in items if item.candidate.endswith("_p6p0")]


def existing_queue_items(queue: list[QueueItem], output_dir: Path, tag: str, *, skip_missing: bool = False) -> list[QueueItem]:
    existing: list[QueueItem] = []
    for item in queue:
        path = candidate_submission_path(item.candidate, output_dir, tag)
        if path.is_file():
            existing.append(item)
            continue
        if not skip_missing:
            raise SystemExit(f"missing candidate CSV: {path}")
    return existing


def _int_field(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _float_field(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def assert_matlab_equivalence_gate(manifest: dict[str, object], *, require: bool = False) -> dict[str, object] | None:
    gate = manifest.get("matlab_equivalence_gate")
    if gate is None:
        if require:
            raise SystemExit("pre-submit manifest is missing matlab_equivalence_gate")
        return None
    if not isinstance(gate, dict):
        raise SystemExit("pre-submit manifest matlab_equivalence_gate must be an object")
    if not bool(gate.get("passed", False)):
        raise SystemExit("MATLAB equivalence gate failed: passed=false")
    if gate.get("equivalence_claim") != "matlab_equivalent":
        raise SystemExit(f"MATLAB equivalence gate failed: equivalence_claim={gate.get('equivalence_claim')}")
    required_pass_fields = ("factor_mask_passed", "raw_bridge_counts_passed", "residual_values_passed")
    failed = [field for field in required_pass_fields if not bool(gate.get(field, False))]
    if failed:
        raise SystemExit(f"MATLAB equivalence gate failed fields: {', '.join(failed)}")
    matlab_only = _as_int(gate.get("residual_total_matlab_only"))
    bridge_only = _as_int(gate.get("residual_total_bridge_only"))
    if matlab_only != 0 or bridge_only != 0:
        raise SystemExit(
            "MATLAB equivalence residual side-only rows are nonzero: "
            f"matlab_only={matlab_only}, bridge_only={bridge_only}"
        )
    max_delta = _as_float(gate.get("residual_overall_max_abs_delta_m"))
    threshold = _as_float(gate.get("residual_max_abs_delta_threshold_m"))
    if threshold > 0.0 and max_delta > threshold:
        raise SystemExit(f"MATLAB equivalence residual max delta failed: {max_delta} > {threshold}")
    return gate


def assert_pre_submit_manifest_gate(
    output_dir: Path,
    candidates: list[str],
    *,
    require_matlab_equivalence: bool = False,
) -> dict[str, object]:
    selected = set(candidates)
    manifest = pre_submit_manifest_payload(output_dir)
    if manifest is None:
        raise SystemExit(f"missing pre-submit manifest in {output_dir / PRE_SUBMIT_MANIFEST}")
    assert_matlab_equivalence_gate(manifest, require=require_matlab_equivalence)

    risk_report = manifest.get("risk_report")
    risk_report = risk_report if isinstance(risk_report, dict) else {}
    try:
        actionable_chunks = int(risk_report.get("candidate_actionable_risky_chunks", -1))
    except (TypeError, ValueError):
        actionable_chunks = -1
    if actionable_chunks != 0:
        raise SystemExit(f"pre-submit manifest risk gate failed: candidate_actionable_risky_chunks={actionable_chunks}")

    manifest_candidates = manifest.get("candidates")
    if not isinstance(manifest_candidates, list):
        raise SystemExit("pre-submit manifest is missing candidates")
    by_candidate: dict[str, dict[str, object]] = {}
    for row in manifest_candidates:
        if isinstance(row, dict) and isinstance(row.get("candidate"), str):
            by_candidate[str(row["candidate"])] = row

    missing = sorted(selected - set(by_candidate))
    if missing:
        raise SystemExit(f"pre-submit manifest is missing candidates: {', '.join(missing)}")

    for candidate in sorted(selected):
        row = by_candidate[candidate]
        try:
            row_actionable = int(row.get("risk_candidate_actionable_chunks", -1))
        except (TypeError, ValueError):
            row_actionable = -1
        if row_actionable != 0:
            raise SystemExit(f"pre-submit manifest candidate risk failed for {candidate}: {row_actionable}")
        try:
            pixel6pro_scale = float(row.get("pixel6pro_scale"))
        except (TypeError, ValueError):
            pixel6pro_scale = float("nan")
        if candidate.endswith("_p6p0") and pixel6pro_scale != 0.0:
            raise SystemExit(f"pre-submit manifest expected pixel6pro_scale=0.0 for {candidate}, got {pixel6pro_scale}")
        output = row.get("output")
        output_sha256 = row.get("output_sha256")
        if isinstance(output, str) and isinstance(output_sha256, str):
            output_path = Path(output)
            if not output_path.is_file():
                raise SystemExit(f"pre-submit manifest output is missing for {candidate}: {output_path}")
            actual_sha256 = sha256_file(output_path)
            if actual_sha256 != output_sha256:
                raise SystemExit(f"pre-submit manifest sha256 mismatch for {candidate}: {actual_sha256} != {output_sha256}")

    trip_csv = output_dir / PRE_SUBMIT_TRIP_CHECKS
    if not trip_csv.is_file():
        raise SystemExit(f"missing pre-submit trip checks in {trip_csv}")
    seen: set[str] = set()
    with trip_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            candidate = row.get("candidate")
            if candidate not in selected:
                continue
            seen.add(candidate)
            if _int_field(row, "rows") <= 0:
                raise SystemExit(f"pre-submit trip check has no rows for {candidate}: {row.get('tripId')}")
            changed_rows = _int_field(row, "input_changed_rows")
            input_max_m = _float_field(row, "input_max_m")
            if changed_rows != 0 or input_max_m != 0.0:
                raise SystemExit(
                    f"pre-submit trip check failed for {candidate} {row.get('tripId')}: "
                    f"input_changed_rows={changed_rows}, input_max_m={input_max_m}"
                )
    missing_trip_checks = sorted(selected - seen)
    if missing_trip_checks:
        raise SystemExit(f"pre-submit trip checks are missing candidates: {', '.join(missing_trip_checks)}")
    return manifest


def assert_submit_risk_gate(output_dir: Path, *, allow_risk: bool = False) -> dict[str, object] | None:
    report = risk_report_payload(output_dir)
    if allow_risk:
        return report
    if report is None:
        raise SystemExit(f"missing risk report in {output_dir / 'build_summary.json'}")
    if not bool(report.get("enabled", False)):
        raise SystemExit("PR proxy risk report was not enabled for this candidate build")
    try:
        risky_chunks = int(report.get("candidate_actionable_risky_chunks", report.get("risky_chunks", 0)))
    except (TypeError, ValueError):
        risky_chunks = -1
    if risky_chunks != 0:
        raise SystemExit(
            f"PR proxy risk gate failed: risky_chunks={risky_chunks}; "
            "rebuild with clean --risk-metrics or pass --allow-risk explicitly"
        )
    return report


def build_ready_report(
    *,
    output_dir: Path,
    tag: str,
    groups: list[str] | None,
    queue: list[QueueItem],
    risk_report: dict[str, object] | None,
    pre_submit_manifest: dict[str, object] | None,
    allow_risk: bool,
) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for item in queue:
        path = candidate_submission_path(item.candidate, output_dir, tag)
        candidates.append(
            {
                "candidate": item.candidate,
                "priority_group": item.priority_group,
                "message": item.message,
                "path": str(path),
                "sha256": sha256_file(path),
                "command": kaggle_submit_command(path, item.message),
            },
        )
    manifest_risk: object | None = None
    matlab_equivalence_gate: object | None = None
    if isinstance(pre_submit_manifest, dict):
        manifest_risk = pre_submit_manifest.get("risk_report")
        matlab_equivalence_gate = pre_submit_manifest.get("matlab_equivalence_gate")
    return {
        "output_dir": str(output_dir),
        "tag": tag,
        "groups": groups or [],
        "allow_risk": allow_risk,
        "ready_count": len(candidates),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "risk_report": risk_report,
        "pre_submit_manifest": {
            "present": pre_submit_manifest is not None,
            "risk_report": manifest_risk,
            "matlab_equivalence_gate": matlab_equivalence_gate,
        },
    }


def write_ready_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    candidates = report.get("candidates", [])
    rows = candidates if isinstance(candidates, list) else []
    fieldnames = ["candidate", "priority_group", "message", "path", "sha256", "command"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            command = row.get("command", [])
            writer.writerow(
                {
                    "candidate": row.get("candidate", ""),
                    "priority_group": row.get("priority_group", ""),
                    "message": row.get("message", ""),
                    "path": row.get("path", ""),
                    "sha256": row.get("sha256", ""),
                    "command": shlex.join(command) if isinstance(command, list) else str(command),
                },
            )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value if value is not None else default))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _format_prepare_command(
    *,
    output_dir: Path,
    tag: str,
    groups: list[str] | None,
    ready_report_path: Path,
    previous_output_dir: Path | None,
    previous_tag: str,
    skip_missing: bool,
) -> str:
    args = ["--output-dir", str(output_dir), "--tag", tag]
    for group in groups or []:
        args.extend(["--group", group])
    args.extend(["--prepare-ready-report", str(ready_report_path)])
    if previous_output_dir is not None:
        args.extend(["--previous-output-dir", str(previous_output_dir)])
        args.extend(["--previous-tag", previous_tag])
    if skip_missing:
        args.append("--skip-missing")
    lines = ["PYTHONPATH=.:python python3 experiments/submit_gsdc2023_pixel5_candidate_queue.py \\"]
    i = 0
    while i < len(args):
        if args[i] == "--skip-missing":
            lines.append("  --skip-missing")
            i += 1
            continue
        lines.append(f"  {args[i]} {shlex.quote(args[i + 1])} \\")
        i += 2
    if lines[-1].endswith(" \\"):
        lines[-1] = lines[-1][:-2]
    return "\n".join(lines)


def write_submit_readiness_doc(
    *,
    output_dir: Path,
    ready_report_path: Path,
    tag: str,
    groups: list[str] | None,
    previous_output_dir: Path | None,
    previous_tag: str,
    skip_missing: bool,
) -> Path:
    report = _read_json_object(ready_report_path)
    manifest = _read_json_object(output_dir / PRE_SUBMIT_MANIFEST)
    ready_csv_rows = _csv_rows(ready_report_path.with_suffix(".csv"))
    trip_rows = _csv_rows(output_dir / PRE_SUBMIT_TRIP_CHECKS)
    manifest_risk = manifest.get("risk_report")
    manifest_risk = manifest_risk if isinstance(manifest_risk, dict) else {}
    matlab_equivalence = manifest.get("matlab_equivalence_gate")
    matlab_equivalence = matlab_equivalence if isinstance(matlab_equivalence, dict) else {}
    report_risk = report.get("risk_report")
    report_risk = report_risk if isinstance(report_risk, dict) else {}
    max_changed = max((_as_int(row.get("input_changed_rows")) for row in trip_rows), default=0)
    max_delta = max((_as_float(row.get("input_max_m")) for row in trip_rows), default=0.0)
    prepare_command = _format_prepare_command(
        output_dir=output_dir,
        tag=tag,
        groups=groups,
        ready_report_path=ready_report_path,
        previous_output_dir=previous_output_dir,
        previous_tag=previous_tag,
        skip_missing=skip_missing,
    )
    audit_command = (
        "PYTHONPATH=.:python python3 experiments/submit_gsdc2023_pixel5_candidate_queue.py \\\n"
        f"  --audit-ready-report {ready_report_path}"
    )
    candidates = report.get("candidates", [])
    candidate_rows = candidates if isinstance(candidates, list) else []
    table_rows = [
        "| Candidate | SHA256 |",
        "| --- | --- |",
    ]
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        table_rows.append(f"| `{row.get('candidate', '')}` | `{row.get('sha256', '')}` |")

    path = output_dir / "submit_readiness.md"
    path.write_text(
        "\n".join(
            [
                "# P6P0 Submit Readiness",
                "",
                "This directory contains the ready-to-audit artifacts for the P6P0 clean SJC-R scale sweep. Kaggle submission has not been run.",
                "",
                "## Regenerate",
                "",
                "```bash",
                prepare_command,
                "```",
                "",
                "Expected result:",
                "",
                "```text",
                f"prepared: {_as_int(report.get('ready_count'))} candidate(s)",
                "```",
                "",
                "## Audit Only",
                "",
                "```bash",
                audit_command,
                "```",
                "",
                "Expected result:",
                "",
                "```text",
                f"audited: {_as_int(report.get('ready_count'))} candidate(s)",
                "```",
                "",
                "## Artifacts",
                "",
                "- `build_summary.json`: candidate build summary and PR proxy risk report.",
                "- `pre_submit_manifest.json`: candidate-level pre-submit manifest.",
                "- `pre_submit_candidate_manifest.csv`: candidate manifest table.",
                "- `pre_submit_trip_delta_checks.csv`: risky Pixel6Pro trip delta checks.",
                "- `submit_ready_report.json`: ready report with candidate paths, SHA256, commands, and gate summaries.",
                "- `submit_ready_report.csv`: compact candidate table for human review.",
                "",
                "## Current Gate State",
                "",
                f"- Ready candidates: `{_as_int(report.get('ready_count'))}`",
                f"- Ready CSV rows: `{len(ready_csv_rows)}`",
                f"- Pre-submit manifest candidates: `{_as_int(manifest.get('candidate_count'))}`",
                f"- Risk actionable chunks: `{_as_int(report_risk.get('candidate_actionable_risky_chunks'))}`",
                f"- Pre-submit manifest actionable chunks: `{_as_int(manifest_risk.get('candidate_actionable_risky_chunks'))}`",
                f"- Risky Pixel6Pro trip delta rows: `{len(trip_rows)}`",
                f"- Max risky Pixel6Pro input changed rows: `{max_changed}`",
                f"- Max risky Pixel6Pro input delta: `{max_delta:.1f} m`",
                f"- MATLAB equivalence: `{matlab_equivalence.get('equivalence_claim', 'not recorded')}`",
                "",
                "## Candidate SHA256",
                "",
                *table_rows,
                "",
                "## Submit Command Source",
                "",
                "Use `submit_ready_report.csv` as the human-readable source for candidate paths and shell-quoted Kaggle commands. Use `submit_ready_report.json` as the machine-audited source.",
                "",
            ],
        ),
        encoding="utf-8",
    )
    return path


def _ready_report_candidates(report: dict[str, Any], report_path: Path) -> list[dict[str, Any]]:
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise SystemExit(f"ready report is missing candidates: {report_path}")
    rows: list[dict[str, Any]] = []
    for row in candidates:
        if not isinstance(row, dict):
            raise SystemExit(f"ready report has a non-object candidate row: {report_path}")
        rows.append(row)
    return rows


def _candidate_rows_by_name(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or not candidate:
            raise SystemExit(f"{label} has a candidate row without candidate name")
        if candidate in by_name:
            raise SystemExit(f"{label} has duplicate candidate: {candidate}")
        by_name[candidate] = row
    return by_name


def _risk_actionable_chunks(report: dict[str, object] | None) -> int | None:
    if not isinstance(report, dict):
        return None
    try:
        return int(report.get("candidate_actionable_risky_chunks", report.get("risky_chunks", 0)))
    except (TypeError, ValueError):
        return None


def assert_ready_report_consistency(report_path: Path) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = _read_json_object(report_path)
    output_dir_raw = report.get("output_dir")
    if not isinstance(output_dir_raw, str):
        raise SystemExit(f"ready report is missing output_dir: {report_path}")
    output_dir = Path(output_dir_raw).expanduser().resolve()
    allow_risk = bool(report.get("allow_risk", False))
    candidates = _ready_report_candidates(report, report_path)
    by_candidate = _candidate_rows_by_name(candidates, label="ready report")

    expected_count = int(report.get("ready_count", report.get("candidate_count", -1)))
    if expected_count != len(candidates):
        raise SystemExit(f"ready report count mismatch: ready_count={expected_count}, rows={len(candidates)}")

    current_risk = assert_submit_risk_gate(output_dir, allow_risk=allow_risk)
    ready_risk = report.get("risk_report")
    ready_actionable = _risk_actionable_chunks(ready_risk if isinstance(ready_risk, dict) else None)
    current_actionable = _risk_actionable_chunks(current_risk)
    if current_actionable is not None and ready_actionable is not None and current_actionable != ready_actionable:
        raise SystemExit(
            "ready report risk mismatch: "
            f"report candidate_actionable_risky_chunks={ready_actionable}, current={current_actionable}"
        )

    p6p0 = [candidate for candidate in by_candidate if candidate.endswith("_p6p0")]
    manifest: dict[str, object] | None = None
    manifest_by_candidate: dict[str, dict[str, Any]] = {}
    if p6p0 and not allow_risk:
        manifest = assert_pre_submit_manifest_gate(output_dir, p6p0)
        manifest_rows = manifest.get("candidates")
        if not isinstance(manifest_rows, list):
            raise SystemExit("pre-submit manifest is missing candidates")
        manifest_by_candidate = _candidate_rows_by_name(
            [row for row in manifest_rows if isinstance(row, dict)],
            label="pre-submit manifest",
        )

    for candidate, row in sorted(by_candidate.items()):
        path_raw = row.get("path")
        sha_raw = row.get("sha256")
        if not isinstance(path_raw, str) or not isinstance(sha_raw, str):
            raise SystemExit(f"ready report row is missing path/sha256 for {candidate}")
        path = Path(path_raw).expanduser()
        if not path.is_absolute():
            cwd_path = path.resolve()
            path = cwd_path if cwd_path.exists() else (report_path.parent / path).resolve()
        if not path.is_file():
            raise SystemExit(f"ready report candidate CSV is missing for {candidate}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != sha_raw:
            raise SystemExit(f"ready report sha256 mismatch for {candidate}: {actual_sha} != {sha_raw}")
        if candidate in manifest_by_candidate:
            manifest_sha = manifest_by_candidate[candidate].get("output_sha256")
            if manifest_sha != sha_raw:
                raise SystemExit(f"ready report/pre-submit sha256 mismatch for {candidate}: {sha_raw} != {manifest_sha}")

    csv_path = report_path.with_suffix(".csv")
    if not csv_path.is_file():
        raise SystemExit(f"ready report CSV is missing: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    if len(csv_rows) != len(candidates):
        raise SystemExit(f"ready report CSV row mismatch: json={len(candidates)}, csv={len(csv_rows)}")
    csv_by_candidate = {row.get("candidate", ""): row for row in csv_rows}
    if set(csv_by_candidate) != set(by_candidate):
        raise SystemExit("ready report CSV candidates differ from JSON candidates")
    for candidate, row in by_candidate.items():
        csv_sha = csv_by_candidate[candidate].get("sha256")
        if csv_sha != row.get("sha256"):
            raise SystemExit(f"ready report CSV sha256 mismatch for {candidate}: {csv_sha} != {row.get('sha256')}")
    return report


def prepare_ready_report(
    *,
    output_dir: Path,
    tag: str,
    groups: list[str] | None,
    ready_report_path: Path,
    build_summary_path: Path | None = None,
    previous_output_dir: Path | None = None,
    previous_tag: str = "20260501",
    risky_trips: tuple[str, ...] = DEFAULT_RISKY_TRIPS,
    matlab_equivalence_summary: Path | None = None,
    require_matlab_equivalence: bool = False,
    skip_missing: bool = False,
    allow_risk: bool = False,
) -> dict[str, Any]:
    build_summary = build_summary_path or output_dir / "build_summary.json"
    build_pre_submit_manifest(
        build_summary,
        output_dir=output_dir,
        previous_output_dir=previous_output_dir,
        previous_tag=previous_tag,
        risky_trips=risky_trips,
        matlab_equivalence_summary=matlab_equivalence_summary,
    )
    queue = selected_queue(set(groups) if groups else None)
    ready_queue = existing_queue_items(queue, output_dir, tag, skip_missing=skip_missing)
    risk_report: dict[str, object] | None = None
    pre_submit_manifest: dict[str, object] | None = None
    if ready_queue:
        risk_report = assert_submit_risk_gate(output_dir, allow_risk=allow_risk)
    clean_p6p0 = p6p0_candidates(ready_queue)
    if clean_p6p0 and not allow_risk:
        pre_submit_manifest = assert_pre_submit_manifest_gate(
            output_dir,
            clean_p6p0,
            require_matlab_equivalence=require_matlab_equivalence,
        )
    report = build_ready_report(
        output_dir=output_dir,
        tag=tag,
        groups=groups,
        queue=ready_queue,
        risk_report=risk_report,
        pre_submit_manifest=pre_submit_manifest,
        allow_risk=allow_risk,
    )
    write_ready_report(ready_report_path, report)
    audited = assert_ready_report_consistency(ready_report_path)
    write_submit_readiness_doc(
        output_dir=output_dir,
        ready_report_path=ready_report_path,
        tag=tag,
        groups=groups,
        previous_output_dir=previous_output_dir,
        previous_tag=previous_tag,
        skip_missing=skip_missing,
    )
    return audited


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--group", action="append", choices=sorted({item.priority_group for item in PENDING_QUEUE}))
    parser.add_argument("--submit", action="store_true", help="run kaggle submissions instead of listing commands")
    parser.add_argument("--check-ready", action="store_true", help="run submit gates without calling Kaggle")
    parser.add_argument("--ready-report", type=Path, help="write check-ready/submit candidate manifest JSON")
    parser.add_argument("--audit-ready-report", type=Path, help="audit a ready-report JSON and its paired CSV")
    parser.add_argument(
        "--prepare-ready-report",
        type=Path,
        help="build pre-submit manifest, write ready-report JSON/CSV, and audit the result",
    )
    parser.add_argument("--build-summary", type=Path, help="build_summary.json for --prepare-ready-report")
    parser.add_argument("--previous-output-dir", type=Path, help="previous candidate output dir for pre-submit manifest")
    parser.add_argument("--previous-tag", default="20260501")
    parser.add_argument("--risky-trip", action="append", dest="risky_trips")
    parser.add_argument("--matlab-equivalence-summary", type=Path)
    parser.add_argument(
        "--require-matlab-equivalence",
        action="store_true",
        help="require a passing MATLAB equivalence summary in the pre-submit manifest for P6P0 candidates",
    )
    parser.add_argument("--skip-missing", action="store_true", help="skip candidates whose CSVs do not exist")
    parser.add_argument(
        "--allow-risk",
        action="store_true",
        help="allow Kaggle submit even when the build risk report is missing or has risky chunks",
    )
    args = parser.parse_args(argv)
    if args.audit_ready_report:
        report = assert_ready_report_consistency(args.audit_ready_report)
        print(f"audited: {int(report.get('ready_count', 0))} candidate(s)")
        return 0
    if args.prepare_ready_report:
        report = prepare_ready_report(
            output_dir=args.output_dir,
            tag=args.tag,
            groups=args.group,
            ready_report_path=args.prepare_ready_report,
            build_summary_path=args.build_summary,
            previous_output_dir=args.previous_output_dir,
            previous_tag=args.previous_tag,
            risky_trips=tuple(args.risky_trips or DEFAULT_RISKY_TRIPS),
            matlab_equivalence_summary=args.matlab_equivalence_summary,
            require_matlab_equivalence=args.require_matlab_equivalence,
            skip_missing=args.skip_missing,
            allow_risk=args.allow_risk,
        )
        print(f"prepared: {int(report.get('ready_count', 0))} candidate(s)")
        return 0

    queue = selected_queue(set(args.group) if args.group else None)
    ready_queue = existing_queue_items(queue, args.output_dir, args.tag, skip_missing=args.skip_missing)
    risk_report: dict[str, object] | None = None
    pre_submit_manifest: dict[str, object] | None = None

    if args.submit or args.check_ready:
        if ready_queue:
            risk_report = assert_submit_risk_gate(args.output_dir, allow_risk=args.allow_risk)
        clean_p6p0 = p6p0_candidates(ready_queue)
        if clean_p6p0 and not args.allow_risk:
            pre_submit_manifest = assert_pre_submit_manifest_gate(
                args.output_dir,
                clean_p6p0,
                require_matlab_equivalence=args.require_matlab_equivalence,
            )
        if args.check_ready:
            print(f"ready: {len(ready_queue)} candidate(s)")
    if args.ready_report:
        report = build_ready_report(
            output_dir=args.output_dir,
            tag=args.tag,
            groups=args.group,
            queue=ready_queue,
            risk_report=risk_report,
            pre_submit_manifest=pre_submit_manifest,
            allow_risk=args.allow_risk,
        )
        write_ready_report(args.ready_report, report)

    for item in ready_queue:
        path = candidate_submission_path(item.candidate, args.output_dir, args.tag)
        command = kaggle_submit_command(path, item.message)
        if args.submit:
            subprocess.run(command, check=True)
        else:
            print(shlex.join(command))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
