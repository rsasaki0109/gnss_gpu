#!/usr/bin/env python3
"""Summarize GSDC2023 source-family ablation candidates into a submit policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_POLICY = {
    "pixel4_fgo": (
        1,
        "first_mtv700_probe",
        "Narrow intended rescue family; test as pixel4 FGO only on a private-floor base.",
    ),
    "pixel6pro_fgo": (
        2,
        "risk_ablation",
        "Known risky phone family; test off/on after pixel4 rescue is isolated.",
    ),
    "pixel5_fgo_no_tdcp": (
        3,
        "risk_ablation",
        "Larger fgo_no_tdcp block; test independently after pixel6pro FGO.",
    ),
    "raw_wls_all": (
        4,
        "hold_spike_risk",
        "Raw WLS rows are narrow but historically spike-prone; do not lead with this.",
    ),
    "interpolated_missing_all": (
        5,
        "hold_fill_artifact",
        "Only 24 rows, but row movement is large; handle as fill policy, not a standalone probe.",
    ),
    "nonbaseline_all": (
        99,
        "reject_too_broad",
        "Broad multi-family patch; this repeats the full-bridge failure mode.",
    ),
}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty manifest: {path}")
    return rows


def _by_family_mode(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        family = str(row.get("family", ""))
        mode = str(row.get("mode", ""))
        if not family or not mode:
            raise SystemExit("manifest rows must include family and mode")
        by_key[(family, mode)] = row
    return by_key


def _summarize_family(family: str, only: dict[str, str] | None, revert: dict[str, str] | None) -> dict[str, Any]:
    source = only or revert
    if source is None:
        raise SystemExit(f"family has no rows: {family}")
    rank, action, reason = DEFAULT_POLICY.get(
        family,
        (50, "review_manually", "No family-specific policy is recorded."),
    )
    return {
        "rank": rank,
        "family": family,
        "recommended_action": action,
        "recommendation_reason": reason,
        "selected_rows": _as_int(source.get("selected_rows")),
        "selected_trip_count": _as_int(source.get("selected_trip_count")),
        "sources": source.get("sources", ""),
        "phones": source.get("phones", ""),
        "selected_score_m": _as_float(source.get("selected_vs_reference_score_m")),
        "selected_p50_m": _as_float(source.get("selected_vs_reference_p50_m")),
        "selected_p95_m": _as_float(source.get("selected_vs_reference_p95_m")),
        "selected_max_m": _as_float(source.get("selected_vs_reference_max_m")),
        "only_output": only.get("output", "") if only else "",
        "only_sha256": only.get("output_sha256", "") if only else "",
        "revert_output": revert.get("output", "") if revert else "",
        "revert_sha256": revert.get("output_sha256", "") if revert else "",
    }


def build_source_family_ranking(
    *,
    manifest_path: Path,
    output_dir: Path,
    tag: str,
) -> list[dict[str, Any]]:
    rows = _read_manifest(manifest_path.expanduser().resolve())
    by_key = _by_family_mode(rows)
    families = sorted({family for family, _mode in by_key})
    summary_rows = [
        _summarize_family(
            family,
            by_key.get((family, "only")),
            by_key.get((family, "revert")),
        )
        for family in families
    ]
    summary_rows.sort(key=lambda row: (int(row["rank"]), str(row["family"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"source_family_ablation_ranking_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / f"source_family_ablation_ranking_{tag}.json"
    json_path.write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "ranking_csv": str(csv_path),
                "rows": summary_rows,
                "submit_policy": {
                    "must_have_private_floor_reference": True,
                    "do_not_submit_full_bridge": True,
                    "one_family_per_probe": True,
                    "first_probe_family": "pixel4_fgo",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    md_path = output_dir / f"source_family_ablation_ranking_{tag}.md"
    md_path.write_text(_render_markdown(summary_rows, manifest_path, csv_path, json_path), encoding="utf-8")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")
    return summary_rows


def _render_markdown(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    csv_path: Path,
    json_path: Path,
) -> str:
    lines = [
        "# GSDC2023 Source-Family Ablation Ranking",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Ranking CSV: `{csv_path}`",
        f"- Ranking JSON: `{json_path}`",
        "",
        "## Submit Policy",
        "",
        "- Do not submit another full raw-bridge export.",
        "- Do not submit these dry-run candidates unless the reference is a recovered/reconstructed private-floor base.",
        "- Require a mtv700/private-floor reference body or a reconstructed private-floor base.",
        "- Probe one source family at a time.",
        "- First probe after a private-floor base is available: `pixel4_fgo` only.",
        "",
        "## Ranking",
        "",
        "| Rank | Family | Action | Rows | Trips | Selected score m | P95 m | Max m | Reason |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | `{family}` | `{recommended_action}` | {selected_rows} | "
            "{selected_trip_count} | {selected_score_m:.3f} | {selected_p95_m:.3f} | "
            "{selected_max_m:.3f} | {recommendation_reason} |".format(**row),
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)

    build_source_family_ranking(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        tag=args.tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
