#!/usr/bin/env python3
"""Replay faithful Recurrence Vector on predeclared blocked spans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("blocked_span_manifest.csv")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_recurrence_full_runs import (  # noqa: E402
    SAFE_MAX_SOURCE_ERROR_M,
    SAFE_MIN_SELECTED_PROBABILITY,
    _chunk_is_complete,
    _recurrence_mode_flags,
    _summarize_epoch_files,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=REPO / "datasets/PPC-Dataset-data")
    parser.add_argument(
        "--source-pos-dir",
        type=Path,
        default=REPO / "experiments/results/libgnss_rtk_pos_v5",
    )
    parser.add_argument("--triangle-cache-dir", type=Path, default=Path("E:/datasets/plateau_cache"))
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/results")
    parser.add_argument("--radius-m", type=float, default=3.0)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resummarize", action="store_true")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="evaluate the ungated paper-method argmax as a counterfactual",
    )
    args = parser.parse_args()

    spans = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    summaries: list[dict[str, object]] = []
    for span in spans:
        city, run, span_id = span["city"], span["run"], span["span_id"]
        start = int(span["start_epoch"])
        count = int(span["end_epoch_exclusive"]) - start
        artifact_prefix = (
            "candidate_3dma_recurrence_raw_blocked"
            if args.raw
            else "candidate_3dma_recurrence_blocked"
        )
        prefix = args.out_dir / f"{artifact_prefix}_{span_id}"
        summary_path = prefix.with_name(prefix.name + "_summary.json")
        epoch_path = prefix.with_name(prefix.name + "_epochs.csv")
        if args.force or not _chunk_is_complete(
            summary_path, epoch_path, start=start, count=count, raw=args.raw
        ):
            command = [
                sys.executable,
                str(Path(__file__).with_name("eval_candidate_3dma_ppc.py")),
                "--data-dir",
                str(args.data_root / city / run),
                "--source-pos",
                str(args.source_pos_dir / f"{city}_{run}_full.pos"),
                "--triangle-cache-npz",
                str(args.triangle_cache_dir / f"{city}_{run}_triangles.npz"),
                "--out-prefix",
                str(prefix),
                "--start-epoch",
                str(start),
                "--max-epochs",
                str(count),
                "--strategy",
                "recurrence_vector",
                "--radius-m",
                str(args.radius_m),
                "--spacing-m",
                str(args.spacing_m),
            ]
            command.extend(_recurrence_mode_flags(args.raw))
            print(f"[{span_id}] replay {count} epochs", flush=True)
            subprocess.run(command, cwd=REPO, check=True)
        original_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if args.force or args.resummarize or "honest_ppc_score_pct" not in original_payload:
            scored = _summarize_epoch_files(
                [epoch_path],
                requested_epochs=count,
                runtime_s=float(original_payload.get("runtime_s", 0.0)),
                reference_path=args.data_root / city / run / "reference.csv",
                start_epoch=start,
                end_epoch=start + count,
            )
            original_payload.update(scored)
            summary_path.write_text(
                json.dumps(original_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        payload = original_payload
        payload["span_id"] = span_id
        payload["evaluation_role"] = span.get(
            "recurrence_evaluation_role", span["evaluation_role"]
        )
        payload["provenance"] = span["provenance"]
        payload["recurrence_mode"] = (
            "raw_counterfactual" if args.raw else "safe_gated"
        )
        payload["recurrence_min_selected_probability"] = (
            0.0 if args.raw else SAFE_MIN_SELECTED_PROBABILITY
        )
        payload["recurrence_max_source_error_m"] = (
            0.0 if args.raw else SAFE_MAX_SOURCE_ERROR_M
        )
        payload["recurrence_allow_boundary"] = bool(args.raw)
        summaries.append(payload)

    fields = [
        "span_id",
        "evaluation_role",
        "provenance",
        "recurrence_mode",
        "recurrence_min_selected_probability",
        "recurrence_max_source_error_m",
        "recurrence_allow_boundary",
        "requested_epochs",
        "evaluated_epochs",
        "skipped_epochs",
        "coverage",
        "recurrence_abstained_epochs",
        "recurrence_acceptance_rate",
        "baseline_p50_m",
        "selected_p50_m",
        "baseline_p95_m",
        "selected_p95_m",
        "baseline_p99_m",
        "selected_p99_m",
        "improved_epochs",
        "worsened_epochs",
        "baseline_honest_ppc_score_pct",
        "honest_ppc_score_pct",
        "pass_distance_m",
        "total_distance_m",
        "runtime_s",
        "runtime_ms_per_evaluated_epoch",
    ]
    output = args.out_dir / (
        "candidate_3dma_recurrence_raw_blocked_spans_summary.csv"
        if args.raw
        else "candidate_3dma_recurrence_blocked_spans_summary.csv"
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fields} for row in summaries)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
