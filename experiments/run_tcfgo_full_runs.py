#!/usr/bin/env python3
"""Replay TC-FGO structural-factor variants on all six official PPC runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_tcfgo_blocked_spans import VARIANTS, _sha256_or_empty, _summarize  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)
PHASE_INIT_STATIC_FIXES = {("nagoya", "run3"): 4}
FIELDS = [
    "scope_id",
    "city",
    "run",
    "evaluation_role",
    "phase_init_static_fixes",
    "variant",
    "run_status",
    "failure_reason",
    "requested_epochs",
    "output_epochs",
    "evaluated_epochs",
    "coverage",
    "pass_0_5m",
    "pass_1m",
    "pass_3m",
    "error_p50_m",
    "error_p95_m",
    "error_p99_m",
    "honest_ppc_score_pct",
    "pass_distance_m",
    "total_distance_m",
    "n_wcp_factors",
    "n_switchable_pseudorange",
    "n_switched_pseudorange",
    "n_switch_integrity_abstained_epochs",
    "n_switch_integrity_abstained_rows",
    "n_switch_shadow_epochs",
    "runtime_s",
    "warmup_epochs",
    "runtime_ms_per_output_epoch",
    "position_sha256",
]


def _phase_init_static_fixes(city: str, run: str) -> int:
    """Use every available causal static FIX, with five as the normal protocol."""
    return PHASE_INIT_STATIC_FIXES.get((city, run), 5)


def _reference_epoch_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def _needs_run(
    *,
    force: bool,
    refresh_switch: bool,
    resummarize: bool,
    variant: str,
    summary_exists: bool,
) -> bool:
    """Keep the read-only resummarization path from launching replays."""
    return not resummarize and (
        force
        or (refresh_switch and variant in {"switch", "wcp_switch"})
        or not summary_exists
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("E:/datasets/PPC-Dataset-data"))
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/results")
    parser.add_argument("--systems", default="G,E,J")
    parser.add_argument("--window-epochs", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resummarize", action="store_true")
    parser.add_argument(
        "--only-scope",
        choices=[f"{city}_{run}_full" for city, run in RUNS],
        help="process one official run only",
    )
    parser.add_argument(
        "--only-variant",
        choices=tuple(VARIANTS),
        help="process one structural variant only",
    )
    parser.add_argument(
        "--refresh-switch",
        action="store_true",
        help="rerun switch-bearing variants while preserving baseline/WCP results",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for city, run in RUNS:
        scope_id = f"{city}_{run}_full"
        if args.only_scope and scope_id != args.only_scope:
            continue
        requested_epochs = _reference_epoch_count(
            args.data_root / city / run / "reference.csv"
        )
        phase_init_static_fixes = _phase_init_static_fixes(city, run)
        for variant, flags in VARIANTS.items():
            if args.only_variant and variant != args.only_variant:
                continue
            stem = f"tcfgo_{variant}_{scope_id}"
            position_path = args.out_dir / f"{stem}.pos"
            telemetry_path = args.out_dir / f"{stem}_telemetry.csv"
            summary_path = args.out_dir / f"{stem}_summary.json"
            needs_run = _needs_run(
                force=args.force,
                refresh_switch=args.refresh_switch,
                resummarize=args.resummarize,
                variant=variant,
                summary_exists=summary_path.exists(),
            )
            if args.resummarize and not (
                position_path.exists() and telemetry_path.exists()
            ):
                print(
                    f"[{scope_id}/{variant}] skip resummarize: artifacts incomplete",
                    flush=True,
                )
                continue
            previous_payload = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else {}
            )
            runtime_s = float(previous_payload.get("runtime_s", 0.0))
            if needs_run:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("wp12_run_tc_fgo.py")),
                    "--run",
                    f"{city}/{run}",
                    "--data-root",
                    str(args.data_root),
                    "--max-epochs",
                    "0",
                    "--export-pos",
                    str(position_path),
                    "--telemetry-csv",
                    str(telemetry_path),
                    "--systems",
                    str(args.systems),
                    "--window-epochs",
                    str(args.window_epochs),
                    "--phase-init-static-fixes",
                    str(phase_init_static_fixes),
                    *flags,
                ]
                print(f"[{scope_id}/{variant}] replay {requested_epochs} epochs", flush=True)
                started = time.perf_counter()
                subprocess.run(command, cwd=REPO, check=True)
                runtime_s = time.perf_counter() - started
            if needs_run or args.resummarize:
                payload = _summarize(
                    telemetry_path,
                    start_epoch=0,
                    end_epoch=requested_epochs,
                    requested_epochs=requested_epochs,
                    runtime_s=runtime_s,
                    reference_path=args.data_root / city / run / "reference.csv",
                )
                summary_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            else:
                payload = previous_payload
            payload["position_sha256"] = _sha256_or_empty(position_path)
            payload.update(
                {
                    "scope_id": scope_id,
                    "city": city,
                    "run": run,
                    "evaluation_role": (
                        "development" if (city, run) == ("tokyo", "run1") else "holdout"
                    ),
                    "phase_init_static_fixes": phase_init_static_fixes,
                    "variant": variant,
                }
            )
            summaries.append(payload)

    output = args.out_dir / "tcfgo_structural_full_runs_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in summaries)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
