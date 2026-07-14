#!/usr/bin/env python3
"""Replay TC-FGO structural factors on predeclared blocked spans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ppc_distance_score import honest_ppc_distance_score  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("blocked_span_manifest.csv")
VARIANTS: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "wcp": ("--wcp",),
    "switch": ("--switchable-pseudorange",),
    "wcp_switch": ("--wcp", "--switchable-pseudorange"),
}

_CAUSAL_INIT_FAILURE = "insufficient static RTK FIX epochs for phase-1 init"


def _sha256_or_empty(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _classify_expected_abstention(stderr: str) -> str | None:
    """Return an honest evaluation abstention reason for a known causal limit."""
    if _CAUSAL_INIT_FAILURE in stderr:
        return "insufficient_causal_static_fix_history"
    return None


def _write_empty_telemetry(path: Path) -> None:
    """Create a readable zero-coverage artifact for a causally unavailable span."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "pos_err_m",
                "n_wcp_factors",
                "n_switchable_pseudorange",
                "n_switched_pseudorange",
            ]
        )


def _finite(values: list[str]) -> np.ndarray:
    data = np.asarray([float(value) for value in values], dtype=np.float64)
    return data[np.isfinite(data)]


def _summarize(
    telemetry_path: Path,
    *,
    start_epoch: int,
    end_epoch: int,
    requested_epochs: int,
    runtime_s: float,
    reference_path: Path | None = None,
) -> dict[str, object]:
    all_rows = list(csv.DictReader(telemetry_path.open(newline="", encoding="utf-8")))
    rows = [
        row
        for row in all_rows
        if start_epoch <= int(row["epoch"]) < end_epoch
    ]
    errors = _finite([row["pos_err_m"] for row in rows])
    result: dict[str, object] = {
        "run_status": "ok",
        "failure_reason": "",
        "requested_epochs": int(requested_epochs),
        "output_epochs": len(rows),
        "evaluated_epochs": int(errors.size),
        "coverage": float(errors.size / requested_epochs) if requested_epochs else 0.0,
        "runtime_s": float(runtime_s),
        "warmup_epochs": int(start_epoch),
        "runtime_ms_per_output_epoch": (
            float(1000.0 * runtime_s / len(rows)) if rows else float("nan")
        ),
        "n_wcp_factors": sum(int(row["n_wcp_factors"]) for row in rows),
        "n_switchable_pseudorange": sum(
            int(row["n_switchable_pseudorange"]) for row in rows
        ),
        "n_switched_pseudorange": sum(int(row["n_switched_pseudorange"]) for row in rows),
        "n_switch_integrity_abstained_epochs": sum(
            int(row.get("n_switch_integrity_abstained_epochs", 0)) for row in rows
        ),
        "n_switch_integrity_abstained_rows": sum(
            int(row.get("n_switch_integrity_abstained_rows", 0)) for row in rows
        ),
        "n_switch_shadow_epochs": sum(
            int(row.get("n_switch_shadow_epochs", 0)) for row in rows
        ),
    }
    if reference_path is not None:
        result.update(
            honest_ppc_distance_score(
                {int(row["epoch"]): float(row["pos_err_m"]) for row in rows},
                reference_path,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            )
        )
    if errors.size:
        result.update(
            {
                "pass_0_5m": float(np.mean(errors <= 0.5)),
                "pass_1m": float(np.mean(errors <= 1.0)),
                "pass_3m": float(np.mean(errors <= 3.0)),
                "error_p50_m": float(np.quantile(errors, 0.50)),
                "error_p95_m": float(np.quantile(errors, 0.95)),
                "error_p99_m": float(np.quantile(errors, 0.99)),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=Path("E:/datasets/PPC-Dataset-data"))
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/results")
    parser.add_argument("--systems", default="G,E,J")
    parser.add_argument("--window-epochs", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resummarize", action="store_true")
    parser.add_argument(
        "--refresh-switch",
        action="store_true",
        help="rerun switch-bearing variants while preserving baseline/WCP results",
    )
    args = parser.parse_args()

    spans = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    summaries: list[dict[str, object]] = []
    for span in spans:
        start = int(span["start_epoch"])
        count = int(span["end_epoch_exclusive"]) - start
        for variant, flags in VARIANTS.items():
            stem = f"tcfgo_{variant}_blocked_{span['span_id']}"
            position_path = args.out_dir / f"{stem}.pos"
            telemetry_path = args.out_dir / f"{stem}_telemetry.csv"
            summary_path = args.out_dir / f"{stem}_summary.json"
            needs_run = (
                args.force
                or (args.refresh_switch and variant in {"switch", "wcp_switch"})
                or not summary_path.exists()
            )
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
                    f"{span['city']}/{span['run']}",
                    "--data-root",
                    str(args.data_root),
                    "--max-epochs",
                    str(start + count),
                    "--export-pos",
                    str(position_path),
                    "--telemetry-csv",
                    str(telemetry_path),
                    "--systems",
                    str(args.systems),
                    "--window-epochs",
                    str(args.window_epochs),
                    *flags,
                ]
                print(
                    f"[{span['span_id']}/{variant}] replay {start} warmup + {count} scored epochs",
                    flush=True,
                )
                started = time.perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=REPO,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if completed.stdout:
                    print(completed.stdout, end="", flush=True)
                runtime_s = time.perf_counter() - started
                abstention_reason = _classify_expected_abstention(completed.stderr)
                if completed.returncode != 0 and abstention_reason is None:
                    if completed.stderr:
                        print(completed.stderr, end="", file=sys.stderr, flush=True)
                    raise subprocess.CalledProcessError(
                        completed.returncode,
                        command,
                        output=completed.stdout,
                        stderr=completed.stderr,
                    )
                if abstention_reason is not None:
                    print(
                        f"[{span['span_id']}/{variant}] abstained: {abstention_reason}",
                        flush=True,
                    )
                    _write_empty_telemetry(telemetry_path)
                elif completed.stderr:
                    print(completed.stderr, end="", file=sys.stderr, flush=True)
            if needs_run or args.resummarize:
                payload = _summarize(
                    telemetry_path,
                    start_epoch=start,
                    end_epoch=start + count,
                    requested_epochs=count,
                    runtime_s=runtime_s,
                    reference_path=(
                        args.data_root / span["city"] / span["run"] / "reference.csv"
                    ),
                )
                if needs_run and abstention_reason is not None:
                    payload["run_status"] = "abstained"
                    payload["failure_reason"] = abstention_reason
                summary_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            else:
                payload = previous_payload
            payload["position_sha256"] = _sha256_or_empty(position_path)
            payload.update(
                {
                    "span_id": span["span_id"],
                    "evaluation_role": span.get(
                        "tcfgo_evaluation_role", span["evaluation_role"]
                    ),
                    "variant": variant,
                }
            )
            summaries.append(payload)

    fields = [
        "span_id",
        "evaluation_role",
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
    output = args.out_dir / "tcfgo_structural_blocked_spans_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in summaries)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
