"""Reproduce the GSDC2023 MATLAB reference final CSV from bridge artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from experiments.materialize_gsdc2023_missing_bridge_timestamp_rows import (
    materialize_missing_bridge_timestamp_rows,
    write_outputs as write_missing_timestamp_outputs,
)
from experiments.reconstruct_gsdc2023_matlab_reference_submission import (
    reconstruct_matlab_reference_submission,
    write_outputs as write_reconstruction_outputs,
)


DEFAULT_REFERENCE_SUBMISSION = Path("../ref/gsdc2023/results/test_parallel/20260501_0526/submission_20260501_0526.csv")
DEFAULT_CANDIDATE_SUBMISSION = Path(
    "experiments/results/source_selection_lowbaseline_submission_probe_20260430/"
    "pixel5_old_gated_fgo_early_raw_late_extra_candidate/"
    "submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv",
)
DEFAULT_BRIDGE_ROOT = Path("../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/test")
DEFAULT_MAX_DELTA_M = 1e-6


def reproduce_matlab_reference_final(
    *,
    reference_submission: Path,
    candidate_submission: Path,
    bridge_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    missing_rows, missing_summary = materialize_missing_bridge_timestamp_rows(
        submission=pd.read_csv(reference_submission),
        bridge_root=bridge_root,
    )
    missing_dir = output_dir / "missing_bridge_timestamp_rows"
    write_missing_timestamp_outputs(missing_dir, missing_rows, missing_summary)

    reconstructed, rows, trips, source_runs, trip_delta, summary = reconstruct_matlab_reference_submission(
        reference_submission=reference_submission,
        candidate_submission=candidate_submission,
        bridge_root=bridge_root,
        override_row_summaries=[("missing_bridge_timestamps", missing_dir / "missing_bridge_timestamp_rows.csv")],
    )
    reconstruction_dir = output_dir / "reconstruction"
    write_reconstruction_outputs(
        reconstruction_dir,
        reconstructed,
        rows,
        trips,
        source_runs,
        trip_delta,
        summary,
    )

    payload = {
        "reference_submission": str(reference_submission),
        "candidate_submission": str(candidate_submission),
        "bridge_root": str(bridge_root),
        "missing_bridge_timestamp_summary": missing_summary,
        "reconstruction_summary": summary,
        "missing_bridge_timestamp_rows_csv": str(missing_dir / "missing_bridge_timestamp_rows.csv"),
        "reconstructed_submission_csv": str(reconstruction_dir / "submission_reconstructed_matlab_reference.csv"),
        "reconstruction_summary_json": str(reconstruction_dir / "summary.json"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def validate_reproduction_exact(payload: dict[str, object], *, max_delta_m: float) -> None:
    delta = payload["reconstruction_summary"]["delta_vs_reference"]
    max_delta = float(delta["max_delta_m"])
    if max_delta > max_delta_m:
        raise ValueError(
            "reconstructed submission is not numerically exact: "
            f"max_delta_m={max_delta:.6g} exceeds threshold {max_delta_m:.6g}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-submission", type=Path, default=DEFAULT_REFERENCE_SUBMISSION)
    parser.add_argument("--candidate-submission", type=Path, default=DEFAULT_CANDIDATE_SUBMISSION)
    parser.add_argument("--bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Fail with exit code 2 if the reconstructed CSV differs from the reference beyond --max-delta-m.",
    )
    parser.add_argument(
        "--max-delta-m",
        type=float,
        default=DEFAULT_MAX_DELTA_M,
        help=f"Maximum allowed haversine delta when --require-exact is set (default: {DEFAULT_MAX_DELTA_M:g}).",
    )
    args = parser.parse_args(argv)

    payload = reproduce_matlab_reference_final(
        reference_submission=args.reference_submission,
        candidate_submission=args.candidate_submission,
        bridge_root=args.bridge_root,
        output_dir=args.output_dir,
    )
    if args.require_exact:
        try:
            validate_reproduction_exact(payload, max_delta_m=args.max_delta_m)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    delta = payload["reconstruction_summary"]["delta_vs_reference"]
    print(
        "reproduced MATLAB reference final: "
        f"rows={delta['rows']} p95={delta['p95_delta_m']:.6g}m max={delta['max_delta_m']:.6g}m "
        f"missing_rows={payload['missing_bridge_timestamp_summary']['rows']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
