"""Run the fixed Taroz full GNSS+IMU parity gate for GSDC2023 smoke devices."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.summarize_gsdc2023_taroz_imu_state_parity import run as summarize_parity


DEFAULT_SUMMARIES: tuple[tuple[str, Path], ...] = (
    (
        "pixel5",
        Path("experiments/results/native_taroz_imu_state_navstate_rj_100_iter1_extpreint_fdjac_gravitynav_default_firstpair_cli_delta_20260606.json"),
    ),
    (
        "pixel6pro",
        Path(
            "experiments/results/"
            "native_taroz_imu_state_navstate_rj_pixel6pro_100_iter1_extpreint_fdjac_gravitynav_default_firstpair_cli_delta_20260606.json"
        ),
    ),
    (
        "sm-g988b",
        Path(
            "experiments/results/"
            "native_taroz_imu_state_navstate_rj_sm_g988b_100_iter1_extpreint_fdjac_gravitynav_default_firstpair_cli_delta_20260606.json"
        ),
    ),
)

DEFAULT_THRESHOLDS: tuple[str, ...] = (
    "position_m.max_norm=0.002",
    "rpy_rad.max_norm=1e-6",
    "velocity_mps.max_norm=1e-5",
    "clock_bias_m.max_norm=0.01",
    "clock_drift_mps.max_norm=5e-6",
    "bias_acc_mps2.max_norm=2e-6",
    "bias_gyro_radps.max_norm=1e-8",
)

DEFAULT_OUTPUT_CSV = Path("experiments/results/taroz_imu_state_parity_summary_20260606.csv")
DEFAULT_OUTPUT_JSON = Path("experiments/results/taroz_imu_state_parity_summary_20260606.json")


def default_summary_specs() -> list[str]:
    return [f"{label}={path}" for label, path in DEFAULT_SUMMARIES]


def run_gate(
    *,
    summaries: list[str] | None = None,
    thresholds: list[str] | None = None,
    output_csv: Path | None = DEFAULT_OUTPUT_CSV,
    output_json: Path | None = DEFAULT_OUTPUT_JSON,
) -> dict[str, object]:
    return summarize_parity(
        argparse.Namespace(
            summary=summaries if summaries is not None else default_summary_specs(),
            threshold=thresholds if thresholds is not None else list(DEFAULT_THRESHOLDS),
            output_csv=output_csv,
            output_json=output_json,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        default=None,
        help="summary JSON path, optionally as label=path; defaults to the three fixed smoke devices",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        help="gate threshold as group.metric=value; defaults to the Taroz full smoke thresholds",
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = run_gate(
        summaries=args.summary,
        thresholds=args.threshold,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(
        "Taroz full parity gate: "
        f"passed={payload['passed']} summaries={payload['summary_count']} "
        f"violations={len(payload['threshold_violations'])} output={args.output_json}"
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
