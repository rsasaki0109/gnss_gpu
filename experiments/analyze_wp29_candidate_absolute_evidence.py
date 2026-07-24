"""Score every saved basin candidate with truth-free absolute DD observations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp29_moving_offset_shadow import (  # noqa: E402
    _assignment_integers,
    _huber_cost,
    _lookup_assignment_integer,
    _position,
    _read_csv,
)
from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402


def _ddpr_cost(position: np.ndarray, obs: DDPseudorangeEpoch, sigma_m: float) -> float:
    residuals = []
    for index in range(obs.n):
        expected, _jac = _dd_expected_and_jacobian_m(
            position,
            obs.sat_ecef_k[index],
            obs.sat_ecef_ref[index],
            obs.base_range_k[index],
            obs.base_range_ref[index],
        )
        residuals.append((float(obs.dd_pseudorange_m[index]) - expected) / sigma_m)
    return float(np.mean(_huber_cost(np.asarray(residuals), 1.5)))


def _carrier_cost(
    position: np.ndarray,
    row: dict[str, str],
    obs: DDCarrierEpoch,
    sigma_cycles: float,
) -> tuple[float, int]:
    if obs.sat_ids is None or obs.ref_sat_ids is None:
        return float("nan"), 0
    assignments = _assignment_integers(row)
    residuals = []
    for index, (ref_sat, sat_id) in enumerate(zip(obs.ref_sat_ids, obs.sat_ids)):
        wavelength = float(obs.wavelengths_m[index])
        integer = _lookup_assignment_integer(
            assignments, str(ref_sat), str(sat_id), wavelength
        )
        if integer is None:
            continue
        expected, _jac = _dd_expected_and_jacobian_m(
            position,
            obs.sat_ecef_k[index],
            obs.sat_ecef_ref[index],
            obs.base_range_k[index],
            obs.base_range_ref[index],
        )
        residuals.append(
            (float(obs.dd_carrier_cycles[index]) - expected / wavelength - integer)
            / sigma_cycles
        )
    if not residuals:
        return float("nan"), 0
    return float(np.mean(_huber_cost(np.asarray(residuals), 1.5))), len(residuals)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    basin_rows = [
        row
        for row in _read_csv(args.basin_trace)
        if args.start <= int(row["epoch"]) < args.end
        and int(row["epoch"]) % args.anchor_stride_epochs == 0
    ]
    by_epoch: dict[int, list[dict[str, str]]] = {}
    for row in basin_rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    systems = ("G", "E", "J", "C")
    pseudorange = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    carrier = DDCarrierComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        observation_cache=cache,
    )
    families = tuple(value for value in args.carrier_families.split(",") if value)
    output: list[dict[str, object]] = []
    for epoch in sorted(by_epoch):
        approximate_row = max(by_epoch[epoch], key=lambda row: float(row["log_weight"]))
        approximate = _position(approximate_row)
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch]),
            np.asarray(data["system_ids"][epoch]),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch]),
            approximate,
            systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        tow = float(data["times"][epoch])
        ddpr_result = pseudorange.compute_dd(
            tow, measurements, rover_position_approx=approximate, min_common_sats=4
        )
        carrier_result = carrier.compute_dd_families(
            tow,
            measurements,
            rover_position_approx=approximate,
            min_common_sats=2,
            carrier_families=families,
        )
        ddpr = None if ddpr_result is None else DDPseudorangeEpoch.from_result(ddpr_result)
        cp = None if carrier_result is None else DDCarrierEpoch.from_result(carrier_result)
        if ddpr is None and cp is None:
            continue
        truth = np.asarray(data["ground_truth"][epoch], dtype=np.float64)
        for row in by_epoch[epoch]:
            position = _position(row)
            cp_cost, cp_rows = (
                (float("nan"), 0)
                if cp is None
                else _carrier_cost(position, row, cp, args.carrier_sigma_cycles)
            )
            output.append(
                {
                    "epoch": epoch,
                    "basin_id": row["basin_id"],
                    "ddpr_cost": (
                        float("nan")
                        if ddpr is None
                        else _ddpr_cost(position, ddpr, args.ddpr_sigma_m)
                    ),
                    "ddpr_rows": 0 if ddpr is None else ddpr.n,
                    "carrier_cost": cp_cost,
                    "carrier_rows": cp_rows,
                    "audit_error_m": float(np.linalg.norm(position - truth)),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--basin-trace", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--anchor-stride-epochs", type=int, default=5)
    parser.add_argument("--ddpr-sigma-m", type=float, default=4.0)
    parser.add_argument("--carrier-sigma-cycles", type=float, default=0.5)
    parser.add_argument("--carrier-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(args)
    if not rows:
        raise RuntimeError("no absolute-evidence rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        {
            "candidate_rows": len(rows),
            "evidence_epochs": len({int(row["epoch"]) for row in rows}),
        }
    )


if __name__ == "__main__":
    main()
