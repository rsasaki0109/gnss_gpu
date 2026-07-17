#!/usr/bin/env python3
"""Audit the FGO-free WP23b DD float-KF seed on PPC Tokyo.

Truth is used only after filtering to score position and top-K candidate
coverage.  LAMBDA problems are accumulated and submitted in one genuine GPU
batch, avoiding the known batch-of-one slow path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
for _path in (_ROOT / "python", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _GPS_L1_WAVELENGTH_M,
    _build_dd_measurements,
    _doppler_centered_wls_velocity,
    _filter_data_by_systems,
    _load_full_reference,
    _reference_position_map,
)
from exp_urbannav_baseline import run_wls  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_float_kf import DDFloatKalmanFilter  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.doppler_signals import (  # noqa: E402
    doppler_wavelengths_m,
    normalize_constellation_clock_drifts,
)
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.lambda_batch import (  # noqa: E402
    HAS_LAMBDA_BATCH,
    lambda_batch_max_n,
    mlambda_batch,
)


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _doppler_velocity(data: dict, i: int, position: np.ndarray) -> tuple[np.ndarray | None, float]:
    sat = np.asarray(data["sat_ecef"][i], dtype=np.float64)
    sat_vel = np.asarray(data["sat_velocity"][i], dtype=np.float64)
    doppler = np.asarray(data["doppler_hz"][i], dtype=np.float64)
    weights = np.asarray(data["weights"][i], dtype=np.float64)
    system_ids = np.asarray(data["system_ids"][i], dtype=np.int32)
    wavelengths = np.full(doppler.shape, _GPS_L1_WAVELENGTH_M, dtype=np.float64)
    if data.get("doppler_codes") is not None:
        wavelengths = doppler_wavelengths_m(data["used_prns"][i], data["doppler_codes"][i])
    valid = (
        np.isfinite(doppler)
        & np.all(np.isfinite(sat), axis=1)
        & np.all(np.isfinite(sat_vel), axis=1)
        & np.isfinite(weights)
        & np.isfinite(wavelengths)
        & (wavelengths > 0.0)
    )
    if int(np.count_nonzero(valid)) < 4:
        return None, float("nan")
    try:
        normalized, fit = normalize_constellation_clock_drifts(
            sat[valid],
            sat_vel[valid],
            doppler[valid],
            wavelengths[valid],
            np.asarray(position, dtype=np.float64),
            system_ids[valid],
            weights=weights[valid],
        )
        velocity, rms = _doppler_centered_wls_velocity(
            sat[valid],
            sat_vel[valid],
            normalized,
            weights[valid],
            position,
            doppler_sign=-1.0,
            wavelength_m=_GPS_L1_WAVELENGTH_M,
        )
        if velocity is None or not np.all(np.isfinite(velocity)):
            return None, float("nan")
        if float(np.linalg.norm(velocity)) > 60.0:
            return None, float(rms)
        return velocity, max(float(rms), float(fit.residual_rms_mps))
    except (ValueError, np.linalg.LinAlgError):
        return None, float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="tokyo/run2")
    parser.add_argument("--max-epochs", type=int, default=1200)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/PPC-Dataset-data"))
    parser.add_argument("--dd-systems", default="G,E,J,C")
    parser.add_argument("--pr-systems", default="G,E,J")
    parser.add_argument("--sigma-dd-pr-m", type=float, default=5.0)
    parser.add_argument("--sigma-dd-cp-cycles", type=float, default=0.10)
    parser.add_argument("--subset-sizes", default="4,6,8,10,12,16")
    parser.add_argument("--out-epochs", type=Path, default=Path("results/wp23b/csv/float_seed_run2_epochs.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/wp23b/csv/float_seed_run2_summary.json"))
    args = parser.parse_args()

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    dd_systems = tuple(x.strip() for x in str(args.dd_systems).split(",") if x.strip())
    pr_systems = tuple(x.strip() for x in str(args.pr_systems).split(",") if x.strip())
    subset_sizes = tuple(
        sorted({int(x.strip()) for x in str(args.subset_sizes).split(",") if x.strip()})
    )
    data = PPCDatasetLoader(run_dir).load_experiment_data(
        max_epochs=int(args.max_epochs),
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    wls_positions, _ = run_wls(_filter_data_by_systems(data, pr_systems))
    truth = _reference_position_map(_load_full_reference(run_dir / "reference.csv"))

    carrier = DDCarrierComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=dd_systems,
    )
    pseudorange = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=dd_systems,
    )

    kf = DDFloatKalmanFilter(
        np.asarray(wls_positions[0, :3], dtype=np.float64),
        position_sigma_m=50.0,
        velocity_sigma_mps=10.0,
        accel_process_sigma_mps2=3.0,
        ambiguity_init_sigma_cycles=40.0,
        max_track_age_epochs=10,
    )
    times = np.asarray(data["times"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    lambda_problems: list[dict[str, object]] = []
    covariance_failures = 0

    for i, tow in enumerate(times):
        if i > 0:
            kf.predict(max(float(times[i] - times[i - 1]), 1.0e-3))
        velocity, doppler_rms = _doppler_velocity(data, i, kf.position_ecef)
        if velocity is not None:
            kf.update_velocity(velocity, sigma_mps=max(0.5, min(float(doppler_rms), 5.0)))

        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["weights"][i], dtype=np.float64),
            kf.position_ecef,
            dd_systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd_pr = pseudorange.compute_dd(
            float(tow), measurements, rover_position_approx=kf.position_ecef, min_common_sats=4
        )
        dd_cp = carrier.compute_dd(
            float(tow), measurements, rover_position_approx=kf.position_ecef, min_common_sats=4
        )
        pr_diag = None
        cp_diag = None
        if dd_pr is not None and int(dd_pr.n_dd) >= 3:
            pr_diag = kf.update_pseudorange(
                dd_pr, sigma_pr_m=float(args.sigma_dd_pr_m), huber_k_sigma=2.5
            )
        if dd_cp is not None and int(dd_cp.n_dd) >= 3:
            cp_diag = kf.update_carrier(
                dd_cp,
                dd_pseudorange_result=dd_pr,
                sigma_cp_cycles=float(args.sigma_dd_cp_cycles),
                huber_k_sigma=3.0,
            )

        tow_key = round(float(tow), 1)
        ref = truth.get(tow_key)
        error = (
            float(np.linalg.norm(kf.position_ecef - ref))
            if ref is not None and np.all(np.isfinite(ref))
            else float("nan")
        )
        min_eig = float(np.min(np.linalg.eigvalsh(kf.covariance)))
        if not np.isfinite(min_eig) or min_eig <= 0.0:
            covariance_failures += 1
        row: dict[str, object] = {
            "epoch": i,
            "tow": float(tow),
            "position_error_m": error,
            "n_dd_pr": 0 if dd_pr is None else int(dd_pr.n_dd),
            "n_dd_cp": 0 if dd_cp is None else int(dd_cp.n_dd),
            "n_float_ambiguities": len(kf.ambiguity_seed().keys),
            "doppler_rms_mps": float(doppler_rms),
            "dd_pr_nis": float("nan") if pr_diag is None else pr_diag.normalized_innovation_sq,
            "dd_cp_nis": float("nan") if cp_diag is None else cp_diag.normalized_innovation_sq,
            "ambiguities_reset": 0 if cp_diag is None else int(cp_diag.ambiguities_reset),
            "covariance_min_eig": min_eig,
            "lambda_n": 0,
            "lambda_best_ratio": float("nan"),
            "top12_min_error_m": float("nan"),
            "top16_min_error_m": float("nan"),
            "top12_has_sub50cm": 0,
            "top16_has_sub50cm": 0,
        }
        for subset_size in subset_sizes:
            row[f"subset{subset_size}_top16_min_error_m"] = float("nan")
            row[f"subset{subset_size}_top16_has_sub50cm"] = 0
        rows.append(row)

        if dd_cp is not None and ref is not None and HAS_LAMBDA_BATCH:
            current_keys = tuple(key for key in kf.ambiguity_seed().keys if key[0:2] in {
                (str(r), str(s)) for r, s in zip(dd_cp.ref_sat_ids, dd_cp.sat_ids)
            })
            seed = kf.ambiguity_seed(current_keys)
            if 3 <= len(seed.keys) <= int(lambda_batch_max_n()):
                base_problem = {
                    "row_index": len(rows) - 1,
                    "position": kf.position_ecef,
                    "truth": np.asarray(ref, dtype=np.float64),
                }
                lambda_problems.append(
                    {
                        **base_problem,
                        "label": "full",
                        "ahat": seed.ahat_cycles,
                        "qahat": seed.qahat_cycles2,
                        "cross": seed.position_ambiguity_cov,
                    }
                )
                variance_order = np.argsort(np.diag(seed.qahat_cycles2))
                for subset_size in subset_sizes:
                    if subset_size >= len(seed.keys):
                        continue
                    idx = np.sort(variance_order[:subset_size])
                    lambda_problems.append(
                        {
                            **base_problem,
                            "label": f"subset{subset_size}",
                            "ahat": seed.ahat_cycles[idx],
                            "qahat": seed.qahat_cycles2[np.ix_(idx, idx)],
                            "cross": seed.position_ambiguity_cov[:, idx],
                        }
                    )

    if lambda_problems:
        batch_results = mlambda_batch(
            [p["ahat"] for p in lambda_problems],
            [p["qahat"] for p in lambda_problems],
            ncands=16,
            parmode=1,
        )
        for problem, result in zip(lambda_problems, batch_results):
            row = rows[int(problem["row_index"])]
            if int(result.status) != 0 or result.s.size < 16:
                continue
            ahat = np.asarray(problem["ahat"], dtype=np.float64)
            qahat = np.asarray(problem["qahat"], dtype=np.float64)
            cross = np.asarray(problem["cross"], dtype=np.float64)
            gain = np.linalg.solve(qahat, cross.T).T
            candidates = np.asarray(result.afix, dtype=np.float64)
            positions = np.asarray(problem["position"], dtype=np.float64)[:, None] + gain @ (
                candidates - ahat[:, None]
            )
            errors = np.linalg.norm(positions.T - np.asarray(problem["truth"]), axis=1)
            ratio = float(result.s[1] / result.s[0]) if result.s[0] > 0.0 else float("inf")
            label = str(problem["label"])
            if label == "full":
                row["lambda_n"] = int(ahat.size)
                row["lambda_best_ratio"] = ratio
                row["top12_min_error_m"] = float(np.min(errors[:12]))
                row["top16_min_error_m"] = float(np.min(errors[:16]))
                row["top12_has_sub50cm"] = int(np.min(errors[:12]) < 0.5)
                row["top16_has_sub50cm"] = int(np.min(errors[:16]) < 0.5)
            else:
                row[f"{label}_top16_min_error_m"] = float(np.min(errors[:16]))
                row[f"{label}_top16_has_sub50cm"] = int(np.min(errors[:16]) < 0.5)

    errors = [float(row["position_error_m"]) for row in rows]
    lambda_rows = [row for row in rows if int(row["lambda_n"]) > 0]
    summary = {
        "run": str(args.run),
        "n_epochs": len(rows),
        "covariance_spd_failures": int(covariance_failures),
        "position_error_m": _finite_stats(errors),
        "dd_pr_nis": _finite_stats([float(row["dd_pr_nis"]) for row in rows]),
        "dd_cp_nis": _finite_stats([float(row["dd_cp_nis"]) for row in rows]),
        "ambiguities_reset_total": int(sum(int(row["ambiguities_reset"]) for row in rows)),
        "lambda_epochs": len(lambda_rows),
        "top12_sub50cm_epochs": int(sum(int(row["top12_has_sub50cm"]) for row in lambda_rows)),
        "top16_sub50cm_epochs": int(sum(int(row["top16_has_sub50cm"]) for row in lambda_rows)),
        "top12_sub50cm_pct": (
            100.0 * sum(int(row["top12_has_sub50cm"]) for row in lambda_rows) / len(lambda_rows)
            if lambda_rows else 0.0
        ),
        "top16_sub50cm_pct": (
            100.0 * sum(int(row["top16_has_sub50cm"]) for row in lambda_rows) / len(lambda_rows)
            if lambda_rows else 0.0
        ),
        "top12_min_error_m": _finite_stats([float(row["top12_min_error_m"]) for row in lambda_rows]),
        "top16_min_error_m": _finite_stats([float(row["top16_min_error_m"]) for row in lambda_rows]),
    }
    summary["partial_ar"] = {
        str(subset_size): {
            "sub50cm_epochs": int(
                sum(int(row[f"subset{subset_size}_top16_has_sub50cm"]) for row in lambda_rows)
            ),
            "sub50cm_pct": (
                100.0
                * sum(int(row[f"subset{subset_size}_top16_has_sub50cm"]) for row in lambda_rows)
                / len(lambda_rows)
                if lambda_rows else 0.0
            ),
            "min_error_m": _finite_stats(
                [float(row[f"subset{subset_size}_top16_min_error_m"]) for row in lambda_rows]
            ),
        }
        for subset_size in subset_sizes
    }
    args.out_epochs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_epochs.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"wrote {args.out_epochs}")
    print(f"wrote {args.out_summary}")


if __name__ == "__main__":
    main()
