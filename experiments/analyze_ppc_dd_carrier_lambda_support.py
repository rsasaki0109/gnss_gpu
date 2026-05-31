#!/usr/bin/env python3
"""Diagnose DD-carrier LAMBDA support on PPC runs.

This is a narrow, non-deploying probe for the post Phase43 outside-ranker path:
before building a carrier-phase anchor candidate, check whether the raw DD
carrier tracks have enough integer support to pass LAMBDA ratio tests when
evaluated around an existing trajectory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exp_ppc_ctrbpf_fgo import _build_dd_measurements, _load_hybrid_pos_file  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.local_fgo import (  # noqa: E402
    DDCarrierEpoch,
    LambdaFixConfig,
    LocalFgoWindow,
    _estimate_lambda_fixes,
)
from ppc_window_geometry import load_ppc_window_geometry  # noqa: E402


RESULTS = _SCRIPT_DIR / "results"
DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _score_pass_rows(
    *,
    ref_path: Path,
    pos_by_tow: dict[float, np.ndarray],
    tows: list[float],
) -> tuple[int, float]:
    if not ref_path.is_file():
        return 0, float("nan")
    from exp_ppc_ctrbpf_fgo import _load_full_reference  # local import keeps CLI fast

    ref = {
        round(float(t), 1): np.asarray(p, dtype=np.float64)
        for t, p in _load_full_reference(ref_path)
    }
    errs: list[float] = []
    for tow in tows:
        key = round(float(tow), 1)
        pos = pos_by_tow.get(key)
        truth = ref.get(key)
        if pos is None or truth is None:
            continue
        errs.append(float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth)))
    if not errs:
        return 0, float("nan")
    vals = np.asarray(errs, dtype=np.float64)
    return int(np.count_nonzero(vals <= 0.5)), float(np.median(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="nagoya/run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--pos",
        type=Path,
        default=(
            RESULTS
            / "libgnss_diag_phase10"
            / "full_ratio15_lock3_trustedseed_rtkout5mlc1c005oG"
            / "nagoya_run2_full.pos"
        ),
        help="Trajectory used to linearize DD carrier ambiguities.",
    )
    parser.add_argument("--start-tow", type=float, default=557046.2)
    parser.add_argument("--end-tow", type=float, default=557051.6)
    parser.add_argument("--max-epochs", type=int, default=12000)
    parser.add_argument("--dd-systems", default="G")
    parser.add_argument("--dd-min-pairs", type=int, default=4)
    parser.add_argument("--dd-min-elevation-deg", type=float, default=-90.0)
    parser.add_argument("--dd-min-snr", type=float, default=0.0)
    parser.add_argument("--dd-keep-best", type=int, default=0)
    parser.add_argument("--dd-base-interp", action="store_true")
    parser.add_argument(
        "--fast-window-loader",
        action="store_true",
        help="Load only the requested TOW window instead of full PPCDatasetLoader data.",
    )
    parser.add_argument(
        "--dd-carrier-families",
        default="",
        help=(
            "Optional comma-separated multi-frequency carrier families for "
            "DDCarrierComputer.compute_dd_families, e.g. L1_E1_B1,L5_E5A_B2A. "
            "Empty keeps the historical single-family compute_dd path."
        ),
    )
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--ratios", default="1.5,2.0,3.0")
    parser.add_argument("--min-epochs", default="2,3,4,6,10")
    parser.add_argument("--max-epoch-gap", type=int, default=6)
    parser.add_argument("--slip-threshold-cycles", type=float, default=1.5)
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=RESULTS / "nr2_dd_carrier_lambda_support_summary.csv",
    )
    parser.add_argument(
        "--out-windows",
        type=Path,
        default=RESULTS / "nr2_dd_carrier_lambda_support_windows.csv",
    )
    args = parser.parse_args()

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    pos_by_tow, _status = _load_hybrid_pos_file(args.pos)
    systems = tuple(s.strip() for s in str(args.dd_systems).split(",") if s.strip())
    if args.fast_window_loader:
        data = load_ppc_window_geometry(
            run_dir,
            start_tow=float(args.start_tow),
            end_tow=float(args.end_tow),
            systems=systems,
        )
    else:
        data = PPCDatasetLoader(run_dir).load_experiment_data(
            max_epochs=int(args.max_epochs),
            include_sat_velocity=False,
            systems=systems,
        )
    times = np.asarray(data["times"], dtype=np.float64)
    sat_ecef = data["sat_ecef"]
    system_ids = data["system_ids"]
    used_prns = data["used_prns"]
    weights = data["weights"]

    dd_computer = DDCarrierComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=tuple(s.strip() for s in str(args.dd_systems).split(",") if s.strip()),
        interpolate_base_epochs=bool(args.dd_base_interp),
    )

    tows: list[float] = []
    positions: list[np.ndarray] = []
    dd_cache: list[DDCarrierEpoch | None] = []
    n_attempt = 0
    n_dd = 0
    for i, tow in enumerate(times):
        tow_r = round(float(tow), 1)
        if tow_r < float(args.start_tow) or tow_r > float(args.end_tow):
            continue
        pos = pos_by_tow.get(tow_r)
        if pos is None:
            continue
        n_attempt += 1
        measurements = _build_dd_measurements(
            np.asarray(sat_ecef[i], dtype=np.float64),
            np.asarray(system_ids[i], dtype=np.int32),
            list(used_prns[i]),
            np.asarray(weights[i], dtype=np.float64),
            np.asarray(pos, dtype=np.float64),
            tuple(s.strip() for s in str(args.dd_systems).split(",") if s.strip()),
            min_elevation_deg=float(args.dd_min_elevation_deg),
            min_snr=float(args.dd_min_snr),
            keep_best=int(args.dd_keep_best),
        )
        dd_epoch = None
        if len(measurements) >= int(args.dd_min_pairs):
            family_names = tuple(
                s.strip() for s in str(args.dd_carrier_families).split(",") if s.strip()
            )
            if family_names:
                result = dd_computer.compute_dd_families(
                    tow_r,
                    measurements,
                    rover_position_approx=np.asarray(pos, dtype=np.float64),
                    min_common_sats=2,
                    carrier_families=family_names,
                )
            else:
                result = dd_computer.compute_dd(
                    tow_r,
                    measurements,
                    rover_position_approx=np.asarray(pos, dtype=np.float64),
                    min_common_sats=int(args.dd_min_pairs),
                )
            if result is not None and int(getattr(result, "n_dd", 0)) > 0:
                dd_epoch = DDCarrierEpoch.from_result(result)
                n_dd += 1
        tows.append(tow_r)
        positions.append(np.asarray(pos, dtype=np.float64))
        dd_cache.append(dd_epoch)

    if not tows:
        raise SystemExit("no trajectory rows matched the requested TOW window")

    pos_arr = np.asarray(positions, dtype=np.float64)
    pass_rows, pos_error_median = _score_pass_rows(
        ref_path=run_dir / "reference.csv",
        pos_by_tow=pos_by_tow,
        tows=tows,
    )

    ratios = _parse_float_list(args.ratios)
    min_epochs_list = _parse_int_list(args.min_epochs)
    win_size = min(max(2, int(args.window_size)), len(tows))
    stride = max(1, int(args.window_stride))

    window_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for ratio in ratios:
        for min_epochs in min_epochs_list:
            total_fixed = 0
            total_fixed_obs = 0
            total_ratio_rejected = 0
            best_ratio = 0.0
            n_windows = 0
            n_windows_with_fix = 0
            for start in range(0, len(tows) - win_size + 1, stride):
                end = start + win_size - 1
                lam_cfg = LambdaFixConfig(
                    ratio_threshold=float(ratio),
                    min_epochs=int(min_epochs),
                    max_epoch_gap=int(args.max_epoch_gap),
                    slip_threshold_cycles=float(args.slip_threshold_cycles),
                )
                fixes, info = _estimate_lambda_fixes(
                    dd_cache[start : end + 1],
                    pos_arr[start : end + 1],
                    LocalFgoWindow(0, win_size - 1),
                    lam_cfg,
                )
                n_fixed = int(info.get("n_fixed", 0))
                n_fixed_obs = int(info.get("n_fixed_observations", 0))
                n_windows += 1
                total_fixed += n_fixed
                total_fixed_obs += n_fixed_obs
                total_ratio_rejected += int(info.get("n_ratio_rejected", 0))
                br = float(info.get("best_ratio", 0.0))
                if np.isinf(br):
                    best_ratio = float("inf")
                elif np.isfinite(br) and not np.isinf(best_ratio):
                    best_ratio = max(best_ratio, br)
                if fixes:
                    n_windows_with_fix += 1
                window_rows.append(
                    {
                        "run": str(args.run),
                        "pos": str(args.pos),
                        "ratio": float(ratio),
                        "min_epochs": int(min_epochs),
                        "start_tow": float(tows[start]),
                        "end_tow": float(tows[end]),
                        "n_dd_epochs": int(
                            sum(1 for d in dd_cache[start : end + 1] if d is not None)
                        ),
                        "n_tracks": int(info.get("n_tracks", 0)),
                        "n_segments": int(info.get("n_segments", 0)),
                        "n_fixed": n_fixed,
                        "n_fixed_observations": n_fixed_obs,
                        "n_ratio_rejected": int(info.get("n_ratio_rejected", 0)),
                        "best_ratio": br,
                        "ratio_median": float(info.get("ratio_median", 0.0)),
                        "ratio_p90": float(info.get("ratio_p90", 0.0)),
                        "segment_n_epochs_median": float(
                            info.get("segment_n_epochs_median", 0.0)
                        ),
                        "segment_n_epochs_max": int(info.get("segment_n_epochs_max", 0)),
                        "segment_abs_frac_median": float(
                            info.get("segment_abs_frac_median", 0.0)
                        ),
                        "segment_abs_frac_p90": float(
                            info.get("segment_abs_frac_p90", 0.0)
                        ),
                    }
                )
            summary_rows.append(
                {
                    "run": str(args.run),
                    "pos": str(args.pos),
                    "start_tow": float(args.start_tow),
                    "end_tow": float(args.end_tow),
                    "n_rows": int(len(tows)),
                    "n_attempt": int(n_attempt),
                    "n_dd_epochs": int(n_dd),
                    "pass_rows": int(pass_rows),
                    "pos_error_median_m": float(pos_error_median),
                    "window_size": int(win_size),
                    "window_stride": int(stride),
                    "ratio": float(ratio),
                    "min_epochs": int(min_epochs),
                    "n_windows": int(n_windows),
                    "n_windows_with_fix": int(n_windows_with_fix),
                    "n_fixed_total": int(total_fixed),
                    "n_fixed_observations_total": int(total_fixed_obs),
                    "n_ratio_rejected_total": int(total_ratio_rejected),
                    "best_ratio": best_ratio,
                }
            )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.out_summary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with args.out_windows.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(window_rows)

    print(
        f"rows={len(tows)} dd_epochs={n_dd}/{n_attempt} pass_rows={pass_rows} "
        f"pos_err_med={pos_error_median:.3f}m"
    )
    top = sorted(
        summary_rows,
        key=lambda r: (
            int(r["n_windows_with_fix"]),
            int(r["n_fixed_observations_total"]),
            float("inf") if np.isinf(float(r["best_ratio"])) else float(r["best_ratio"]),
        ),
        reverse=True,
    )[:10]
    for row in top:
        print(
            "ratio={ratio:.1f} min_epochs={min_epochs} windows_with_fix={n_windows_with_fix} "
            "fixed_obs={n_fixed_observations_total} best_ratio={best_ratio}".format(**row)
        )
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_windows}")


if __name__ == "__main__":
    main()
