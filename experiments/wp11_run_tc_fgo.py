#!/usr/bin/env python3
"""WP11 driver: tightly-coupled GNSS+IMU float FGO on PPC Tokyo runs.

Two-phase initialization (static RTK FIX + IMU alignment, then heading from
velocity), sliding fixed-lag window solver, and IMU propagation through
GNSS-missing epochs for 100% rover-epoch coverage.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _PROJECT_ROOT, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evaluate import ecef_to_lla  # noqa: E402
from exp_ppc_ctrbpf_fgo import _build_dd_measurements  # noqa: E402
from gsdc2023_imu import imu_preintegration_segment_with_bias_jacobians  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.ins_ekf import INSEKF, INSConfig  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from gnss_gpu.tc_fgo import (  # noqa: E402
    PhaseInitConfig,
    TcFgoConfig,
    TcFgoEpochObs,
    TcFgoNavState,
    TcFgoWindowProblem,
    collapse_imu_preintegration_segment,
    collect_static_imu_samples,
    ecef_to_enu,
    enu_to_ecef,
    is_static_epoch,
    naive_marginalization_prior,
    propagate_nav_state_with_imu,
    run_two_phase_initialization,
    solve_tc_fgo_window,
)
from ppc_imu_adapter import build_ppc_imu_preintegration  # noqa: E402
from ppc_window_geometry import load_ppc_window_geometry  # noqa: E402
from wp4_run_local_fgo_full import parse_rover_tows_from_obs  # noqa: E402
from wp5_run_anchored_fgo import load_rtk_pos_with_status  # noqa: E402

DEFAULT_BASELINE_POS = _PROJECT_ROOT / "results/wp10/sweep/run1/a0_baseline_no_wp10.pos"
DATA_ROOT_CANDIDATES = (
    Path("datasets/PPC-Dataset-data"),
    Path("E:/datasets/PPC-Dataset-data"),
)


def resolve_data_root() -> Path:
    for root in DATA_ROOT_CANDIDATES:
        if (root / "tokyo" / "run1" / "rover.obs").exists():
            return root
    raise FileNotFoundError("PPC dataset not found under datasets/ or E:/datasets/")


def resolve_run_dir(data_root: Path, run_spec: str) -> Path:
    run_dir = data_root / run_spec
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    return run_dir


def write_tc_pos_file(
    path: Path,
    tows: np.ndarray,
    positions_ecef: np.ndarray,
    *,
    status: int = 5,
) -> None:
    """RTKLIB-like .pos with lat/lon filled and Q=5 (float) for WP11."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("% WP11 TC-FGO float trajectory\n")
        fh.write(
            "%  GPST_week   tow(s)      x-ecef(m)        y-ecef(m)        z-ecef(m)"
            "   lat(deg)   lon(deg)  height(m)   Q  ns   sdx    sdy    sdz   age  ratio\n"
        )
        for tow, pos in zip(tows, positions_ecef, strict=True):
            lat, lon, height = ecef_to_lla(float(pos[0]), float(pos[1]), float(pos[2]))
            fh.write(
                f"2324 {float(tow):14.4f} "
                f"{pos[0]:16.4f} {pos[1]:16.4f} {pos[2]:16.4f}  "
                f"{lat:10.6f} {lon:11.6f} {height:8.3f} {int(status)}   0  "
                f"0.000  0.000  0.000  0.00  0.0\n"
            )


def collect_all_rtk_fixes(
    tows: np.ndarray,
    rtk_by_tow: dict[float, tuple[np.ndarray, int]],
    *,
    fix_status: int = 4,
) -> list[tuple[float, np.ndarray]]:
    fixes: list[tuple[float, np.ndarray]] = []
    for tow in np.asarray(tows, dtype=np.float64):
        hit = rtk_by_tow.get(round(float(tow), 1))
        if hit is None:
            continue
        ecef, status = hit
        if int(status) == int(fix_status):
            fixes.append((float(tow), np.asarray(ecef, dtype=np.float64)))
    return fixes


def collect_rtk_fixes_while_static(
    tows: np.ndarray,
    rtk_by_tow: dict[float, tuple[np.ndarray, int]],
    *,
    speed_thresh_mps: float = 1.0,
    fix_status: int = 4,
) -> list[tuple[float, np.ndarray]]:
    fixes: list[tuple[float, np.ndarray]] = []
    prev_pos: np.ndarray | None = None
    prev_tow: float | None = None
    for tow in np.asarray(tows, dtype=np.float64):
        hit = rtk_by_tow.get(round(float(tow), 1))
        if hit is None:
            prev_pos = None
            prev_tow = None
            continue
        ecef, status = hit
        if int(status) != int(fix_status):
            continue
        speed = 0.0
        if prev_pos is not None and prev_tow is not None:
            dt = float(tow - prev_tow)
            if dt > 0.0:
                speed = float(np.linalg.norm(ecef - prev_pos) / dt)
        if speed <= float(speed_thresh_mps):
            fixes.append((float(tow), np.asarray(ecef, dtype=np.float64)))
        prev_pos = ecef
        prev_tow = float(tow)
    return fixes


def _dd_measurement_kwargs() -> dict[str, float | int]:
    return {
        "min_elevation_deg": 15.0,
        "min_snr": 25.0,
        "keep_best": 12,
    }


def build_dd_measurements_for_epoch(
    data: dict,
    epoch_index: int,
    rover_pos_ecef: np.ndarray,
    systems: tuple[str, ...],
    *,
    min_elevation_deg: float = 15.0,
    min_snr: float = 25.0,
    keep_best: int = 12,
) -> list:
    """Elevation/SNR-gated measurement list for one epoch at ``rover_pos_ecef``."""

    return _build_dd_measurements(
        np.asarray(data["sat_ecef"][epoch_index], dtype=np.float64),
        np.asarray(data["system_ids"][epoch_index], dtype=np.int32),
        list(data["used_prns"][epoch_index]),
        np.asarray(data["weights"][epoch_index], dtype=np.float64),
        np.asarray(rover_pos_ecef, dtype=np.float64),
        systems,
        min_elevation_deg=float(min_elevation_deg),
        min_snr=float(min_snr),
        keep_best=int(keep_best),
    )


def build_dd_pr_epoch_at_index(
    dd_pr_computer: DDPseudorangeComputer,
    data: dict,
    epoch_index: int,
    rover_pos_ecef: np.ndarray,
    systems: tuple[str, ...],
    *,
    min_common_sats: int = 2,
    min_elevation_deg: float = 15.0,
    min_snr: float = 25.0,
    keep_best: int = 12,
) -> DDPseudorangeEpoch | None:
    """Build one DD pseudorange epoch from the current rover position (WP12b)."""

    tow = float(np.asarray(data["times"], dtype=np.float64)[epoch_index])
    pos = np.asarray(rover_pos_ecef, dtype=np.float64)
    measurements = build_dd_measurements_for_epoch(
        data,
        epoch_index,
        pos,
        systems,
        min_elevation_deg=min_elevation_deg,
        min_snr=min_snr,
        keep_best=keep_best,
    )
    if len(measurements) < int(min_common_sats):
        return None
    result = dd_pr_computer.compute_dd(
        round(tow, 1),
        measurements,
        rover_position_approx=pos,
        min_common_sats=int(min_common_sats),
        rover_weights=np.asarray(data["weights"][epoch_index], dtype=np.float64),
    )
    if result is not None and int(getattr(result, "n_dd", 0)) > 0:
        return DDPseudorangeEpoch.from_result(result)
    return None


def build_dd_carrier_epoch_at_index(
    dd_cp_computer: DDCarrierComputer,
    data: dict,
    epoch_index: int,
    rover_pos_ecef: np.ndarray,
    systems: tuple[str, ...],
    *,
    min_common_sats: int = 2,
    min_elevation_deg: float = 15.0,
    min_snr: float = 25.0,
    keep_best: int = 12,
) -> DDCarrierEpoch | None:
    """Build one DD carrier epoch from the current rover position (WP12b)."""

    tow = float(np.asarray(data["times"], dtype=np.float64)[epoch_index])
    pos = np.asarray(rover_pos_ecef, dtype=np.float64)
    measurements = build_dd_measurements_for_epoch(
        data,
        epoch_index,
        pos,
        systems,
        min_elevation_deg=min_elevation_deg,
        min_snr=min_snr,
        keep_best=keep_best,
    )
    if len(measurements) < int(min_common_sats):
        return None
    result = dd_cp_computer.compute_dd(
        round(tow, 1),
        measurements,
        rover_position_approx=pos,
        min_common_sats=int(min_common_sats),
    )
    if result is not None and int(getattr(result, "n_dd", 0)) > 0:
        return DDCarrierEpoch.from_result(result)
    return None


def make_dd_computers(
    run_dir: Path,
    data: dict,
    systems: tuple[str, ...],
) -> tuple[DDPseudorangeComputer, DDCarrierComputer]:
    """Shared DD pseudorange + carrier computers for a PPC run."""

    common = dict(
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=systems,
        interpolate_base_epochs=True,
    )
    return (
        DDPseudorangeComputer(run_dir / "base.obs", **common),
        DDCarrierComputer(run_dir / "base.obs", **common),
    )


def build_dd_pr_epochs(
    run_dir: Path,
    data: dict,
    seed_positions: np.ndarray,
    systems: tuple[str, ...],
) -> list[DDPseudorangeEpoch | None]:
    dd_pr_computer, _ = make_dd_computers(run_dir, data, systems)
    kwargs = _dd_measurement_kwargs()
    epochs: list[DDPseudorangeEpoch | None] = []
    for i in range(len(np.asarray(data["times"], dtype=np.float64))):
        pos = np.asarray(seed_positions[i], dtype=np.float64)
        epochs.append(
            build_dd_pr_epoch_at_index(dd_pr_computer, data, i, pos, systems, **kwargs)
        )
    return epochs


def imu_rows_between(
    imu_times_s: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro_dps: np.ndarray,
    t0: float,
    t1: float,
) -> np.ndarray:
    mask = (imu_times_s >= float(t0)) & (imu_times_s <= float(t1))
    if not mask.any():
        return np.zeros((0, 7), dtype=np.float64)
    return np.column_stack(
        [
            imu_times_s[mask],
            imu_acc[mask],
            imu_gyro_dps[mask],
        ]
    )


def run_tc_fgo_sequence(
    *,
    tows: np.ndarray,
    data: dict,
    dd_epochs: list[DDPseudorangeEpoch | None],
    imu_preint,
    imu_times_s: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro_dps: np.ndarray,
    init_state: TcFgoNavState,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    config: TcFgoConfig,
    phase2_idx: int,
) -> tuple[np.ndarray, list[TcFgoNavState]]:
    n = int(tows.size)
    window = max(1, int(config.window_epochs))
    output_ecef = np.zeros((n, 3), dtype=np.float64)
    epoch_states: list[TcFgoNavState] = []
    marginal_prior: TcFgoNavState | None = None
    marginal_sigmas: np.ndarray | None = None

    for i in range(n):
        if i == 0:
            epoch_states.append(init_state.copy())
        else:
            imu_rows = imu_rows_between(
                imu_times_s,
                imu_acc,
                imu_gyro_dps,
                float(tows[i - 1]),
                float(tows[i]),
            )
            epoch_states.append(propagate_nav_state_with_imu(epoch_states[-1], imu_rows))

        start = max(0, len(epoch_states) - window)
        win_states = [s.copy() for s in epoch_states[start:]]
        imu_segments: list = []
        for j in range(len(win_states) - 1):
            g0 = start + j
            seg_raw = imu_preintegration_segment_with_bias_jacobians(imu_preint, g0, g0 + 2)
            imu_segments.append(
                collapse_imu_preintegration_segment(
                    seg_raw[0],
                    seg_raw[1],
                    seg_raw[2],
                    seg_raw[3],
                    seg_raw[4],
                    seg_raw[5],
                    seg_raw[6],
                    seg_raw[7],
                )
            )

        observations: list[TcFgoEpochObs] = []
        for j in range(len(win_states)):
            gi = start + j
            speed = float(np.linalg.norm(win_states[j].v_enu[0:2]))
            static = is_static_epoch(speed) and gi < phase2_idx
            dd = dd_epochs[gi] if gi < len(dd_epochs) else None
            observations.append(
                TcFgoEpochObs(
                    dd_pseudorange=dd,
                    enable_nhc=(gi >= phase2_idx and not static),
                    enable_zupt=static,
                )
            )

        problem = TcFgoWindowProblem(
            initial_states=win_states,
            imu_segments=imu_segments,
            observations=observations,
            origin_ecef=origin_ecef,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            marginal_prior=marginal_prior,
            marginal_prior_sigmas=marginal_sigmas,
        )
        if len(win_states) >= 2:
            result = solve_tc_fgo_window(problem, config=config)
            epoch_states[-1] = result.states[-1].copy()
            if len(result.states) >= 2 and i >= window - 1:
                marginal_prior, marginal_sigmas = naive_marginalization_prior(result.states[1], config)

        output_ecef[i] = enu_to_ecef(epoch_states[-1].p_enu, origin_ecef, origin_lat, origin_lon)

    return output_ecef, epoch_states


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WP11 TC-FGO float runner")
    parser.add_argument("--run", default="tokyo/run1", help="PPC run path under dataset root")
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit epochs (0 = all)")
    parser.add_argument("--export-pos", type=Path, required=True, help="Output RTKLIB-like .pos path")
    parser.add_argument("--baseline-pos", type=Path, default=DEFAULT_BASELINE_POS)
    parser.add_argument("--systems", default="G", help="Comma-separated GNSS systems")
    parser.add_argument("--window-epochs", type=int, default=5, help="Sliding window size (~1 s at 5 Hz)")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root is not None else resolve_data_root()
    run_dir = resolve_run_dir(data_root, str(args.run))
    systems = tuple(s.strip() for s in str(args.systems).split(",") if s.strip())

    all_tows = parse_rover_tows_from_obs(run_dir / "rover.obs")
    if int(args.max_epochs) > 0:
        tows = all_tows[: int(args.max_epochs)]
    else:
        tows = all_tows
    if tows.size == 0:
        raise ValueError("no rover epochs")

    t0 = time.perf_counter()
    data = load_ppc_window_geometry(
        run_dir,
        start_tow=float(tows[0]),
        end_tow=float(tows[-1]),
        systems=systems,
    )
    if len(data["times"]) != tows.size:
        # Align to requested TOW list length when geometry loader clipped differently.
        n = min(len(data["times"]), tows.size)
        tows = tows[:n]
        for key in ("times", "sat_ecef", "weights", "system_ids", "used_prns", "truth"):
            if key in data and hasattr(data[key], "__len__"):
                data[key] = data[key][:n]

    rtk_by_tow = load_rtk_pos_with_status(Path(args.baseline_pos))
    static_fixes = collect_rtk_fixes_while_static(tows, rtk_by_tow)
    all_fixes = collect_all_rtk_fixes(tows, rtk_by_tow)
    if len(static_fixes) < 5:
        raise ValueError(f"insufficient static RTK FIX epochs for phase-1 init: {len(static_fixes)}")
    if len(all_fixes) < 5:
        raise ValueError(f"insufficient RTK FIX epochs for phase-2 heading: {len(all_fixes)}")

    loader = PPCDatasetLoader(run_dir)
    imu_data = loader.load_imu()
    imu_times_s = np.asarray(imu_data["time"], dtype=np.float64)
    imu_acc = np.column_stack(
        [imu_data["acc_x"], imu_data["acc_y"], imu_data["acc_z"]],
    )
    imu_gyro_dps = np.column_stack(
        [imu_data["gyro_x"], imu_data["gyro_y"], imu_data["gyro_z"]],
    )

    origin_ecef = np.asarray(data["base_ecef"], dtype=np.float64)
    origin_lat, origin_lon, _ = ecef_to_lla(float(origin_ecef[0]), float(origin_ecef[1]), float(origin_ecef[2]))

    seed_positions = np.vstack(
        [
            rtk_by_tow.get(round(float(t), 1), (origin_ecef, 0))[0]
            for t in tows
        ]
    )
    for i, tow in enumerate(tows):
        hit = rtk_by_tow.get(round(float(tow), 1))
        if hit is not None and int(hit[1]) != 0:
            seed_positions[i] = hit[0]

    imu_preint = build_ppc_imu_preintegration(
        imu_data,
        np.asarray(data["times"], dtype=np.float64),
        seed_positions,
        delta_frame="body",
    )

    static_imu = collect_static_imu_samples(
        imu_times_s,
        imu_acc,
        imu_gyro_dps,
        float(static_fixes[0][0]),
        float(static_fixes[min(len(static_fixes) - 1, 4)][0]),
    )
    ins = INSEKF(INSConfig())
    init_state, phase2_idx = run_two_phase_initialization(
        ins,
        epoch_times_s=np.asarray(data["times"], dtype=np.float64),
        rtk_fix_positions_ecef=all_fixes,
        static_fix_positions_ecef=static_fixes,
        imu_samples_static=static_imu,
        origin_ecef=origin_ecef,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )

    dd_epochs = build_dd_pr_epochs(run_dir, data, seed_positions, systems)
    config = TcFgoConfig(window_epochs=int(args.window_epochs))

    positions_ecef, _states = run_tc_fgo_sequence(
        tows=tows,
        data=data,
        dd_epochs=dd_epochs,
        imu_preint=imu_preint,
        imu_times_s=imu_times_s,
        imu_acc=imu_acc,
        imu_gyro_dps=imu_gyro_dps,
        init_state=init_state,
        origin_ecef=origin_ecef,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        config=config,
        phase2_idx=phase2_idx,
    )

    write_tc_pos_file(Path(args.export_pos), tows, positions_ecef, status=5)
    elapsed = time.perf_counter() - t0
    print(
        f"WP11 TC-FGO: {tows.size} epochs -> {args.export_pos} "
        f"(phase2@{phase2_idx}, static_fixes={len(static_fixes)}, all_fixes={len(all_fixes)}, {elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
