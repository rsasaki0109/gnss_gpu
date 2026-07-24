#!/usr/bin/env python3
# ruff: noqa: E402
"""WP21/WP21b PF ablation on PPC: CV vs heuristic-IMU-guide vs preint-IMU-guide.

Arms (see internal_docs/task_wp21_imu_preint.md, internal_docs/task_wp21b_preint_payoff.md):
  (a) cv         -- current CV predict, no IMU (velocity=None).
  (b) heuristic  -- current velocity-guide IMU: gnss_gpu.imu.IMUPredictor
                    (open-loop accel+gyro dead-reckoning, no wheel data on
                    PPC; WP21b fixed the pre-existing gravity-sign bug --
                    see gnss_gpu.imu.IMUPredictor's docstring), same
                    sigma_pos as (a) for a fair "guide-only" test.
  (c) preint_v1  -- WP21 Phase A gnss_gpu.pf_imu_preint_adapter.ImuPreintPfGuide:
                    preintegrates the 100 Hz IMU between GNSS epochs and
                    feeds pf.predict() a scalar velocity guide + sigma_pos
                    derived from the accel/gyro-only preintegration
                    covariance (rbpf_velocity_kf=False). Kept byte-identical
                    to the WP21 Phase A code path for continuity.
  (d) preint_v2  -- WP21b: same preintegration, but (1) sigma_pos also folds
                    in heading uncertainty via the cross-track lever
                    |displacement|*sigma_heading, and (2) the segment's
                    delta_v covariance feeds the per-particle velocity KF
                    (pf.set_velocity_covariance + predict(rbpf_velocity_kf=True))
                    instead of a one-shot scalar guide.

All arms use the identical GNSS pseudorange stream, particle count, seed,
and scoring (experiments/score_vs_inuex35.py), so any difference is
attributable to the predict-step guide alone.

Heading source (arms b, c, d): per-epoch robust_spp point fixes are used as
the causal (no ground truth) "GNSS bearing" reference; arms (c)/(d)
additionally feed them to a gnss_gpu.imu.ComplementaryHeadingFilter shared
with the preintegration adapter, per the WP21 spec ("use the existing
INSEKF or ComplementaryHeadingFilter outside the particle state").
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu import ParticleFilterDevice  # noqa: E402
from gnss_gpu.imu import ComplementaryHeadingFilter, IMUPredictor  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.pf_imu_preint_adapter import (  # noqa: E402
    ImuPreintPfGuide,
    ecef_to_enu_rotation,
    ecef_to_lla_rad,
    imu_preint_predict,
    imu_preint_predict_velocity_kf,
)
from gnss_gpu.robust_spp import robust_spp  # noqa: E402

from score_vs_inuex35 import (  # noqa: E402
    ScoreResult,
    TrajectoryEpoch,
    load_reference_grid,
    score_trajectory,
)

DEG2RAD = math.pi / 180.0
_DATA_ROOT = Path("datasets/PPC-Dataset-data")
RESULTS_DIR = _SCRIPT_DIR / "results" / "wp21"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ppc_imu_for_predictor(run_dir: Path) -> dict[str, np.ndarray]:
    """Load PPC imu.csv into the gnss_gpu.imu-compatible dict.

    imu.py's helpers expect {tow, accel(N,3), gyro(N,3) [rad/s], wheel_vel}.
    PPC gyro columns are deg/s (converted here); PPC has no wheel channel.
    """

    raw = PPCDatasetLoader(run_dir).load_imu()
    n = raw["time"].size
    return {
        "tow": np.asarray(raw["time"], dtype=np.float64),
        "accel": np.column_stack([raw["acc_x"], raw["acc_y"], raw["acc_z"]]).astype(np.float64),
        "gyro": (
            np.column_stack([raw["gyro_x"], raw["gyro_y"], raw["gyro_z"]]).astype(np.float64)
            * DEG2RAD
        ),
        "wheel_vel": np.full(n, np.nan, dtype=np.float64),
    }


def compute_epoch_spp_fixes(data: dict) -> np.ndarray:
    """Causal (no ground truth) per-epoch robust_spp position fixes, shape (n_epochs, 3)."""

    n = data["n_epochs"]
    fixes = np.full((n, 3), np.nan, dtype=np.float64)
    # robust_spp's IRLS needs a reasonable seed (satellite-cloud centroid is
    # ~20000 km off and does not converge); origin_ecef is a coarse
    # dataset-provided receiver-vicinity estimate, not ground truth.
    init = np.asarray(data["origin_ecef"], dtype=np.float64).copy()
    for i in range(n):
        pos = robust_spp(
            data["sat_ecef"][i],
            data["pseudoranges"][i],
            weights=data["weights"][i],
            init_pos=init,
            weight_func="cauchy",
            threshold=15.0,
        )
        if pos is not None:
            fixes[i] = pos
            init = pos
    return fixes


def spp_velocity_and_heading(
    fixes: np.ndarray, times: np.ndarray, i: int
) -> tuple[np.ndarray | None, float | None, float | None]:
    """Finite-difference ECEF velocity + heading (rad, north-clockwise) from fixes[i-1]->fixes[i].

    Also returns the consecutive-fix displacement magnitude [m] (WP21b item
    1: converted into a per-epoch SPP-heading measurement uncertainty by
    ``ImuPreintPfGuide`` when ``use_heading_uncertainty=True``).
    """

    if i <= 0 or not np.all(np.isfinite(fixes[i])) or not np.all(np.isfinite(fixes[i - 1])):
        return None, None, None
    dt = float(times[i] - times[i - 1])
    if dt <= 0.0:
        return None, None, None
    displacement = fixes[i] - fixes[i - 1]
    displacement_m = float(np.linalg.norm(displacement))
    v_ecef = displacement / dt
    lat, lon = ecef_to_lla_rad(fixes[i])
    v_enu = ecef_to_enu_rotation(lat, lon) @ v_ecef
    speed = math.hypot(float(v_enu[0]), float(v_enu[1]))
    heading = math.atan2(float(v_enu[0]), float(v_enu[1])) if speed > 0.5 else None
    return v_ecef, heading, displacement_m


def initial_position_and_clock_bias(
    sat0: np.ndarray, pr0: np.ndarray, w0: np.ndarray, init_guess: np.ndarray
) -> tuple[np.ndarray, float]:
    pos = robust_spp(sat0, pr0, weights=w0, init_pos=init_guess, weight_func="cauchy", threshold=15.0)
    if pos is None:
        pos = np.asarray(init_guess, dtype=np.float64)
    ranges = np.linalg.norm(sat0 - pos, axis=1)
    cb = float(np.median(pr0 - ranges))
    return np.asarray(pos, dtype=np.float64), cb


# ---------------------------------------------------------------------------
# Per-arm PF run
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    name: str
    positions: np.ndarray
    ess_ratio: np.ndarray
    resampled: np.ndarray
    wall_s: float
    n_epochs: int


def _new_pf(n_particles: int, sigma_pos: float, sigma_cb: float, sigma_pr: float,
            ess_threshold: float, resampling: str, seed: int) -> ParticleFilterDevice:
    return ParticleFilterDevice(
        n_particles=n_particles,
        sigma_pos=sigma_pos,
        sigma_cb=sigma_cb,
        sigma_pr=sigma_pr,
        resampling=resampling,
        ess_threshold=ess_threshold,
        seed=seed,
    )


def _step_update(pf: ParticleFilterDevice, sat_i, pr_i, w_i) -> tuple[float, bool]:
    """Weight update without auto-resample, then explicit resample_if_needed for stats."""

    pf.update(sat_i, pr_i, weights=w_i, resample=False)
    ess = float(pf.get_ess())
    did_resample = pf.resample_if_needed()
    return ess, did_resample


def run_arm_cv(data: dict, *, n_particles: int, sigma_pos: float, sigma_cb: float,
               sigma_pr: float, ess_threshold: float, resampling: str, seed: int) -> ArmResult:
    n = data["n_epochs"]
    times = data["times"]
    positions = np.zeros((n, 3), dtype=np.float64)
    ess_ratio = np.zeros(n, dtype=np.float64)
    resampled = np.zeros(n, dtype=bool)

    init_pos, init_cb = initial_position_and_clock_bias(
        data["sat_ecef"][0], data["pseudoranges"][0], data["weights"][0], data["origin_ecef"]
    )
    pf = _new_pf(n_particles, sigma_pos, sigma_cb, sigma_pr, ess_threshold, resampling, seed)
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    t0 = time.perf_counter()
    for i in range(n):
        dt = float(times[i] - times[i - 1]) if i > 0 else 0.0
        if i > 0:
            pf.predict(velocity=None, dt=dt)
        ess, did_resample = _step_update(pf, data["sat_ecef"][i], data["pseudoranges"][i], data["weights"][i])
        positions[i] = np.asarray(pf.estimate())[:3]
        ess_ratio[i] = ess / n_particles
        resampled[i] = did_resample
    wall_s = time.perf_counter() - t0
    return ArmResult("cv", positions, ess_ratio, resampled, wall_s, n)


def run_arm_heuristic_imu(data: dict, imu_dict: dict, *, n_particles: int, sigma_pos: float,
                           sigma_cb: float, sigma_pr: float, ess_threshold: float,
                           resampling: str, seed: int) -> ArmResult:
    n = data["n_epochs"]
    times = data["times"]
    positions = np.zeros((n, 3), dtype=np.float64)
    ess_ratio = np.zeros(n, dtype=np.float64)
    resampled = np.zeros(n, dtype=bool)

    init_pos, init_cb = initial_position_and_clock_bias(
        data["sat_ecef"][0], data["pseudoranges"][0], data["weights"][0], data["origin_ecef"]
    )
    pf = _new_pf(n_particles, sigma_pos, sigma_cb, sigma_pr, ess_threshold, resampling, seed)
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)
    predictor = IMUPredictor(imu_dict, initial_heading=0.0)

    t0 = time.perf_counter()
    for i in range(n):
        dt = float(times[i] - times[i - 1]) if i > 0 else 0.0
        if i > 0:
            vel_enu = predictor.get_velocity_enu(float(times[i - 1]), float(times[i]))
            velocity_guide = None
            if vel_enu is not None:
                lat, lon = ecef_to_lla_rad(positions[i - 1])
                velocity_guide = predictor.velocity_enu_to_ecef(vel_enu, lat, lon)
            pf.predict(velocity=velocity_guide, dt=dt, sigma_pos=sigma_pos)
        ess, did_resample = _step_update(pf, data["sat_ecef"][i], data["pseudoranges"][i], data["weights"][i])
        positions[i] = np.asarray(pf.estimate())[:3]
        ess_ratio[i] = ess / n_particles
        resampled[i] = did_resample
    wall_s = time.perf_counter() - t0
    return ArmResult("heuristic_imu", positions, ess_ratio, resampled, wall_s, n)


def run_arm_preint_v1(data: dict, imu_dict: dict, spp_fixes: np.ndarray, *, n_particles: int,
                       sigma_pos_floor: float, sigma_pos_scale: float, sigma_accel: float,
                       sigma_gyro: float, velocity_blend_alpha: float, sigma_cb: float,
                       sigma_pr: float, ess_threshold: float, resampling: str, seed: int) -> ArmResult:
    """WP21 Phase A preint arm: scalar sigma_pos guide, rbpf_velocity_kf=False.

    Kept byte-identical to the original WP21 code path (``use_heading_uncertainty``
    left at its False default) so this arm's numbers reproduce
    ``results/wp21/WP21_REPORT.md``'s Sec. 5 table for continuity.
    """
    n = data["n_epochs"]
    times = data["times"]
    positions = np.zeros((n, 3), dtype=np.float64)
    ess_ratio = np.zeros(n, dtype=np.float64)
    resampled = np.zeros(n, dtype=bool)
    sigma_pos_used = np.zeros(n, dtype=np.float64)

    init_pos, init_cb = initial_position_and_clock_bias(
        data["sat_ecef"][0], data["pseudoranges"][0], data["weights"][0], data["origin_ecef"]
    )
    # PF's own sigma_pos (used only when a segment has no samples and we
    # fall back to CV for that one epoch); the covariance-derived sigma_pos
    # is passed explicitly per predict() call otherwise.
    pf = _new_pf(n_particles, max(sigma_pos_floor, 0.1), sigma_cb, sigma_pr, ess_threshold, resampling, seed)
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    heading_filter = ComplementaryHeadingFilter(imu_dict, alpha=0.05)
    guide = ImuPreintPfGuide(
        heading_filter,
        sigma_accel_mps2_sqrthz=sigma_accel,
        sigma_gyro_radps_sqrthz=sigma_gyro,
        sigma_pos_floor=sigma_pos_floor,
        sigma_pos_scale=sigma_pos_scale,
        velocity_blend_alpha=velocity_blend_alpha,
    )

    imu_tow = imu_dict["tow"]
    imu_accel = imu_dict["accel"]
    imu_gyro = imu_dict["gyro"]
    imu_dt = np.diff(imu_tow)
    imu_dt = np.concatenate([imu_dt, imu_dt[-1:]]) if imu_dt.size else np.zeros(0)

    t0 = time.perf_counter()
    for i in range(n):
        dt = float(times[i] - times[i - 1]) if i > 0 else 0.0
        if i > 0:
            t_prev, t_cur = float(times[i - 1]), float(times[i])
            heading_filter.update_heading_gyro(t_prev, t_cur)
            idx0 = int(np.searchsorted(imu_tow, t_prev, side="left"))
            idx1 = int(np.searchsorted(imu_tow, t_cur, side="left"))
            for k in range(idx0, idx1):
                guide.add_sample(imu_accel[k], imu_gyro[k], float(imu_dt[k]))
            v_gnss, spp_heading, _ = spp_velocity_and_heading(spp_fixes, times, i)
            p_i = np.asarray(pf.estimate())[:3]
            velocity_guide, sigma_pos_eff = guide.close_segment(
                p_i, dt, v_gnss_ecef=v_gnss, spp_heading_rad=spp_heading
            )
            if velocity_guide is None:
                pf.predict(velocity=None, dt=dt, sigma_pos=sigma_pos_floor)
                sigma_pos_used[i] = sigma_pos_floor
            else:
                pf.predict(
                    velocity=velocity_guide,
                    dt=dt,
                    sigma_pos=sigma_pos_eff,
                    sigma_vel=0.0,
                    velocity_guide_alpha=1.0,
                    rbpf_velocity_kf=False,
                    velocity_process_noise=0.0,
                )
                sigma_pos_used[i] = sigma_pos_eff
            guide.reset_segment()
        ess, did_resample = _step_update(pf, data["sat_ecef"][i], data["pseudoranges"][i], data["weights"][i])
        positions[i] = np.asarray(pf.estimate())[:3]
        ess_ratio[i] = ess / n_particles
        resampled[i] = did_resample
    wall_s = time.perf_counter() - t0
    result = ArmResult("preint_v1", positions, ess_ratio, resampled, wall_s, n)
    result.mean_sigma_pos = float(np.mean(sigma_pos_used[1:])) if n > 1 else float("nan")  # type: ignore[attr-defined]
    return result


def run_arm_preint_v2(data: dict, imu_dict: dict, spp_fixes: np.ndarray, *, n_particles: int,
                       sigma_pos_floor: float, sigma_pos_scale: float, sigma_accel: float,
                       sigma_gyro: float, velocity_blend_alpha: float, sigma_spp_pos_m: float,
                       velocity_process_noise: float, sigma_cb: float, sigma_pr: float,
                       ess_threshold: float, resampling: str, seed: int) -> ArmResult:
    """WP21b preint-v2 arm: items 1 (heading-uncertainty -> sigma_pos) +
    2 (per-particle velocity-KF fed from the preintegration's delta_v
    covariance, rbpf_velocity_kf=True) combined.
    """
    n = data["n_epochs"]
    times = data["times"]
    positions = np.zeros((n, 3), dtype=np.float64)
    ess_ratio = np.zeros(n, dtype=np.float64)
    resampled = np.zeros(n, dtype=bool)
    sigma_pos_used = np.zeros(n, dtype=np.float64)

    init_pos, init_cb = initial_position_and_clock_bias(
        data["sat_ecef"][0], data["pseudoranges"][0], data["weights"][0], data["origin_ecef"]
    )
    pf = _new_pf(n_particles, max(sigma_pos_floor, 0.1), sigma_cb, sigma_pr, ess_threshold, resampling, seed)
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    heading_filter = ComplementaryHeadingFilter(imu_dict, alpha=0.05)
    guide = ImuPreintPfGuide(
        heading_filter,
        sigma_accel_mps2_sqrthz=sigma_accel,
        sigma_gyro_radps_sqrthz=sigma_gyro,
        sigma_pos_floor=sigma_pos_floor,
        sigma_pos_scale=sigma_pos_scale,
        velocity_blend_alpha=velocity_blend_alpha,
        use_heading_uncertainty=True,
        sigma_spp_pos_m=sigma_spp_pos_m,
    )

    imu_tow = imu_dict["tow"]
    imu_accel = imu_dict["accel"]
    imu_gyro = imu_dict["gyro"]
    imu_dt = np.diff(imu_tow)
    imu_dt = np.concatenate([imu_dt, imu_dt[-1:]]) if imu_dt.size else np.zeros(0)

    t0 = time.perf_counter()
    for i in range(n):
        dt = float(times[i] - times[i - 1]) if i > 0 else 0.0
        if i > 0:
            t_prev, t_cur = float(times[i - 1]), float(times[i])
            heading_filter.update_heading_gyro(t_prev, t_cur)
            idx0 = int(np.searchsorted(imu_tow, t_prev, side="left"))
            idx1 = int(np.searchsorted(imu_tow, t_cur, side="left"))
            for k in range(idx0, idx1):
                guide.add_sample(imu_accel[k], imu_gyro[k], float(imu_dt[k]))
            v_gnss, spp_heading, spp_disp_m = spp_velocity_and_heading(spp_fixes, times, i)
            p_i = np.asarray(pf.estimate())[:3]
            velocity_guide, sigma_pos_eff = guide.close_segment(
                p_i, dt, v_gnss_ecef=v_gnss, spp_heading_rad=spp_heading,
                spp_displacement_m=spp_disp_m,
            )
            if velocity_guide is None:
                pf.predict(velocity=None, dt=dt, sigma_pos=sigma_pos_floor)
                sigma_pos_used[i] = sigma_pos_floor
            else:
                vel_cov = guide.velocity_covariance_ecef
                if vel_cov is not None:
                    pf.set_velocity_covariance(vel_cov)
                pf.predict(
                    velocity=velocity_guide,
                    dt=dt,
                    sigma_pos=sigma_pos_eff,
                    velocity_guide_alpha=1.0,
                    rbpf_velocity_kf=True,
                    velocity_process_noise=velocity_process_noise,
                )
                sigma_pos_used[i] = sigma_pos_eff
            guide.reset_segment()
        ess, did_resample = _step_update(pf, data["sat_ecef"][i], data["pseudoranges"][i], data["weights"][i])
        positions[i] = np.asarray(pf.estimate())[:3]
        ess_ratio[i] = ess / n_particles
        resampled[i] = did_resample
    wall_s = time.perf_counter() - t0
    result = ArmResult("preint_v2", positions, ess_ratio, resampled, wall_s, n)
    result.mean_sigma_pos = float(np.mean(sigma_pos_used[1:])) if n > 1 else float("nan")  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Scoring + reporting
# ---------------------------------------------------------------------------


def score_arm(arm: ArmResult, data: dict, reference: dict[float, np.ndarray], city: str, run: str) -> ScoreResult:
    epochs = [
        TrajectoryEpoch(tow=float(t), ecef=pos, is_fix=False)
        for t, pos in zip(data["times"], arm.positions)
    ]
    return score_trajectory(
        epochs, reference, city=city, run=run, traj_path=Path(f"wp21_{arm.name}"), fmt="csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="tokyo")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    parser.add_argument(
        "--systems",
        default="G",
        help=(
            "Comma-separated constellations. Default G-only (matches "
            "experiments/exp_ppc_pf_ablation_sweep.py): the PF carries a single scalar "
            "clock-bias particle state with no inter-system-bias term, so this avoids an "
            "extra unmodeled confound on top of the (already dominant, see "
            "WP21_REPORT.md) unmodeled-ionosphere SPP-level pseudorange error."
        ),
    )
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--n-particles", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resampling", default="megopolis")
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--sigma-cb", type=float, default=300.0)
    parser.add_argument("--sigma-pr", type=float, default=5.0)
    parser.add_argument("--sigma-pos-cv", type=float, default=2.0)
    parser.add_argument("--sigma-pos-heuristic", type=float, default=2.0)
    parser.add_argument("--preint-sigma-accel", type=float, default=0.05)
    parser.add_argument("--preint-sigma-gyro", type=float, default=0.005)
    parser.add_argument(
        "--preint-sigma-pos-floor", type=float, default=0.3,
        help=(
            "preint_v1 (WP21 Phase A) only: hand-tuned-in-effect floor kept at its "
            "original 0.3 default for exact continuity with WP21_REPORT.md. "
            "preint_v2 uses --preint-v2-sigma-pos-floor instead (default 0.05, a "
            "small numerical-stability floor -- WP21b item 1)."
        ),
    )
    parser.add_argument("--preint-sigma-pos-scale", type=float, default=1.0)
    parser.add_argument("--preint-velocity-blend-alpha", type=float, default=0.3)
    parser.add_argument(
        "--preint-v2-sigma-pos-floor", type=float, default=0.05,
        help="WP21b item 1: small numerical-stability floor for preint_v2 (not a tuning knob).",
    )
    parser.add_argument(
        "--preint-v2-sigma-spp-pos-m", type=float, default=30.0,
        help=(
            "WP21b item 1: documented raw-SPP horizontal sigma [m] used to convert "
            "consecutive-fix displacement into a per-epoch SPP-heading measurement "
            "uncertainty (see ImuPreintPfGuide docstring / WP21_REPORT.md Sec. 7)."
        ),
    )
    parser.add_argument(
        "--preint-v2-velocity-process-noise", type=float, default=0.0,
        help="WP21b item 2: velocity_process_noise passed to predict(rbpf_velocity_kf=True).",
    )
    parser.add_argument(
        "--arms", default="cv,heuristic,preint_v1,preint_v2",
        help="comma-separated subset to run: cv,heuristic,preint_v1,preint_v2",
    )
    parser.add_argument("--results-prefix", default="wp21_imu_rbpf")
    args = parser.parse_args()

    systems = tuple(s.strip().upper() for s in args.systems.split(",") if s.strip())
    arms_to_run = {s.strip() for s in args.arms.split(",") if s.strip()}
    run_dir = args.data_root / args.city / args.run
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  WP21/WP21b IMU-RBPF ablation")
    print("=" * 78)
    print(f"  {args.city}/{args.run}  systems={systems}  "
          f"start_epoch={args.start_epoch}  max_epochs={args.max_epochs}  "
          f"n_particles={args.n_particles}")

    t_load = time.perf_counter()
    loader = PPCDatasetLoader(run_dir)
    data = loader.load_experiment_data(
        max_epochs=args.max_epochs, start_epoch=args.start_epoch, systems=systems
    )
    imu_dict = load_ppc_imu_for_predictor(run_dir)
    print(f"  loaded {data['n_epochs']} epochs "
          f"(tow {data['times'][0]:.1f}-{data['times'][-1]:.1f}s, "
          f"dt~{float(np.median(np.diff(data['times']))):.3f}s) "
          f"and {imu_dict['tow'].size} IMU samples in {time.perf_counter() - t_load:.1f}s")

    reference = load_reference_grid(args.city, args.run, data_root=args.data_root)

    print("  computing causal per-epoch robust_spp fixes (heading/velocity reference) ...")
    t_spp = time.perf_counter()
    spp_fixes = compute_epoch_spp_fixes(data)
    n_valid_fixes = int(np.isfinite(spp_fixes).all(axis=1).sum())
    print(f"  {n_valid_fixes}/{data['n_epochs']} valid SPP fixes in {time.perf_counter() - t_spp:.1f}s")

    arm_results: dict[str, ArmResult] = {}
    common = dict(
        n_particles=args.n_particles,
        sigma_cb=args.sigma_cb,
        sigma_pr=args.sigma_pr,
        ess_threshold=args.ess_threshold,
        resampling=args.resampling,
        seed=args.seed,
    )

    if "cv" in arms_to_run:
        print("\n  [arm a] cv (no IMU) ...")
        arm_results["cv"] = run_arm_cv(data, sigma_pos=args.sigma_pos_cv, **common)
        print(f"    wall={arm_results['cv'].wall_s:.1f}s")

    if "heuristic" in arms_to_run:
        print("\n  [arm b] heuristic velocity-guide IMU (IMUPredictor) ...")
        arm_results["heuristic"] = run_arm_heuristic_imu(
            data, imu_dict, sigma_pos=args.sigma_pos_heuristic, **common
        )
        print(f"    wall={arm_results['heuristic'].wall_s:.1f}s")

    if "preint_v1" in arms_to_run:
        print("\n  [arm c] preint_v1 (WP21 Phase A: scalar sigma_pos guide) ...")
        arm_results["preint_v1"] = run_arm_preint_v1(
            data, imu_dict, spp_fixes,
            sigma_pos_floor=args.preint_sigma_pos_floor,
            sigma_pos_scale=args.preint_sigma_pos_scale,
            sigma_accel=args.preint_sigma_accel,
            sigma_gyro=args.preint_sigma_gyro,
            velocity_blend_alpha=args.preint_velocity_blend_alpha,
            **common,
        )
        print(f"    wall={arm_results['preint_v1'].wall_s:.1f}s  "
              f"mean_sigma_pos={getattr(arm_results['preint_v1'], 'mean_sigma_pos', float('nan')):.3f}m")

    if "preint_v2" in arms_to_run:
        print("\n  [arm d] preint_v2 (WP21b: heading-uncertainty sigma_pos + velocity-KF) ...")
        arm_results["preint_v2"] = run_arm_preint_v2(
            data, imu_dict, spp_fixes,
            sigma_pos_floor=args.preint_v2_sigma_pos_floor,
            sigma_pos_scale=args.preint_sigma_pos_scale,
            sigma_accel=args.preint_sigma_accel,
            sigma_gyro=args.preint_sigma_gyro,
            velocity_blend_alpha=args.preint_velocity_blend_alpha,
            sigma_spp_pos_m=args.preint_v2_sigma_spp_pos_m,
            velocity_process_noise=args.preint_v2_velocity_process_noise,
            **common,
        )
        print(f"    wall={arm_results['preint_v2'].wall_s:.1f}s  "
              f"mean_sigma_pos={getattr(arm_results['preint_v2'], 'mean_sigma_pos', float('nan')):.3f}m")

    print("\n" + "=" * 78)
    print("  Scoring")
    print("=" * 78)

    rows: list[dict[str, object]] = []
    for name, arm in arm_results.items():
        score = score_arm(arm, data, reference, args.city, args.run)
        mean_ess_ratio = float(np.mean(arm.ess_ratio))
        resample_rate = float(np.mean(arm.resampled))
        row = {
            "arm": name,
            "n_epochs": arm.n_epochs,
            "n_scored": score.n_scored,
            "coverage_pct": score.coverage_pct,
            "all_rms_m": score.all_rms_m,
            "fix_rms_m": score.fix_rms_m,
            "lt50cm_pct": score.lt50cm_pct,
            "lt50cm_full_pct": score.lt50cm_full_pct,
            "ppc_official_pct": score.ppc_official_pct,
            "mean_ess_ratio": mean_ess_ratio,
            "resample_rate": resample_rate,
            "wall_s": arm.wall_s,
            "epochs_per_s": arm.n_epochs / arm.wall_s if arm.wall_s > 0 else float("nan"),
        }
        rows.append(row)
        print(
            f"  {name:14s} n_scored={score.n_scored:5d} AllRMS={score.all_rms_m:7.3f}m "
            f"<50cm%={score.lt50cm_pct:6.2f} <50cm_full%={score.lt50cm_full_pct:6.2f} "
            f"mean_ESS/N={mean_ess_ratio:5.3f} resample_rate={resample_rate:5.3f} "
            f"wall={arm.wall_s:6.1f}s"
        )

    out_csv = RESULTS_DIR / f"{args.results_prefix}_{args.city}_{args.run}.csv"
    if rows:
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved: {out_csv}")

    # Persist raw per-epoch positions for post-hoc diagnosis.
    for name, arm in arm_results.items():
        traj_path = RESULTS_DIR / f"{args.results_prefix}_{args.city}_{args.run}_{name}_traj.csv"
        with traj_path.open("w", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["tow", "ecef_x", "ecef_y", "ecef_z", "ess_ratio", "resampled"])
            for t, pos, ess_r, rs in zip(data["times"], arm.positions, arm.ess_ratio, arm.resampled):
                writer.writerow([f"{t:.3f}", *[f"{v:.4f}" for v in pos], f"{ess_r:.4f}", int(rs)])


if __name__ == "__main__":
    main()
