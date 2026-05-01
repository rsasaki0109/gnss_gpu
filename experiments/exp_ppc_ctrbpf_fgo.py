#!/usr/bin/env python3
# ruff: noqa: E402
"""CT-RBPF-FGO PPC port — Phase 0 scaffolding.

Implements the PPC-specific particle filter pipeline planned in
``internal_docs/ctrbpf_fgo_ppc_design.md``. Phase 0 only wires:

  - PPCDatasetLoader -> PR/Doppler/sat_velocity per epoch
  - WLS init -> ParticleFilterDevice predict/update loop
  - Optional RBPF velocity-KF Doppler update (--enable-rbpf-velocity-kf)
  - Optional WLS position-update soft constraint (--enable-position-update)
  - Honest score_ppc2024 over the FULL reference.csv denominator
    (missing PF epochs are filled with [0,0,0]; pattern from
     experiments/exp_ppc_libgnss_hybrid.py)

Subsequent phases will add region-aware gates, local FGO bridge, LAMBDA,
and B-spline post-smoothing per the design doc.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_urbannav_baseline import run_wls  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.ppc_score import score_ppc2024

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

_FULL_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

_GPS_L1_WAVELENGTH_M = 0.19029367279836488

# Per-system PR weight scale: GPS=1, GLO=0.7, others=0.5. Empirically
# matches what gnss_solve does in libgnss++.
_SYSTEM_WEIGHT_SCALE = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.5, 4: 0.7}

_SYS_ID_TO_CHAR = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}

_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141316e-3


@dataclass(frozen=True)
class CTRBPFConfig:
    n_particles: int = 50_000
    sigma_pr: float = 8.0
    enable_pr_gmm: bool = False
    pr_gmm_statuses: tuple[int, ...] = (1, 3)
    pr_gmm_w_los: float = 0.7
    pr_gmm_mu_nlos_m: float = 15.0
    pr_gmm_sigma_nlos_m: float = 30.0
    pr_gmm_hybrid_loose_sigma_m: float = 5.0
    pr_gmm_clock_quantile: float = 0.35
    pr_weight_mode: str = "raw"
    pr_weight_ref_cn0: float = 45.0
    pr_weight_min: float = 0.25
    pr_weight_max: float = 1.5
    pr_prefit_gate_m: float = 0.0
    pr_prefit_gate_min_sats: int = 6
    pr_prefit_gate_keep_best: int = 0
    pr_prefit_ref: str = "pf"
    pr_skip_statuses: tuple[int, ...] = ()
    defer_epoch_resample: bool = False
    # Phase 10i: RTK-diagnostic candidate as a PF pseudo-observation.
    # The v5 libgnss++ hybrid remains the passthrough floor. A relaxed RTK
    # candidate is injected into the PF only when its diagnostics pass the
    # residual gate, and only those epochs emit the PF estimate.
    enable_rtkdiag_pf_rescue: bool = False
    rtkdiag_candidate_sigma_m: float = 0.02
    rtkdiag_candidate_ratio_min: float = 1.5
    rtkdiag_candidate_residual_rms_max: float = 1.8
    rtkdiag_candidate_emit_max_diff_m: float = 0.4
    rtkdiag_candidate_recenter_max_shift_m: float = 10000.0
    rtkdiag_candidate_select_mode: str = "residual"
    rtkdiag_candidate_emit_mode: str = "pf"
    sigma_pos: float = 2.0
    sigma_cb: float = 50.0
    spread_pos_init: float = 50.0
    spread_cb_init: float = 500.0
    sigma_doppler_mps: float = 0.5
    velocity_init_sigma: float = 1.0
    velocity_process_noise: float = 1.0
    enable_rbpf_velocity_kf: bool = False
    enable_position_update: bool = False
    position_update_sigma_m: float = 30.0
    enable_correct_clock_bias: bool = True
    enable_dd_carrier_afv: bool = False
    dd_sigma_cycles: float = 0.05
    dd_min_pairs: int = 4
    dd_min_pairs_update: int = 3
    dd_systems: tuple[str, ...] = ("G", "E", "J", "C")
    dd_base_interp: bool = False
    # Phase 2: region-aware gate for the RBPF velocity-KF (Doppler) update.
    # ``None`` disables a gate. DD-pair gate uses 0 if no DD computer is
    # available, so it implicitly skips the KF update unless DD is wired.
    rbpf_kf_gate_min_dd_pairs: int | None = None
    rbpf_kf_gate_min_ess_ratio: float | None = None
    rbpf_kf_gate_max_spread_m: float | None = None
    # Phase 6: libgnss++ hybrid (50.91% baseline) position update. When
    # enabled, ``pf.position_update(hybrid_pos, sigma=hybrid_sigma_m)`` is
    # applied per epoch (if a hybrid sample exists for the rover TOW),
    # placing the cloud within 1 m of the hybrid baseline before the
    # Doppler-KF/DD-AFV updates so fractional DD residuals become meaningful.
    enable_hybrid_pu: bool = False
    hybrid_sigma_m: float = 1.0
    # Phase 7: derive a per-epoch velocity guide from hybrid pos finite
    # differences and feed it to ``pf.predict(velocity=...)`` so the cloud
    # can independently track the trajectory. Without this, the cloud is
    # stationary while the truth moves and hybrid PU at sigma=1m cannot
    # rescue particles that are several meters from the moving baseline.
    enable_hybrid_velocity_guide: bool = False
    # Phase 7 output mode. Default ``passthrough`` emits the hybrid pos
    # directly when available (Phase 6 MVP); ``pf`` emits the PF estimate
    # so the run can show whether DD-AFV / Doppler-KF correction beats
    # plain hybrid.
    hybrid_emit_pf_estimate: bool = False
    # Phase 4: post-process FGO + LAMBDA partial fix. After the PF loop
    # completes, slide a window over the trajectory and call
    # ``solve_local_fgo_with_lambda`` per window using cached DD-carrier
    # observations. Where LAMBDA accepts at least ``fgo_min_fixed_to_apply``
    # integer fixes via the ratio test, the FGO positions replace the PF /
    # hybrid output for that window. This is fixed-lag (not strictly
    # realtime) but the latency = window_size * dt is bounded.
    enable_fgo_lambda: bool = False
    fgo_window_size: int = 30
    fgo_window_stride: int = 15
    fgo_lambda_ratio: float = 3.0
    fgo_lambda_min_epochs: int = 10
    fgo_min_fixed_to_apply: int = 3
    fgo_prior_sigma_m: float = 0.5
    fgo_dd_sigma_cycles: float = 0.20
    # C2: per-epoch gate. If non-empty, FGO output is only written back to
    # ``positions[i]`` when the hybrid Status at ``times[i]`` is one of these
    # values; epochs with other Status values keep the hybrid passthrough.
    # Empty tuple = apply Phase 4 to every epoch (legacy behavior).
    # Default ``(1, 3)`` skips Status=4 (cm-class libgnss++ output) and only
    # rewrites Status=1/3 (m-class).
    fgo_apply_hybrid_statuses: tuple[int, ...] = (1, 3)
    # D2: per-epoch prior sigmas inside the FGO solve. Status values that
    # are NOT in ``fgo_apply_hybrid_statuses`` (e.g. Status=4 = cm-class
    # hybrid) get the tight ``fgo_anchor_sigma_m`` so the FGO treats them
    # as cm-class anchors. Status values that ARE in the apply set
    # (m-class hybrid) get ``fgo_loose_sigma_m`` so DD carrier + DD PR
    # drive the local solve. Set ``fgo_anchor_sigma_m`` to a non-positive
    # value to disable per-epoch priors entirely (fall back to legacy
    # endpoint-only priors at ``fgo_prior_sigma_m``).
    fgo_anchor_sigma_m: float = 0.05
    fgo_loose_sigma_m: float = 5.0
    # D2b "minimum-correction gate": skip rewrites where FGO output is
    # within ``fgo_min_correction_m`` of the hybrid passthrough. Empirical
    # evidence (tokyo/run2 first 2000 ep) showed Phase 4 nudges many
    # cm-class hybrid passes (~5cm) across the 0.5m PPC threshold; small
    # rewrites cost more pass than they recover. Set to 0.0 to disable.
    fgo_min_correction_m: float = 0.5
    # Phase 8: TDCP-anchored hybrid smoother. After the PF loop, run a
    # per-coordinate forward+backward Kalman smoother over the trajectory,
    # using the rover-side TDCP velocity as the motion model and the hybrid
    # passthrough as the observation. Each epoch's observation sigma is
    # derived from its hybrid Status (anchor sigma for "good" Status, loose
    # sigma for "rewritable" Status, huge sigma when hybrid is missing).
    enable_tdcp_smoother: bool = False
    tdcp_sigma_mps: float = 0.05
    tdcp_postfit_max_m: float = 1.0
    tdcp_min_sats: int = 5
    tdcp_obs_anchor_sigma_m: float = 0.05
    tdcp_obs_loose_sigma_m: float = 5.0
    tdcp_obs_missing_sigma_m: float = 1000.0
    # Phase 9a: ZUPT (zero-velocity update) using PPC IMU. Per rover epoch
    # we look at the specific-force / angular-rate norms over the IMU
    # samples that fall in [t_i, t_{i+1}). When the accel norm is close
    # to gravity AND the gyro norm is small, the vehicle is stopped and
    # the rover position must equal the last position. We use this to
    # damp hybrid jitter on Status=1/3 epochs that sit inside a stop.
    enable_zupt: bool = False
    zupt_acc_norm_low_mps2: float = 9.78
    zupt_acc_norm_high_mps2: float = 9.85
    zupt_gyro_norm_max_dps: float = 0.3
    # Number of consecutive static epochs required (including the current
    # one) before ZUPT actually rewrites. Single-epoch static glitches are
    # discarded; 5 epochs at 5 Hz = 1 s of confirmed stop.
    zupt_min_consecutive: int = 5
    # Maximum |base - anchor| disagreement [m] tolerated before ZUPT
    # rewrites. If the current passthrough already differs from the static
    # anchor by more than this, the anchor is probably stale and we skip
    # the rewrite (we cannot tell at runtime whether base drifted or the
    # vehicle moved).
    zupt_max_anchor_drift_m: float = 0.5
    # Only ZUPT-rewrite epochs whose hybrid Status is in this set; the
    # Status=4 cm-class epochs stay fixed to their hybrid value.
    zupt_apply_hybrid_statuses: tuple[int, ...] = (1, 3)
    # Phase 9b: tight-coupled IMU. Unlike Phase 9a (post-process ZUPT) or
    # Phase 4/8 (post-process FGO/TDCP), IMU evidence enters the PF *during*
    # the loop as a per-particle position pseudo-observation
    # (``pf.position_update(imu_predicted_pos, sigma=imu_sigma)``). The
    # IMU prediction is derived from a sliding pre-integration window
    # anchored at the most recent Status=4 (cm-class) hybrid epoch:
    #   - body-frame accel is rotated into ENU using a yaw derived from the
    #     anchor's recent velocity (course-over-ground, pitch/roll = 0),
    #   - gravity (9.81 m/s^2) is removed from the body-z axis,
    #   - acceleration is double-integrated to give Δp_imu since anchor.
    # Output emission is gated on hybrid Status: Status=4 epochs always emit
    # the hybrid passthrough (cm-class anchor); Status in
    # ``imu_tc_emit_pf_hybrid_statuses`` (default (1, 3)) emit the PF
    # estimate so the IMU likelihood actually surfaces in the PPC score.
    enable_imu_tc: bool = False
    imu_tc_emit_pf_hybrid_statuses: tuple[int, ...] = (1, 3)
    # Position pseudo-observation sigma at t=0s after the anchor reset.
    imu_tc_pos_sigma_base_m: float = 0.5
    # Sigma growth per second of dead-reckoning (linear; reflects
    # accel-bias drift). At 5s elapsed, default sigma = 0.5 + 5*0.5 = 3.0 m.
    imu_tc_pos_sigma_per_s: float = 0.5
    # Maximum dead-reckoning duration [s]. Past this, IMU drift dominates
    # and we skip the PU + emission switch (epoch falls back to hybrid).
    imu_tc_max_dr_seconds: float = 5.0
    # If |IMU-predicted pos - hybrid pos| exceeds this when both are
    # available, the IMU prediction is presumed wrong (yaw drift, axis
    # error) and we skip the PU + emission switch. Conservative default
    # 30 m: PPC NLOS hybrid jumps can exceed 20 m, so this is loose.
    imu_tc_max_disagreement_m: float = 30.0
    # When the emission switch fires we still bound the PF estimate to the
    # hybrid ± this distance; if the PF wandered farther we keep hybrid.
    # Protects against PF cloud collapse drift in long NLOS gaps.
    imu_tc_emit_max_diff_m: float = 20.0
    # Hybrid PU sigma override on Status values that ARE in
    # ``imu_tc_emit_pf_hybrid_statuses``. When set > 0, the hybrid pos
    # update on those (m-class) epochs uses this sigma instead of
    # ``hybrid_sigma_m`` (default 1.0). Looser hybrid PU lets the IMU
    # pre-integration drive the cloud through NLOS without being clamped
    # to the m-class hybrid baseline. Set to 0 to keep the global sigma.
    imu_tc_hybrid_loose_sigma_m: float = 5.0
    # Static-detection thresholds for the *anchor*: when the vehicle is
    # IMU-static at a Status=4 epoch, we set the anchor velocity to zero
    # (instead of a noisy hybrid finite difference). Reuses the Phase 9a
    # ZUPT thresholds so a single `--zupt-*` knob set covers both.
    imu_tc_anchor_static_acc_low_mps2: float = 9.6
    imu_tc_anchor_static_acc_high_mps2: float = 9.95
    imu_tc_anchor_static_gyro_max_dps: float = 1.5
    # Phase 9c: full 15-state INS-GNSS EKF. This coexists with Phase 9b
    # but is enabled through separate method labels so the yaw-only
    # pre-integration baseline remains reproducible.
    enable_ins_tc: bool = False
    ins_tc_emit_pf_hybrid_statuses: tuple[int, ...] = (1, 3)
    ins_tc_obs_status_4_sigma_m: float = 0.05
    ins_tc_obs_status_3_sigma_m: float = 0.0
    ins_tc_max_dr_seconds: float = 10.0
    ins_tc_max_disagreement_m: float = 30.0
    ins_tc_emit_max_diff_m: float = 1.0
    ins_tc_pf_pu_floor_sigma_m: float = 0.1
    ins_tc_pf_pu_ceiling_sigma_m: float = 5.0
    ins_tc_use_particle_imu_predict: bool = True
    ins_tc_particle_imu_sigma_pos_m: float = 0.02
    ins_tc_particle_imu_sigma_acc_mps2: float = 0.10
    ins_tc_particle_imu_sigma_gyro_rps: float = 0.005
    ins_tc_particle_imu_acc_bias_rw: float = 1.0e-4
    ins_tc_particle_imu_gyro_bias_rw: float = 1.0e-5
    ins_tc_particle_imu_att_spread_rad: float = math.radians(2.0)
    ins_tc_particle_imu_acc_bias_spread: float = 0.05
    ins_tc_particle_imu_gyro_bias_spread_rps: float = math.radians(0.1)
    ins_tc_particle_imu_velocity_spread_mps: float = 0.5
    ins_tc_recenter_status4: bool = False
    ins_tc_recenter_max_shift_m: float = 5000.0
    ins_tc_use_motion_predict: bool = True
    ins_tc_predict_sigma_pos_m: float = 0.2
    ins_tc_predict_velocity_alpha: float = 1.0
    ins_tc_align_acc_low: float = 9.6
    ins_tc_align_acc_high: float = 9.95
    ins_tc_align_gyro_max_dps: float = 1.5
    ins_tc_align_min_samples: int = 50
    ins_tc_yaw_init_min_speed_mps: float = 1.0
    systems: tuple[str, ...] = ("G", "R", "E", "C", "J")
    method_label: str = "PF-PR"


@dataclass
class _PPCMeasurement:
    """Light-weight measurement object compatible with DDCarrierComputer.

    Only the attributes consumed by ``compute_dd`` (in rover_obs_path mode)
    are populated: ``system_id``, ``prn``, ``satellite_ecef``, ``elevation``,
    ``snr``. ``carrier_phase`` is read directly from the rover RINEX, not
    from this object.
    """

    system_id: int
    prn: int
    satellite_ecef: np.ndarray
    elevation: float
    snr: float


def _ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float, float]:
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - N
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * N / max(N + h, 1.0)))
    sin_lat = math.sin(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1e-12) - N
    return lat, lon, alt


def _elevation_rad(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    lat, lon, _ = _ecef_to_llh(float(rx_ecef[0]), float(rx_ecef[1]), float(rx_ecef[2]))
    dx = sat_ecef - rx_ecef
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    e = -sin_lon * dx[0] + cos_lon * dx[1]
    n = -sin_lat * cos_lon * dx[0] - sin_lat * sin_lon * dx[1] + cos_lat * dx[2]
    u = cos_lat * cos_lon * dx[0] + cos_lat * sin_lon * dx[1] + sin_lat * dx[2]
    return math.atan2(float(u), math.sqrt(float(e) * float(e) + float(n) * float(n)))


def _build_dd_measurements(
    sat_ecef: np.ndarray,
    system_ids: np.ndarray,
    sat_id_strs: list[str],
    weights: np.ndarray,
    rover_pos: np.ndarray,
    allowed_systems: tuple[str, ...],
) -> list[_PPCMeasurement]:
    """Build measurement objects from per-epoch arrays for compute_dd()."""
    out: list[_PPCMeasurement] = []
    sids = np.asarray(system_ids, dtype=np.int32)
    for k in range(int(sat_ecef.shape[0])):
        sys_char = _SYS_ID_TO_CHAR.get(int(sids[k]))
        if sys_char is None or sys_char not in allowed_systems:
            continue
        sat_id_str = sat_id_strs[k] if k < len(sat_id_strs) else ""
        prn_str = sat_id_str[1:].lstrip("0") if sat_id_str else ""
        try:
            prn = int(prn_str) if prn_str else 0
        except ValueError:
            prn = 0
        if prn <= 0:
            continue
        sat_pos = np.asarray(sat_ecef[k], dtype=np.float64)
        if sat_pos.size != 3 or not np.all(np.isfinite(sat_pos)):
            continue
        elev = _elevation_rad(rover_pos, sat_pos)
        out.append(
            _PPCMeasurement(
                system_id=int(sids[k]),
                prn=prn,
                satellite_ecef=sat_pos,
                elevation=float(elev),
                snr=float(weights[k]) if k < len(weights) else 1.0,
            )
        )
    return out


@dataclass
class _DDStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    pairs_total: int = 0


@dataclass
class _PRObsStats:
    epochs_gaussian: int = 0
    epochs_gmm: int = 0
    deferred_resample_epochs: int = 0
    prefit_epochs: int = 0
    prefit_sats_kept: int = 0
    prefit_sats_dropped: int = 0
    epochs_skipped: int = 0


@dataclass
class _RBPFGateStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    skipped_min_dd_pairs: int = 0
    skipped_min_ess_ratio: int = 0
    skipped_max_spread: int = 0


@dataclass
class _HybridStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    epochs_lookup_missing: int = 0


@dataclass
class _RTKDiagPFStats:
    epochs_evaluated: int = 0
    gate_pass: int = 0
    candidate_missing: int = 0
    candidate_options_total: int = 0
    pu_applied: int = 0
    recenter_applied: int = 0
    recenter_skipped: int = 0
    emit_pf_estimate: int = 0
    emit_candidate: int = 0
    emit_skipped_pf_drift: int = 0
    selected_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _FGOStats:
    windows_attempted: int = 0
    windows_solved: int = 0
    windows_applied: int = 0
    n_fixed_total: int = 0
    n_fixed_observations_total: int = 0
    epochs_replaced: int = 0


@dataclass
class _TDCPSmootherStats:
    pairs_attempted: int = 0
    pairs_accepted: int = 0
    pairs_rejected_min_sats: int = 0
    pairs_rejected_postfit: int = 0


@dataclass
class _ZUPTStats:
    epochs_evaluated: int = 0
    epochs_static: int = 0
    epochs_rewritten: int = 0
    epochs_no_imu: int = 0


@dataclass
class _IMUTCStats:
    """Per-run counters for Phase 9b tight-coupled IMU."""

    epochs_evaluated: int = 0  # rover epochs reached after the first anchor
    anchor_resets: int = 0  # Status=4 epochs that re-set the anchor
    anchor_resets_static: int = 0  # subset where the anchor was IMU-static
    pu_applied: int = 0  # epochs where pf.position_update(imu_pred) fired
    pu_skipped_no_imu: int = 0
    pu_skipped_no_anchor: int = 0
    pu_skipped_dr_too_long: int = 0
    pu_skipped_disagreement: int = 0
    emit_pf_estimate: int = 0  # epochs where output[i] = pf.estimate (vs hybrid)
    emit_skipped_pf_drift: int = 0  # PF wandered too far from hybrid; kept hybrid
    avg_dr_seconds: float = 0.0
    _dr_seconds_sum: float = 0.0
    _dr_seconds_n: int = 0

    def record_dr_seconds(self, dr: float) -> None:
        if not (dr is None or dr != dr):  # noqa: SIM114 — NaN safe
            self._dr_seconds_sum += float(dr)
            self._dr_seconds_n += 1
            self.avg_dr_seconds = self._dr_seconds_sum / max(self._dr_seconds_n, 1)


@dataclass
class _INSTCStats:
    """Per-run counters for Phase 9c INS-GNSS tight coupling."""

    aligned_at_epoch: int = -1
    yaw_initialized_at_epoch: int = -1
    epochs_evaluated: int = 0
    pu_applied: int = 0
    pu_skipped_not_aligned: int = 0
    pu_skipped_no_yaw: int = 0
    pu_skipped_dr_too_long: int = 0
    pu_skipped_disagreement: int = 0
    particle_imu_initialized: int = 0
    particle_imu_predict_used: int = 0
    recenter_applied: int = 0
    recenter_skipped: int = 0
    recenter_shift_sum_m: float = 0.0
    motion_predict_used: int = 0
    obs_status_4_used: int = 0
    obs_status_3_used: int = 0
    emit_pf_estimate: int = 0
    emit_skipped_pf_drift: int = 0
    final_acc_bias_norm: float = 0.0
    final_gyro_bias_norm_dps: float = 0.0
    final_pos_sigma_m: float = 0.0


def _build_imu_per_epoch_stats(
    imu: dict[str, np.ndarray],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per rover-epoch accel norm, gyro norm and sample count.

    Each rover epoch ``i`` aggregates the IMU samples whose timestamp falls in
    ``[times[i] - dt/2, times[i] + dt/2)`` where ``dt`` is the median rover
    epoch spacing. Epochs with zero IMU samples have NaN stats.
    """
    n = int(times.size)
    acc_norm_per = np.full(n, np.nan, dtype=np.float64)
    gyro_norm_per = np.full(n, np.nan, dtype=np.float64)
    n_samples = np.zeros(n, dtype=np.int32)
    if not imu or "time" not in imu or imu["time"].size == 0:
        return acc_norm_per, gyro_norm_per, n_samples

    t_imu = np.asarray(imu["time"], dtype=np.float64)
    ax = np.asarray(imu.get("acc_x", np.full_like(t_imu, np.nan)), dtype=np.float64)
    ay = np.asarray(imu.get("acc_y", np.full_like(t_imu, np.nan)), dtype=np.float64)
    az = np.asarray(imu.get("acc_z", np.full_like(t_imu, np.nan)), dtype=np.float64)
    gx = np.asarray(imu.get("gyro_x", np.full_like(t_imu, np.nan)), dtype=np.float64)
    gy = np.asarray(imu.get("gyro_y", np.full_like(t_imu, np.nan)), dtype=np.float64)
    gz = np.asarray(imu.get("gyro_z", np.full_like(t_imu, np.nan)), dtype=np.float64)
    acc_norm_imu = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_norm_imu = np.sqrt(gx * gx + gy * gy + gz * gz)

    if n >= 2:
        dt_med = float(np.median(np.diff(times)))
    else:
        dt_med = 0.2
    half = 0.5 * dt_med

    for i in range(n):
        t = float(times[i])
        lo = t - half
        hi = t + half
        # Binary search for sample range.
        i0 = int(np.searchsorted(t_imu, lo, side="left"))
        i1 = int(np.searchsorted(t_imu, hi, side="left"))
        if i1 <= i0:
            continue
        a_slice = acc_norm_imu[i0:i1]
        g_slice = gyro_norm_imu[i0:i1]
        finite_a = np.isfinite(a_slice)
        finite_g = np.isfinite(g_slice)
        if finite_a.any():
            acc_norm_per[i] = float(np.mean(a_slice[finite_a]))
        if finite_g.any():
            gyro_norm_per[i] = float(np.mean(g_slice[finite_g]))
        n_samples[i] = int(min(int(np.sum(finite_a)), int(np.sum(finite_g))))
    return acc_norm_per, gyro_norm_per, n_samples


def _apply_zupt(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    imu: dict[str, np.ndarray] | None,
    config: CTRBPFConfig,
    hybrid_status: dict[float, int] | None,
    stats: _ZUPTStats,
) -> np.ndarray:
    """Hold position constant on rover epochs where IMU reports static.

    ``positions[i] := positions[i-1]`` when:
      - ``imu`` provides finite accel / gyro norms for epoch ``i``,
      - the accel norm is in ``[zupt_acc_norm_low_mps2, zupt_acc_norm_high_mps2]``,
      - the gyro norm is below ``zupt_gyro_norm_max_dps``,
      - the hybrid Status at this epoch is in ``zupt_apply_hybrid_statuses``
        (so cm-class Status=4 anchors are never moved).

    ``positions`` is treated as the hybrid passthrough output; this
    function returns a modified copy.
    """
    if imu is None or "time" not in imu or imu["time"].size == 0:
        return positions

    n = int(positions.shape[0])
    if n < 2:
        return positions

    acc_per, gyro_per, _ = _build_imu_per_epoch_stats(imu, times)

    rewrite_set = set(int(s) for s in config.zupt_apply_hybrid_statuses)
    out = np.array(positions, dtype=np.float64, copy=True)

    # ZUPT can only copy from a past anchor position if the vehicle has
    # been continuously static since that anchor. Track:
    #   - last_static_anchor: pos at the most recent Status=4 epoch where
    #     the vehicle was also IMU-static
    #   - static_streak: consecutive static epochs since last motion
    last_static_anchor: np.ndarray | None = None
    static_streak: int = 0
    min_streak = max(1, int(config.zupt_min_consecutive))
    max_drift = float(config.zupt_max_anchor_drift_m)

    for i in range(n):
        anchor_here = False
        if hybrid_status is not None and config.zupt_apply_hybrid_statuses:
            t_key = round(float(times[i]), 1)
            st = hybrid_status.get(t_key)
            if st is not None and st not in rewrite_set:
                anchor_here = True

        a = acc_per[i] if i < acc_per.size else float("nan")
        g = gyro_per[i] if i < gyro_per.size else float("nan")
        is_static = (
            np.isfinite(a)
            and np.isfinite(g)
            and float(config.zupt_acc_norm_low_mps2) <= a <= float(config.zupt_acc_norm_high_mps2)
            and g <= float(config.zupt_gyro_norm_max_dps)
        )

        if i > 0:
            stats.epochs_evaluated += 1
            if not (np.isfinite(a) and np.isfinite(g)):
                stats.epochs_no_imu += 1
            elif is_static:
                stats.epochs_static += 1

        if not is_static:
            last_static_anchor = None
            static_streak = 0
            continue

        static_streak += 1

        if anchor_here:
            last_static_anchor = np.asarray(out[i], dtype=np.float64).copy()
            continue

        if (
            last_static_anchor is None
            or static_streak < min_streak
            or i == 0
        ):
            continue
        # Skip the rewrite if base drifted away from the anchor by more
        # than the allowed tolerance -- the anchor is probably stale.
        drift = float(np.linalg.norm(out[i] - last_static_anchor))
        if drift > max_drift:
            continue
        out[i] = last_static_anchor
        stats.epochs_rewritten += 1

    return out


# ---------------------------------------------------------------------------
# Phase 9b: tight-coupled IMU helpers
# ---------------------------------------------------------------------------


def _ecef_to_enu_rotation(lat: float, lon: float) -> np.ndarray:
    """3x3 rotation matrix that maps an ECEF delta to local ENU at (lat, lon)."""
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so,             co,            0.0],
            [-sl * co, -sl * so,  cl],
            [ cl * co,  cl * so,  sl],
        ],
        dtype=np.float64,
    )


def _ecef_velocity_to_enu(pos_ecef: np.ndarray, vel_ecef: np.ndarray) -> np.ndarray:
    lat, lon, _ = _ecef_to_llh(float(pos_ecef[0]), float(pos_ecef[1]), float(pos_ecef[2]))
    R = _ecef_to_enu_rotation(lat, lon)
    return R @ np.asarray(vel_ecef, dtype=np.float64)


def _enu_delta_to_ecef(pos_ecef: np.ndarray, enu: np.ndarray) -> np.ndarray:
    lat, lon, _ = _ecef_to_llh(float(pos_ecef[0]), float(pos_ecef[1]), float(pos_ecef[2]))
    R = _ecef_to_enu_rotation(lat, lon)
    return R.T @ np.asarray(enu, dtype=np.float64)


def _ecef_to_enu_at_origin(
    pos_ecef: np.ndarray,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R @ (
        np.asarray(pos_ecef, dtype=np.float64).reshape(3)
        - np.asarray(origin_ecef, dtype=np.float64).reshape(3)
    )


def _ecef_velocity_to_enu_at_origin(
    vel_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R @ np.asarray(vel_ecef, dtype=np.float64).reshape(3)


def _enu_velocity_to_ecef_at_origin(
    vel_enu: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R.T @ np.asarray(vel_enu, dtype=np.float64).reshape(3)


def _gravity_ecef_at_origin(origin_lat: float, origin_lon: float) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R.T @ np.array([0.0, 0.0, -9.81], dtype=np.float64)


def _slice_imu_samples(
    imu: dict[str, np.ndarray],
    imu_t: np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Return INS EKF IMU rows [t, ax, ay, az, gx, gy, gz] in [start, end]."""
    if not math.isfinite(t_start) or not math.isfinite(t_end) or t_end < t_start:
        return np.empty((0, 7), dtype=np.float64)
    i0 = int(np.searchsorted(imu_t, float(t_start), side="left"))
    i1 = int(np.searchsorted(imu_t, float(t_end), side="right"))
    if i1 <= i0:
        return np.empty((0, 7), dtype=np.float64)
    return np.column_stack(
        (
            np.asarray(imu_t[i0:i1], dtype=np.float64),
            np.asarray(imu["acc_x"][i0:i1], dtype=np.float64),
            np.asarray(imu["acc_y"][i0:i1], dtype=np.float64),
            np.asarray(imu["acc_z"][i0:i1], dtype=np.float64),
            np.asarray(imu["gyro_x"][i0:i1], dtype=np.float64),
            np.asarray(imu["gyro_y"][i0:i1], dtype=np.float64),
            np.asarray(imu["gyro_z"][i0:i1], dtype=np.float64),
        )
    )


@dataclass
class _IMUAnchor:
    """State of the IMU dead-reckoning anchor.

    The anchor is the most recent rover epoch where the position is trusted
    (Status=4 hybrid, or first epoch). Pre-integration accumulates body-frame
    accelerometer samples since this time and projects them into ENU using
    the anchor's heading (yaw). The first ``valid_yaw`` is set as soon as we
    have a usable course-over-ground estimate; until then the IMU PU is
    skipped.
    """

    pos_ecef: np.ndarray
    time: float
    yaw_enu_rad: float  # CCW from East in ENU; vehicle forward = +yaw direction
    valid_yaw: bool
    velocity_enu: np.ndarray  # (east, north, up) m/s, used as initial dr velocity


def _heading_from_velocity_enu(vel_enu: np.ndarray, min_speed_mps: float = 0.7) -> tuple[float, bool]:
    """Yaw (CCW from East) derived from ENU velocity.

    Returns (yaw_rad, valid). ``valid`` is False when the planar speed is
    below ``min_speed_mps`` (heading cannot be reliably inferred).
    """
    east = float(vel_enu[0]) if vel_enu.size > 0 else 0.0
    north = float(vel_enu[1]) if vel_enu.size > 1 else 0.0
    speed = math.hypot(east, north)
    if not math.isfinite(speed) or speed < float(min_speed_mps):
        return 0.0, False
    return math.atan2(north, east), True


def _build_imu_segment_index(imu: dict[str, np.ndarray] | None) -> np.ndarray | None:
    """Sorted IMU timestamp array used for binary search; ``None`` if no IMU."""
    if imu is None or "time" not in imu:
        return None
    t = np.asarray(imu["time"], dtype=np.float64)
    if t.size == 0:
        return None
    return t


def _integrate_imu_between(
    imu: dict[str, np.ndarray],
    imu_t: np.ndarray,
    t_start: float,
    t_end: float,
    yaw_enu_rad: float,
    initial_velocity_enu: np.ndarray,
    *,
    gravity_mps2: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pre-integrate body-frame IMU between two rover epochs.

    Returns ``(delta_pos_enu, end_velocity_enu, n_samples)``. The body frame
    convention assumed: x=forward, y=left, z=up (right-handed, +z aligned
    with gravity reaction). Pitch and roll are taken as zero so accel_z is
    just the vertical specific force. Yaw rotates the horizontal body axes
    into ENU (east, north).

    Returns zero deltas (and propagated initial velocity * dt) when no
    samples fall in [t_start, t_end).
    """
    dt_total = float(t_end - t_start)
    if not math.isfinite(dt_total) or dt_total <= 0.0:
        return np.zeros(3, dtype=np.float64), np.asarray(initial_velocity_enu, dtype=np.float64).copy(), 0

    i0 = int(np.searchsorted(imu_t, t_start, side="left"))
    i1 = int(np.searchsorted(imu_t, t_end, side="left"))
    if i1 <= i0:
        # No IMU samples in window -- propagate at constant velocity.
        v = np.asarray(initial_velocity_enu, dtype=np.float64).copy()
        dp = v * dt_total
        return dp, v, 0

    ax = np.asarray(imu["acc_x"][i0:i1], dtype=np.float64)
    ay = np.asarray(imu["acc_y"][i0:i1], dtype=np.float64)
    az = np.asarray(imu["acc_z"][i0:i1], dtype=np.float64)
    ts = np.asarray(imu_t[i0:i1], dtype=np.float64)

    # Replace NaNs with zeros (rare, treat as no acceleration sample).
    ax = np.where(np.isfinite(ax), ax, 0.0)
    ay = np.where(np.isfinite(ay), ay, 0.0)
    az = np.where(np.isfinite(az), az, 0.0)

    # Body -> ENU (yaw only; pitch/roll = 0). x=forward, y=left, z=up.
    cy = math.cos(yaw_enu_rad)
    sy = math.sin(yaw_enu_rad)
    east_acc = ax * cy - ay * sy
    north_acc = ax * sy + ay * cy
    up_acc = az - float(gravity_mps2)

    # Per-sample dt: use the gaps between successive timestamps, plus an
    # opening dt from t_start to ts[0] and a closing dt from ts[-1] to t_end.
    n = ts.size
    dts = np.empty(n, dtype=np.float64)
    if n == 1:
        dts[0] = dt_total
    else:
        # Each sample is "valid" for the time slice ending at the next sample.
        diffs = np.diff(ts)
        dts[:-1] = diffs
        # Last sample carries from ts[-1] to t_end.
        dts[-1] = max(0.0, t_end - float(ts[-1]))
        # Adjust the first sample to also cover t_start..ts[0] so we don't
        # lose the opening fraction.
        opening = max(0.0, float(ts[0]) - t_start)
        dts[0] = dts[0] + opening
    # Clamp negative dts (out-of-order samples) to zero.
    dts = np.where(dts > 0.0, dts, 0.0)

    # Trapezoidal-ish integration: assume the sample's accel applies for its
    # owned dt. v_end = v_start + sum(a * dt); p_end = p_start + sum(v_mid * dt).
    v_east = float(initial_velocity_enu[0])
    v_north = float(initial_velocity_enu[1])
    v_up = float(initial_velocity_enu[2])
    dp_east = 0.0
    dp_north = 0.0
    dp_up = 0.0
    for k in range(n):
        dt_k = float(dts[k])
        if dt_k <= 0.0:
            continue
        # Mid-point velocity for position step (trapezoidal rule).
        ve_mid = v_east + 0.5 * float(east_acc[k]) * dt_k
        vn_mid = v_north + 0.5 * float(north_acc[k]) * dt_k
        vu_mid = v_up + 0.5 * float(up_acc[k]) * dt_k
        dp_east += ve_mid * dt_k
        dp_north += vn_mid * dt_k
        dp_up += vu_mid * dt_k
        v_east += float(east_acc[k]) * dt_k
        v_north += float(north_acc[k]) * dt_k
        v_up += float(up_acc[k]) * dt_k

    delta = np.array([dp_east, dp_north, dp_up], dtype=np.float64)
    v_end = np.array([v_east, v_north, v_up], dtype=np.float64)
    return delta, v_end, int(n)


@dataclass
class _TDCPMeasurement:
    """Light-weight measurement object compatible with the TDCP module.

    The TDCP solver pulls ``system_id``, ``prn``, ``satellite_ecef``,
    ``carrier_phase`` (cycles), ``satellite_velocity`` (m/s), and
    ``clock_drift`` (s/s). The PPC loader supplies all of these when
    ``include_sat_velocity=True``.
    """

    system_id: int
    prn: int
    satellite_ecef: np.ndarray
    carrier_phase: float
    satellite_velocity: np.ndarray
    clock_drift: float
    elevation: float = 0.0


def _build_tdcp_measurements(
    sat_ecef: np.ndarray,
    system_ids: np.ndarray,
    sat_id_strs: list[str],
    carrier_phase: np.ndarray,
    sat_velocity: np.ndarray,
    clock_drift: np.ndarray,
    rover_pos: np.ndarray,
) -> list[_TDCPMeasurement]:
    out: list[_TDCPMeasurement] = []
    sids = np.asarray(system_ids, dtype=np.int32)
    for k in range(int(sat_ecef.shape[0])):
        sys_char = _SYS_ID_TO_CHAR.get(int(sids[k]))
        if sys_char is None:
            continue
        sat_id_str = sat_id_strs[k] if k < len(sat_id_strs) else ""
        prn_str = sat_id_str[1:].lstrip("0") if sat_id_str else ""
        try:
            prn = int(prn_str) if prn_str else 0
        except ValueError:
            prn = 0
        if prn <= 0:
            continue
        sat_pos = np.asarray(sat_ecef[k], dtype=np.float64)
        if sat_pos.size != 3 or not np.all(np.isfinite(sat_pos)):
            continue
        cp = float(carrier_phase[k]) if k < len(carrier_phase) else float("nan")
        if not np.isfinite(cp) or abs(cp) < 1e3:
            continue
        sv = (
            np.asarray(sat_velocity[k], dtype=np.float64)
            if k < len(sat_velocity)
            else np.zeros(3, dtype=np.float64)
        )
        if sv.size != 3 or not np.all(np.isfinite(sv)):
            continue
        cd = float(clock_drift[k]) if k < len(clock_drift) else 0.0
        if not np.isfinite(cd):
            cd = 0.0
        elev = _elevation_rad(rover_pos, sat_pos)
        out.append(
            _TDCPMeasurement(
                system_id=int(sids[k]),
                prn=prn,
                satellite_ecef=sat_pos,
                carrier_phase=cp,
                satellite_velocity=sv,
                clock_drift=cd,
                elevation=float(elev),
            )
        )
    return out


def _apply_tdcp_smoother(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    data: dict,
    config: CTRBPFConfig,
    hybrid_pos: dict[float, np.ndarray] | None,
    hybrid_status: dict[float, int] | None,
    stats: _TDCPSmootherStats,
) -> np.ndarray:
    """Forward+backward Kalman smoother over ``positions`` using TDCP velocity.

    State per coordinate is scalar (position only). The motion model is
    ``x[t+1] = x[t] + v_tdcp[t] * dt[t]`` and the observation is the
    hybrid passthrough (``hybrid_pos[t]``) with sigma derived from the
    hybrid Status field. Where TDCP is rejected we fall through to a
    very loose motion model (sigma = 10 m) so the observation alone
    drives the state. Where hybrid is missing we use a huge observation
    sigma so the prediction propagates unchanged.

    Decoupled per-coordinate scalar Kalman/RTS — fine for trajectory
    smoothing on the 0.5 m PPC threshold question; cross-axis
    correlations from satellite geometry are second-order at this scale.
    """
    from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics

    n = int(positions.shape[0])
    if n < 2:
        return positions

    sat_ecef = data["sat_ecef"]
    system_ids = data.get("system_ids")
    used_prns = data.get("used_prns") or [[] for _ in range(n)]
    carrier_phase = data.get("carrier_phase")
    sat_velocity = data.get("sat_velocity")
    clock_drift = data.get("clock_drift")
    if (
        carrier_phase is None
        or sat_velocity is None
        or clock_drift is None
        or system_ids is None
    ):
        return positions

    # TDCP velocity per consecutive pair (between epoch i and i+1).
    tdcp_v = [None] * (n - 1)
    for i in range(n - 1):
        stats.pairs_attempted += 1
        dt_i = float(times[i + 1] - times[i])
        if not np.isfinite(dt_i) or dt_i <= 0.0:
            continue
        prev_meas = _build_tdcp_measurements(
            np.asarray(sat_ecef[i], dtype=np.float64),
            np.asarray(system_ids[i], dtype=np.int32),
            list(used_prns[i]) if i < len(used_prns) else [],
            np.asarray(carrier_phase[i], dtype=np.float64),
            np.asarray(sat_velocity[i], dtype=np.float64),
            np.asarray(clock_drift[i], dtype=np.float64),
            np.asarray(positions[i], dtype=np.float64),
        )
        cur_meas = _build_tdcp_measurements(
            np.asarray(sat_ecef[i + 1], dtype=np.float64),
            np.asarray(system_ids[i + 1], dtype=np.int32),
            list(used_prns[i + 1]) if (i + 1) < len(used_prns) else [],
            np.asarray(carrier_phase[i + 1], dtype=np.float64),
            np.asarray(sat_velocity[i + 1], dtype=np.float64),
            np.asarray(clock_drift[i + 1], dtype=np.float64),
            np.asarray(positions[i + 1], dtype=np.float64),
        )
        if len(prev_meas) < int(config.tdcp_min_sats) or len(cur_meas) < int(config.tdcp_min_sats):
            stats.pairs_rejected_min_sats += 1
            continue
        v, postfit = estimate_velocity_from_tdcp_with_metrics(
            np.asarray(positions[i], dtype=np.float64),
            prev_meas,
            cur_meas,
            dt_i,
            min_sats=int(config.tdcp_min_sats),
            max_postfit_rms_m=float(config.tdcp_postfit_max_m),
        )
        if v is None:
            stats.pairs_rejected_postfit += 1
            continue
        tdcp_v[i] = (v, dt_i, float(postfit))
        stats.pairs_accepted += 1

    # Per-epoch observation sigma from hybrid Status.
    obs_sigma = np.full(n, float(config.tdcp_obs_missing_sigma_m), dtype=np.float64)
    if hybrid_pos is not None:
        anchor_set = set(int(s) for s in config.fgo_apply_hybrid_statuses)
        for i in range(n):
            t_key = round(float(times[i]), 1)
            if t_key not in hybrid_pos:
                continue
            st = hybrid_status.get(t_key) if hybrid_status is not None else None
            if st is None:
                obs_sigma[i] = float(config.tdcp_obs_loose_sigma_m)
            elif st in anchor_set:
                obs_sigma[i] = float(config.tdcp_obs_loose_sigma_m)
            else:
                obs_sigma[i] = float(config.tdcp_obs_anchor_sigma_m)

    # Anchor mask: Status=4 (or any status NOT in fgo_apply_hybrid_statuses)
    # are HARD-PINNED to the hybrid passthrough. The Kalman smoother only
    # operates on non-anchor epochs, bridging them via TDCP velocity from
    # the surrounding anchors. This protects cm-class hybrid passes from
    # being corrupted by TDCP postfit noise.
    anchor_mask = np.zeros(n, dtype=bool)
    if hybrid_pos is not None and hybrid_status is not None:
        rewrite_set = set(int(s) for s in config.fgo_apply_hybrid_statuses)
        for i in range(n):
            t_key = round(float(times[i]), 1)
            st = hybrid_status.get(t_key)
            if st is not None and st not in rewrite_set:
                anchor_mask[i] = True

    # Per-coordinate scalar Kalman forward + RTS backward.
    sigma_v = float(config.tdcp_sigma_mps)
    out = np.array(positions, dtype=np.float64, copy=True)
    for axis in range(3):
        x = np.zeros(n, dtype=np.float64)
        P = np.zeros(n, dtype=np.float64)
        # Initialise: anchors get the hybrid value at near-zero variance,
        # non-anchors take the passthrough as initial state with the
        # status-derived obs sigma.
        x[0] = positions[0, axis]
        P[0] = (1e-3) ** 2 if anchor_mask[0] else float(obs_sigma[0]) ** 2

        for i in range(1, n):
            tv = tdcp_v[i - 1]
            if tv is not None:
                v_axis = float(tv[0][axis])
                dt_i = tv[1]
                Q = (sigma_v * dt_i) ** 2
                x_pred = x[i - 1] + v_axis * dt_i
            else:
                # No TDCP: predict by inheriting last state, big process noise.
                Q = 100.0
                x_pred = x[i - 1]
            P_pred = P[i - 1] + Q
            if anchor_mask[i]:
                # Hard pin to hybrid value.
                x[i] = positions[i, axis]
                P[i] = (1e-3) ** 2
            else:
                R = obs_sigma[i] ** 2
                K = P_pred / (P_pred + R)
                z = positions[i, axis]
                x[i] = x_pred + K * (z - x_pred)
                P[i] = (1.0 - K) * P_pred

        # Backward (RTS smoother) — keep anchors hard-pinned.
        xs = np.array(x, copy=True)
        Ps = np.array(P, copy=True)
        for i in range(n - 2, -1, -1):
            if anchor_mask[i]:
                xs[i] = positions[i, axis]
                Ps[i] = (1e-3) ** 2
                continue
            tv = tdcp_v[i]
            if tv is not None:
                v_axis = float(tv[0][axis])
                dt_i = tv[1]
                Q = (sigma_v * dt_i) ** 2
                x_pred = x[i] + v_axis * dt_i
            else:
                Q = 100.0
                x_pred = x[i]
            P_pred = P[i] + Q
            if P_pred > 0.0:
                G = P[i] / P_pred
                xs[i] = x[i] + G * (xs[i + 1] - x_pred)
                Ps[i] = P[i] + G * G * (Ps[i + 1] - P_pred)
        out[:, axis] = xs

    # Hard pin anchors in the final output (defensive; the loop above
    # already keeps them at the hybrid value).
    for i in range(n):
        if anchor_mask[i]:
            out[i] = positions[i]

    return out


def _build_hybrid_velocity_guide(
    hybrid_pos: dict[float, np.ndarray],
    times: np.ndarray,
    max_dt_s: float = 0.5,
) -> dict[float, np.ndarray]:
    """Per-epoch ECEF velocity guide derived from hybrid pos finite-diffs.

    For rover TOW ``t``, the velocity is ``(hp[t] - hp[t_prev]) / (t - t_prev)``
    where ``t_prev`` is the most recent hybrid sample within ``max_dt_s``.
    Epochs without a usable predecessor receive zero velocity (so PF stays
    stationary that step rather than extrapolating an unreliable guide).
    """
    out: dict[float, np.ndarray] = {}
    if not hybrid_pos:
        return out
    sorted_keys = np.array(sorted(hybrid_pos.keys()), dtype=np.float64)
    for t in times:
        t_key = round(float(t), 1)
        hp_now = hybrid_pos.get(t_key)
        if hp_now is None:
            continue
        idx = int(np.searchsorted(sorted_keys, t_key))
        # find the most recent strictly-prior key
        prev_key = None
        for j in range(idx - 1, -1, -1):
            cand = float(sorted_keys[j])
            if cand >= t_key:
                continue
            if (t_key - cand) > max_dt_s:
                break
            prev_key = cand
            break
        if prev_key is None:
            continue
        hp_prev = hybrid_pos.get(prev_key)
        if hp_prev is None:
            continue
        dt = t_key - prev_key
        if dt <= 0.0:
            continue
        v = (hp_now - hp_prev) / dt
        if not np.all(np.isfinite(v)):
            continue
        out[t_key] = v
    return out


def _load_hybrid_pos_file(
    path: Path,
) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Parse a libgnss++ ``.pos`` file into TOW-keyed ECEF + Status lookups.

    File format (whitespace-separated, '%' comments):
        GPS_Week  GPS_TOW  X  Y  Z  Lat  Lon  Height  Status  ...

    Returns ``(positions, statuses)``. Rows with non-finite coords or
    status==0 (no fix) are dropped. Keys are rounded to 0.1 s to match
    the rover-epoch TOW grid used elsewhere.

    Note: libgnss++ Status semantics are inverted from RTKLIB; on the
    Phase 6 baseline pos files Status=4 is the high-quality (cm-class)
    bucket, while Status in {1, 3} are m-class regions where Phase 4
    FGO+LAMBDA is more likely to help than to hurt.
    """
    positions: dict[float, np.ndarray] = {}
    statuses: dict[float, int] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                tow = round(float(parts[1]), 1)
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                status = int(float(parts[8]))
            except (ValueError, IndexError):
                continue
            if status == 0:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            positions[tow] = np.array([x, y, z], dtype=np.float64)
            statuses[tow] = int(status)
    return positions, statuses


def _load_rtk_diag_file(path: Path) -> dict[float, dict[str, str]]:
    """Parse gnss_solve --diagnostics-csv output keyed by rounded TOW."""
    rows: dict[float, dict[str, str]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tow = round(float(row["tow"]), 1)
            except (KeyError, TypeError, ValueError):
                continue
            rows[tow] = row
    return rows


def _diag_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _rtkdiag_candidate_gate(
    row: dict[str, str] | None,
    *,
    ratio_min: float,
    residual_rms_max: float,
) -> bool:
    if row is None:
        return False
    try:
        output_added = int(row.get("output_added", "0")) == 1
        final_status = int(row.get("final_status", "0")) == 4
    except ValueError:
        return False
    return (
        output_added
        and final_status
        and _diag_float(row, "final_ratio") >= float(ratio_min)
        and _diag_float(row, "final_residual_rms") <= float(residual_rms_max)
    )


def _rtkdiag_candidate_sort_key(
    row: dict[str, str],
    *,
    mode: str,
) -> tuple[float, float]:
    """Rank gated RTK diagnostic candidates; smaller tuple is better."""
    ratio = _diag_float(row, "final_ratio")
    residual = _diag_float(row, "final_residual_rms")
    update_rows = _diag_float(row, "final_update_rows")
    if mode == "ratio":
        return (-ratio, residual)
    if mode == "score":
        return (residual / max(ratio, 1.0e-6), residual)
    if mode == "maxabs":
        return (_diag_float(row, "final_residual_abs_max"), residual)
    if mode == "nrows":
        return (-update_rows, residual)
    # Conservative default: prefer the tightest measurement update residual.
    return (residual, -ratio)


def _load_full_reference(path: Path) -> list[tuple[float, np.ndarray]]:
    """Load every reference.csv row in order (TOW seconds, ECEF)."""
    rows: list[tuple[float, np.ndarray]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 1)
            ecef = np.array(
                [float(row[5]), float(row[6]), float(row[7])],
                dtype=np.float64,
            )
            rows.append((tow, ecef))
    return rows


def _scale_weights_per_system(weights: np.ndarray, system_ids: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float64).copy()
    sids = np.asarray(system_ids, dtype=np.int32)
    for sys_id, scale in _SYSTEM_WEIGHT_SCALE.items():
        mask = sids == sys_id
        if mask.any() and scale != 1.0:
            out[mask] *= scale
    return out


def _pr_likelihood_weights(weights: np.ndarray, config: CTRBPFConfig) -> np.ndarray:
    """Convert PPC C/N0-like values into PF likelihood multipliers."""
    w = np.asarray(weights, dtype=np.float64)
    mode = str(config.pr_weight_mode).strip().lower()
    if mode == "raw":
        return w.astype(np.float64, copy=True)
    elif mode == "unit":
        out = np.ones_like(w, dtype=np.float64)
    elif mode in {"cn0-relative", "relative"}:
        ref = max(float(config.pr_weight_ref_cn0), 1.0)
        out = w / ref
    else:
        raise ValueError(f"unsupported pr_weight_mode: {config.pr_weight_mode}")

    lo = float(config.pr_weight_min)
    hi = float(config.pr_weight_max)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        out = np.clip(out, lo, hi)
    return out.astype(np.float64, copy=False)


def _pr_prefit_gate_mask(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    ref_pos_ecef: np.ndarray,
    *,
    gate_m: float,
    clock_quantile: float,
    min_sats: int,
    keep_best: int,
) -> np.ndarray:
    """Gate pseudoranges by robust prefit residual around a reference position."""
    n_sat = int(len(pseudoranges))
    mask = np.ones(n_sat, dtype=bool)
    if n_sat < 4 or float(gate_m) <= 0.0:
        return mask

    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    ref = np.asarray(ref_pos_ecef, dtype=np.float64).ravel()
    ranges = np.linalg.norm(sat - ref[:3], axis=1)
    residuals = pr - ranges
    finite = np.isfinite(residuals)
    if int(finite.sum()) < 4:
        return mask

    q = float(np.clip(float(clock_quantile), 0.0, 1.0))
    cb = float(np.quantile(residuals[finite], q))
    abs_prefit = np.abs(residuals - cb)
    mask = finite & (abs_prefit <= float(gate_m))

    min_keep = max(4, min(int(min_sats), n_sat))
    finite_order = np.argsort(np.where(finite, abs_prefit, np.inf))
    if int(mask.sum()) < min_keep:
        mask = np.zeros(n_sat, dtype=bool)
        mask[finite_order[:min_keep]] = True
    elif int(keep_best) > 0 and int(mask.sum()) > int(keep_best):
        gated_order = np.array([idx for idx in finite_order if mask[idx]], dtype=np.int32)
        mask = np.zeros(n_sat, dtype=bool)
        mask[gated_order[: int(keep_best)]] = True
    return mask


def _build_pf(config: CTRBPFConfig):
    from gnss_gpu import ParticleFilterDevice

    pf = ParticleFilterDevice(
        n_particles=config.n_particles,
        sigma_pos=config.sigma_pos,
        sigma_cb=config.sigma_cb,
        sigma_pr=config.sigma_pr,
        resampling="megopolis",
        seed=42,
    )
    return pf


def _run_ctrbpf_on_segment(
    data: dict,
    wls_positions: np.ndarray,
    config: CTRBPFConfig,
    dd_computer=None,
    dd_pr_computer=None,
    hybrid_pos: dict[float, np.ndarray] | None = None,
    hybrid_velocity: dict[float, np.ndarray] | None = None,
    hybrid_status: dict[float, int] | None = None,
    rtkdiag_candidates: list[
        tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]
    ] | None = None,
    imu: dict[str, np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    float,
    _PRObsStats,
    _DDStats,
    _RBPFGateStats,
    _HybridStats,
    _RTKDiagPFStats,
    _FGOStats,
    _TDCPSmootherStats,
    _ZUPTStats,
    _IMUTCStats,
    _INSTCStats,
]:
    """Run the PF on the loaded PPC segment and return (positions, ms/epoch).

    ``positions`` is shape ``(n_epochs, 3)``, aligned with ``data['times']``.
    """
    n_epochs = int(data["n_epochs"])
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    system_ids = data.get("system_ids")
    sat_velocity = data.get("sat_velocity")
    doppler_hz = data.get("doppler_hz")
    used_prns = data.get("used_prns") or [[] for _ in range(n_epochs)]
    times = np.asarray(data["times"], dtype=np.float64)
    pr_obs_stats = _PRObsStats()
    dd_stats = _DDStats()
    gate_stats = _RBPFGateStats()
    hybrid_stats = _HybridStats()
    rtkdiag_pf_stats = _RTKDiagPFStats()
    gate_active = config.enable_rbpf_velocity_kf and (
        config.rbpf_kf_gate_min_dd_pairs is not None
        or config.rbpf_kf_gate_min_ess_ratio is not None
        or config.rbpf_kf_gate_max_spread_m is not None
    )
    need_dd_compute = (
        config.enable_dd_carrier_afv
        or (gate_active and config.rbpf_kf_gate_min_dd_pairs is not None)
        or config.enable_fgo_lambda
    )
    use_hybrid = config.enable_hybrid_pu and hybrid_pos is not None
    use_rtkdiag_pf = (
        bool(config.enable_rtkdiag_pf_rescue)
        and rtkdiag_candidates is not None
        and len(rtkdiag_candidates) > 0
    )
    use_vguide = (
        config.enable_hybrid_velocity_guide
        and hybrid_velocity is not None
        and len(hybrid_velocity) > 0
    )
    fgo_dd_cache: list = [None] * n_epochs  # holds DDCarrierEpoch | None
    fgo_dd_pr_cache: list = [None] * n_epochs  # holds DDPseudorangeEpoch | None
    fgo_stats = _FGOStats()
    tdcp_stats = _TDCPSmootherStats()
    zupt_stats = _ZUPTStats()
    imu_tc_stats = _IMUTCStats()
    ins_tc_stats = _INSTCStats()

    # Phase 9b: tight-coupled IMU state (in-loop pre-integration since the
    # last Status=4 hybrid anchor). Initialized lazily on the first epoch
    # we have a usable position + heading.
    if config.enable_imu_tc and config.enable_ins_tc:
        raise ValueError("enable_imu_tc and enable_ins_tc are mutually exclusive")
    use_imu_tc = config.enable_imu_tc and imu is not None
    use_ins_tc = config.enable_ins_tc and imu is not None
    imu_t = _build_imu_segment_index(imu) if (use_imu_tc or use_ins_tc) else None
    if use_imu_tc and imu_t is None:
        use_imu_tc = False  # IMU file present but empty
    if use_ins_tc and imu_t is None:
        use_ins_tc = False
    imu_tc_anchor: _IMUAnchor | None = None
    imu_tc_anchor_acc: np.ndarray | None = None
    imu_tc_anchor_gyro: np.ndarray | None = None
    imu_tc_emit_set = set(int(s) for s in config.imu_tc_emit_pf_hybrid_statuses)
    imu_tc_anchor_set = set(int(s) for s in config.zupt_apply_hybrid_statuses)
    if use_imu_tc and hybrid_status is not None:
        # Pre-compute per-epoch IMU stats so the anchor-static check matches
        # the Phase 9a ZUPT logic (re-using the same windowing).
        imu_tc_anchor_acc, imu_tc_anchor_gyro, _ = _build_imu_per_epoch_stats(imu, times)

    ins_ekf = None
    ins_origin_ecef: np.ndarray | None = None
    ins_origin_lat: float | None = None
    ins_origin_lon: float | None = None
    ins_last_obs_t: float | None = None
    ins_particle_imu_initialized = False
    ins_particle_imu_last_t: float | None = None
    ins_particle_imu_last_accel: np.ndarray | None = None
    ins_particle_imu_last_gyro_dps: np.ndarray | None = None
    ins_gravity_ecef: np.ndarray | None = None
    ins_particle_imu_available = bool(config.ins_tc_use_particle_imu_predict)
    ins_tc_emit_set = set(int(s) for s in config.ins_tc_emit_pf_hybrid_statuses)
    if use_ins_tc:
        from gnss_gpu.ins_ekf import INSEKF, INSConfig

        ins_ekf = INSEKF(
            INSConfig(
                static_acc_low=float(config.ins_tc_align_acc_low),
                static_acc_high=float(config.ins_tc_align_acc_high),
                static_gyro_max_dps=float(config.ins_tc_align_gyro_max_dps),
                align_min_static_samples=int(config.ins_tc_align_min_samples),
                yaw_init_min_speed_mps=float(config.ins_tc_yaw_init_min_speed_mps),
            )
        )

    positions = np.zeros((n_epochs, 3), dtype=np.float64)
    init_pos = np.asarray(wls_positions[0, :3], dtype=np.float64)
    init_cb = float(wls_positions[0, 3])
    init_spread = float(config.spread_pos_init)
    if use_hybrid:
        hybrid_init = hybrid_pos.get(round(float(times[0]), 1))
        if hybrid_init is not None and np.all(np.isfinite(hybrid_init)):
            init_pos = np.asarray(hybrid_init, dtype=np.float64)
            # Hybrid is already cm-class for "fix" status epochs and m-class
            # for float, so a 5m spread captures both regimes.
            init_spread = max(5.0, float(config.hybrid_sigma_m) * 5.0)

    pf = _build_pf(config)
    pf.initialize(
        init_pos,
        clock_bias=init_cb,
        spread_pos=init_spread,
        spread_cb=config.spread_cb_init,
        velocity_init_sigma=config.velocity_init_sigma if config.enable_rbpf_velocity_kf else 0.0,
    )

    t0 = time.perf_counter()
    for i in range(n_epochs):
        t_now = float(times[i])
        t_key = round(t_now, 1)
        hp_prefetched = hybrid_pos.get(t_key) if use_hybrid else None
        hp_prefetched_valid = (
            hp_prefetched is not None
            and np.all(np.isfinite(hp_prefetched))
            and not np.all(np.asarray(hp_prefetched, dtype=np.float64) == 0.0)
        )

        if i == 0:
            dt = float(data.get("dt", 0.2))
        else:
            dt = float(times[i] - times[i - 1])
            if not np.isfinite(dt) or dt <= 0.0:
                dt = float(data.get("dt", 0.2))

        ins_epoch_prepared = False
        if use_ins_tc and ins_ekf is not None:
            if ins_origin_ecef is None and hp_prefetched_valid:
                ins_origin_ecef = np.asarray(hp_prefetched, dtype=np.float64).copy()
                lat, lon, _ = _ecef_to_llh(
                    float(ins_origin_ecef[0]),
                    float(ins_origin_ecef[1]),
                    float(ins_origin_ecef[2]),
                )
                ins_origin_lat = lat
                ins_origin_lon = lon
                ins_gravity_ecef = _gravity_ecef_at_origin(lat, lon)
                ins_ekf.initialize_position(np.zeros(3, dtype=np.float64))
                ins_last_obs_t = t_now

            if (
                ins_origin_ecef is not None
                and ins_origin_lat is not None
                and ins_origin_lon is not None
            ):
                ins_tc_stats.epochs_evaluated += 1
                if i > 0 and imu_t is not None and imu is not None:
                    imu_window = _slice_imu_samples(
                        imu,
                        imu_t,
                        float(times[i - 1]),
                        t_now,
                    )
                else:
                    imu_window = np.empty((0, 7), dtype=np.float64)

                if not ins_ekf.aligned:
                    for sample in imu_window:
                        ins_ekf.feed_imu_for_alignment(sample[0], sample[1:4], sample[4:7])
                    if ins_ekf.aligned:
                        if ins_tc_stats.aligned_at_epoch < 0:
                            ins_tc_stats.aligned_at_epoch = i
                        if hp_prefetched_valid:
                            ins_ekf.initialize_position(
                                _ecef_to_enu_at_origin(
                                    np.asarray(hp_prefetched, dtype=np.float64),
                                    ins_origin_ecef,
                                    ins_origin_lat,
                                    ins_origin_lon,
                                )
                            )
                            ins_last_obs_t = t_now

                if ins_ekf.aligned and imu_window.size > 0:
                    ins_ekf.propagate(imu_window)

                if ins_ekf.aligned and not ins_ekf.yaw_initialized:
                    v_guide_now = hybrid_velocity.get(t_key) if hybrid_velocity else None
                    if v_guide_now is not None and np.all(np.isfinite(v_guide_now)):
                        v_enu = _ecef_velocity_to_enu_at_origin(
                            np.asarray(v_guide_now, dtype=np.float64),
                            ins_origin_lat,
                            ins_origin_lon,
                        )
                        if ins_ekf.initialize_yaw_from_velocity(v_enu):
                            if ins_tc_stats.yaw_initialized_at_epoch < 0:
                                ins_tc_stats.yaw_initialized_at_epoch = i

                ins_epoch_prepared = True

        pf_predict_done = False
        if (
            use_ins_tc
            and ins_particle_imu_available
            and ins_ekf is not None
            and ins_ekf.aligned
            and ins_ekf.yaw_initialized
            and ins_origin_lat is not None
            and ins_origin_lon is not None
            and ins_gravity_ecef is not None
        ):
            if not ins_particle_imu_initialized:
                q_body_to_ecef = ins_ekf.attitude_quat_body_to_ecef(
                    ins_origin_lat,
                    ins_origin_lon,
                )
                v_body_ecef = ins_ekf.velocity_ecef(ins_origin_lat, ins_origin_lon)
                try:
                    pf.set_inertial_state(
                        q_body_to_ecef,
                        ins_ekf.accel_bias_body(),
                        ins_ekf.gyro_bias_body_radps(),
                        v_body_ecef,
                        attitude_spread_rad=float(config.ins_tc_particle_imu_att_spread_rad),
                        accel_bias_spread=float(config.ins_tc_particle_imu_acc_bias_spread),
                        gyro_bias_spread=float(config.ins_tc_particle_imu_gyro_bias_spread_rps),
                        velocity_spread=float(config.ins_tc_particle_imu_velocity_spread_mps),
                    )
                    ins_particle_imu_initialized = True
                    ins_tc_stats.particle_imu_initialized += 1
                    ins_particle_imu_last_t = t_now
                    if imu_window.size > 0:
                        last_sample = imu_window[-1]
                        ins_particle_imu_last_accel = np.asarray(last_sample[1:4], dtype=np.float64)
                        ins_particle_imu_last_gyro_dps = np.asarray(last_sample[4:7], dtype=np.float64)
                except RuntimeError:
                    ins_particle_imu_available = False
            elif ins_particle_imu_initialized:
                if imu_window.size > 0:
                    for sample in imu_window:
                        sample_t = float(sample[0])
                        sample_accel = np.asarray(sample[1:4], dtype=np.float64)
                        sample_gyro_dps = np.asarray(sample[4:7], dtype=np.float64)
                        if (
                            ins_particle_imu_last_t is not None
                            and ins_particle_imu_last_accel is not None
                            and ins_particle_imu_last_gyro_dps is not None
                            and sample_t > ins_particle_imu_last_t
                        ):
                            dt_imu = sample_t - float(ins_particle_imu_last_t)
                            pf.predict_imu(
                                ins_particle_imu_last_accel,
                                ins_particle_imu_last_gyro_dps * (math.pi / 180.0),
                                ins_gravity_ecef,
                                dt_imu,
                                sigma_pos=float(config.ins_tc_particle_imu_sigma_pos_m),
                                sigma_acc=float(config.ins_tc_particle_imu_sigma_acc_mps2),
                                sigma_gyro=float(config.ins_tc_particle_imu_sigma_gyro_rps),
                                sigma_acc_bias_rw=float(config.ins_tc_particle_imu_acc_bias_rw),
                                sigma_gyro_bias_rw=float(config.ins_tc_particle_imu_gyro_bias_rw),
                            )
                            ins_tc_stats.particle_imu_predict_used += 1
                            pf_predict_done = True
                        ins_particle_imu_last_t = sample_t
                        ins_particle_imu_last_accel = sample_accel
                        ins_particle_imu_last_gyro_dps = sample_gyro_dps
                if (
                    ins_particle_imu_last_t is not None
                    and ins_particle_imu_last_accel is not None
                    and ins_particle_imu_last_gyro_dps is not None
                    and t_now > ins_particle_imu_last_t
                ):
                    pf.predict_imu(
                        ins_particle_imu_last_accel,
                        ins_particle_imu_last_gyro_dps * (math.pi / 180.0),
                        ins_gravity_ecef,
                        t_now - float(ins_particle_imu_last_t),
                        sigma_pos=float(config.ins_tc_particle_imu_sigma_pos_m),
                        sigma_acc=float(config.ins_tc_particle_imu_sigma_acc_mps2),
                        sigma_gyro=float(config.ins_tc_particle_imu_sigma_gyro_rps),
                        sigma_acc_bias_rw=float(config.ins_tc_particle_imu_acc_bias_rw),
                        sigma_gyro_bias_rw=float(config.ins_tc_particle_imu_gyro_bias_rw),
                    )
                    ins_tc_stats.particle_imu_predict_used += 1
                    pf_predict_done = True
                    ins_particle_imu_last_t = t_now

        v_guide = None
        v_guide_from_ins = False
        if (
            not pf_predict_done
            and
            use_ins_tc
            and bool(config.ins_tc_use_motion_predict)
            and ins_ekf is not None
            and ins_ekf.aligned
            and ins_ekf.yaw_initialized
            and ins_origin_lat is not None
            and ins_origin_lon is not None
        ):
            v_ins_ecef = _enu_velocity_to_ecef_at_origin(
                ins_ekf.velocity_enu(),
                ins_origin_lat,
                ins_origin_lon,
            )
            if np.all(np.isfinite(v_ins_ecef)):
                v_guide = v_ins_ecef
                v_guide_from_ins = True
                ins_tc_stats.motion_predict_used += 1
        if use_vguide:
            if v_guide is None:
                v_guide = hybrid_velocity.get(round(float(times[i]), 1))
        if pf_predict_done:
            pass
        elif v_guide is not None and np.all(np.isfinite(v_guide)):
            pf.predict(
                velocity=np.asarray(v_guide, dtype=np.float64),
                dt=dt,
                sigma_pos=(
                    float(config.ins_tc_predict_sigma_pos_m)
                    if v_guide_from_ins
                    else None
                ),
                velocity_guide_alpha=(
                    float(config.ins_tc_predict_velocity_alpha)
                    if v_guide_from_ins
                    else None
                ),
                rbpf_velocity_kf=config.enable_rbpf_velocity_kf,
                velocity_process_noise=config.velocity_process_noise,
            )
        else:
            pf.predict(
                dt=dt,
                rbpf_velocity_kf=config.enable_rbpf_velocity_kf,
                velocity_process_noise=config.velocity_process_noise,
            )

        sat_i = np.asarray(sat_ecef[i], dtype=np.float64)
        pr_i = np.asarray(pseudoranges[i], dtype=np.float64)
        w_i = np.asarray(weights[i], dtype=np.float64)
        if system_ids is not None:
            w_i = _scale_weights_per_system(w_i, system_ids[i])
        w_i = _pr_likelihood_weights(w_i, config)

        finite = (
            np.all(np.isfinite(sat_i), axis=1)
            & np.isfinite(pr_i)
            & np.isfinite(w_i)
            & (pr_i > 1e6)
        )
        if int(finite.sum()) < 4:
            est = np.asarray(pf.estimate(), dtype=np.float64)
            positions[i] = est[:3]
            continue
        sat_i = sat_i[finite]
        pr_i = pr_i[finite]
        w_i = w_i[finite]
        sids_i = None if system_ids is None else np.asarray(system_ids[i])[finite]
        st_obs = hybrid_status.get(t_key) if hybrid_status is not None else None
        skip_pr_here = (
            st_obs is not None
            and int(st_obs) in {int(s) for s in config.pr_skip_statuses}
        )

        clock_quantile = (
            float(config.pr_gmm_clock_quantile)
            if bool(config.enable_pr_gmm)
            else 0.5
        )
        if (not skip_pr_here) and float(config.pr_prefit_gate_m) > 0.0:
            ref_mode = str(config.pr_prefit_ref).strip().lower()
            if ref_mode == "hybrid" and hp_prefetched_valid:
                prefit_ref = np.asarray(hp_prefetched, dtype=np.float64)
            else:
                prefit_ref = np.asarray(pf.estimate(), dtype=np.float64)[:3]
            gate_mask = _pr_prefit_gate_mask(
                sat_i,
                pr_i,
                prefit_ref,
                gate_m=float(config.pr_prefit_gate_m),
                clock_quantile=clock_quantile,
                min_sats=int(config.pr_prefit_gate_min_sats),
                keep_best=int(config.pr_prefit_gate_keep_best),
            )
            if int(gate_mask.sum()) >= 4 and int(gate_mask.sum()) < int(gate_mask.size):
                pr_obs_stats.prefit_epochs += 1
                pr_obs_stats.prefit_sats_kept += int(gate_mask.sum())
                pr_obs_stats.prefit_sats_dropped += int(gate_mask.size - gate_mask.sum())
                sat_i = sat_i[gate_mask]
                pr_i = pr_i[gate_mask]
                w_i = w_i[gate_mask]
                if sids_i is not None:
                    sids_i = sids_i[gate_mask]

        if (not skip_pr_here) and config.enable_correct_clock_bias and i % 5 == 0:
            pf.correct_clock_bias(sat_i, pr_i, quantile=clock_quantile)

        use_pr_gmm_here = bool(config.enable_pr_gmm)
        if use_pr_gmm_here and st_obs is not None and config.pr_gmm_statuses:
            use_pr_gmm_here = int(st_obs) in {int(s) for s in config.pr_gmm_statuses}
        if skip_pr_here:
            pr_obs_stats.epochs_skipped += 1
        elif use_pr_gmm_here:
            pf.update_gmm(
                sat_i,
                pr_i,
                weights=w_i,
                sigma_pr=float(config.sigma_pr),
                w_los=float(config.pr_gmm_w_los),
                mu_nlos=float(config.pr_gmm_mu_nlos_m),
                sigma_nlos=float(config.pr_gmm_sigma_nlos_m),
                resample=not bool(config.defer_epoch_resample),
            )
            pr_obs_stats.epochs_gmm += 1
        else:
            pf.update(
                sat_i,
                pr_i,
                weights=w_i,
                resample=not bool(config.defer_epoch_resample),
            )
            pr_obs_stats.epochs_gaussian += 1

        # Phase 1+2: compute DD once (cached) so both the Doppler-KF gate and
        # the AFV update can read its pair count.
        dd_result = None
        if need_dd_compute and dd_computer is not None:
            rover_pos_now = np.asarray(pf.estimate(), dtype=np.float64)[:3]
            sat_full = np.asarray(sat_ecef[i], dtype=np.float64)
            sids_full = (
                np.asarray(system_ids[i], dtype=np.int32)
                if system_ids is not None
                else np.zeros(int(sat_full.shape[0]), dtype=np.int32)
            )
            sat_id_strs = list(used_prns[i]) if i < len(used_prns) else []
            w_full_meas = np.asarray(weights[i], dtype=np.float64)
            measurements = _build_dd_measurements(
                sat_full,
                sids_full,
                sat_id_strs,
                w_full_meas,
                rover_pos_now,
                config.dd_systems,
            )
            if len(measurements) >= 2:
                dd_result = dd_computer.compute_dd(
                    float(times[i]),
                    measurements,
                    rover_position_approx=rover_pos_now,
                    min_common_sats=config.dd_min_pairs,
                )
                # Phase 4 cache: save the DD carrier observation for the
                # post-process FGO + LAMBDA pass after the loop.
                if (
                    config.enable_fgo_lambda
                    and dd_result is not None
                    and int(getattr(dd_result, "n_dd", 0)) > 0
                ):
                    from gnss_gpu.local_fgo import DDCarrierEpoch

                    fgo_dd_cache[i] = DDCarrierEpoch.from_result(dd_result)
                    # Phase 4 v2: also cache DD pseudorange so FGO has an
                    # absolute-position anchor (DD carrier alone is
                    # cm-relative; the integer ambiguities can absorb a
                    # systematic absolute bias and pass the ratio test).
                    if dd_pr_computer is not None:
                        try:
                            dd_pr_result = dd_pr_computer.compute_dd(
                                float(times[i]),
                                measurements,
                                rover_position_approx=rover_pos_now,
                                min_common_sats=config.dd_min_pairs,
                                rover_weights=[float(m.snr) for m in measurements],
                            )
                        except Exception:
                            dd_pr_result = None
                        if (
                            dd_pr_result is not None
                            and int(getattr(dd_pr_result, "n_dd", 0)) > 0
                        ):
                            from gnss_gpu.local_fgo import DDPseudorangeEpoch

                            fgo_dd_pr_cache[i] = DDPseudorangeEpoch.from_result(dd_pr_result)

        if config.enable_rbpf_velocity_kf and sat_velocity is not None and doppler_hz is not None:
            sv_full = np.asarray(sat_velocity[i], dtype=np.float64)
            dop_full = np.asarray(doppler_hz[i], dtype=np.float64)
            sat_full = np.asarray(sat_ecef[i], dtype=np.float64)
            w_full = np.asarray(weights[i], dtype=np.float64)
            if system_ids is not None:
                w_full = _scale_weights_per_system(w_full, system_ids[i])
            dop_finite = (
                np.isfinite(dop_full)
                & np.all(np.isfinite(sv_full), axis=1)
                & np.all(np.isfinite(sat_full), axis=1)
                & np.isfinite(w_full)
            )
            if int(dop_finite.sum()) >= 4:
                if gate_active:
                    gate_stats.epochs_attempted += 1
                gate_skipped = False
                if config.rbpf_kf_gate_min_dd_pairs is not None:
                    n_dd_now = int(getattr(dd_result, "n_dd", 0)) if dd_result is not None else 0
                    if n_dd_now < int(config.rbpf_kf_gate_min_dd_pairs):
                        gate_stats.skipped_min_dd_pairs += 1
                        gate_skipped = True
                if not gate_skipped and config.rbpf_kf_gate_min_ess_ratio is not None:
                    ess = float(pf.get_ess())
                    ess_ratio = ess / max(int(config.n_particles), 1)
                    if ess_ratio < float(config.rbpf_kf_gate_min_ess_ratio):
                        gate_stats.skipped_min_ess_ratio += 1
                        gate_skipped = True
                if not gate_skipped and config.rbpf_kf_gate_max_spread_m is not None:
                    spread = float(pf.get_position_spread())
                    if spread > float(config.rbpf_kf_gate_max_spread_m):
                        gate_stats.skipped_max_spread += 1
                        gate_skipped = True
                if not gate_skipped:
                    pf.update_doppler_kf(
                        sat_full[dop_finite],
                        sv_full[dop_finite],
                        dop_full[dop_finite],
                        weights=w_full[dop_finite],
                        wavelength=_GPS_L1_WAVELENGTH_M,
                        sigma_mps=config.sigma_doppler_mps,
                        resample=not bool(config.defer_epoch_resample),
                    )
                    if gate_active:
                        gate_stats.epochs_applied += 1

        if config.enable_dd_carrier_afv and dd_computer is not None:
            dd_stats.epochs_attempted += 1
            if (
                dd_result is not None
                and int(getattr(dd_result, "n_dd", 0)) >= int(config.dd_min_pairs_update)
            ):
                pf.update_dd_carrier_afv(
                    dd_result,
                    sigma_cycles=float(config.dd_sigma_cycles),
                    resample=not bool(config.defer_epoch_resample),
                )
                dd_stats.epochs_applied += 1
                dd_stats.pairs_total += int(dd_result.n_dd)

        if config.enable_position_update:
            ref = wls_positions[i, :3]
            if np.all(np.isfinite(ref)) and not np.all(ref == 0.0):
                pf.position_update(ref, sigma_pos=config.position_update_sigma_m)

        # Phase 6: hybrid (libgnss++ 50.91%) blend. The runtime PF cannot
        # currently follow the trajectory without a velocity guide (predict
        # step keeps particles stationary, hybrid PU at sigma~1m cannot
        # rescue a cloud that is several meters from the moving hybrid
        # baseline). MVP behavior: emit the hybrid position directly when
        # available, and only fall through to the PF estimate when hybrid
        # is missing for that TOW. This gives us the hybrid baseline as the
        # floor and lets the PF cover hybrid-gap epochs. A future revision
        # will add a velocity guide so the PF can independently correct
        # hybrid (e.g., DD-carrier-driven cm-pull), but until then the PF
        # alone diverges over hundreds of meters.
        hp = None
        if use_hybrid:
            hybrid_stats.epochs_attempted += 1
            hp = hp_prefetched
            if hp is None:
                hybrid_stats.epochs_lookup_missing += 1
            elif np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                # Phase 9b: relax the hybrid PU sigma on m-class Status
                # epochs so IMU pre-integration can dominate. Without this
                # the tight 1m hybrid PU clamps the cloud to the (wrong-by-
                # several-meters) hybrid passthrough and the IMU rescue
                # never surfaces in pf.estimate().
                hybrid_sigma_now = float(config.hybrid_sigma_m)
                st_now_pu = hybrid_status.get(round(float(times[i]), 1)) if hybrid_status is not None else None
                if (
                    config.enable_pr_gmm
                    and float(config.pr_gmm_hybrid_loose_sigma_m) > 0.0
                ):
                    gmm_status_match = True
                    if st_now_pu is not None and config.pr_gmm_statuses:
                        gmm_status_match = int(st_now_pu) in {int(s) for s in config.pr_gmm_statuses}
                    if gmm_status_match:
                        hybrid_sigma_now = float(config.pr_gmm_hybrid_loose_sigma_m)
                if (
                    config.enable_imu_tc
                    and float(config.imu_tc_hybrid_loose_sigma_m) > 0.0
                    and hybrid_status is not None
                ):
                    if (
                        st_now_pu is not None
                        and int(st_now_pu) in {int(s) for s in config.imu_tc_emit_pf_hybrid_statuses}
                    ):
                        hybrid_sigma_now = float(config.imu_tc_hybrid_loose_sigma_m)
                pf.position_update(hp, sigma_pos=hybrid_sigma_now)
                hybrid_stats.epochs_applied += 1

        rtkdiag_pf_emit_here = False
        rtkdiag_pf_ref: np.ndarray | None = None
        if use_rtkdiag_pf:
            rtkdiag_pf_stats.epochs_evaluated += 1
            best_candidate: tuple[str, np.ndarray, dict[str, str]] | None = None
            best_key: tuple[float, float] | None = None
            gated_options = 0
            for label, candidate_pos, candidate_diag in rtkdiag_candidates or []:
                diag_row = candidate_diag.get(t_key)
                if not _rtkdiag_candidate_gate(
                    diag_row,
                    ratio_min=float(config.rtkdiag_candidate_ratio_min),
                    residual_rms_max=float(config.rtkdiag_candidate_residual_rms_max),
                ):
                    continue
                gated_options += 1
                cand = candidate_pos.get(t_key)
                if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                    continue
                select_mode = str(config.rtkdiag_candidate_select_mode)
                if select_mode == "hybrid_anchor" and hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                    cand_arr = np.asarray(cand, dtype=np.float64)
                    dist_to_hyb = float(np.linalg.norm(cand_arr - np.asarray(hp, dtype=np.float64)))
                    candidate_key = (dist_to_hyb, _diag_float(diag_row, "final_residual_rms"))
                else:
                    if select_mode == "hybrid_anchor":
                        # Hybrid floor missing — fall back to residual.
                        candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode="residual")
                    else:
                        candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode=select_mode)
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_candidate = (label, np.asarray(cand, dtype=np.float64), diag_row)

            if gated_options > 0:
                rtkdiag_pf_stats.gate_pass += 1
                rtkdiag_pf_stats.candidate_options_total += int(gated_options)
                if best_candidate is None:
                    rtkdiag_pf_stats.candidate_missing += 1
                else:
                    label, selected_pos, _selected_diag = best_candidate
                    rtkdiag_pf_ref = selected_pos
                    rtkdiag_pf_stats.selected_counts[label] = (
                        rtkdiag_pf_stats.selected_counts.get(label, 0) + 1
                    )
                    recenter_max = float(config.rtkdiag_candidate_recenter_max_shift_m)
                    if recenter_max > 0.0:
                        shift_norm, recentered = pf.recenter_position(
                            rtkdiag_pf_ref,
                            max_shift_m=recenter_max,
                        )
                        if recentered:
                            rtkdiag_pf_stats.recenter_applied += 1
                        else:
                            rtkdiag_pf_stats.recenter_skipped += 1
                    pf.position_update(
                        rtkdiag_pf_ref,
                        sigma_pos=max(float(config.rtkdiag_candidate_sigma_m), 0.01),
                    )
                    rtkdiag_pf_stats.pu_applied += 1
                    rtkdiag_pf_emit_here = True

        # Phase 9b: tight-coupled IMU. After hybrid PU but BEFORE estimate /
        # emission, integrate IMU since the anchor and apply a per-particle
        # position pseudo-observation. This is the "tight" piece: every
        # particle gets an IMU log-likelihood, not a global rewrite. We
        # also evaluate whether to emit the PF estimate (vs hybrid) on
        # this epoch based on the hybrid Status.
        imu_tc_emit_pf_here = False
        ins_tc_emit_pf_here = False
        if use_imu_tc:
            t_now = float(times[i])
            st_now = (
                hybrid_status.get(round(t_now, 1)) if hybrid_status is not None else None
            )
            anchor_eligible_here = (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(hp == 0.0)
                and (
                    st_now is None
                    or imu_tc_anchor_set
                    and st_now not in {int(s) for s in config.imu_tc_emit_pf_hybrid_statuses}
                )
            )
            # Lazily seed the anchor on the very first epoch with a usable
            # position so the loop has something to integrate against.
            if imu_tc_anchor is None:
                seed_pos = None
                if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                    seed_pos = np.asarray(hp, dtype=np.float64)
                elif np.all(np.isfinite(positions[i])) and not np.all(positions[i] == 0.0):
                    seed_pos = np.asarray(positions[i], dtype=np.float64)
                if seed_pos is not None:
                    imu_tc_anchor = _IMUAnchor(
                        pos_ecef=seed_pos.copy(),
                        time=t_now,
                        yaw_enu_rad=0.0,
                        valid_yaw=False,
                        velocity_enu=np.zeros(3, dtype=np.float64),
                    )
                    imu_tc_stats.anchor_resets += 1

            if imu_tc_anchor is not None:
                imu_tc_stats.epochs_evaluated += 1

                # Refresh yaw from a recent velocity when we don't have one yet.
                if not imu_tc_anchor.valid_yaw:
                    v_guide_now = (
                        hybrid_velocity.get(round(t_now, 1)) if hybrid_velocity else None
                    )
                    if v_guide_now is not None and np.all(np.isfinite(v_guide_now)):
                        v_enu = _ecef_velocity_to_enu(imu_tc_anchor.pos_ecef, v_guide_now)
                        yaw, valid = _heading_from_velocity_enu(v_enu)
                        if valid:
                            imu_tc_anchor = _IMUAnchor(
                                pos_ecef=imu_tc_anchor.pos_ecef,
                                time=imu_tc_anchor.time,
                                yaw_enu_rad=yaw,
                                valid_yaw=True,
                                velocity_enu=v_enu,
                            )

                dr_seconds = t_now - imu_tc_anchor.time
                imu_tc_stats.record_dr_seconds(dr_seconds)

                run_pu = imu_tc_anchor.valid_yaw
                if not run_pu:
                    imu_tc_stats.pu_skipped_no_anchor += 1
                if run_pu and dr_seconds > float(config.imu_tc_max_dr_seconds):
                    imu_tc_stats.pu_skipped_dr_too_long += 1
                    run_pu = False

                imu_pred_pos = None
                if run_pu and imu_t is not None and imu is not None:
                    delta_enu, _vel_enu_end, n_samp = _integrate_imu_between(
                        imu,
                        imu_t,
                        imu_tc_anchor.time,
                        t_now,
                        imu_tc_anchor.yaw_enu_rad,
                        imu_tc_anchor.velocity_enu,
                    )
                    if n_samp == 0:
                        imu_tc_stats.pu_skipped_no_imu += 1
                        run_pu = False
                    else:
                        delta_ecef = _enu_delta_to_ecef(imu_tc_anchor.pos_ecef, delta_enu)
                        imu_pred_pos = imu_tc_anchor.pos_ecef + delta_ecef
                        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                            disagreement = float(np.linalg.norm(imu_pred_pos - hp))
                            if disagreement > float(config.imu_tc_max_disagreement_m):
                                imu_tc_stats.pu_skipped_disagreement += 1
                                run_pu = False

                if run_pu and imu_pred_pos is not None:
                    sigma_imu = (
                        float(config.imu_tc_pos_sigma_base_m)
                        + float(config.imu_tc_pos_sigma_per_s) * float(max(dr_seconds, 0.0))
                    )
                    pf.position_update(imu_pred_pos, sigma_pos=max(sigma_imu, 0.05))
                    imu_tc_stats.pu_applied += 1

                # Decide emission policy for this epoch (post-IMU-PU).
                if (
                    st_now is not None
                    and int(st_now) in imu_tc_emit_set
                ):
                    imu_tc_emit_pf_here = True

                # Anchor-reset on Status=4 (or anything NOT in the emit-PF set).
                if (
                    anchor_eligible_here
                    and st_now is not None
                    and int(st_now) not in imu_tc_emit_set
                ):
                    # Derive anchor velocity from hybrid velocity guide when
                    # available; otherwise fall back to whatever the previous
                    # anchor velocity was (will get refreshed lazily).
                    v_anchor_enu = imu_tc_anchor.velocity_enu
                    yaw_anchor = imu_tc_anchor.yaw_enu_rad
                    valid_yaw_anchor = imu_tc_anchor.valid_yaw
                    v_guide_now = (
                        hybrid_velocity.get(round(t_now, 1)) if hybrid_velocity else None
                    )
                    if v_guide_now is not None and np.all(np.isfinite(v_guide_now)):
                        v_anchor_enu = _ecef_velocity_to_enu(
                            np.asarray(hp, dtype=np.float64), v_guide_now
                        )
                        yaw_new, valid_new = _heading_from_velocity_enu(v_anchor_enu)
                        if valid_new:
                            yaw_anchor = yaw_new
                            valid_yaw_anchor = True
                    # Static check: if IMU says vehicle is stopped at this
                    # anchor, force zero velocity (matches Phase 9a logic).
                    is_static_anchor = False
                    if imu_tc_anchor_acc is not None and imu_tc_anchor_gyro is not None and i < imu_tc_anchor_acc.size:
                        a = imu_tc_anchor_acc[i]
                        g = imu_tc_anchor_gyro[i]
                        if (
                            np.isfinite(a)
                            and np.isfinite(g)
                            and float(config.imu_tc_anchor_static_acc_low_mps2) <= a <= float(config.imu_tc_anchor_static_acc_high_mps2)
                            and g <= float(config.imu_tc_anchor_static_gyro_max_dps)
                        ):
                            is_static_anchor = True
                            v_anchor_enu = np.zeros(3, dtype=np.float64)
                    imu_tc_anchor = _IMUAnchor(
                        pos_ecef=np.asarray(hp, dtype=np.float64).copy(),
                        time=t_now,
                        yaw_enu_rad=yaw_anchor,
                        valid_yaw=valid_yaw_anchor,
                        velocity_enu=v_anchor_enu,
                    )
                    imu_tc_stats.anchor_resets += 1
                    if is_static_anchor:
                        imu_tc_stats.anchor_resets_static += 1

        # Phase 9c: full INS-GNSS EKF. The EKF owns attitude and IMU bias
        # estimation in local ENU; the PF still receives only a position
        # pseudo-observation through the existing GPU position_update path.
        if use_ins_tc and ins_ekf is not None:
            t_now = float(times[i])
            t_key = round(t_now, 1)
            st_now = hybrid_status.get(t_key) if hybrid_status is not None else None
            hp_valid = (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(np.asarray(hp, dtype=np.float64) == 0.0)
            )

            if (
                ins_origin_ecef is not None
                and ins_origin_lat is not None
                and ins_origin_lon is not None
            ):
                if ins_ekf.aligned and hp_valid:
                    p_meas_enu = _ecef_to_enu_at_origin(
                        np.asarray(hp, dtype=np.float64),
                        ins_origin_ecef,
                        ins_origin_lat,
                        ins_origin_lon,
                    )
                    if st_now == 4 and float(config.ins_tc_obs_status_4_sigma_m) > 0.0:
                        sigma = float(config.ins_tc_obs_status_4_sigma_m)
                        ins_ekf.update_position_enu(p_meas_enu, (sigma, sigma, sigma * 2.0))
                        ins_last_obs_t = t_now
                        ins_tc_stats.obs_status_4_used += 1
                        if bool(config.ins_tc_recenter_status4):
                            shift_norm, recentered = pf.recenter_position(
                                np.asarray(hp, dtype=np.float64),
                                max_shift_m=float(config.ins_tc_recenter_max_shift_m),
                            )
                            if recentered:
                                ins_tc_stats.recenter_applied += 1
                                ins_tc_stats.recenter_shift_sum_m += float(shift_norm)
                            else:
                                ins_tc_stats.recenter_skipped += 1
                    elif st_now == 3 and float(config.ins_tc_obs_status_3_sigma_m) > 0.0:
                        sigma = float(config.ins_tc_obs_status_3_sigma_m)
                        ins_ekf.update_position_enu(p_meas_enu, (sigma, sigma, sigma * 2.0))
                        ins_last_obs_t = t_now
                        ins_tc_stats.obs_status_3_used += 1

                ins_valid_for_emit = False
                if not ins_ekf.aligned:
                    ins_tc_stats.pu_skipped_not_aligned += 1
                elif not ins_ekf.yaw_initialized:
                    ins_tc_stats.pu_skipped_no_yaw += 1
                else:
                    dr_seconds = (
                        t_now - ins_last_obs_t
                        if ins_last_obs_t is not None
                        else float("inf")
                    )
                    if dr_seconds > float(config.ins_tc_max_dr_seconds):
                        ins_tc_stats.pu_skipped_dr_too_long += 1
                    else:
                        p_ins_ecef = ins_ekf.position_ecef(
                            ins_origin_ecef,
                            ins_origin_lat,
                            ins_origin_lon,
                        )
                        if hp_valid and float(np.linalg.norm(p_ins_ecef - hp)) > float(config.ins_tc_max_disagreement_m):
                            ins_tc_stats.pu_skipped_disagreement += 1
                        else:
                            sigma_ins = max(
                                float(config.ins_tc_pf_pu_floor_sigma_m),
                                min(
                                    float(config.ins_tc_pf_pu_ceiling_sigma_m),
                                    float(ins_ekf.position_sigma_m()),
                                ),
                            )
                            pf.position_update(p_ins_ecef, sigma_pos=sigma_ins)
                            if not bool(config.defer_epoch_resample):
                                pf.resample_if_needed()
                            ins_tc_stats.pu_applied += 1
                            ins_valid_for_emit = True

                if (
                    ins_valid_for_emit
                    and st_now is not None
                    and int(st_now) in ins_tc_emit_set
                ):
                    ins_tc_emit_pf_here = True

        est = np.asarray(pf.estimate(), dtype=np.float64)
        if ins_tc_emit_pf_here:
            if (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(hp == 0.0)
                and float(np.linalg.norm(est[:3] - hp)) > float(config.ins_tc_emit_max_diff_m)
            ):
                positions[i] = np.asarray(hp, dtype=np.float64)
                ins_tc_stats.emit_skipped_pf_drift += 1
            else:
                positions[i] = est[:3]
                ins_tc_stats.emit_pf_estimate += 1
        elif imu_tc_emit_pf_here:
            # Phase 9b: emit PF estimate on Status in emit_pf set, as long as
            # the PF didn't wander too far from hybrid (cloud-collapse guard).
            if (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(hp == 0.0)
                and float(np.linalg.norm(est[:3] - hp)) > float(config.imu_tc_emit_max_diff_m)
            ):
                positions[i] = np.asarray(hp, dtype=np.float64)
                imu_tc_stats.emit_skipped_pf_drift += 1
            else:
                positions[i] = est[:3]
                imu_tc_stats.emit_pf_estimate += 1
        elif rtkdiag_pf_emit_here and rtkdiag_pf_ref is not None:
            # Phase 10i: a trusted relaxed-RTK candidate is a PF
            # pseudo-observation. The conservative default emits the PF only
            # when the weighted cloud follows the candidate; newer modes let
            # diagnostics-passing candidates fill the epochs the PF guard would
            # otherwise throw back to the hybrid floor.
            emit_mode = str(config.rtkdiag_candidate_emit_mode)
            candidate_delta_m = float(np.linalg.norm(est[:3] - rtkdiag_pf_ref))
            pf_close_to_candidate = candidate_delta_m <= float(
                config.rtkdiag_candidate_emit_max_diff_m
            )
            if emit_mode == "candidate":
                positions[i] = np.asarray(rtkdiag_pf_ref, dtype=np.float64)
                rtkdiag_pf_stats.emit_candidate += 1
            elif pf_close_to_candidate:
                positions[i] = est[:3]
                rtkdiag_pf_stats.emit_pf_estimate += 1
            elif emit_mode == "candidate-on-drift":
                positions[i] = np.asarray(rtkdiag_pf_ref, dtype=np.float64)
                rtkdiag_pf_stats.emit_candidate += 1
                rtkdiag_pf_stats.emit_skipped_pf_drift += 1
            else:
                if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                    positions[i] = np.asarray(hp, dtype=np.float64)
                else:
                    positions[i] = est[:3]
                rtkdiag_pf_stats.emit_skipped_pf_drift += 1
        elif (
            use_hybrid
            and not config.hybrid_emit_pf_estimate
            and hp is not None
            and np.all(np.isfinite(hp))
            and not np.all(hp == 0.0)
        ):
            # Phase 6 passthrough: trust hybrid as the floor.
            positions[i] = np.asarray(hp, dtype=np.float64)
        else:
            # Phase 7 / non-hybrid: emit the PF's own weighted-mean estimate
            # so any DD-AFV / Doppler-KF correction shows up in the score.
            positions[i] = est[:3]

        if config.defer_epoch_resample and pf.resample_if_needed():
            pr_obs_stats.deferred_resample_epochs += 1

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)

    # Phase 9a: ZUPT before any smoother / FGO so jitter on stops is
    # damped first.
    if config.enable_zupt:
        positions = _apply_zupt(
            positions=positions,
            times=times,
            imu=imu,
            config=config,
            hybrid_status=hybrid_status,
            stats=zupt_stats,
        )

    # Phase 8: TDCP-anchored hybrid smoother. Runs BEFORE Phase 4 so
    # Phase 4 can operate on the smoothed trajectory. With TDCP off this
    # is a no-op.
    if config.enable_tdcp_smoother:
        positions = _apply_tdcp_smoother(
            positions=positions,
            times=times,
            data=data,
            config=config,
            hybrid_pos=hybrid_pos,
            hybrid_status=hybrid_status,
            stats=tdcp_stats,
        )

    # Phase 4: post-process FGO + LAMBDA partial fix over sliding windows.
    if config.enable_fgo_lambda and any(c is not None for c in fgo_dd_cache):
        # Build per-epoch indices that are eligible for FGO replacement
        # based on the hybrid Status gate (C2). When the gate is empty or
        # no hybrid status was loaded, every epoch is eligible.
        protect_indices: set[int] = set()
        if hybrid_status is not None and config.fgo_apply_hybrid_statuses:
            allowed = set(int(s) for s in config.fgo_apply_hybrid_statuses)
            for i in range(n_epochs):
                st = hybrid_status.get(round(float(times[i]), 1))
                # Protect epoch (keep hybrid passthrough) when it has a
                # status NOT in the allowed-rewrite set.
                if st is not None and st not in allowed:
                    protect_indices.add(i)

        # D2: build per-epoch prior sigma array. Status values NOT in the
        # apply set get the tight anchor sigma; Status values in the apply
        # set get the loose sigma; missing-status epochs default to loose.
        prior_sigmas_arr: np.ndarray | None = None
        if (
            hybrid_status is not None
            and float(config.fgo_anchor_sigma_m) > 0.0
            and config.fgo_apply_hybrid_statuses
        ):
            allowed = set(int(s) for s in config.fgo_apply_hybrid_statuses)
            prior_sigmas_arr = np.full(n_epochs, float(config.fgo_loose_sigma_m), dtype=np.float64)
            for i in range(n_epochs):
                st = hybrid_status.get(round(float(times[i]), 1))
                if st is not None and st not in allowed:
                    prior_sigmas_arr[i] = float(config.fgo_anchor_sigma_m)

        positions = _apply_fgo_lambda(
            positions=positions,
            dd_cache=fgo_dd_cache,
            dd_pr_cache=fgo_dd_pr_cache,
            config=config,
            stats=fgo_stats,
            protect_indices=protect_indices,
            prior_sigmas=prior_sigmas_arr,
        )

    if ins_ekf is not None:
        ins_tc_stats.final_acc_bias_norm = float(np.linalg.norm(ins_ekf.b_a))
        ins_tc_stats.final_gyro_bias_norm_dps = float(
            np.linalg.norm(ins_ekf.b_g) * 180.0 / math.pi
        )
        ins_tc_stats.final_pos_sigma_m = float(ins_ekf.position_sigma_m())

    return (
        positions,
        ms_per_epoch,
        pr_obs_stats,
        dd_stats,
        gate_stats,
        hybrid_stats,
        rtkdiag_pf_stats,
        fgo_stats,
        tdcp_stats,
        zupt_stats,
        imu_tc_stats,
        ins_tc_stats,
    )


def _apply_fgo_lambda(
    *,
    positions: np.ndarray,
    dd_cache: list,
    dd_pr_cache: list | None,
    config: CTRBPFConfig,
    stats: _FGOStats,
    protect_indices: set[int] | None = None,
    prior_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Slide a window over the trajectory and run solve_local_fgo_with_lambda.

    Replaces ``positions[start:end+1]`` with the FGO output whenever LAMBDA
    accepts at least ``fgo_min_fixed_to_apply`` integer fixes via the ratio
    test. The replacement is per-window; overlapping windows write the most
    recent solve, so a stride < window_size effectively re-solves earlier
    epochs with potentially better integer support.
    """
    from gnss_gpu.local_fgo import (
        LambdaFixConfig,
        LocalFgoConfig,
        LocalFgoProblem,
        LocalFgoWindow,
        solve_local_fgo_with_lambda,
    )

    n = int(positions.shape[0])
    win_size = max(2, int(config.fgo_window_size))
    stride = max(1, int(config.fgo_window_stride))
    if win_size > n:
        return positions

    base_cfg = LocalFgoConfig(
        prior_sigma_m=float(config.fgo_prior_sigma_m),
        dd_sigma_cycles=float(config.fgo_dd_sigma_cycles),
    )
    lam_cfg = LambdaFixConfig(
        ratio_threshold=float(config.fgo_lambda_ratio),
        min_epochs=int(config.fgo_lambda_min_epochs),
    )

    out = positions.copy()
    start = 0
    while start + win_size <= n:
        end = start + win_size - 1
        window_dd = dd_cache[start : end + 1]
        n_dd_epochs = sum(1 for d in window_dd if d is not None)
        if n_dd_epochs < int(config.fgo_lambda_min_epochs):
            start += stride
            continue

        stats.windows_attempted += 1
        slice_pos = np.asarray(out[start : end + 1], dtype=np.float64).copy()
        window_dd_pr = (
            dd_pr_cache[start : end + 1] if dd_pr_cache is not None else None
        )
        slice_prior_sigmas = (
            np.asarray(prior_sigmas[start : end + 1], dtype=np.float64).copy()
            if prior_sigmas is not None
            else None
        )
        problem = LocalFgoProblem(
            initial_positions_ecef=slice_pos,
            window=LocalFgoWindow(0, win_size - 1),
            dd_carrier=window_dd,
            dd_pseudorange=window_dd_pr,
            prior_positions_ecef=slice_pos,
            prior_sigmas_m=slice_prior_sigmas,
        )
        try:
            result, summary = solve_local_fgo_with_lambda(problem, base_cfg, lam_cfg)
        except Exception:
            start += stride
            continue
        stats.windows_solved += 1
        n_fixed = int(summary.get("n_fixed", 0))
        n_fixed_obs = int(summary.get("n_fixed_observations", 0))
        stats.n_fixed_total += n_fixed
        stats.n_fixed_observations_total += n_fixed_obs
        if n_fixed >= int(config.fgo_min_fixed_to_apply):
            new_positions = np.asarray(result.positions_ecef, dtype=np.float64)
            if (
                new_positions.shape == slice_pos.shape
                and np.all(np.isfinite(new_positions))
            ):
                # C2 per-epoch gate: only overwrite indices that are NOT
                # protected (i.e., not Status=4 / cm-class hybrid).
                # D2b: also skip rewrites where the FGO output disagrees
                # with the hybrid passthrough by less than fgo_min_correction_m;
                # those small rewrites tend to nudge cm-class hybrid passes
                # across the 0.5 m PPC threshold without recovering far-fail
                # epochs.
                min_corr = float(config.fgo_min_correction_m)
                replaced = 0
                for rel_i in range(win_size):
                    abs_i = start + rel_i
                    if protect_indices is not None and abs_i in protect_indices:
                        continue
                    if min_corr > 0.0:
                        delta = float(np.linalg.norm(new_positions[rel_i] - slice_pos[rel_i]))
                        if delta < min_corr:
                            continue
                    out[abs_i] = new_positions[rel_i]
                    replaced += 1
                if replaced > 0:
                    stats.windows_applied += 1
                    stats.epochs_replaced += replaced
        start += stride

    return out


def _write_pos_file(
    path: Path,
    times: np.ndarray,
    positions: np.ndarray,
    status: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("% CT-RBPF-FGO PPC port (gnss_gpu)\n")
        fh.write("% GPST                tow         x-ecef(m)        y-ecef(m)        z-ecef(m)   Q  ns   sdx    sdy    sdz   age  ratio\n")
        for tow, pos in zip(times, positions, strict=True):
            fh.write(
                f"0 {float(tow):14.4f} "
                f"{pos[0]:16.4f} {pos[1]:16.4f} {pos[2]:16.4f}  "
                f"   1   0  0.000  0.000  0.000  0.00  0.0   {status}\n"
            )


def _parse_path_list(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _parse_label_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_run_label_blocks(raw: str) -> dict[tuple[str, str], set[str]]:
    blocks: dict[tuple[str, str], set[str]] = {}
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"invalid run label block spec {spec!r}; expected city/run=label+label"
            )
        run_key, labels_raw = spec.split("=", 1)
        if "/" not in run_key:
            raise ValueError(
                f"invalid run label block key {run_key!r}; expected city/run"
            )
        city, run = (part.strip() for part in run_key.split("/", 1))
        labels = {p.strip() for p in labels_raw.replace(",", "+").split("+") if p.strip()}
        if not city or not run or not labels:
            raise ValueError(
                f"invalid run label block spec {spec!r}; expected city/run=label+label"
            )
        blocks[(city, run)] = set(labels)
    return blocks


def _emission_count(
    full_ref_rows: list[tuple[float, np.ndarray]],
    times_used: np.ndarray,
) -> int:
    pos_map = {round(float(t), 1): True for t in times_used}
    return int(sum(1 for tow, _ in full_ref_rows if pos_map.get(round(float(tow), 1))))


def _config_variants(args: argparse.Namespace) -> list[CTRBPFConfig]:
    variants: list[CTRBPFConfig] = []
    dd_systems = tuple(s.strip() for s in args.dd_systems.split(",") if s.strip())
    base = dict(
        n_particles=args.n_particles,
        sigma_pr=args.sigma_pr,
        pr_gmm_statuses=tuple(
            int(s.strip()) for s in args.pr_gmm_statuses.split(",") if s.strip()
        ),
        pr_gmm_w_los=args.pr_gmm_w_los,
        pr_gmm_mu_nlos_m=args.pr_gmm_mu_nlos_m,
        pr_gmm_sigma_nlos_m=args.pr_gmm_sigma_nlos_m,
        pr_gmm_hybrid_loose_sigma_m=args.pr_gmm_hybrid_loose_sigma_m,
        pr_gmm_clock_quantile=args.pr_gmm_clock_quantile,
        pr_weight_mode=args.pr_weight_mode,
        pr_weight_ref_cn0=args.pr_weight_ref_cn0,
        pr_weight_min=args.pr_weight_min,
        pr_weight_max=args.pr_weight_max,
        pr_prefit_gate_m=args.pr_prefit_gate_m,
        pr_prefit_gate_min_sats=args.pr_prefit_gate_min_sats,
        pr_prefit_gate_keep_best=args.pr_prefit_gate_keep_best,
        pr_prefit_ref=args.pr_prefit_ref,
        pr_skip_statuses=tuple(
            int(s.strip()) for s in args.pr_skip_statuses.split(",") if s.strip()
        ),
        defer_epoch_resample=bool(args.defer_epoch_resample),
        rtkdiag_candidate_sigma_m=args.rtkdiag_candidate_sigma_m,
        rtkdiag_candidate_ratio_min=args.rtkdiag_candidate_ratio_min,
        rtkdiag_candidate_residual_rms_max=args.rtkdiag_candidate_residual_rms_max,
        rtkdiag_candidate_emit_max_diff_m=args.rtkdiag_candidate_emit_max_diff_m,
        rtkdiag_candidate_recenter_max_shift_m=args.rtkdiag_candidate_recenter_max_shift_m,
        rtkdiag_candidate_select_mode=args.rtkdiag_candidate_select_mode,
        rtkdiag_candidate_emit_mode=args.rtkdiag_candidate_emit_mode,
        sigma_pos=args.sigma_pos,
        sigma_cb=args.sigma_cb,
        spread_pos_init=args.spread_pos_init,
        spread_cb_init=args.spread_cb_init,
        sigma_doppler_mps=args.sigma_doppler_mps,
        velocity_init_sigma=args.velocity_init_sigma,
        velocity_process_noise=args.velocity_process_noise,
        position_update_sigma_m=args.position_update_sigma_m,
        enable_correct_clock_bias=not args.disable_correct_clock_bias,
        dd_sigma_cycles=args.dd_sigma_cycles,
        dd_min_pairs=args.dd_min_pairs,
        dd_min_pairs_update=args.dd_min_pairs_update,
        dd_systems=dd_systems,
        dd_base_interp=bool(args.dd_base_interp),
        hybrid_sigma_m=args.hybrid_sigma_m,
        fgo_window_size=args.fgo_window_size,
        fgo_window_stride=args.fgo_window_stride,
        fgo_lambda_ratio=args.fgo_lambda_ratio,
        fgo_lambda_min_epochs=args.fgo_lambda_min_epochs,
        fgo_min_fixed_to_apply=args.fgo_min_fixed_to_apply,
        fgo_prior_sigma_m=args.fgo_prior_sigma_m,
        fgo_dd_sigma_cycles=args.fgo_dd_sigma_cycles,
        fgo_apply_hybrid_statuses=tuple(
            int(s.strip()) for s in args.fgo_apply_hybrid_statuses.split(",") if s.strip()
        ),
        fgo_anchor_sigma_m=args.fgo_anchor_sigma_m,
        fgo_loose_sigma_m=args.fgo_loose_sigma_m,
        fgo_min_correction_m=args.fgo_min_correction_m,
        tdcp_sigma_mps=args.tdcp_sigma_mps,
        tdcp_postfit_max_m=args.tdcp_postfit_max_m,
        tdcp_min_sats=args.tdcp_min_sats,
        tdcp_obs_anchor_sigma_m=args.tdcp_obs_anchor_sigma_m,
        tdcp_obs_loose_sigma_m=args.tdcp_obs_loose_sigma_m,
        tdcp_obs_missing_sigma_m=args.tdcp_obs_missing_sigma_m,
        zupt_acc_norm_low_mps2=args.zupt_acc_norm_low_mps2,
        zupt_acc_norm_high_mps2=args.zupt_acc_norm_high_mps2,
        zupt_gyro_norm_max_dps=args.zupt_gyro_norm_max_dps,
        zupt_apply_hybrid_statuses=tuple(
            int(s.strip()) for s in args.zupt_apply_hybrid_statuses.split(",") if s.strip()
        ),
        zupt_min_consecutive=args.zupt_min_consecutive,
        zupt_max_anchor_drift_m=args.zupt_max_anchor_drift_m,
        imu_tc_emit_pf_hybrid_statuses=tuple(
            int(s.strip()) for s in args.imu_tc_emit_pf_hybrid_statuses.split(",") if s.strip()
        ),
        imu_tc_pos_sigma_base_m=args.imu_tc_pos_sigma_base_m,
        imu_tc_pos_sigma_per_s=args.imu_tc_pos_sigma_per_s,
        imu_tc_max_dr_seconds=args.imu_tc_max_dr_seconds,
        imu_tc_max_disagreement_m=args.imu_tc_max_disagreement_m,
        imu_tc_emit_max_diff_m=args.imu_tc_emit_max_diff_m,
        imu_tc_hybrid_loose_sigma_m=args.imu_tc_hybrid_loose_sigma_m,
        imu_tc_anchor_static_acc_low_mps2=args.zupt_acc_norm_low_mps2,
        imu_tc_anchor_static_acc_high_mps2=args.zupt_acc_norm_high_mps2,
        imu_tc_anchor_static_gyro_max_dps=args.zupt_gyro_norm_max_dps,
        ins_tc_emit_pf_hybrid_statuses=tuple(
            int(s.strip()) for s in args.ins_tc_emit_pf_hybrid_statuses.split(",") if s.strip()
        ),
        ins_tc_obs_status_4_sigma_m=args.ins_tc_obs_status_4_sigma_m,
        ins_tc_obs_status_3_sigma_m=args.ins_tc_obs_status_3_sigma_m,
        ins_tc_max_dr_seconds=args.ins_tc_max_dr_seconds,
        ins_tc_max_disagreement_m=args.ins_tc_max_disagreement_m,
        ins_tc_emit_max_diff_m=args.ins_tc_emit_max_diff_m,
        ins_tc_pf_pu_floor_sigma_m=args.ins_tc_pf_pu_floor_sigma_m,
        ins_tc_pf_pu_ceiling_sigma_m=args.ins_tc_pf_pu_ceiling_sigma_m,
        ins_tc_use_particle_imu_predict=not args.ins_tc_disable_particle_imu_predict,
        ins_tc_particle_imu_sigma_pos_m=args.ins_tc_particle_imu_sigma_pos_m,
        ins_tc_particle_imu_sigma_acc_mps2=args.ins_tc_particle_imu_sigma_acc_mps2,
        ins_tc_particle_imu_sigma_gyro_rps=args.ins_tc_particle_imu_sigma_gyro_rps,
        ins_tc_particle_imu_acc_bias_rw=args.ins_tc_particle_imu_acc_bias_rw,
        ins_tc_particle_imu_gyro_bias_rw=args.ins_tc_particle_imu_gyro_bias_rw,
        ins_tc_particle_imu_att_spread_rad=args.ins_tc_particle_imu_att_spread_rad,
        ins_tc_particle_imu_acc_bias_spread=args.ins_tc_particle_imu_acc_bias_spread,
        ins_tc_particle_imu_gyro_bias_spread_rps=args.ins_tc_particle_imu_gyro_bias_spread_rps,
        ins_tc_particle_imu_velocity_spread_mps=args.ins_tc_particle_imu_velocity_spread_mps,
        ins_tc_recenter_status4=bool(args.ins_tc_enable_recenter_status4),
        ins_tc_recenter_max_shift_m=args.ins_tc_recenter_max_shift_m,
        ins_tc_use_motion_predict=not args.ins_tc_disable_motion_predict,
        ins_tc_predict_sigma_pos_m=args.ins_tc_predict_sigma_pos_m,
        ins_tc_predict_velocity_alpha=args.ins_tc_predict_velocity_alpha,
        ins_tc_align_acc_low=args.ins_tc_align_acc_low,
        ins_tc_align_acc_high=args.ins_tc_align_acc_high,
        ins_tc_align_gyro_max_dps=args.ins_tc_align_gyro_max_dps,
        ins_tc_align_min_samples=args.ins_tc_align_min_samples,
        ins_tc_yaw_init_min_speed_mps=args.ins_tc_yaw_init_min_speed_mps,
        # NOTE: rbpf_kf_gate_* defaults stay None in `base` so that bare
        # `rbpf` / `rbpf+dd` variants run without a gate (true baseline).
        # Only the `rbpf+dd+gate*` variants below opt in via `aaa_gate`.
        # `enable_hybrid_pu` also stays False here; only `*+hybrid` opts in.
        systems=tuple(args.systems.split(",")),
    )
    if "pf" in args.methods:
        variants.append(CTRBPFConfig(**base, method_label="PF-PR"))
    if "pf+pu" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_position_update=True,
            method_label="PF-PR+PU",
        ))
    if "rbpf" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            method_label="RBPF-velKF",
        ))
    if "rbpf+pu" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_position_update=True,
            method_label="RBPF-velKF+PU",
        ))
    if "pf+dd" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_dd_carrier_afv=True,
            method_label="PF-PR+DD",
        ))
    if "rbpf+dd" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            method_label="RBPF-velKF+DD",
        ))
    if "rbpf+dd+pu" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            method_label="RBPF-velKF+DD+PU",
        ))
    # Phase 2 variants: same as rbpf+dd / rbpf+dd+pu but force the
    # AAA-style region-aware gate defaults so a single CLI flag is enough
    # to opt in. Per-knob CLI overrides still apply via args.*.
    aaa_gate = dict(
        rbpf_kf_gate_min_dd_pairs=(
            args.rbpf_velocity_kf_gate_min_dd_pairs
            if args.rbpf_velocity_kf_gate_min_dd_pairs is not None
            else 15
        ),
        rbpf_kf_gate_min_ess_ratio=(
            args.rbpf_velocity_kf_gate_min_ess_ratio
            if args.rbpf_velocity_kf_gate_min_ess_ratio is not None
            else 0.02
        ),
        rbpf_kf_gate_max_spread_m=args.rbpf_velocity_kf_gate_max_spread_m,
    )
    if "rbpf+dd+gate" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            method_label="RBPF-velKF+DD+gate",
        ))
    if "rbpf+dd+gate+pu" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            method_label="RBPF-velKF+DD+gate+PU",
        ))
    # Phase 6 variants: layer hybrid (libgnss++ 50.91% baseline) PU on top.
    if "pf+hybrid" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_hybrid_pu=True,
            method_label="PF-PR+hybrid",
        ))
    if "rbpf+dd+hybrid" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            method_label="RBPF-velKF+DD+hybrid",
        ))
    if "rbpf+dd+gate+hybrid" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            method_label="RBPF-velKF+DD+gate+hybrid",
        ))
    # Phase 10a: switch Status=1/3 pseudorange likelihood from Gaussian to
    # LOS/NLOS positive-bias GMM, while keeping Status=4 on the sharp model.
    if "rbpf+dd+gate+hybrid+gmm" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_pr_gmm=True,
            method_label="RBPF-velKF+DD+gate+hybrid+gmm",
        ))
    # Phase 10i: keep v5 hybrid as the floor, but inject the relaxed RTK
    # candidate into the PF on diagnostics-passing epochs and emit PF only
    # there. This is the PF version of the Phase 10h dual-profile chooser.
    if "rbpf+dd+gate+hybrid+rtkdiag_pf" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_rtkdiag_pf_rescue=True,
            method_label="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf",
        ))
    # Phase 7: full stack (vguide + hybrid PU + DD AFV + RBPF gate) emitting
    # the PF's own estimate. Goal: show DD-AFV cm-correction beat the hybrid
    # baseline floor (Phase 6 passthrough).
    if "rbpf+dd+gate+phase7" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            hybrid_emit_pf_estimate=True,
            method_label="RBPF-velKF+DD+gate+phase7",
        ))
    if "rbpf+dd+gate+phase7+gmm" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            hybrid_emit_pf_estimate=True,
            enable_pr_gmm=True,
            method_label="RBPF-velKF+DD+gate+phase7+gmm",
        ))
    # Phase 9a: hybrid passthrough + ZUPT (IMU stop detection).
    if "rbpf+dd+gate+hybrid+zupt" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_zupt=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt",
        ))
    # Phase 9b: tight-coupled IMU (in-loop pre-integration; per-particle
    # position pseudo-observation; emission switches to PF estimate on
    # Status=1/3 epochs). Requires hybrid for the Status=4 anchor reset.
    if "rbpf+dd+gate+hybrid+imu_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_imu_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+imu_tc",
        ))
    # Phase 9a + Phase 9b stacked. ZUPT damps stops, IMU-TC handles motion
    # NLOS rescue. Same anchor IMU-static thresholds for both.
    if "rbpf+dd+gate+hybrid+zupt+imu_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_zupt=True,
            enable_imu_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt+imu_tc",
        ))
    # Phase 9c: full INS-GNSS EKF (15-state; online accel/gyro bias).
    if "rbpf+dd+gate+hybrid+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+ins_tc",
        ))
    # Phase 10a stacked with per-particle INS/IMU propagation.
    if "rbpf+dd+gate+hybrid+gmm+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_pr_gmm=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+gmm+ins_tc",
        ))
    # Phase 9a + Phase 9c stacked.
    if "rbpf+dd+gate+hybrid+zupt+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_zupt=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt+ins_tc",
        ))
    # Phase 9a + Phase 8 stacked.
    if "rbpf+dd+gate+hybrid+zupt+tdcp" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_zupt=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt+tdcp",
        ))
    # Phase 8: hybrid passthrough + TDCP-anchored Kalman smoother.
    if "rbpf+dd+gate+hybrid+tdcp" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+gate+hybrid+tdcp",
        ))
    # Phase 8 + Phase 4 stacked.
    if "rbpf+dd+gate+hybrid+tdcp+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_tdcp_smoother=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+hybrid+tdcp+phase4",
        ))
    # Phase 4: hybrid passthrough + post-process FGO + LAMBDA partial fix.
    # The PF supplies cached DD carrier observations (no hybrid bias on the
    # FGO solve), and where LAMBDA accepts integer fixes we replace the
    # hybrid passthrough with the cm-pulled FGO trajectory.
    if "rbpf+dd+gate+hybrid+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+hybrid+phase4",
        ))
    # Phase 4 without hybrid: pure PF + FGO + LAMBDA. Useful to see whether
    # LAMBDA alone (independent of hybrid) is enough to cm-pull the
    # trajectory.
    if "rbpf+dd+gate+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+phase4",
        ))
    if not variants:
        raise ValueError(f"no valid methods: {args.methods}")
    return variants


_RTKDIAG_POLICIES = {
    "phase10o", "phase10p", "phase10r",
    "phase11h", "phase11i", "phase11l", "phase11n",
    "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au",
}


def _apply_rtkdiag_run_index_policy(
    variant: CTRBPFConfig,
    *,
    run: str,
    policy: str,
    city: str | None = None,
) -> CTRBPFConfig:
    if (
        policy not in _RTKDIAG_POLICIES
        or not bool(variant.enable_rtkdiag_pf_rescue)
    ):
        return variant
    if policy in {"phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"}:
        # Phase 11aa: extends Phase 11z by switching tokyo/run1 selector
        # to hybrid_anchor (consensus selector showed +0.20pp on this run /
        # +0.05pp aggregate). Other runs unchanged from Phase 11z.
        # Phase 11ab: same gates as 11aa, but r15ga (--glonass-ar autocal,
        # ratio 1.5) is added to the candidate pool — restricted via
        # blocked_labels to {(tokyo, run3), (nagoya, run1)} where offline
        # simulation showed positive aggregate (+88m / +0.19pp).
        # Phase 11ac: extends 11ab with two more selectively-eligible
        # candidates: r20ga (ratio2.0 + glonass-ar) for {(tokyo, run3),
        # (nagoya, run2)} and em10 (--elevation-mask-deg 10) for {(tokyo,
        # run3)}. Combined offline gain over 11ab is +81m / +0.18pp.
        # Phase 11ad: extends 11ac with two more candidates restricted to
        # {(tokyo, run3)}: psig1 (--pseudorange-sigma 1.0) and holdrlx
        # (--min-hold-count 3 --hold-ratio-threshold 1.5). Offline shows
        # +146m on tokyo/run3 alone (additive).
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="hybrid_anchor",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11z logic if city unknown.
    if policy == "phase11z":
        # Phase 11z: extends Phase 11y by pushing rms_max to 50 on
        # tokyo/run3 + nagoya/run2 and rms_max=30 on nagoya/run3 (extended2
        # gate sweep showed plateau between rms=50 and rms=100, with most
        # gain on nagoya/run2 +0.79pp).
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11y/phase11x logic if city unknown.
    if policy == "phase11y":
        # Phase 11y: extends Phase 11x by widening rms_max to 20 on the
        # heavy-NLOS runs (tokyo/run3, nagoya/run2, nagoya/run3) and
        # dropping ratio_min to 1.0 where the extended gate sweep showed
        # selector improvement. Other runs unchanged from Phase 11x.
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11x logic if city unknown.
    if policy == "phase11x":
        # Phase 11x: city x run gate from offline gate sweep on Phase 11v
        # candidate pool. Selector mode is unchanged from phase11n.
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.3,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.5,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.5,
                rtkdiag_candidate_residual_rms_max=7.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through: city unknown, behave like phase11n.
    if run == "run1":
        return replace(
            variant,
            rtkdiag_candidate_select_mode="nrows",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=1.4,
            rtkdiag_candidate_emit_mode="candidate",
        )
    if run == "run2":
        if policy in {"phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa"}:
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=6.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        return replace(
            variant,
            rtkdiag_candidate_select_mode="maxabs",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=6.0 if policy == "phase10p" else 5.0,
            rtkdiag_candidate_emit_mode="candidate",
        )
    if run == "run3":
        return replace(
            variant,
            rtkdiag_candidate_select_mode="score",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=7.0 if policy in {"phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa"} else (
                6.0 if policy == "phase10p" else 5.0
            ),
            rtkdiag_candidate_emit_mode="candidate",
        )
    return variant


def _filter_rtkdiag_candidates_by_policy(
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]],
    *,
    city: str,
    run: str,
    policy: str,
    blocked_labels: set[str] | None = None,
) -> list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]]:
    effective_blocked_labels = set(blocked_labels or set())
    if policy in {"phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and city == "nagoya" and run == "run2":
        effective_blocked_labels.update({"r15g15", "r20g15", "r25g15", "r30g15"})
    if policy in {"phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (
        (city, run) in {("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run2")}
    ):
        effective_blocked_labels.update({"r30", "r30g"})
    if policy in {"phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (
        (city, run) in {("nagoya", "run1"), ("nagoya", "run2")}
    ):
        effective_blocked_labels.add("r20g10")
    if policy in {"phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and city == "nagoya":
        effective_blocked_labels.update({"r15g10", "r25g10"})
    # Phase 11ab: r15ga (--glonass-ar autocal ratio 1.5) only helps tokyo/run3
    # and nagoya/run1 in offline simulation; block on the other 4 runs to
    # avoid the -30/-19/-5/-2 m/run regressions seen in the sim.
    if policy in {"phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("r15ga")
    # Phase 11ac: r20ga only helps tokyo/run3 (+12.6m) and nagoya/run2
    # (+47.7m) on top of 11ab base. Block elsewhere.
    if policy in {"phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("r20ga")
    # Phase 11ac: em10 (elev mask 10°) only helps tokyo/run3 (+21m). Block
    # elsewhere. Phase 11af also allows em10 on nagoya/run1 (+2.0m offline).
    if policy in {"phase11ac", "phase11ad", "phase11ae"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("em10")
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("em10")
    # Phase 11ad: psig1 (--pseudorange-sigma 1.0) gives +125m only on
    # tokyo/run3. Phase 11af also allows psig1 on nagoya/run3 (+4.1m offline).
    if policy in {"phase11ad", "phase11ae"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig1")
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("psig1")
    # Phase 11ad: holdrlx (relaxed hold ambiguity) gives +72m only on
    # tokyo/run3.
    if policy in {"phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("holdrlx")
    # Phase 11ae: r12ga (--ratio 1.2 + --glonass-ar autocal) gives +103m
    # additive on tokyo/run3 only. Phase 11af keeps it (no harm).
    if policy in {"phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("r12ga")
    # Phase 11ae: psig2 (--pseudorange-sigma 2.0) gives +44.5m additive on
    # tokyo/run3 only. Phase 11af/ag keeps it (no harm).
    if policy in {"phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig2")
    # Phase 11af: psig1hr (psig1 + holdrlx combined config) gives +90.8m
    # additive on tokyo/run3 only. Negative on nagoya runs.
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig1hr")
    # Phase 11af: nobds (--no-beidou) gives +29.7m on nagoya/run2 (largest),
    # +9.8m on tokyo/run3, +4.2m on tokyo/run1. Negative on nagoya/run1, run3.
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}:
        effective_blocked_labels.add("nobds")
    # Phase 11ag: csig05 (--carrier-phase-sigma 0.0005) gives massive gains:
    # tokyo/run3 +95.8m, nagoya/run2 +95.8m, tokyo/run1 +9.8m. Negative on
    # nagoya/run1/run3 originally; Phase 11am also enables nagoya/run3 (+2.6m
    # additive after the other csig05* variants).
    if policy in {"phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1")}:
        effective_blocked_labels.add("csig05")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05")
    # Phase 11ag: noglo (--no-glonass) gives +40.5m on tokyo/run1 only.
    if policy in {"phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("noglo")
    # Phase 11ag: csig1 (--carrier-phase-sigma 0.001) gives +0.6m on
    # nagoya/run1. Phase 11ah extends to {nagoya/run1, nagoya/run2, tokyo/run3}
    # (additive after csig05).
    if policy == "phase11ag" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("csig1")
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("nagoya", "run1"), ("nagoya", "run2"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig1")
    # Phase 11ah: rout30 (--rtk-update-outlier-threshold 30) gives +3.6m
    # on nagoya/run3 only.
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("rout30")
    # Phase 11ah: rout20 (--rtk-update-outlier-threshold 20) gives +2.2m
    # on nagoya/run1 only.
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rout20")
    # Phase 11ai: csig05hr (csig05 + holdrlx combined) gives massive gains:
    # tokyo/run1 +509.8m, tokyo/run3 +147.1m, nagoya/run3 +128.3m. Block on
    # nagoya/run2 (-7.7m) and nagoya/run1 (+0.4m, near-zero).
    if policy in {"phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05hr")
    # Phase 11ai: csig05ps (csig05 + psig1 combined) gives +22.5m on
    # tokyo/run3 and +3.4m on nagoya/run1. Phase 11aj also enables on
    # nagoya/run3 (+11.7m additive).
    # Phase 11ai/ak: csig05ps (csig05 + psig1) for {tokyo/run3, nagoya/run1}.
    # Phase 11aj added nagoya/run3 (-0.12pp PF on that run, so removed in 11ak).
    if policy in {"phase11ai", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("csig05ps")
    if policy == "phase11aj" and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05ps")
    # Phase 11aj/ak: csig01 (--carrier-phase-sigma 0.0001). Phase 11ak narrows
    # to {tokyo/run3, nagoya/run2} only — nagoya/run3 was -0.12pp in PF.
    if policy == "phase11aj" and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig01")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("csig01")
    # Phase 11aj: rout100 hurt nagoya/run1 by -1.49pp in PF — block in phase11ak.
    if policy == "phase11aj" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rout100")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"}:
        effective_blocked_labels.add("rout100")
    # Phase 11aj: csig05nb (csig05 + nobds) +0.8m offline but -0.02pp PF.
    # Block entirely in phase11ak.
    if policy == "phase11aj" and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("csig05nb")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"}:
        effective_blocked_labels.add("csig05nb")
    # Phase 11ak: csig05hvr (csig05 + holdvrlx very loose hold) gives +53.2m
    # on tokyo/run1 only. Phase 11am also enables tokyo/run3 (+29.4m offline).
    if policy in {"phase11ak", "phase11al"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("csig05hvr")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig05hvr")
    # Phase 11ak: csig05psh (csig05 + psig1 + holdrlx triple) gives +44m on
    # tokyo/run3 and +21.5m on nagoya/run3. Phase 11am also enables tokyo/run1 (+4.1m).
    if policy in {"phase11ak", "phase11al"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05psh")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3"), ("tokyo", "run1")}:
        effective_blocked_labels.add("csig05psh")
    # Phase 11ak: csig05em (csig05 + em10) gives +2.5m on nagoya/run1.
    if policy == "phase11ak" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("csig05em")
    if policy == "phase11al":
        effective_blocked_labels.add("csig05em")
    # Phase 11am: csig05em allowed only on tokyo/run3 (+20.5m offline). Block elsewhere.
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("csig05em")
    # Phase 11ak: csig01hr (csig01 + holdrlx) gives +42.8m on nagoya/run2.
    # Phase 11am widens to {tokyo/run1 +7.2m, tokyo/run3 +23.1m, nagoya/run2 +42.8m}.
    if policy in {"phase11ak", "phase11al"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("csig01hr")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("nagoya", "run2"), ("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig01hr")
    # Phase 11an: c5p1hvr (csig05+psig1+holdvrlx 4-knob) gives +104.6m on
    # tokyo/run1 and +28.4m on nagoya/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("c5p1hvr")
    # Phase 11an: c1p1hr (csig01+psig1+holdrlx triple) gives +7.3m on tokyo/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("c1p1hr")
    # Phase 11an: c5hrem (csig05+holdrlx+em10 triple) gives +18.2m on tokyo/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("c5hrem")
    # Phase 11ao: csig005 (--carrier-phase-sigma 0.00005, super-tight) gives
    # +36.1m on nagoya/run2 (the lowest run). Marginal elsewhere.
    if policy in {"phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("csig005")
    # Phase 11ao: c5nbhr (csig05+nobds+holdrlx triple). Marginal +9.8m on
    # tokyo/run1 and +6.8m on nagoya/run1. Negative on others.
    if policy in {"phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c5nbhr")
    # Phase 11ap: c005hr (csig005+holdrlx) marginal +2.6m on tokyo/run1, +9.5m on nagoya/run1.
    if policy in {"phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c005hr")
    # Phase 11aq: r05 (--ratio 0.5) marginal +5.0/+3.9/+2.3m on tokyo/run1, tokyo/run3, nagoya/run1.
    if policy in {"phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("r05")
    # Phase 11aq: c005ga (csig005+glonassar) marginal +1.8/+3.3m on tokyo/run1, tokyo/run3.
    if policy in {"phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("c005ga")
    # Phase 11ar: onlyG (--no-glonass --no-beidou, only G+E+J). +9.4m tokyo/run1
    # and **+20.2m on nagoya/run2** (the lowest-scoring run). Negative on nagoya/run3.
    if policy in {"phase11ar", "phase11as", "phase11at", "phase11au"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run2")}:
        effective_blocked_labels.add("onlyG")
    # Phase 11as: oGc05 (onlyG + csig05) +14.3m on tokyo/run1.
    if policy in {"phase11as", "phase11at", "phase11au"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGc05")
    # Phase 11as: oGc005 (onlyG + csig005) +10.4m on nagoya/run2.
    if policy in {"phase11as", "phase11at", "phase11au"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("oGc005")
    # Phase 11as: oGp1 (onlyG + psig1) +2.1/+3.0m on nagoya/run1, nagoya/run3.
    if policy in {"phase11as", "phase11at", "phase11au"} and (city, run) not in {("nagoya", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("oGp1")
    # Phase 11at: oGp1hr (onlyG + psig1 + holdrlx) +15.4m on tokyo/run1.
    if policy in {"phase11at", "phase11au"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGp1hr")
    # Phase 11at: oGp1c05 (onlyG + psig1 + csig05) +9.2m nagoya/run3, +2.0m nagoya/run1.
    if policy in {"phase11at", "phase11au"} and (city, run) not in {("nagoya", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("oGp1c05")
    # Phase 11at: oGr05 (onlyG + ratio 0.5) marginal +2.2/+3.7m on tokyo/run3, nagoya/run2.
    if policy in {"phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("oGr05")
    # Phase 11at: oGc01 (onlyG + csig01) marginal +3.1m on nagoya/run1.
    if policy in {"phase11at", "phase11au"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("oGc01")
    # Phase 11at: oGem10 (onlyG + em10) marginal +1.5m/+1.1m on tokyo/run3, nagoya/run3.
    if policy in {"phase11at", "phase11au"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("oGem10")
    # Phase 11au: oGc005p1 (onlyG + csig005 + psig1) +37m tokyo/run2, +6.6m tokyo/run1, +1.2m nagoya/run1.
    if policy == "phase11au" and (city, run) not in {("tokyo", "run2"), ("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("oGc005p1")
    # Phase 11au: c005p1 (csig005 + psig1, no onlyG) +13.9m tokyo/run1, +15.1m tokyo/run2, +1.5m nagoya/run1.
    if policy == "phase11au" and (city, run) not in {("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c005p1")
    # Phase 11au: oGc01p1 (onlyG + csig01 + psig1) +21.4m tokyo/run2, +3.4m tokyo/run3.
    if policy == "phase11au" and (city, run) not in {("tokyo", "run2"), ("tokyo", "run3")}:
        effective_blocked_labels.add("oGc01p1")
    if not effective_blocked_labels:
        return candidates
    return [
        candidate for candidate in candidates
        if candidate[0] not in effective_blocked_labels
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-RBPF-FGO PPC port (Phase 0 scaffolding)")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-prefix", type=str, default="ppc_ctrbpf_fgo")
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=RESULTS_DIR / "libgnss_ctrbpf_pos",
    )
    parser.add_argument("--n-particles", type=int, default=50_000)
    parser.add_argument("--sigma-pr", type=float, default=8.0)
    parser.add_argument("--pr-gmm-statuses", type=str, default="1,3",
                        help="Hybrid Status values where PR-GMM likelihood is used (default '1,3'; empty means all)")
    parser.add_argument("--pr-gmm-w-los", type=float, default=0.7,
                        help="LOS mixture weight for PR-GMM likelihood (default 0.7)")
    parser.add_argument("--pr-gmm-mu-nlos-m", type=float, default=15.0,
                        help="Positive NLOS pseudorange bias mean for PR-GMM [m] (default 15)")
    parser.add_argument("--pr-gmm-sigma-nlos-m", type=float, default=30.0,
                        help="NLOS pseudorange sigma for PR-GMM [m] (default 30)")
    parser.add_argument("--pr-gmm-hybrid-loose-sigma-m", type=float, default=5.0,
                        help="Hybrid PU sigma on PR-GMM statuses; <=0 keeps global hybrid sigma (default 5)")
    parser.add_argument("--pr-gmm-clock-quantile", type=float, default=0.35,
                        help="Clock-bias correction residual quantile when PR-GMM is active (default 0.35)")
    parser.add_argument("--pr-weight-mode", choices=("raw", "unit", "cn0-relative"), default="raw",
                        help="PF pseudorange likelihood weight transform. raw preserves legacy C/N0-as-weight behavior")
    parser.add_argument("--pr-weight-ref-cn0", type=float, default=45.0,
                        help="Reference C/N0 for --pr-weight-mode cn0-relative (default 45)")
    parser.add_argument("--pr-weight-min", type=float, default=0.25,
                        help="Minimum transformed PR likelihood weight when clipping is active (default 0.25)")
    parser.add_argument("--pr-weight-max", type=float, default=1.5,
                        help="Maximum transformed PR likelihood weight when clipping is active (default 1.5)")
    parser.add_argument("--pr-prefit-gate-m", type=float, default=0.0,
                        help="Drop satellites whose robust clock-centered PR prefit residual exceeds this [m]; <=0 disables")
    parser.add_argument("--pr-prefit-gate-min-sats", type=int, default=6,
                        help="Minimum satellites kept by the PR prefit gate (default 6)")
    parser.add_argument("--pr-prefit-gate-keep-best", type=int, default=0,
                        help="If >0, keep at most this many smallest-prefit satellites after gating")
    parser.add_argument("--pr-prefit-ref", choices=("pf", "hybrid"), default="pf",
                        help="Reference position for PR prefit residuals (default pf; hybrid uses libgnss++ position when available)")
    parser.add_argument("--pr-skip-statuses", type=str, default="",
                        help="Comma-separated hybrid Status values where undifferenced PR update is skipped")
    parser.add_argument("--defer-epoch-resample", action="store_true",
                        help="Accumulate PR/DD/Doppler/PU likelihoods within an epoch and resample only after emission")
    parser.add_argument("--sigma-pos", type=float, default=2.0)
    parser.add_argument("--sigma-cb", type=float, default=50.0)
    parser.add_argument("--spread-pos-init", type=float, default=50.0)
    parser.add_argument("--spread-cb-init", type=float, default=500.0)
    parser.add_argument("--sigma-doppler-mps", type=float, default=0.5)
    parser.add_argument("--velocity-init-sigma", type=float, default=1.0)
    parser.add_argument("--velocity-process-noise", type=float, default=1.0)
    parser.add_argument("--position-update-sigma-m", type=float, default=30.0)
    parser.add_argument("--disable-correct-clock-bias", action="store_true")
    parser.add_argument("--systems", type=str, default="G,R,E,C,J")
    parser.add_argument(
        "--methods",
        type=str,
        default="pf,pf+pu,rbpf,rbpf+pu",
        help=(
            "Comma-separated subset of {pf, pf+pu, rbpf, rbpf+pu, pf+dd, "
            "rbpf+dd, rbpf+dd+pu, rbpf+dd+gate, rbpf+dd+gate+pu, "
            "pf+hybrid, rbpf+dd+hybrid, rbpf+dd+gate+hybrid, "
            "rbpf+dd+gate+hybrid+gmm, "
            "rbpf+dd+gate+hybrid+rtkdiag_pf, "
            "rbpf+dd+gate+phase7, rbpf+dd+gate+phase7+gmm, "
            "rbpf+dd+gate+phase4, "
            "rbpf+dd+gate+hybrid+phase4, "
            "rbpf+dd+gate+hybrid+tdcp, rbpf+dd+gate+hybrid+tdcp+phase4, "
            "rbpf+dd+gate+hybrid+zupt, rbpf+dd+gate+hybrid+zupt+tdcp, "
            "rbpf+dd+gate+hybrid+imu_tc, rbpf+dd+gate+hybrid+zupt+imu_tc, "
            "rbpf+dd+gate+hybrid+ins_tc, rbpf+dd+gate+hybrid+gmm+ins_tc, "
            "rbpf+dd+gate+hybrid+zupt+ins_tc}"
        ),
    )
    parser.add_argument("--dd-sigma-cycles", type=float, default=0.05,
                        help="DD carrier AFV sigma in cycles (default 0.05)")
    parser.add_argument("--dd-min-pairs", type=int, default=4,
                        help="Min common rover/base sats to attempt DD (default 4)")
    parser.add_argument("--dd-min-pairs-update", type=int, default=3,
                        help="Min DD pairs required to apply pf.update_dd_carrier_afv (default 3)")
    parser.add_argument("--dd-systems", type=str, default="G,E,J,C",
                        help="Constellations used for DD (GLONASS skipped, default G,E,J,C)")
    parser.add_argument("--dd-base-interp", action="store_true",
                        help="Interpolate base RINEX between epochs when rover TOW falls between two")
    # Phase 2: region-aware gate on RBPF velocity-KF (Doppler) update.
    # Mirrors the AAA gate knobs from internal_docs/plan.md §10.1.
    parser.add_argument("--rbpf-velocity-kf-gate-min-dd-pairs", type=int, default=None,
                        help="Skip Doppler KF update unless DD pair count >= N (default off)")
    parser.add_argument("--rbpf-velocity-kf-gate-min-ess-ratio", type=float, default=None,
                        help="Skip Doppler KF update unless ESS / n_particles >= ratio (default off)")
    parser.add_argument("--rbpf-velocity-kf-gate-max-spread-m", type=float, default=None,
                        help="Skip Doppler KF update if PF position spread exceeds this [m] (default off)")
    # Phase 6: libgnss++ hybrid position update (uses .pos files from
    # experiments/results/libgnss_rtk_pos_v5/ — the 50.91% baseline).
    parser.add_argument("--hybrid-pos-dir", type=Path, default=None,
                        help="Directory of libgnss++ .pos files used as hybrid PU baseline "
                             "(expects {city}_{run}_full.pos)")
    parser.add_argument("--hybrid-pos-suffix", type=str, default="_full.pos",
                        help="Suffix used to find pos files (default _full.pos)")
    parser.add_argument("--hybrid-sigma-m", type=float, default=1.0,
                        help="Sigma [m] for the hybrid position_update soft constraint (default 1.0)")
    parser.add_argument("--hybrid-vguide-max-dt-s", type=float, default=0.5,
                        help="Max gap [s] between consecutive hybrid samples for finite-diff velocity (default 0.5)")
    # Phase 10i: PF rescue using RTK diagnostics from gnss_solve
    parser.add_argument("--rtkdiag-candidate-pos-dir", type=Path, default=None,
                        help="Directory of relaxed RTK candidate .pos files for rtkdiag_pf "
                             "(expects {city}_{run}_full.pos)")
    parser.add_argument("--rtkdiag-candidate-diag-dir", type=Path, default=None,
                        help="Directory of relaxed RTK diagnostics CSV files for rtkdiag_pf "
                             "(expects {city}_{run}_full.csv)")
    parser.add_argument("--rtkdiag-candidate-pos-dirs", type=str, default="",
                        help="Comma-separated candidate .pos directories for multi-candidate rtkdiag_pf")
    parser.add_argument("--rtkdiag-candidate-diag-dirs", type=str, default="",
                        help="Comma-separated diagnostics CSV directories for multi-candidate rtkdiag_pf")
    parser.add_argument("--rtkdiag-candidate-labels", type=str, default="",
                        help="Optional comma-separated labels for multi-candidate rtkdiag_pf")
    parser.add_argument("--rtkdiag-candidate-block-labels", type=str, default="",
                        help="Comma-separated candidate labels to block for all runs")
    parser.add_argument("--rtkdiag-candidate-block-labels-by-run", type=str, default="",
                        help=(
                            "Semicolon-separated run-specific candidate label blocks, "
                            "e.g. 'nagoya/run2=r20g15+r15g15'"
                        ))
    parser.add_argument("--rtkdiag-candidate-pos-suffix", type=str, default="_full.pos",
                        help="Suffix used for RTK diagnostic candidate pos files (default _full.pos)")
    parser.add_argument("--rtkdiag-candidate-diag-suffix", type=str, default="_full.csv",
                        help="Suffix used for RTK diagnostic CSV files (default _full.csv)")
    parser.add_argument("--rtkdiag-candidate-sigma-m", type=float, default=0.02,
                        help="PF position_update sigma for diagnostics-passing RTK candidate [m] (default 0.02)")
    parser.add_argument("--rtkdiag-candidate-ratio-min", type=float, default=1.5,
                        help="Minimum candidate final_ratio for rtkdiag_pf gate (default 1.5)")
    parser.add_argument("--rtkdiag-candidate-residual-rms-max", type=float, default=1.8,
                        help="Maximum candidate final_residual_rms for rtkdiag_pf gate (default 1.8)")
    parser.add_argument("--rtkdiag-candidate-emit-max-diff-m", type=float, default=0.4,
                        help="Emit PF only if |PF - candidate| <= this [m] after candidate PU (default 0.4)")
    parser.add_argument("--rtkdiag-candidate-recenter-max-shift-m", type=float, default=10000.0,
                        help="Recenter PF cloud to candidate before candidate PU when shift <= this [m] (default 10000)")
    parser.add_argument("--rtkdiag-candidate-select-mode",
                        choices=("residual", "ratio", "score", "maxabs", "nrows", "hybrid_anchor"),
                        default="residual",
                        help="How to choose among multiple gated RTK candidates (default residual). "
                             "hybrid_anchor picks the candidate closest to the hybrid floor; "
                             "useful when hybrid is reliable")
    parser.add_argument("--rtkdiag-candidate-emit-mode",
                        choices=("pf", "candidate-on-drift", "candidate"),
                        default="pf",
                        help=(
                            "Output policy for diagnostics-passing RTK candidate epochs: "
                            "pf=emit PF only when close to candidate and otherwise keep hybrid; "
                            "candidate-on-drift=emit PF when close, otherwise emit candidate; "
                            "candidate=always emit selected candidate (default pf)"
                        ))
    parser.add_argument("--rtkdiag-candidate-run-index-policy",
                        choices=("none", "phase10o", "phase10p", "phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au"),
                        default="none",
                        help=(
                            "Experimental per-run-index RTKDiag policy. phase10o uses "
                            "run1=nrows/rms1.4, run2=maxabs/rms5.0, run3=score/rms5.0, "
                            "all with candidate emit. phase10p uses the same family split "
                            "but run2/run3 rms5.0 -> rms6.0 for the r30 candidate set "
                            "phase10r uses run1=nrows/rms1.4, run2=score/ratio1.7/rms6.0, "
                            "run3=score/rms7.0 for the r15 no-hold candidate set. "
                            "phase11h follows phase10r but filters gate15 candidates on "
                            "nagoya/run2. phase11i follows phase11h but also filters "
                            "r30/r30g on tokyo/run2, nagoya/run1, and nagoya/run2. "
                            "phase11l follows phase11i but filters r20g10 on "
                            "nagoya/run1 and nagoya/run2. "
                            "phase11n follows phase11l but additionally filters "
                            "r15g10 and r25g10 on all nagoya runs. "
                            "phase11x extends phase11n with city x run gate values "
                            "from the offline gate sweep on Phase 11v candidates "
                            "(tokyo/run1 ratio2.5; tokyo/run2 rms10; tokyo/run3 ratio1.3 rms10; "
                            "nagoya/run1 rms1.0; nagoya/run2 ratio1.5 rms7; nagoya/run3 ratio1.7 rms10). "
                            "phase11y extends phase11x by widening rms_max to 20 on the "
                            "heavy-NLOS runs (tokyo/run3, nagoya/run2, nagoya/run3 with ratio_min "
                            "dropped to 1.0 on tokyo/run3 and nagoya/run2) "
                            "(default none)"
                        ))
    # Phase 4: post-process FGO + LAMBDA partial fix
    parser.add_argument("--fgo-window-size", type=int, default=30,
                        help="FGO window size in epochs (default 30)")
    parser.add_argument("--fgo-window-stride", type=int, default=15,
                        help="FGO window stride in epochs (default 15, 50%% overlap)")
    parser.add_argument("--fgo-lambda-ratio", type=float, default=3.0,
                        help="LAMBDA ratio test threshold (default 3.0)")
    parser.add_argument("--fgo-lambda-min-epochs", type=int, default=10,
                        help="Min epochs with DD obs in window to run LAMBDA (default 10)")
    parser.add_argument("--fgo-min-fixed-to-apply", type=int, default=3,
                        help="Min ratio-passed integer fixes per window to apply FGO output (default 3)")
    parser.add_argument("--fgo-prior-sigma-m", type=float, default=0.5,
                        help="FGO prior sigma [m] for initial positions (default 0.5)")
    parser.add_argument("--fgo-dd-sigma-cycles", type=float, default=0.20,
                        help="FGO DD carrier float sigma [cycles] (default 0.20)")
    parser.add_argument("--fgo-apply-hybrid-statuses", type=str, default="1,3",
                        help="Comma-separated hybrid Status values where Phase 4 may overwrite "
                             "the hybrid passthrough; default '1,3' protects Status=4 cm-class "
                             "epochs. Use empty string '' to disable the gate (apply everywhere).")
    parser.add_argument("--fgo-anchor-sigma-m", type=float, default=0.05,
                        help="Per-epoch FGO prior sigma [m] applied to non-rewritten Status epochs "
                             "(D2: tight anchor, default 0.05). Set <=0 to fall back to legacy "
                             "endpoint-only priors at --fgo-prior-sigma-m.")
    parser.add_argument("--fgo-loose-sigma-m", type=float, default=5.0,
                        help="Per-epoch FGO prior sigma [m] applied to rewritten Status epochs "
                             "(D2: loose, lets DD drive the solve, default 5.0).")
    parser.add_argument("--fgo-min-correction-m", type=float, default=0.5,
                        help="Minimum |FGO - hybrid| disagreement [m] required to overwrite the "
                             "hybrid passthrough at a given epoch (D2b: filters out small noisy "
                             "rewrites that cost cm-class passes; default 0.5, set 0 to disable).")
    # Phase 8: TDCP-anchored hybrid smoother
    parser.add_argument("--tdcp-sigma-mps", type=float, default=0.05,
                        help="TDCP velocity sigma [m/s] used as the smoother process noise (default 0.05)")
    parser.add_argument("--tdcp-postfit-max-m", type=float, default=1.0,
                        help="Reject TDCP velocity if postfit RMS exceeds this [m] (default 1.0)")
    parser.add_argument("--tdcp-min-sats", type=int, default=5,
                        help="Min satellites tracked through both epochs for TDCP (default 5)")
    parser.add_argument("--tdcp-obs-anchor-sigma-m", type=float, default=0.05,
                        help="Smoother observation sigma [m] for protected hybrid epochs "
                             "(Status NOT in --fgo-apply-hybrid-statuses; default 0.05)")
    parser.add_argument("--tdcp-obs-loose-sigma-m", type=float, default=5.0,
                        help="Smoother observation sigma [m] for rewritable hybrid epochs (default 5.0)")
    parser.add_argument("--tdcp-obs-missing-sigma-m", type=float, default=1000.0,
                        help="Smoother observation sigma [m] when hybrid is missing entirely (default 1000)")
    # Phase 9a: ZUPT (zero-velocity update) using PPC IMU
    parser.add_argument("--zupt-acc-norm-low-mps2", type=float, default=9.6,
                        help="Lower bound on accel norm [m/s^2] for static detection (default 9.6)")
    parser.add_argument("--zupt-acc-norm-high-mps2", type=float, default=9.95,
                        help="Upper bound on accel norm [m/s^2] for static detection (default 9.95)")
    parser.add_argument("--zupt-gyro-norm-max-dps", type=float, default=1.5,
                        help="Max gyro norm [deg/s] for static detection (default 1.5)")
    parser.add_argument("--zupt-apply-hybrid-statuses", type=str, default="1,3",
                        help="Hybrid Status values where ZUPT may overwrite (default '1,3', protects 4)")
    parser.add_argument("--zupt-min-consecutive", type=int, default=5,
                        help="Min consecutive static epochs (incl. current) required before ZUPT rewrites (default 5)")
    parser.add_argument("--zupt-max-anchor-drift-m", type=float, default=0.5,
                        help="Max |base - anchor| disagreement [m] tolerated before ZUPT skips (default 0.5)")
    # Phase 9b: tight-coupled IMU. In-loop pre-integration; per-particle
    # position pseudo-observation; emission switch on Status=1/3.
    parser.add_argument("--imu-tc-emit-pf-hybrid-statuses", type=str, default="1,3",
                        help="Hybrid Status values where IMU-TC emits the PF estimate (default '1,3', protects 4)")
    parser.add_argument("--imu-tc-pos-sigma-base-m", type=float, default=0.5,
                        help="IMU position pseudo-obs sigma at t=0s after anchor reset (default 0.5)")
    parser.add_argument("--imu-tc-pos-sigma-per-s", type=float, default=0.5,
                        help="IMU sigma growth per s of dead-reckoning (default 0.5)")
    parser.add_argument("--imu-tc-max-dr-seconds", type=float, default=5.0,
                        help="Max IMU dead-reckoning duration before falling back to hybrid (default 5.0)")
    parser.add_argument("--imu-tc-max-disagreement-m", type=float, default=30.0,
                        help="Skip IMU PU when |IMU-pred - hybrid| > this [m] (default 30)")
    parser.add_argument("--imu-tc-emit-max-diff-m", type=float, default=20.0,
                        help="Bound PF emission to hybrid +- this [m]; else keep hybrid (default 20)")
    parser.add_argument("--imu-tc-hybrid-loose-sigma-m", type=float, default=5.0,
                        help="Hybrid PU sigma on m-class Status epochs when IMU-TC is on (default 5.0; 0 disables)")
    # Phase 9c: full 15-state INS-GNSS EKF tight coupling.
    parser.add_argument("--ins-tc-emit-pf-hybrid-statuses", type=str, default="1,3",
                        help="Hybrid Status values where INS-TC emits the PF estimate (default '1,3', protects 4)")
    parser.add_argument("--ins-tc-obs-status-4-sigma-m", type=float, default=0.05,
                        help="INS position-observation horizontal sigma for Status=4 hybrid epochs (default 0.05)")
    parser.add_argument("--ins-tc-obs-status-3-sigma-m", type=float, default=0.0,
                        help="INS loose observation sigma for Status=3 hybrid epochs; 0 disables (default 0)")
    parser.add_argument("--ins-tc-max-dr-seconds", type=float, default=10.0,
                        help="Max INS dead-reckoning time since a position observation (default 10)")
    parser.add_argument("--ins-tc-max-disagreement-m", type=float, default=30.0,
                        help="Skip INS PF PU when |INS - hybrid| > this [m] (default 30)")
    parser.add_argument("--ins-tc-emit-max-diff-m", type=float, default=1.0,
                        help="Bound PF emission to hybrid +- this [m]; else keep hybrid (default 1)")
    parser.add_argument("--ins-tc-pf-pu-floor-sigma-m", type=float, default=0.1,
                        help="Lower bound on sigma fed to pf.position_update from INS (default 0.1)")
    parser.add_argument("--ins-tc-pf-pu-ceiling-sigma-m", type=float, default=5.0,
                        help="Upper bound on sigma fed to pf.position_update from INS (default 5.0)")
    parser.add_argument("--ins-tc-disable-particle-imu-predict", action="store_true",
                        help="Disable per-particle PF strapdown IMU prediction")
    parser.add_argument("--ins-tc-particle-imu-sigma-pos-m", type=float, default=0.02,
                        help="Per IMU-step PF position noise for strapdown predict [m] (default 0.02)")
    parser.add_argument("--ins-tc-particle-imu-sigma-acc-mps2", type=float, default=0.10,
                        help="PF strapdown accelerometer noise [m/s^2] (default 0.10)")
    parser.add_argument("--ins-tc-particle-imu-sigma-gyro-rps", type=float, default=0.005,
                        help="PF strapdown gyro noise [rad/s] (default 0.005)")
    parser.add_argument("--ins-tc-particle-imu-acc-bias-rw", type=float, default=1.0e-4,
                        help="PF accel-bias random walk [m/s^2/sqrt(s)] (default 1e-4)")
    parser.add_argument("--ins-tc-particle-imu-gyro-bias-rw", type=float, default=1.0e-5,
                        help="PF gyro-bias random walk [rad/s/sqrt(s)] (default 1e-5)")
    parser.add_argument("--ins-tc-particle-imu-att-spread-rad", type=float, default=math.radians(2.0),
                        help="Initial per-particle attitude spread [rad] (default 2deg)")
    parser.add_argument("--ins-tc-particle-imu-acc-bias-spread", type=float, default=0.05,
                        help="Initial accel-bias particle spread [m/s^2] (default 0.05)")
    parser.add_argument("--ins-tc-particle-imu-gyro-bias-spread-rps", type=float, default=math.radians(0.1),
                        help="Initial gyro-bias particle spread [rad/s] (default 0.1deg/s)")
    parser.add_argument("--ins-tc-particle-imu-velocity-spread-mps", type=float, default=0.5,
                        help="Initial PF inertial velocity spread [m/s] (default 0.5)")
    parser.add_argument("--ins-tc-enable-recenter-status4", action="store_true",
                        help="Enable PF cloud recentering on trusted Status=4 INS/GNSS anchors")
    parser.add_argument("--ins-tc-recenter-max-shift-m", type=float, default=5000.0,
                        help="Skip Status=4 PF recenter if required shift exceeds this [m] (default 5000)")
    parser.add_argument("--ins-tc-disable-motion-predict", action="store_true",
                        help="Disable INS velocity as the PF predict motion guide")
    parser.add_argument("--ins-tc-predict-sigma-pos-m", type=float, default=0.2,
                        help="PF predict position noise [m] when using INS velocity guide (default 0.2)")
    parser.add_argument("--ins-tc-predict-velocity-alpha", type=float, default=1.0,
                        help="Blend factor for INS velocity in PF predict (default 1.0)")
    parser.add_argument("--ins-tc-align-acc-low", type=float, default=9.6,
                        help="Lower accel-norm bound for INS static alignment (default 9.6)")
    parser.add_argument("--ins-tc-align-acc-high", type=float, default=9.95,
                        help="Upper accel-norm bound for INS static alignment (default 9.95)")
    parser.add_argument("--ins-tc-align-gyro-max-dps", type=float, default=1.5,
                        help="Max gyro norm [deg/s] for INS static alignment (default 1.5)")
    parser.add_argument("--ins-tc-align-min-samples", type=int, default=50,
                        help="Consecutive static IMU samples required for INS alignment (default 50)")
    parser.add_argument("--ins-tc-yaw-init-min-speed-mps", type=float, default=1.0,
                        help="Planar ENU speed required before yaw is initialized from hybrid velocity (default 1.0)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Cap epochs per run (smoke / debug). None = full run.")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--runs",
        type=str,
        default="all",
        help="Comma-separated run filter, e.g. 'tokyo/run1,nagoya/run3'. 'all' = 6 runs.",
    )
    args = parser.parse_args()

    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.systems_tuple = tuple(s.strip() for s in args.systems.split(",") if s.strip())

    if args.runs == "all":
        runs = _FULL_RUNS
    else:
        wanted = {r.strip() for r in args.runs.split(",") if r.strip()}
        runs = tuple((c, r) for c, r in _FULL_RUNS if f"{c}/{r}" in wanted)
        if not runs:
            raise SystemExit(f"no matching runs: {args.runs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.pos_dir.mkdir(parents=True, exist_ok=True)

    variants = _config_variants(args)
    rtkdiag_candidate_pos_dirs = (
        _parse_path_list(args.rtkdiag_candidate_pos_dirs)
        if str(args.rtkdiag_candidate_pos_dirs).strip()
        else ([args.rtkdiag_candidate_pos_dir] if args.rtkdiag_candidate_pos_dir else [])
    )
    rtkdiag_candidate_diag_dirs = (
        _parse_path_list(args.rtkdiag_candidate_diag_dirs)
        if str(args.rtkdiag_candidate_diag_dirs).strip()
        else ([args.rtkdiag_candidate_diag_dir] if args.rtkdiag_candidate_diag_dir else [])
    )
    rtkdiag_candidate_labels = _parse_label_list(args.rtkdiag_candidate_labels)
    if not rtkdiag_candidate_labels:
        rtkdiag_candidate_labels = [
            p.name or f"candidate{i}"
            for i, p in enumerate(rtkdiag_candidate_pos_dirs)
        ]
    rtkdiag_candidate_block_labels = set(
        _parse_label_list(args.rtkdiag_candidate_block_labels)
    )
    rtkdiag_candidate_block_labels_by_run = _parse_run_label_blocks(
        args.rtkdiag_candidate_block_labels_by_run
    )

    any_dd = any(v.enable_dd_carrier_afv for v in variants)
    any_dd_for_gate = any(
        (v.enable_rbpf_velocity_kf and v.rbpf_kf_gate_min_dd_pairs is not None)
        for v in variants
    )
    any_gate = any(
        v.enable_rbpf_velocity_kf
        and (
            v.rbpf_kf_gate_min_dd_pairs is not None
            or v.rbpf_kf_gate_min_ess_ratio is not None
            or v.rbpf_kf_gate_max_spread_m is not None
        )
        for v in variants
    )
    any_hybrid = any(v.enable_hybrid_pu for v in variants)
    if any_hybrid and args.hybrid_pos_dir is None:
        raise SystemExit(
            "--hybrid-pos-dir is required when any *+hybrid method is selected"
        )
    any_rtkdiag_pf = any(v.enable_rtkdiag_pf_rescue for v in variants)
    if any_rtkdiag_pf and (
        not rtkdiag_candidate_pos_dirs
        or not rtkdiag_candidate_diag_dirs
        or len(rtkdiag_candidate_pos_dirs) != len(rtkdiag_candidate_diag_dirs)
        or len(rtkdiag_candidate_labels) != len(rtkdiag_candidate_pos_dirs)
    ):
        raise SystemExit(
            "*+rtkdiag_pf requires matching candidate pos/diag dirs and labels "
            "(use either singular --rtkdiag-candidate-pos-dir/diag-dir or "
            "comma-separated --rtkdiag-candidate-pos-dirs/diag-dirs)"
        )

    print("=" * 72)
    print("  CT-RBPF-FGO PPC port — Phase 6 (hybrid blend)")
    print(f"  Methods: {[v.method_label for v in variants]}")
    print(f"  Particles: {args.n_particles}, Systems: {args.systems_tuple}")
    if any_dd or any_dd_for_gate:
        print(
            f"  DD: sigma_cycles={args.dd_sigma_cycles}, "
            f"min_pairs={args.dd_min_pairs}/{args.dd_min_pairs_update}, "
            f"systems={args.dd_systems}, base_interp={bool(args.dd_base_interp)}"
        )
    if any_gate:
        print(
            f"  RBPF gate: min_dd_pairs={args.rbpf_velocity_kf_gate_min_dd_pairs}, "
            f"min_ess_ratio={args.rbpf_velocity_kf_gate_min_ess_ratio}, "
            f"max_spread_m={args.rbpf_velocity_kf_gate_max_spread_m} "
            "(per-variant overrides via method labels)"
        )
    if any_hybrid:
        print(
            f"  Hybrid PU: dir={args.hybrid_pos_dir}, "
            f"suffix={args.hybrid_pos_suffix}, sigma_m={args.hybrid_sigma_m}"
        )
    if any_rtkdiag_pf:
        print(
            f"  RTKDiag PF: candidates={rtkdiag_candidate_labels}, "
            f"sigma={args.rtkdiag_candidate_sigma_m}, "
            f"ratio>={args.rtkdiag_candidate_ratio_min}, "
            f"rms<={args.rtkdiag_candidate_residual_rms_max}, "
            f"select={args.rtkdiag_candidate_select_mode}, "
            f"emit={args.rtkdiag_candidate_emit_mode}, "
            f"run_policy={args.rtkdiag_candidate_run_index_policy}"
        )
    print("=" * 72)

    rows: list[dict[str, object]] = []
    agg_pass: dict[str, float] = {v.method_label: 0.0 for v in variants}
    agg_total: dict[str, float] = {v.method_label: 0.0 for v in variants}

    for city, run in runs:
        run_dir = args.data_root / city / run
        if not run_dir.is_dir():
            print(f"  [skip] {city}/{run}: missing {run_dir}")
            continue
        print(f"\n[{city}/{run}] loading PPC data ...", flush=True)
        loader = PPCDatasetLoader(run_dir)
        data = loader.load_experiment_data(
            max_epochs=args.max_epochs,
            start_epoch=args.start_epoch,
            systems=args.systems_tuple,
            include_sat_velocity=any(
                v.enable_rbpf_velocity_kf or v.enable_tdcp_smoother
                for v in variants
            ),
        )
        full_ref = _load_full_reference(run_dir / "reference.csv")
        n_emit = _emission_count(full_ref, np.asarray(data["times"], dtype=np.float64))
        print(
            f"  data: {data['n_epochs']} usable / {len(full_ref)} ref epochs "
            f"({100.0 * n_emit / max(len(full_ref), 1):.1f}% coverage), "
            f"constellations={data['constellations']}",
            flush=True,
        )

        dd_computer = None
        dd_pr_computer = None
        any_fgo = any(v.enable_fgo_lambda for v in variants)
        if any_dd or any_dd_for_gate or any_fgo:
            from gnss_gpu.dd_carrier import DDCarrierComputer

            base_obs_path = run_dir / "base.obs"
            rover_obs_path = run_dir / "rover.obs"
            if not base_obs_path.is_file() or not rover_obs_path.is_file():
                print(
                    f"  [DD] WARNING: missing rover.obs/base.obs in {run_dir}, "
                    "DD updates skipped for this run"
                )
            else:
                dd_systems_run = tuple(
                    s.strip() for s in args.dd_systems.split(",") if s.strip()
                )
                dd_computer = DDCarrierComputer(
                    base_obs_path,
                    rover_obs_path=rover_obs_path,
                    base_position=np.asarray(data["base_ecef"], dtype=np.float64),
                    allowed_systems=dd_systems_run,
                    interpolate_base_epochs=bool(args.dd_base_interp),
                )
                if any_fgo:
                    from gnss_gpu.dd_pseudorange import DDPseudorangeComputer

                    dd_pr_computer = DDPseudorangeComputer(
                        base_obs_path,
                        rover_obs_path=rover_obs_path,
                        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
                        allowed_systems=dd_systems_run,
                        interpolate_base_epochs=bool(args.dd_base_interp),
                    )
                    print(
                        f"  [DD-PR] Loaded for FGO absolute anchor: "
                        f"base_systems={dd_systems_run}"
                    )

        hybrid_pos_run: dict[float, np.ndarray] | None = None
        hybrid_velocity_run: dict[float, np.ndarray] | None = None
        hybrid_status_run: dict[float, int] | None = None
        rtkdiag_candidates_run: list[
            tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]
        ] = []
        imu_run: dict[str, np.ndarray] | None = None
        if any(v.enable_zupt or v.enable_imu_tc or v.enable_ins_tc for v in variants):
            try:
                imu_run = loader.load_imu()
            except FileNotFoundError as exc:
                print(
                    f"  [IMU] WARNING: {exc}, ZUPT / Phase 9b/9c IMU-TC "
                    "disabled for this run"
                )
                imu_run = None
        if (any_hybrid or any_fgo) and args.hybrid_pos_dir is not None:
            hybrid_path = args.hybrid_pos_dir / f"{city}_{run}{args.hybrid_pos_suffix}"
            if not hybrid_path.is_file():
                print(
                    f"  [hybrid] WARNING: missing {hybrid_path}, hybrid PU disabled for this run"
                )
            else:
                hybrid_pos_run, hybrid_status_run = _load_hybrid_pos_file(hybrid_path)
                from collections import Counter
                status_dist = Counter(hybrid_status_run.values())
                print(
                    f"  [hybrid] {hybrid_path.name}: "
                    f"{len(hybrid_pos_run)} usable rows, status={dict(status_dist)}"
                )
                hybrid_velocity_run = _build_hybrid_velocity_guide(
                    hybrid_pos_run,
                    np.asarray(data["times"], dtype=np.float64),
                    max_dt_s=float(args.hybrid_vguide_max_dt_s),
                )
                print(
                    f"  [hybrid] velocity guide: {len(hybrid_velocity_run)} epochs "
                    f"(max_dt={args.hybrid_vguide_max_dt_s}s)"
                )
        if any_rtkdiag_pf:
            for label, pos_dir, diag_dir in zip(
                rtkdiag_candidate_labels,
                rtkdiag_candidate_pos_dirs,
                rtkdiag_candidate_diag_dirs,
                strict=True,
            ):
                candidate_pos_path = pos_dir / f"{city}_{run}{args.rtkdiag_candidate_pos_suffix}"
                candidate_diag_path = diag_dir / f"{city}_{run}{args.rtkdiag_candidate_diag_suffix}"
                if not candidate_pos_path.is_file() or not candidate_diag_path.is_file():
                    print(
                        f"  [rtkdiag:{label}] WARNING: missing {candidate_pos_path} or "
                        f"{candidate_diag_path}; candidate skipped for this run"
                    )
                    continue
                candidate_pos_run, _candidate_status = _load_hybrid_pos_file(candidate_pos_path)
                candidate_diag_run = _load_rtk_diag_file(candidate_diag_path)
                rtkdiag_candidates_run.append((label, candidate_pos_run, candidate_diag_run))
                print(
                    f"  [rtkdiag:{label}] {candidate_pos_path.name}: "
                    f"{len(candidate_pos_run)} candidate rows, "
                    f"{len(candidate_diag_run)} diagnostic rows"
                )
        rtkdiag_candidates_run_policy = _filter_rtkdiag_candidates_by_policy(
            rtkdiag_candidates_run,
            city=city,
            run=run,
            policy=str(args.rtkdiag_candidate_run_index_policy),
            blocked_labels=(
                rtkdiag_candidate_block_labels
                | rtkdiag_candidate_block_labels_by_run.get((city, run), set())
            ),
        )
        if len(rtkdiag_candidates_run_policy) != len(rtkdiag_candidates_run):
            kept_labels = {label for label, _, _ in rtkdiag_candidates_run_policy}
            removed_labels = [
                label for label, _, _ in rtkdiag_candidates_run
                if label not in kept_labels
            ]
            print(
                f"  [rtkdiag-policy:{args.rtkdiag_candidate_run_index_policy}] "
                f"filtered candidates for {city}/{run}: "
                f"removed={removed_labels}",
                flush=True,
            )

        wls_positions, wls_ms = run_wls(data)
        print(f"  WLS init done ({wls_ms:.2f} ms/epoch)", flush=True)

        for configured_variant in variants:
            variant = _apply_rtkdiag_run_index_policy(
                configured_variant,
                run=run,
                policy=str(args.rtkdiag_candidate_run_index_policy),
                city=city,
            )
            print(f"  [{variant.method_label}] running ...", flush=True)
            need_dd_for_variant = variant.enable_dd_carrier_afv or (
                variant.enable_rbpf_velocity_kf
                and variant.rbpf_kf_gate_min_dd_pairs is not None
            )
            dd_for_variant = dd_computer if need_dd_for_variant else None
            dd_pr_for_variant = dd_pr_computer if variant.enable_fgo_lambda else None
            hybrid_for_variant = hybrid_pos_run if variant.enable_hybrid_pu else None
            # Phase 9b also consumes the hybrid velocity guide -- it derives
            # the IMU anchor yaw from the rover's course-over-ground at the
            # most recent Status=4 anchor. The PF predict step only uses it
            # when ``enable_hybrid_velocity_guide=True``; for IMU-TC alone we
            # pass the dict but the predict path stays on its constant-velocity
            # default (variant.enable_hybrid_velocity_guide=False).
            hybrid_v_for_variant = (
                hybrid_velocity_run
                if (
                    variant.enable_hybrid_velocity_guide
                    or variant.enable_imu_tc
                    or variant.enable_ins_tc
                )
                else None
            )
            hybrid_status_for_variant = (
                hybrid_status_run
                if (
                    variant.enable_fgo_lambda
                    or variant.enable_tdcp_smoother
                    or variant.enable_zupt
                    or variant.enable_imu_tc
                    or variant.enable_ins_tc
                    or variant.enable_pr_gmm
                    or variant.pr_skip_statuses
                    or (
                        float(variant.pr_prefit_gate_m) > 0.0
                        and str(variant.pr_prefit_ref).strip().lower() == "hybrid"
                    )
                )
                else None
            )
            imu_for_variant = (
                imu_run
                if (variant.enable_zupt or variant.enable_imu_tc or variant.enable_ins_tc)
                else None
            )
            rtkdiag_candidates_for_variant = (
                rtkdiag_candidates_run_policy if variant.enable_rtkdiag_pf_rescue else None
            )
            rtkdiag_candidate_labels_for_variant = (
                [label for label, _, _ in rtkdiag_candidates_for_variant]
                if rtkdiag_candidates_for_variant is not None
                else rtkdiag_candidate_labels
            )
            positions, ms_per_epoch, pr_obs_stats, dd_stats, gate_stats, hybrid_stats, rtkdiag_pf_stats, fgo_stats, tdcp_stats, zupt_stats, imu_tc_stats, ins_tc_stats = _run_ctrbpf_on_segment(
                data, wls_positions, variant,
                dd_computer=dd_for_variant,
                dd_pr_computer=dd_pr_for_variant,
                hybrid_pos=hybrid_for_variant,
                hybrid_velocity=hybrid_v_for_variant,
                hybrid_status=hybrid_status_for_variant,
                rtkdiag_candidates=rtkdiag_candidates_for_variant,
                imu=imu_for_variant,
            )
            score = score_ppc2024(
                np.asarray([p for p in _aligned_positions(full_ref, data["times"], positions)],
                           dtype=np.float64),
                np.array([t for _, t in full_ref], dtype=np.float64),
            )
            pos_path = args.pos_dir / f"{city}_{run}_{variant.method_label}.pos"
            _write_pos_file(pos_path, np.asarray(data["times"]), positions)

            variant_gate_active = variant.enable_rbpf_velocity_kf and (
                variant.rbpf_kf_gate_min_dd_pairs is not None
                or variant.rbpf_kf_gate_min_ess_ratio is not None
                or variant.rbpf_kf_gate_max_spread_m is not None
            )
            row = {
                "city": city,
                "run": run,
                "method": variant.method_label,
                "n_particles": variant.n_particles,
                "n_ref_epochs": len(full_ref),
                "n_pf_epochs": int(data["n_epochs"]),
                "coverage_pct": float(100.0 * n_emit / max(len(full_ref), 1)),
                "honest_ppc_pct": float(score.score_pct),
                "honest_pass_m": float(score.pass_distance_m),
                "honest_total_m": float(score.total_distance_m),
                "ms_per_epoch": float(ms_per_epoch),
                "rbpf_velocity_kf": int(variant.enable_rbpf_velocity_kf),
                "position_update": int(variant.enable_position_update),
                "defer_epoch_resample": int(variant.defer_epoch_resample),
                "deferred_resample_epochs": int(pr_obs_stats.deferred_resample_epochs),
                "pr_weight_mode": str(variant.pr_weight_mode),
                "pr_weight_ref_cn0": float(variant.pr_weight_ref_cn0),
                "pr_weight_min": float(variant.pr_weight_min),
                "pr_weight_max": float(variant.pr_weight_max),
                "pr_prefit_gate_m": float(variant.pr_prefit_gate_m),
                "pr_prefit_gate_min_sats": int(variant.pr_prefit_gate_min_sats),
                "pr_prefit_gate_keep_best": int(variant.pr_prefit_gate_keep_best),
                "pr_prefit_ref": str(variant.pr_prefit_ref),
                "pr_prefit_epochs": int(pr_obs_stats.prefit_epochs),
                "pr_prefit_sats_kept": int(pr_obs_stats.prefit_sats_kept),
                "pr_prefit_sats_dropped": int(pr_obs_stats.prefit_sats_dropped),
                "pr_skip_statuses": ",".join(str(int(s)) for s in variant.pr_skip_statuses),
                "pr_skipped_epochs": int(pr_obs_stats.epochs_skipped),
                "pr_gmm": int(variant.enable_pr_gmm),
                "pr_gmm_epochs": int(pr_obs_stats.epochs_gmm),
                "pr_gaussian_epochs": int(pr_obs_stats.epochs_gaussian),
                "pr_gmm_w_los": float(variant.pr_gmm_w_los),
                "pr_gmm_mu_nlos_m": float(variant.pr_gmm_mu_nlos_m),
                "pr_gmm_sigma_nlos_m": float(variant.pr_gmm_sigma_nlos_m),
                "pr_gmm_hybrid_loose_sigma_m": float(variant.pr_gmm_hybrid_loose_sigma_m),
                "pr_gmm_clock_quantile": float(variant.pr_gmm_clock_quantile),
                "dd_carrier_afv": int(variant.enable_dd_carrier_afv),
                "dd_epochs_attempted": int(dd_stats.epochs_attempted),
                "dd_epochs_applied": int(dd_stats.epochs_applied),
                "dd_pairs_total": int(dd_stats.pairs_total),
                "rbpf_kf_gate_active": int(variant_gate_active),
                "rbpf_kf_gate_min_dd_pairs": (
                    -1 if variant.rbpf_kf_gate_min_dd_pairs is None
                    else int(variant.rbpf_kf_gate_min_dd_pairs)
                ),
                "rbpf_kf_gate_min_ess_ratio": (
                    -1.0 if variant.rbpf_kf_gate_min_ess_ratio is None
                    else float(variant.rbpf_kf_gate_min_ess_ratio)
                ),
                "rbpf_kf_gate_max_spread_m": (
                    -1.0 if variant.rbpf_kf_gate_max_spread_m is None
                    else float(variant.rbpf_kf_gate_max_spread_m)
                ),
                "rbpf_kf_attempted": int(gate_stats.epochs_attempted),
                "rbpf_kf_applied": int(gate_stats.epochs_applied),
                "rbpf_kf_skip_min_dd_pairs": int(gate_stats.skipped_min_dd_pairs),
                "rbpf_kf_skip_min_ess_ratio": int(gate_stats.skipped_min_ess_ratio),
                "rbpf_kf_skip_max_spread": int(gate_stats.skipped_max_spread),
                "hybrid_pu": int(variant.enable_hybrid_pu),
                "hybrid_sigma_m": float(variant.hybrid_sigma_m),
                "hybrid_attempted": int(hybrid_stats.epochs_attempted),
                "hybrid_applied": int(hybrid_stats.epochs_applied),
                "hybrid_lookup_missing": int(hybrid_stats.epochs_lookup_missing),
                "rtkdiag_pf": int(variant.enable_rtkdiag_pf_rescue),
                "rtkdiag_candidate_sigma_m": float(variant.rtkdiag_candidate_sigma_m),
                "rtkdiag_candidate_ratio_min": float(variant.rtkdiag_candidate_ratio_min),
                "rtkdiag_candidate_residual_rms_max": float(
                    variant.rtkdiag_candidate_residual_rms_max
                ),
                "rtkdiag_candidate_emit_max_diff_m": float(
                    variant.rtkdiag_candidate_emit_max_diff_m
                ),
                "rtkdiag_candidate_recenter_max_shift_m": float(
                    variant.rtkdiag_candidate_recenter_max_shift_m
                ),
                "rtkdiag_candidate_select_mode": str(variant.rtkdiag_candidate_select_mode),
                "rtkdiag_candidate_emit_mode": str(variant.rtkdiag_candidate_emit_mode),
                "rtkdiag_candidate_run_index_policy": str(
                    args.rtkdiag_candidate_run_index_policy
                ),
                "rtkdiag_candidate_block_labels": ",".join(
                    sorted(rtkdiag_candidate_block_labels)
                ),
                "rtkdiag_candidate_block_labels_by_run": str(
                    args.rtkdiag_candidate_block_labels_by_run
                ),
                "rtkdiag_candidate_labels": ",".join(rtkdiag_candidate_labels_for_variant),
                "rtkdiag_pf_evaluated": int(rtkdiag_pf_stats.epochs_evaluated),
                "rtkdiag_pf_gate_pass": int(rtkdiag_pf_stats.gate_pass),
                "rtkdiag_pf_candidate_options_total": int(
                    rtkdiag_pf_stats.candidate_options_total
                ),
                "rtkdiag_pf_candidate_missing": int(rtkdiag_pf_stats.candidate_missing),
                "rtkdiag_pf_pu_applied": int(rtkdiag_pf_stats.pu_applied),
                "rtkdiag_pf_recenter_applied": int(rtkdiag_pf_stats.recenter_applied),
                "rtkdiag_pf_recenter_skipped": int(rtkdiag_pf_stats.recenter_skipped),
                "rtkdiag_pf_emit_pf_estimate": int(rtkdiag_pf_stats.emit_pf_estimate),
                "rtkdiag_pf_emit_candidate": int(rtkdiag_pf_stats.emit_candidate),
                "rtkdiag_pf_emit_skipped_pf_drift": int(
                    rtkdiag_pf_stats.emit_skipped_pf_drift
                ),
                "rtkdiag_pf_selected_counts": ",".join(
                    f"{k}:{v}" for k, v in sorted(rtkdiag_pf_stats.selected_counts.items())
                ),
                "fgo_lambda": int(variant.enable_fgo_lambda),
                "fgo_windows_attempted": int(fgo_stats.windows_attempted),
                "fgo_windows_solved": int(fgo_stats.windows_solved),
                "fgo_windows_applied": int(fgo_stats.windows_applied),
                "fgo_n_fixed_total": int(fgo_stats.n_fixed_total),
                "fgo_epochs_replaced": int(fgo_stats.epochs_replaced),
                "tdcp_smoother": int(variant.enable_tdcp_smoother),
                "tdcp_pairs_attempted": int(tdcp_stats.pairs_attempted),
                "tdcp_pairs_accepted": int(tdcp_stats.pairs_accepted),
                "tdcp_pairs_rejected_min_sats": int(tdcp_stats.pairs_rejected_min_sats),
                "tdcp_pairs_rejected_postfit": int(tdcp_stats.pairs_rejected_postfit),
                "zupt": int(variant.enable_zupt),
                "zupt_evaluated": int(zupt_stats.epochs_evaluated),
                "zupt_static": int(zupt_stats.epochs_static),
                "zupt_rewritten": int(zupt_stats.epochs_rewritten),
                "zupt_no_imu": int(zupt_stats.epochs_no_imu),
                "imu_tc": int(variant.enable_imu_tc),
                "imu_tc_evaluated": int(imu_tc_stats.epochs_evaluated),
                "imu_tc_anchor_resets": int(imu_tc_stats.anchor_resets),
                "imu_tc_anchor_resets_static": int(imu_tc_stats.anchor_resets_static),
                "imu_tc_pu_applied": int(imu_tc_stats.pu_applied),
                "imu_tc_pu_skipped_no_imu": int(imu_tc_stats.pu_skipped_no_imu),
                "imu_tc_pu_skipped_no_anchor": int(imu_tc_stats.pu_skipped_no_anchor),
                "imu_tc_pu_skipped_dr_too_long": int(imu_tc_stats.pu_skipped_dr_too_long),
                "imu_tc_pu_skipped_disagreement": int(imu_tc_stats.pu_skipped_disagreement),
                "imu_tc_emit_pf_estimate": int(imu_tc_stats.emit_pf_estimate),
                "imu_tc_emit_skipped_pf_drift": int(imu_tc_stats.emit_skipped_pf_drift),
                "imu_tc_avg_dr_seconds": float(imu_tc_stats.avg_dr_seconds),
                "ins_tc": int(variant.enable_ins_tc),
                "ins_tc_aligned_at_epoch": int(ins_tc_stats.aligned_at_epoch),
                "ins_tc_yaw_initialized_at_epoch": int(ins_tc_stats.yaw_initialized_at_epoch),
                "ins_tc_evaluated": int(ins_tc_stats.epochs_evaluated),
                "ins_tc_pu_applied": int(ins_tc_stats.pu_applied),
                "ins_tc_pu_skipped_not_aligned": int(ins_tc_stats.pu_skipped_not_aligned),
                "ins_tc_pu_skipped_no_yaw": int(ins_tc_stats.pu_skipped_no_yaw),
                "ins_tc_pu_skipped_dr_too_long": int(ins_tc_stats.pu_skipped_dr_too_long),
                "ins_tc_pu_skipped_disagreement": int(ins_tc_stats.pu_skipped_disagreement),
                "ins_tc_particle_imu_initialized": int(ins_tc_stats.particle_imu_initialized),
                "ins_tc_particle_imu_predict_used": int(ins_tc_stats.particle_imu_predict_used),
                "ins_tc_recenter_applied": int(ins_tc_stats.recenter_applied),
                "ins_tc_recenter_skipped": int(ins_tc_stats.recenter_skipped),
                "ins_tc_recenter_avg_shift_m": (
                    float(ins_tc_stats.recenter_shift_sum_m) / max(int(ins_tc_stats.recenter_applied), 1)
                ),
                "ins_tc_motion_predict_used": int(ins_tc_stats.motion_predict_used),
                "ins_tc_obs_status_4_used": int(ins_tc_stats.obs_status_4_used),
                "ins_tc_obs_status_3_used": int(ins_tc_stats.obs_status_3_used),
                "ins_tc_emit_pf_estimate": int(ins_tc_stats.emit_pf_estimate),
                "ins_tc_emit_skipped_pf_drift": int(ins_tc_stats.emit_skipped_pf_drift),
                "ins_tc_final_acc_bias_norm": float(ins_tc_stats.final_acc_bias_norm),
                "ins_tc_final_gyro_bias_norm_dps": float(ins_tc_stats.final_gyro_bias_norm_dps),
                "ins_tc_final_pos_sigma_m": float(ins_tc_stats.final_pos_sigma_m),
            }
            rows.append(row)
            agg_pass[variant.method_label] += row["honest_pass_m"]
            agg_total[variant.method_label] += row["honest_total_m"]
            dd_msg = ""
            if variant.enable_dd_carrier_afv:
                applied = dd_stats.epochs_applied
                attempted = max(dd_stats.epochs_attempted, 1)
                avg_pairs = dd_stats.pairs_total / max(applied, 1)
                dd_msg = (
                    f", DD applied {applied}/{dd_stats.epochs_attempted} "
                    f"({100.0 * applied / attempted:.1f}%, avg pairs={avg_pairs:.1f})"
                )
            pr_msg = ""
            if variant.enable_pr_gmm:
                pr_total = max(pr_obs_stats.epochs_gmm + pr_obs_stats.epochs_gaussian, 1)
                pr_msg = (
                    f", PR-GMM {pr_obs_stats.epochs_gmm}/{pr_total} "
                    f"(w_los={variant.pr_gmm_w_los:.2f}, "
                    f"mu={variant.pr_gmm_mu_nlos_m:.1f}m, "
                    f"hloose={variant.pr_gmm_hybrid_loose_sigma_m:.1f}m, "
                    f"cbq={variant.pr_gmm_clock_quantile:.2f})"
                )
            defer_msg = ""
            if variant.defer_epoch_resample:
                defer_msg = (
                    f", defer-resample {pr_obs_stats.deferred_resample_epochs}"
                )
            pr_weight_msg = ""
            if str(variant.pr_weight_mode) != "raw":
                pr_weight_msg = f", prw={variant.pr_weight_mode}"
            prefit_msg = ""
            if float(variant.pr_prefit_gate_m) > 0.0:
                avg_drop = (
                    pr_obs_stats.prefit_sats_dropped
                    / max(pr_obs_stats.prefit_epochs, 1)
                )
                prefit_msg = (
                    f", PR-prefit {pr_obs_stats.prefit_epochs}ep "
                    f"(drop {avg_drop:.1f}/ep, ref={variant.pr_prefit_ref})"
                )
            pr_skip_msg = ""
            if variant.pr_skip_statuses:
                pr_skip_msg = f", PR-skip {pr_obs_stats.epochs_skipped}"
            gate_msg = ""
            if variant_gate_active and gate_stats.epochs_attempted > 0:
                gate_msg = (
                    f", KF gate {gate_stats.epochs_applied}/"
                    f"{gate_stats.epochs_attempted} "
                    f"(skip dd={gate_stats.skipped_min_dd_pairs}, "
                    f"ess={gate_stats.skipped_min_ess_ratio}, "
                    f"spread={gate_stats.skipped_max_spread})"
                )
            hybrid_msg = ""
            if variant.enable_hybrid_pu and hybrid_stats.epochs_attempted > 0:
                hybrid_msg = (
                    f", hybrid {hybrid_stats.epochs_applied}/"
                    f"{hybrid_stats.epochs_attempted} "
                    f"(missing {hybrid_stats.epochs_lookup_missing})"
                )
            rtkdiag_pf_msg = ""
            if variant.enable_rtkdiag_pf_rescue and rtkdiag_pf_stats.epochs_evaluated > 0:
                selected_counts = ",".join(
                    f"{k}:{v}" for k, v in sorted(rtkdiag_pf_stats.selected_counts.items())
                )
                rtkdiag_pf_msg = (
                    f", RTKDiag-PF gate={rtkdiag_pf_stats.gate_pass}/"
                    f"{rtkdiag_pf_stats.epochs_evaluated} "
                    f"opts={rtkdiag_pf_stats.candidate_options_total} "
                    f"pu={rtkdiag_pf_stats.pu_applied} "
                    f"emit_pf={rtkdiag_pf_stats.emit_pf_estimate} "
                    f"emit_cand={rtkdiag_pf_stats.emit_candidate} "
                    f"(miss={rtkdiag_pf_stats.candidate_missing}, "
                    f"drift_skip={rtkdiag_pf_stats.emit_skipped_pf_drift}, "
                    f"rec={rtkdiag_pf_stats.recenter_applied}/"
                    f"{rtkdiag_pf_stats.recenter_skipped}, "
                    f"sel={selected_counts})"
                )
            zupt_msg = ""
            if variant.enable_zupt and zupt_stats.epochs_evaluated > 0:
                zupt_msg = (
                    f", ZUPT static {zupt_stats.epochs_static}/"
                    f"{zupt_stats.epochs_evaluated} "
                    f"(rewritten {zupt_stats.epochs_rewritten}, no_imu {zupt_stats.epochs_no_imu})"
                )
            imu_tc_msg = ""
            if variant.enable_imu_tc and imu_tc_stats.epochs_evaluated > 0:
                imu_tc_msg = (
                    f", IMU-TC pu={imu_tc_stats.pu_applied}/"
                    f"{imu_tc_stats.epochs_evaluated} "
                    f"emit_pf={imu_tc_stats.emit_pf_estimate} "
                    f"(anchors={imu_tc_stats.anchor_resets} "
                    f"static={imu_tc_stats.anchor_resets_static}, "
                    f"skip dr={imu_tc_stats.pu_skipped_dr_too_long} "
                    f"dis={imu_tc_stats.pu_skipped_disagreement} "
                    f"no_anc={imu_tc_stats.pu_skipped_no_anchor}, "
                    f"avg_dr={imu_tc_stats.avg_dr_seconds:.2f}s, "
                    f"pf_drift_skip={imu_tc_stats.emit_skipped_pf_drift})"
                )
            ins_tc_msg = ""
            if variant.enable_ins_tc and ins_tc_stats.epochs_evaluated > 0:
                ins_tc_msg = (
                    f", INS-TC align={ins_tc_stats.aligned_at_epoch} "
                    f"yaw={ins_tc_stats.yaw_initialized_at_epoch} "
                    f"pu={ins_tc_stats.pu_applied}/{ins_tc_stats.epochs_evaluated} "
                    f"emit_pf={ins_tc_stats.emit_pf_estimate} "
                    f"(pimu_init={ins_tc_stats.particle_imu_initialized}, "
                    f"pimu={ins_tc_stats.particle_imu_predict_used}, "
                    f"rec={ins_tc_stats.recenter_applied}/{ins_tc_stats.recenter_skipped}, "
                    f"motion={ins_tc_stats.motion_predict_used}, "
                    f"obs4={ins_tc_stats.obs_status_4_used}, "
                    f"obs3={ins_tc_stats.obs_status_3_used}, "
                    f"skip align={ins_tc_stats.pu_skipped_not_aligned} "
                    f"yaw={ins_tc_stats.pu_skipped_no_yaw} "
                    f"dr={ins_tc_stats.pu_skipped_dr_too_long} "
                    f"dis={ins_tc_stats.pu_skipped_disagreement}, "
                    f"ba={ins_tc_stats.final_acc_bias_norm:.3f}m/s2, "
                    f"bg={ins_tc_stats.final_gyro_bias_norm_dps:.3f}dps, "
                    f"possig={ins_tc_stats.final_pos_sigma_m:.2f}m, "
                    f"pf_drift_skip={ins_tc_stats.emit_skipped_pf_drift})"
                )
            tdcp_msg = ""
            if variant.enable_tdcp_smoother and tdcp_stats.pairs_attempted > 0:
                tdcp_msg = (
                    f", TDCP {tdcp_stats.pairs_accepted}/"
                    f"{tdcp_stats.pairs_attempted} pairs "
                    f"(reject min_sats={tdcp_stats.pairs_rejected_min_sats}, "
                    f"postfit={tdcp_stats.pairs_rejected_postfit})"
                )
            fgo_msg = ""
            if variant.enable_fgo_lambda and fgo_stats.windows_attempted > 0:
                fgo_msg = (
                    f", FGO solved {fgo_stats.windows_solved}/"
                    f"{fgo_stats.windows_attempted} applied {fgo_stats.windows_applied} "
                    f"(fixed {fgo_stats.n_fixed_total}, replaced {fgo_stats.epochs_replaced} ep)"
                )
            print(
                f"    PPC honest: {row['honest_ppc_pct']:5.2f}%  "
                f"(pass {row['honest_pass_m']:.0f} / total {row['honest_total_m']:.0f}m, "
                f"{ms_per_epoch:.1f} ms/epoch{pr_msg}{pr_weight_msg}{prefit_msg}{pr_skip_msg}{defer_msg}{dd_msg}{gate_msg}{hybrid_msg}{rtkdiag_pf_msg}{zupt_msg}{imu_tc_msg}{ins_tc_msg}{tdcp_msg}{fgo_msg})",
                flush=True,
            )

    if not rows:
        raise SystemExit("no rows produced")

    out_csv = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 72)
    print("  Honest aggregates (denominator = full rover-epoch arc length;")
    print("  TURING target is 85.6%, libgnss++ hybrid baseline is 50.91%)")
    for method in [v.method_label for v in variants]:
        total = agg_total[method]
        if total <= 0:
            continue
        pct = 100.0 * agg_pass[method] / total
        print(
            f"  {method:18s}: {pct:5.2f}%   "
            f"(pass {agg_pass[method]:.0f}m / total {total:.0f}m)"
        )
    print(f"  Saved: {out_csv}")
    print("=" * 72)


def _aligned_positions(
    full_ref: list[tuple[float, np.ndarray]],
    times_used: np.ndarray,
    positions: np.ndarray,
) -> list[np.ndarray]:
    """Yield estimated positions aligned to full_ref order, [0,0,0] if missing."""
    pos_map = {round(float(t), 1): p for t, p in zip(times_used, positions, strict=True)}
    out: list[np.ndarray] = []
    for tow, _t in full_ref:
        match = pos_map.get(round(float(tow), 1))
        if match is None or not np.all(np.isfinite(match)):
            out.append(np.zeros(3, dtype=np.float64))
        else:
            out.append(np.asarray(match, dtype=np.float64))
    return out


if __name__ == "__main__":
    main()
