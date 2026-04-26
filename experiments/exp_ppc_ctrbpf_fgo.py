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
from dataclasses import dataclass
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
class _FGOStats:
    windows_attempted: int = 0
    windows_solved: int = 0
    windows_applied: int = 0
    n_fixed_total: int = 0
    n_fixed_observations_total: int = 0
    epochs_replaced: int = 0


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
) -> tuple[np.ndarray, float, _DDStats, _RBPFGateStats, _HybridStats, _FGOStats]:
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
    dd_stats = _DDStats()
    gate_stats = _RBPFGateStats()
    hybrid_stats = _HybridStats()
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
    use_vguide = (
        config.enable_hybrid_velocity_guide
        and hybrid_velocity is not None
        and len(hybrid_velocity) > 0
    )
    fgo_dd_cache: list = [None] * n_epochs  # holds DDCarrierEpoch | None
    fgo_dd_pr_cache: list = [None] * n_epochs  # holds DDPseudorangeEpoch | None
    fgo_stats = _FGOStats()

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
        if i == 0:
            dt = float(data.get("dt", 0.2))
        else:
            dt = float(times[i] - times[i - 1])
            if not np.isfinite(dt) or dt <= 0.0:
                dt = float(data.get("dt", 0.2))

        v_guide = None
        if use_vguide:
            v_guide = hybrid_velocity.get(round(float(times[i]), 1))
        if v_guide is not None and np.all(np.isfinite(v_guide)):
            pf.predict(
                velocity=np.asarray(v_guide, dtype=np.float64),
                dt=dt,
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

        if config.enable_correct_clock_bias and i % 5 == 0:
            pf.correct_clock_bias(sat_i, pr_i)

        pf.update(sat_i, pr_i, weights=w_i)

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
            hp = hybrid_pos.get(round(float(times[i]), 1))
            if hp is None:
                hybrid_stats.epochs_lookup_missing += 1
            elif np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                pf.position_update(hp, sigma_pos=float(config.hybrid_sigma_m))
                hybrid_stats.epochs_applied += 1

        est = np.asarray(pf.estimate(), dtype=np.float64)
        if (
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

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)

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

    return positions, ms_per_epoch, dd_stats, gate_stats, hybrid_stats, fgo_stats


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
            "rbpf+dd+gate+phase7, rbpf+dd+gate+phase4, "
            "rbpf+dd+gate+hybrid+phase4}"
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
    # Phase 4: post-process FGO + LAMBDA partial fix
    parser.add_argument("--fgo-window-size", type=int, default=30,
                        help="FGO window size in epochs (default 30)")
    parser.add_argument("--fgo-window-stride", type=int, default=15,
                        help="FGO window stride in epochs (default 15, 50% overlap)")
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
            include_sat_velocity=any(v.enable_rbpf_velocity_kf for v in variants),
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

        wls_positions, wls_ms = run_wls(data)
        print(f"  WLS init done ({wls_ms:.2f} ms/epoch)", flush=True)

        for variant in variants:
            print(f"  [{variant.method_label}] running ...", flush=True)
            need_dd_for_variant = variant.enable_dd_carrier_afv or (
                variant.enable_rbpf_velocity_kf
                and variant.rbpf_kf_gate_min_dd_pairs is not None
            )
            dd_for_variant = dd_computer if need_dd_for_variant else None
            dd_pr_for_variant = dd_pr_computer if variant.enable_fgo_lambda else None
            hybrid_for_variant = hybrid_pos_run if variant.enable_hybrid_pu else None
            hybrid_v_for_variant = (
                hybrid_velocity_run if variant.enable_hybrid_velocity_guide else None
            )
            hybrid_status_for_variant = (
                hybrid_status_run if variant.enable_fgo_lambda else None
            )
            positions, ms_per_epoch, dd_stats, gate_stats, hybrid_stats, fgo_stats = _run_ctrbpf_on_segment(
                data, wls_positions, variant,
                dd_computer=dd_for_variant,
                dd_pr_computer=dd_pr_for_variant,
                hybrid_pos=hybrid_for_variant,
                hybrid_velocity=hybrid_v_for_variant,
                hybrid_status=hybrid_status_for_variant,
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
                "fgo_lambda": int(variant.enable_fgo_lambda),
                "fgo_windows_attempted": int(fgo_stats.windows_attempted),
                "fgo_windows_solved": int(fgo_stats.windows_solved),
                "fgo_windows_applied": int(fgo_stats.windows_applied),
                "fgo_n_fixed_total": int(fgo_stats.n_fixed_total),
                "fgo_epochs_replaced": int(fgo_stats.epochs_replaced),
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
                f"{ms_per_epoch:.1f} ms/epoch{dd_msg}{gate_msg}{hybrid_msg}{fgo_msg})",
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
