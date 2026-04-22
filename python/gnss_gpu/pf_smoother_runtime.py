"""Runtime setup helpers for PF smoother evaluation runs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gnss_gpu import ParticleFilterDevice
from gnss_gpu.imu import ComplementaryHeadingFilter, load_imu_csv
from gnss_gpu.pf_smoother_config import (
    DopplerConfig,
    ObservationConfig,
    ParticleFilterRuntimeConfig,
    RobustMeasurementConfig,
)


@dataclass(frozen=True)
class RunDataset:
    epochs: object
    spp_lookup: dict[float, np.ndarray]
    gt: np.ndarray
    our_times: np.ndarray
    first_pos: np.ndarray
    init_cb: float
    imu_data: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class ObservationComputers:
    base_obs_path: Path | None
    dd_pr_computer: object | None = None
    wl_computer: object | None = None
    dd_computer: object | None = None
    carrier_bias_tracker: dict[tuple[int, int], object] = field(default_factory=dict)


@dataclass
class ForwardRunBuffers:
    forward_aligned: list[np.ndarray] = field(default_factory=list)
    gt_aligned: list[np.ndarray] = field(default_factory=list)
    aligned_indices: list[int] = field(default_factory=list)
    aligned_stop_flags: list[bool] = field(default_factory=list)
    aligned_epoch_diagnostics: list[dict[str, object]] = field(default_factory=list)
    stored_motion_deltas: list[np.ndarray] = field(default_factory=list)
    stored_tdcp_motion_deltas: list[np.ndarray] = field(default_factory=list)
    stored_dd_carrier: list[Any | None] = field(default_factory=list)
    stored_dd_pseudorange: list[Any | None] = field(default_factory=list)
    stored_undiff_pr: list[Any | None] = field(default_factory=list)

    @property
    def n_stored(self) -> int:
        return len(self.stored_dd_carrier)

    def append_smoother_motion(
        self,
        velocity: np.ndarray | None,
        tdcp_motion_velocity: np.ndarray | None,
        *,
        dt: float,
        need_tdcp_motion: bool,
    ) -> None:
        if self.n_stored > 0:
            self.stored_motion_deltas.append(_motion_delta_or_nan(velocity, dt))
            if need_tdcp_motion:
                self.stored_tdcp_motion_deltas.append(
                    _motion_delta_or_nan(tdcp_motion_velocity, dt)
                )
            else:
                self.stored_tdcp_motion_deltas.append(np.full(3, np.nan, dtype=np.float64))

    def append_smoother_observations(
        self,
        dd_carrier_epoch: Any | None,
        dd_pseudorange_epoch: Any | None,
        undiff_pr_epoch: Any | None,
    ) -> None:
        self.stored_dd_carrier.append(dd_carrier_epoch)
        self.stored_dd_pseudorange.append(dd_pseudorange_epoch)
        self.stored_undiff_pr.append(undiff_pr_epoch)

    def append_aligned_epoch(
        self,
        forward_position: np.ndarray,
        ground_truth: np.ndarray,
        *,
        stop_flag: bool,
        use_smoother: bool,
    ) -> int:
        self.forward_aligned.append(np.asarray(forward_position, dtype=np.float64).copy())
        self.gt_aligned.append(np.asarray(ground_truth, dtype=np.float64).copy())
        self.aligned_stop_flags.append(bool(stop_flag))
        if use_smoother:
            self.aligned_indices.append(self.n_stored - 1)
        return len(self.forward_aligned) - 1


def _motion_delta_or_nan(
    velocity: np.ndarray | None,
    dt: float,
) -> np.ndarray:
    if velocity is not None and dt > 0 and np.isfinite(velocity).all():
        return np.asarray(velocity, dtype=np.float64).ravel()[:3] * float(dt)
    return np.full(3, np.nan, dtype=np.float64)


def coerce_run_dataset(dataset: RunDataset | Mapping[str, Any]) -> RunDataset:
    if isinstance(dataset, RunDataset):
        return dataset

    spp_lookup = {
        float(k): np.asarray(v, dtype=np.float64)
        for k, v in dict(dataset["spp_lookup"]).items()
    }
    imu_data = dataset.get("imu_data")
    return RunDataset(
        epochs=dataset["epochs"],
        spp_lookup=spp_lookup,
        gt=np.asarray(dataset["gt"], dtype=np.float64),
        our_times=np.asarray(dataset["our_times"], dtype=np.float64),
        first_pos=np.asarray(dataset["first_pos"], dtype=np.float64),
        init_cb=float(dataset["init_cb"]),
        imu_data=imu_data,
    )


def resolve_run_dataset(
    run_dir: Path,
    rover_source: str,
    dataset: RunDataset | Mapping[str, Any] | None,
    loader: Callable[[Path, str], Mapping[str, Any]],
) -> RunDataset:
    if dataset is None:
        return coerce_run_dataset(loader(run_dir, rover_source))
    return coerce_run_dataset(dataset)


def find_base_obs_path(run_dir: Path) -> Path | None:
    for name in ("base_trimble.obs", "base.obs"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def spp_heading_from_velocity(
    spp_vel_ecef: np.ndarray,
    lat: float,
    lon: float,
) -> float | None:
    """Compute heading in radians from north, clockwise, from ECEF velocity."""

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    ve = -sin_lon * spp_vel_ecef[0] + cos_lon * spp_vel_ecef[1]
    vn = (
        -sin_lat * cos_lon * spp_vel_ecef[0]
        - sin_lat * sin_lon * spp_vel_ecef[1]
        + cos_lat * spp_vel_ecef[2]
    )
    speed = math.sqrt(ve**2 + vn**2)
    if speed < 0.5:
        return None
    return math.atan2(ve, vn)


def initialize_imu_filter(
    run_dir: Path,
    predict_guide: str,
    dataset: RunDataset,
    ecef_to_lla_func: Callable[[float, float, float], tuple[float, float, float]],
    *,
    alpha: float = 0.05,
) -> ComplementaryHeadingFilter | None:
    if predict_guide not in ("imu", "imu_spp_blend"):
        return None

    imu_data = dataset.imu_data
    if imu_data is None:
        imu_path = run_dir / "imu.csv"
        if imu_path.exists():
            imu_data = load_imu_csv(imu_path)
        else:
            raise RuntimeError(
                f"predict_guide={predict_guide} requires IMU data but "
                f"imu.csv not found in {run_dir}"
            )

    imu_filter = ComplementaryHeadingFilter(imu_data, alpha=alpha)
    tow_keys = sorted(dataset.spp_lookup.keys())
    if len(tow_keys) >= 2:
        p0 = np.asarray(dataset.spp_lookup[tow_keys[0]][:3], dtype=np.float64)
        p1 = np.asarray(dataset.spp_lookup[tow_keys[1]][:3], dtype=np.float64)
        lat0, lon0, _ = ecef_to_lla_func(float(p0[0]), float(p0[1]), float(p0[2]))
        dt_init = float(tow_keys[1] - tow_keys[0])
        if dt_init > 0:
            spp_vel_init = (p1 - p0) / dt_init
            heading = spp_heading_from_velocity(spp_vel_init, lat0, lon0)
            if heading is not None:
                imu_filter.heading = heading
    return imu_filter


def build_observation_computers(
    run_dir: Path,
    rover_source: str,
    observations: ObservationConfig,
) -> ObservationComputers:
    needs_base = (
        observations.dd_pseudorange.enabled
        or observations.dd_carrier.enabled
        or observations.widelane.enabled
    )
    base_obs_path = find_base_obs_path(run_dir) if needs_base else None
    rover_obs_path = run_dir / f"rover_{rover_source}.obs"

    dd_pr_computer = None
    if observations.dd_pseudorange.enabled:
        from gnss_gpu.dd_pseudorange import DDPseudorangeComputer

        if base_obs_path is None:
            raise RuntimeError(
                "dd_pseudorange requires base station RINEX "
                f"(expected base_trimble.obs or base.obs in {run_dir})"
            )
        dd_pr_computer = DDPseudorangeComputer(
            base_obs_path,
            rover_obs_path=rover_obs_path,
            interpolate_base_epochs=observations.dd_pseudorange.base_interp,
        )
        print(f"  [DD-PR] base_pos = {dd_pr_computer.base_position}")

    wl_computer = None
    if observations.widelane.enabled:
        from gnss_gpu.widelane import WidelaneDDPseudorangeComputer

        if base_obs_path is None:
            raise RuntimeError(
                "widelane requires base station RINEX "
                f"(expected base_trimble.obs or base.obs in {run_dir})"
            )
        wl_computer = WidelaneDDPseudorangeComputer(
            base_obs_path,
            rover_obs_path=rover_obs_path,
            interpolate_base_epochs=bool(
                observations.dd_carrier.base_interp
                or observations.dd_pseudorange.base_interp
            ),
            ratio_threshold=observations.widelane.ratio_threshold,
            min_fix_rate=observations.widelane.min_fix_rate,
        )
        print(
            f"  [WL] base_pos = {wl_computer.base_position}, "
            f"min_fix_rate={observations.widelane.min_fix_rate:.2f}, "
            f"ratio={observations.widelane.ratio_threshold:.1f}"
        )

    dd_computer = None
    if observations.dd_carrier.enabled:
        from gnss_gpu.dd_carrier import DDCarrierComputer

        if base_obs_path is not None:
            dd_computer = DDCarrierComputer(
                base_obs_path,
                rover_obs_path=rover_obs_path,
                interpolate_base_epochs=observations.dd_carrier.base_interp,
            )
            print(f"  [DD] base_pos = {dd_computer.base_position}")
        else:
            print(f"  [DD] WARNING: no base station RINEX found in {run_dir}, DD disabled")

    return ObservationComputers(
        base_obs_path=base_obs_path,
        dd_pr_computer=dd_pr_computer,
        wl_computer=wl_computer,
        dd_computer=dd_computer,
    )


def initialize_particle_filter(
    first_pos: np.ndarray,
    init_cb: float,
    particle_filter: ParticleFilterRuntimeConfig,
    robust: RobustMeasurementConfig,
    doppler: DopplerConfig,
    *,
    sigma_cb: float,
    seed: int = 42,
) -> ParticleFilterDevice:
    pf = ParticleFilterDevice(
        n_particles=particle_filter.n_particles,
        sigma_pos=particle_filter.sigma_pos,
        sigma_cb=sigma_cb,
        sigma_pr=particle_filter.sigma_pr,
        resampling=particle_filter.resampling,
        seed=seed,
        per_particle_nlos_gate=robust.per_particle_nlos_gate,
        per_particle_nlos_dd_pr_threshold_m=robust.per_particle_nlos_dd_pr_threshold_m,
        per_particle_nlos_dd_carrier_threshold_cycles=(
            robust.per_particle_nlos_dd_carrier_threshold_cycles
        ),
        per_particle_nlos_undiff_pr_threshold_m=(
            robust.per_particle_nlos_undiff_pr_threshold_m
        ),
        per_particle_huber=robust.per_particle_huber,
        per_particle_huber_dd_pr_k=robust.per_particle_huber_dd_pr_k,
        per_particle_huber_dd_carrier_k=robust.per_particle_huber_dd_carrier_k,
        per_particle_huber_undiff_pr_k=robust.per_particle_huber_undiff_pr_k,
        sigma_vel=particle_filter.pf_sigma_vel,
        velocity_guide_alpha=particle_filter.pf_velocity_guide_alpha,
        rbpf_velocity_kf=doppler.rbpf_velocity_kf,
        velocity_process_noise=doppler.rbpf_velocity_process_noise,
    )
    pf.initialize(
        np.asarray(first_pos, dtype=np.float64),
        clock_bias=float(init_cb),
        spread_pos=10.0,
        spread_cb=100.0,
        spread_vel=(
            0.0 if doppler.rbpf_velocity_kf else particle_filter.pf_init_spread_vel
        ),
        velocity_init_sigma=(
            doppler.rbpf_velocity_init_sigma if doppler.rbpf_velocity_kf else 0.0
        ),
    )
    if particle_filter.use_smoother:
        pf.enable_smoothing()
    return pf
