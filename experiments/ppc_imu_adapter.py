#!/usr/bin/env python3
"""TASK_D D3.1 adapter: PPCDatasetLoader.load_imu() -> IMUPreintegration.

Wires PPC-Dataset (taroz/PPC-Dataset) 100 Hz ``imu.csv`` samples into the
GSDC2023 IMU preintegration helpers (``experiments/gsdc2023_imu.py``) so the
resulting ``imu_delta_p`` / ``imu_delta_v`` (+ bias Jacobians) can be sliced
per-chunk with ``imu_preintegration_segment_with_bias_jacobians`` and fed to
the native ``fgo_gnss_lm_vd`` solver as loosely-coupled priors between GNSS
epochs.

Assumptions (documented per TASK_D D3.1; revisit if scoring regresses):

- **Units**: PPC ``imu.csv`` accelerometer columns are already SI
  (``Acc {X,Y,Z} (m/s^2)``); gyro columns are ``Ang Rate {X,Y,Z} (deg/s)``
  and are converted to rad/s here (mirrors
  ``experiments/exp_ppc_imu_fusion.py:_load_imu``).
- **Timebase**: PPC's ``imu.csv`` time column reuses the same GPS-TOW aliases
  as ``reference.csv`` (see ``gnss_gpu/io/ppc.py:_IMU_ALIASES["time"]``), i.e.
  the PPC dataset ships IMU samples *already synchronized* to GPS time of
  week -- no separate IMU/GNSS clock alignment is performed here beyond
  converting seconds to the milliseconds unit expected by
  ``preintegrate_processed_imu``.
- **Axes/mounting**: the PPC dataset does not document the IMU->vehicle
  mounting rotation. We assume a zero static mounting angle (IMU axes
  == vehicle body axes) and estimate heading purely from the GNSS-derived
  velocity direction (``delta_frame="ecef"`` mode of
  ``preintegrate_processed_imu``, i.e. a flat-vehicle / zero roll-pitch
  assumption). This is a documented approximation, not a calibrated
  boresight; see WP3B_REPORT.md D3 findings for the resulting sensitivity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiments.gsdc2023_imu import (  # noqa: E402
    IMU_GRAVITY_MPS2,
    IMUPreintegration,
    ProcessedIMU,
    preintegrate_processed_imu,
)
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402

DEG2RAD = np.pi / 180.0

# Documented mounting assumption (see module docstring): identity rotation.
PPC_IMU_MOUNTING_ANGLE_RAD = np.zeros(3, dtype=np.float64)


def ppc_imu_to_processed(
    imu_data: dict[str, np.ndarray],
) -> tuple[ProcessedIMU, ProcessedIMU]:
    """Convert a ``PPCDatasetLoader.load_imu()`` dict into (acc, gyro) ``ProcessedIMU``.

    Gyro columns are degrees/second in the PPC CSV and are converted to
    radians/second. Accelerometer columns are already m/s^2.
    """
    times_s = np.asarray(imu_data["time"], dtype=np.float64)
    times_ms = times_s * 1000.0
    acc_xyz = np.column_stack(
        [
            np.asarray(imu_data["acc_x"], dtype=np.float64),
            np.asarray(imu_data["acc_y"], dtype=np.float64),
            np.asarray(imu_data["acc_z"], dtype=np.float64),
        ]
    )
    gyro_xyz_dps = np.column_stack(
        [
            np.asarray(imu_data["gyro_x"], dtype=np.float64),
            np.asarray(imu_data["gyro_y"], dtype=np.float64),
            np.asarray(imu_data["gyro_z"], dtype=np.float64),
        ]
    )
    gyro_xyz = gyro_xyz_dps * DEG2RAD

    finite = (
        np.isfinite(times_ms)
        & np.isfinite(acc_xyz).all(axis=1)
        & np.isfinite(gyro_xyz).all(axis=1)
    )
    times_ms = times_ms[finite]
    acc_xyz = acc_xyz[finite]
    gyro_xyz = gyro_xyz[finite]

    dt_s = np.zeros(times_ms.size, dtype=np.float64)
    if times_ms.size > 1:
        dt_s[1:] = np.diff(times_ms) / 1000.0

    zeros3 = np.zeros((times_ms.size, 3), dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=times_ms,
        xyz=acc_xyz,
        dt_s=dt_s,
        norm_3d=np.linalg.norm(acc_xyz, axis=1) if acc_xyz.size else np.zeros(0),
        norm_std=np.zeros(times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
        bias=zeros3.copy(),
    )
    gyro = ProcessedIMU(
        times_ms=times_ms,
        xyz=gyro_xyz,
        dt_s=dt_s,
        norm_3d=np.linalg.norm(gyro_xyz, axis=1) if gyro_xyz.size else np.zeros(0),
        norm_std=np.zeros(times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
        bias=zeros3.copy(),
    )
    return acc, gyro


def build_ppc_imu_preintegration(
    imu_data: dict[str, np.ndarray],
    epoch_times_s: np.ndarray,
    reference_ecef: np.ndarray,
    *,
    delta_frame: str = "ecef",
    gravity_mps2: float = IMU_GRAVITY_MPS2,
    mounting_angle_rad: np.ndarray = PPC_IMU_MOUNTING_ANGLE_RAD,
) -> IMUPreintegration:
    """Build a full-timeline ``IMUPreintegration`` aligned to GNSS epoch times.

    ``epoch_times_s``: ``(T,)`` GNSS epoch GPS-TOW seconds (same series used
    to index ``sat_ecef`` / ``pseudorange`` / ``fgo_state``).
    ``reference_ecef``: ``(T, 3)`` receiver ECEF position per epoch, used only
    to derive local heading (via GNSS-velocity-implied yaw) and the local
    gravity direction for the ``"ecef"`` delta frame. WLS or FGO positions
    both work; this does not need to be the final answer, just a reasonable
    per-epoch position for the ENU frame construction.
    """
    acc, gyro = ppc_imu_to_processed(imu_data)
    epoch_times_ms = np.asarray(epoch_times_s, dtype=np.float64) * 1000.0
    return preintegrate_processed_imu(
        acc,
        gyro,
        epoch_times_ms,
        delta_frame=delta_frame,
        reference_xyz_ecef=np.asarray(reference_ecef, dtype=np.float64),
        gravity_mps2=gravity_mps2,
        mounting_angle_rad=np.asarray(mounting_angle_rad, dtype=np.float64),
        sample_dt_mode="bounded",
    )


def load_ppc_imu_preintegration(
    run_dir: Path,
    epoch_times_s: np.ndarray,
    reference_ecef: np.ndarray,
    *,
    delta_frame: str = "ecef",
    gravity_mps2: float = IMU_GRAVITY_MPS2,
    mounting_angle_rad: np.ndarray = PPC_IMU_MOUNTING_ANGLE_RAD,
) -> IMUPreintegration:
    """Load ``run_dir/imu.csv`` and preintegrate it against ``epoch_times_s``."""
    imu_data = PPCDatasetLoader(run_dir).load_imu()
    return build_ppc_imu_preintegration(
        imu_data,
        epoch_times_s,
        reference_ecef,
        delta_frame=delta_frame,
        gravity_mps2=gravity_mps2,
        mounting_angle_rad=mounting_angle_rad,
    )
