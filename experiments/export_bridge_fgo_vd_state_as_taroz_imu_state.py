#!/usr/bin/env python3
"""Convert exported native FGO VD state into Taroz ``phone_data_imu_state`` schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.compare_gsdc2023_taroz_imu_state import (
    infer_origin_ecef_from_first_pair,
    taroz_preprocessing_origin_ecef,
)
from experiments.gsdc2023_imu import ecef_to_enu_relative, enu_to_ecef_relative, rotvec_to_rotm
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT


def gtsam_rotm_to_rzryrx(rotm: np.ndarray) -> np.ndarray:
    """Invert GTSAM ``Rot3.RzRyRx(roll, pitch, yaw)`` for a stack of matrices."""

    rot = np.asarray(rotm, dtype=np.float64).reshape(-1, 3, 3)
    out = np.full((rot.shape[0], 3), np.nan, dtype=np.float64)
    for idx, r in enumerate(rot):
        if not np.isfinite(r).all():
            continue
        pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
        cp = np.cos(pitch)
        if abs(cp) > 1.0e-12:
            roll = np.arctan2(r[2, 1], r[2, 2])
            yaw = np.arctan2(r[1, 0], r[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-r[0, 1], r[1, 1])
        out[idx] = [roll, pitch, yaw]
    return out


def _clock_bias_columns(frame: pd.DataFrame, prefix: str = "FgoVdClockBiasMeters") -> list[str]:
    cols = [col for col in frame.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda col: int(col.removeprefix(prefix)))


def _state_extra(frame: pd.DataFrame, idx: int) -> np.ndarray:
    col = f"FgoVdStateExtra{idx}"
    if col not in frame.columns:
        return np.full(frame.shape[0], np.nan, dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)


def infer_origin_ecef_from_bridge_fgo_vd_state(
    bridge_vd_state: pd.DataFrame,
    template_state: pd.DataFrame,
    *,
    initial_origin_ecef: np.ndarray | None = None,
) -> np.ndarray:
    """Infer Taroz ENU origin from the first common native pose/MATLAB state."""

    required_bridge = {"UnixTimeMillis", "FgoVdEcefXMeters", "FgoVdEcefYMeters", "FgoVdEcefZMeters"}
    required_template = {"utcTimeMillis", "position_x", "position_y", "position_z"}
    missing_bridge = sorted(required_bridge - set(bridge_vd_state.columns))
    missing_template = sorted(required_template - set(template_state.columns))
    if missing_bridge:
        raise ValueError(f"bridge VD state is missing columns: {missing_bridge}")
    if missing_template:
        raise ValueError(f"template state is missing columns: {missing_template}")

    bridge = bridge_vd_state.copy()
    bridge["utcTimeMillis"] = pd.to_numeric(bridge["UnixTimeMillis"], errors="coerce").round().astype("Int64")
    bridge = bridge.dropna(subset=["utcTimeMillis"]).copy()
    bridge["utcTimeMillis"] = bridge["utcTimeMillis"].astype(np.int64)
    native_cols = ["FgoVdEcefXMeters", "FgoVdEcefYMeters", "FgoVdEcefZMeters"]
    if all(f"FgoVdStateExtra{idx}" in bridge.columns for idx in range(12)):
        native_cols = [f"FgoVdStateExtra{idx}" for idx in range(3)]
    native = bridge[["utcTimeMillis", *native_cols]].rename(
        columns={native_cols[0]: "native_x", native_cols[1]: "native_y", native_cols[2]: "native_z"}
    )

    template = template_state[["utcTimeMillis", "position_x", "position_y", "position_z"]].copy()
    template["utcTimeMillis"] = pd.to_numeric(template["utcTimeMillis"], errors="coerce").round().astype("Int64")
    template = template.dropna(subset=["utcTimeMillis"]).copy()
    template["utcTimeMillis"] = template["utcTimeMillis"].astype(np.int64)
    joined = native.merge(template, on="utcTimeMillis", how="inner", sort=False)
    if joined.empty:
        raise ValueError("bridge VD state and template state have no common utcTimeMillis")
    native_xyz = joined[["native_x", "native_y", "native_z"]].to_numpy(dtype=np.float64)
    matlab_enu = joined[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
    finite = np.isfinite(native_xyz).all(axis=1) & np.isfinite(matlab_enu).all(axis=1)
    if not finite.any():
        raise ValueError("joined state table has no finite native/MATLAB state pair")
    idx = int(np.flatnonzero(finite)[0])
    return infer_origin_ecef_from_first_pair(
        native_xyz[idx],
        matlab_enu[idx],
        initial_origin_ecef=initial_origin_ecef,
    )


def bridge_fgo_vd_state_to_taroz_imu_state(
    bridge_vd_state: pd.DataFrame,
    *,
    origin_ecef: np.ndarray,
    template_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a Taroz-compatible state table from ``bridge_fgo_vd_state.csv``."""

    required = {
        "UnixTimeMillis",
        "FgoVdEcefXMeters",
        "FgoVdEcefYMeters",
        "FgoVdEcefZMeters",
        "FgoVdVelocityXMps",
        "FgoVdVelocityYMps",
        "FgoVdVelocityZMps",
        "FgoVdClockDriftMps",
    }
    missing = sorted(required - set(bridge_vd_state.columns))
    if missing:
        raise ValueError(f"bridge VD state is missing columns: {missing}")

    source = bridge_vd_state.copy()
    source["utcTimeMillis"] = pd.to_numeric(source["UnixTimeMillis"], errors="coerce").round().astype("Int64")
    source = source.dropna(subset=["utcTimeMillis"]).copy()
    source["utcTimeMillis"] = source["utcTimeMillis"].astype(np.int64)
    source = source.sort_values("utcTimeMillis", kind="mergesort").drop_duplicates("utcTimeMillis", keep="first")

    if template_state is not None and not template_state.empty:
        template = template_state[["epoch_index", "utcTimeMillis"]].copy()
        template["utcTimeMillis"] = pd.to_numeric(template["utcTimeMillis"], errors="coerce").round().astype("Int64")
        template = template.dropna(subset=["utcTimeMillis"]).copy()
        template["utcTimeMillis"] = template["utcTimeMillis"].astype(np.int64)
        template = template.sort_values("utcTimeMillis", kind="mergesort").drop_duplicates("utcTimeMillis", keep="first")
        joined = template.merge(source, on="utcTimeMillis", how="inner", sort=False)
    else:
        joined = source.copy()
        joined.insert(0, "epoch_index", np.arange(1, joined.shape[0] + 1, dtype=np.int64))

    origin = np.asarray(origin_ecef, dtype=np.float64).reshape(3)
    extra_cols = [col for col in joined.columns if str(col).startswith("FgoVdStateExtra")]
    split_pose = len(extra_cols) >= 12
    xyz = joined[["FgoVdEcefXMeters", "FgoVdEcefYMeters", "FgoVdEcefZMeters"]].to_numpy(dtype=np.float64)
    if split_pose:
        pose_xyz = np.column_stack([_state_extra(joined, idx) for idx in range(3)])
        finite_pose = np.isfinite(pose_xyz).all(axis=1)
        xyz = xyz.copy()
        xyz[finite_pose] = pose_xyz[finite_pose]
    vel_ecef = joined[["FgoVdVelocityXMps", "FgoVdVelocityYMps", "FgoVdVelocityZMps"]].to_numpy(dtype=np.float64)
    pos_enu = ecef_to_enu_relative(xyz, origin)
    vel_enu = ecef_to_enu_relative(origin.reshape(1, 3) + vel_ecef, origin)

    attitude_start_idx = 3 if split_pose else 0
    attitude_rotvec = np.column_stack([_state_extra(joined, attitude_start_idx + idx) for idx in range(3)])
    rpy = np.full_like(attitude_rotvec, np.nan)
    finite_att = np.isfinite(attitude_rotvec).all(axis=1)
    if finite_att.any():
        enu_basis_ecef = enu_to_ecef_relative(np.eye(3, dtype=np.float64), origin) - origin
        rot_enu_ecef = enu_basis_ecef
        rpy[finite_att] = gtsam_rotm_to_rzryrx(
            np.stack([rot_enu_ecef @ rotvec_to_rotm(row) for row in attitude_rotvec[finite_att]])
        )

    out = pd.DataFrame(
        {
            "epoch_index": pd.to_numeric(joined["epoch_index"], errors="coerce").astype(np.int64),
            "utcTimeMillis": joined["utcTimeMillis"].astype(np.int64),
            "position_x": pos_enu[:, 0],
            "position_y": pos_enu[:, 1],
            "position_z": pos_enu[:, 2],
            "roll": rpy[:, 0],
            "pitch": rpy[:, 1],
            "yaw": rpy[:, 2],
            "velocity_x": vel_enu[:, 0],
            "velocity_y": vel_enu[:, 1],
            "velocity_z": vel_enu[:, 2],
        }
    )
    for clock_idx, col in enumerate(_clock_bias_columns(joined)):
        out[f"clock_bias_m_{clock_idx}"] = pd.to_numeric(joined[col], errors="coerce").to_numpy(dtype=np.float64)
    out["clock_drift_mps"] = pd.to_numeric(joined["FgoVdClockDriftMps"], errors="coerce").to_numpy(dtype=np.float64)
    bias_acc_start_idx = attitude_start_idx + 3
    bias_gyro_start_idx = attitude_start_idx + 6
    for prefix, start_idx in (("bias_acc", bias_acc_start_idx), ("bias_gyro", bias_gyro_start_idx)):
        out[f"{prefix}_x"] = _state_extra(joined, start_idx + 0)
        out[f"{prefix}_y"] = _state_extra(joined, start_idx + 1)
        out[f"{prefix}_z"] = _state_extra(joined, start_idx + 2)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-fgo-vd-state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--template-imu-state", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", default="train/2021-12-08-20-28-us-ca-lax-c/pixel5")
    parser.add_argument(
        "--origin-mode",
        choices=("taroz_preprocessing", "first_pair"),
        default="taroz_preprocessing",
        help="how to choose the Taroz ENU origin when --origin-ecef is not supplied",
    )
    parser.add_argument("--origin-ecef", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    bridge = pd.read_csv(args.bridge_fgo_vd_state)
    template = pd.read_csv(args.template_imu_state) if args.template_imu_state is not None else None
    if args.origin_ecef is not None:
        origin = np.asarray(args.origin_ecef, dtype=np.float64)
    else:
        preprocessing_origin = taroz_preprocessing_origin_ecef(Path(args.data_root) / str(args.trip))
        if args.origin_mode == "first_pair":
            if template is None:
                raise ValueError("--origin-mode first_pair requires --template-imu-state")
            origin = infer_origin_ecef_from_bridge_fgo_vd_state(
                bridge,
                template,
                initial_origin_ecef=preprocessing_origin,
            )
        else:
            origin = preprocessing_origin
    out = bridge_fgo_vd_state_to_taroz_imu_state(bridge, origin_ecef=origin, template_state=template)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} rows={out.shape[0]} cols={out.shape[1]}")


if __name__ == "__main__":
    main()
