#!/usr/bin/env python3
"""Compare Taroz/GTSAM optimizer deltas against native FGO deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    _enu_to_ecef_relative,
    _taroz_preprocessing_origin_ecef_from_trip_dir,
)


NATIVE_DELTA_COLUMNS = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    *(f"clock_bias_m_{idx}" for idx in range(7)),
    "clock_drift_mps",
    "pose_tangent_x",
    "pose_tangent_y",
    "pose_tangent_z",
    "attitude_tangent_x",
    "attitude_tangent_y",
    "attitude_tangent_z",
    "bias_acc_x",
    "bias_acc_y",
    "bias_acc_z",
    "bias_gyro_x",
    "bias_gyro_y",
    "bias_gyro_z",
)

DELTA_GROUPS = {
    "position_m": ("position_x", "position_y", "position_z"),
    "velocity_mps": ("velocity_x", "velocity_y", "velocity_z"),
    "clock_bias_m": tuple(f"clock_bias_m_{idx}" for idx in range(7)),
    "clock_drift_mps": ("clock_drift_mps",),
    "pose_tangent_m": ("pose_tangent_x", "pose_tangent_y", "pose_tangent_z"),
    "attitude_tangent_rad": ("attitude_tangent_x", "attitude_tangent_y", "attitude_tangent_z"),
    "bias_acc_mps2": ("bias_acc_x", "bias_acc_y", "bias_acc_z"),
    "bias_gyro_radps": ("bias_gyro_x", "bias_gyro_y", "bias_gyro_z"),
}


def enu_basis_ecef(origin_ecef: np.ndarray) -> np.ndarray:
    """Return rows containing ECEF unit vectors for Taroz ENU axes."""

    origin = np.asarray(origin_ecef, dtype=np.float64).reshape(3)
    return _enu_to_ecef_relative(np.eye(3, dtype=np.float64), origin) - origin


def load_delta_matrix(path: Path, *, width: int = 26) -> np.ndarray:
    values = np.loadtxt(Path(path), delimiter=",", dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % int(width) != 0:
            raise ValueError(f"{path} has {arr.size} values, not a multiple of width={width}")
        arr = arr.reshape(-1, int(width))
    if arr.ndim != 2 or arr.shape[1] != int(width):
        raise ValueError(f"{path} must be a [T,{width}] CSV")
    return arr


def taroz_gtsam_delta_to_native_delta(
    taroz_delta: np.ndarray,
    *,
    origin_ecef: np.ndarray,
) -> np.ndarray:
    """Convert Taroz ``x,v,c,d,p,b`` GTSAM delta rows to native VD ordering.

    Taroz export order is ``x(3), v(3), c(7), d(1), Pose3 tangent(6),
    ConstantBias(6)``.  ``Pose3.localCoordinates`` stores rotation tangent
    before translation tangent.  Native split-pose state stores translation
    tangent before attitude tangent, while GNSS point and velocity deltas are
    in local ECEF coordinates.
    """

    src = np.asarray(taroz_delta, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 26:
        raise ValueError("taroz_delta must be [T,26]")
    basis = enu_basis_ecef(origin_ecef)
    out = np.zeros_like(src)
    out[:, 0:3] = src[:, 0:3] @ basis
    out[:, 3:6] = src[:, 3:6] @ basis
    out[:, 6:14] = src[:, 6:14]
    out[:, 14:17] = src[:, 17:20]
    out[:, 17:20] = src[:, 14:17]
    out[:, 20:26] = src[:, 20:26]
    return out


def delta_comparison_frame(native_delta: np.ndarray, taroz_native_delta: np.ndarray) -> pd.DataFrame:
    native = np.asarray(native_delta, dtype=np.float64)
    taroz = np.asarray(taroz_native_delta, dtype=np.float64)
    if native.shape != taroz.shape:
        raise ValueError(f"delta shapes differ: native={native.shape} taroz={taroz.shape}")
    if native.ndim != 2 or native.shape[1] != len(NATIVE_DELTA_COLUMNS):
        raise ValueError(f"delta matrices must be [T,{len(NATIVE_DELTA_COLUMNS)}]")
    out = pd.DataFrame({"epoch_index": np.arange(1, native.shape[0] + 1, dtype=np.int64)})
    for idx, col in enumerate(NATIVE_DELTA_COLUMNS):
        out[f"native_{col}"] = native[:, idx]
        out[f"taroz_{col}"] = taroz[:, idx]
        out[f"delta_{col}"] = native[:, idx] - taroz[:, idx]
    return out


def summarize_delta_comparison(comparison: pd.DataFrame) -> dict[str, object]:
    groups: dict[str, object] = {}
    for group_name, cols in DELTA_GROUPS.items():
        delta_cols = [f"delta_{col}" for col in cols]
        values = comparison[delta_cols].to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        if not finite.any():
            groups[group_name] = {
                "finite_rows": 0,
                "component_rms": None,
                "component_max_abs": None,
                "mean_norm": None,
                "max_norm": None,
            }
            continue
        finite_values = values[finite]
        norms = np.linalg.norm(finite_values, axis=1)
        groups[group_name] = {
            "finite_rows": int(np.count_nonzero(finite)),
            "component_rms": float(np.sqrt(np.mean(finite_values * finite_values))),
            "component_max_abs": float(np.max(np.abs(finite_values))),
            "mean_norm": float(np.mean(norms)),
            "max_norm": float(np.max(norms)),
        }
    column_rank: list[dict[str, object]] = []
    for col in NATIVE_DELTA_COLUMNS:
        values = pd.to_numeric(comparison[f"delta_{col}"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        column_rank.append(
            {
                "column": col,
                "rms": float(np.sqrt(np.mean(values[finite] * values[finite]))),
                "max_abs": float(np.max(np.abs(values[finite]))),
                "mean": float(np.mean(values[finite])),
            }
        )
    column_rank.sort(key=lambda item: float(item["rms"]), reverse=True)
    return {
        "matched_rows": int(comparison.shape[0]),
        "groups": groups,
        "column_rank": column_rank,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-delta", required=True, type=Path)
    parser.add_argument("--taroz-delta", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", default="train/2021-12-08-20-28-us-ca-lax-c/pixel5")
    parser.add_argument("--origin-ecef", nargs=3, type=float, default=None)
    parser.add_argument("--output", type=Path, default=None, help="optional detailed CSV output")
    parser.add_argument("--summary-output", type=Path, default=None, help="optional summary JSON output")
    return parser


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    native_delta = load_delta_matrix(Path(args.native_delta))
    taroz_delta = load_delta_matrix(Path(args.taroz_delta))
    if args.origin_ecef is not None:
        origin = np.asarray(args.origin_ecef, dtype=np.float64)
    else:
        origin = _taroz_preprocessing_origin_ecef_from_trip_dir(Path(args.data_root) / str(args.trip))
    taroz_native_delta = taroz_gtsam_delta_to_native_delta(taroz_delta, origin_ecef=origin)
    comparison = delta_comparison_frame(native_delta, taroz_native_delta)
    summary = {
        "native_delta_path": str(args.native_delta),
        "taroz_delta_path": str(args.taroz_delta),
        "origin_ecef": [float(x) for x in np.asarray(origin, dtype=np.float64).reshape(3)],
        "delta_stats": summarize_delta_comparison(comparison),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
        if args.summary_output is None:
            args.summary_output = args.output.with_suffix(".json")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    print(json.dumps(run_comparison(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
