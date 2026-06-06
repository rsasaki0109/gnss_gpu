"""Compare native bridge states against Taroz MATLAB IMU optimizer states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.evaluate import ecef_to_lla, lla_to_ecef
from experiments.gsdc2023_imu import ecef_to_enu_relative, enu_to_ecef_relative
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT


SOURCE_PREFIXES = {
    "baseline": "Baseline",
    "raw_wls": "RawWls",
    "rawwls": "RawWls",
    "fgo": "Fgo",
    "fgo_vd": "FgoVd",
    "fgovd": "FgoVd",
    "selected": "Selected",
    "ground_truth": "GroundTruth",
    "groundtruth": "GroundTruth",
}

TAROZ_IMU_STATE_GROUPS = {
    "position_m": ("position_x", "position_y", "position_z"),
    "rpy_rad": ("roll", "pitch", "yaw"),
    "velocity_mps": ("velocity_x", "velocity_y", "velocity_z"),
    "clock_bias_m": tuple(f"clock_bias_m_{idx}" for idx in range(7)),
    "clock_drift_mps": ("clock_drift_mps",),
    "bias_acc_mps2": ("bias_acc_x", "bias_acc_y", "bias_acc_z"),
    "bias_gyro_radps": ("bias_gyro_x", "bias_gyro_y", "bias_gyro_z"),
}


def source_prefix(source: str) -> str:
    key = str(source).strip().lower()
    if key in SOURCE_PREFIXES:
        return SOURCE_PREFIXES[key]
    if str(source) in set(SOURCE_PREFIXES.values()):
        return str(source)
    raise ValueError(f"unsupported state source: {source}")


def state_ecef_columns(source: str) -> tuple[str, str, str]:
    prefix = source_prefix(source)
    return (
        f"{prefix}EcefXMeters",
        f"{prefix}EcefYMeters",
        f"{prefix}EcefZMeters",
    )


def load_bridge_states(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "UnixTimeMillis" not in frame.columns:
        raise ValueError(f"{path} is missing UnixTimeMillis")
    return frame


def load_matlab_imu_state(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"utcTimeMillis", "position_x", "position_y", "position_z"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def taroz_preprocessing_origin_ecef(trip_dir: Path) -> np.ndarray:
    path = trip_dir / "device_gnss.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    cols = [
        "utcTimeMillis",
        "BiasUncertaintyNanos",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]
    frame = pd.read_csv(path, usecols=lambda col: col in cols)
    if "BiasUncertaintyNanos" in frame.columns:
        frame = frame[~(frame["BiasUncertaintyNanos"].to_numpy(dtype=np.float64) > 1.0e4)]
    frame = frame.sort_values("utcTimeMillis", kind="mergesort").drop_duplicates("utcTimeMillis", keep="first")
    xyz_cols = ["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]
    if not set(xyz_cols).issubset(frame.columns):
        raise ValueError(f"{path} is missing WLS ECEF columns")
    xyz = frame[xyz_cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        raise ValueError(f"{path} has no finite WLS ECEF origin candidate")
    return xyz[np.flatnonzero(finite)[0]].astype(np.float64)


def origin_ecef_from_llh_deg(llh_deg: Iterable[float]) -> np.ndarray:
    lat_deg, lon_deg, alt_m = [float(value) for value in llh_deg]
    return np.array(
        lla_to_ecef(np.deg2rad(lat_deg), np.deg2rad(lon_deg), alt_m),
        dtype=np.float64,
    )


def llh_deg_from_origin_ecef(origin_ecef: np.ndarray) -> list[float]:
    lat_rad, lon_rad, alt_m = ecef_to_lla(
        float(origin_ecef[0]),
        float(origin_ecef[1]),
        float(origin_ecef[2]),
    )
    return [float(np.rad2deg(lat_rad)), float(np.rad2deg(lon_rad)), float(alt_m)]


def infer_origin_ecef_from_first_pair(
    native_xyz_ecef: np.ndarray,
    matlab_enu_m: np.ndarray,
    *,
    initial_origin_ecef: np.ndarray | None = None,
    max_iter: int = 20,
) -> np.ndarray:
    """Infer the ENU origin that maps one MATLAB ENU state onto one ECEF state."""

    native_xyz = np.asarray(native_xyz_ecef, dtype=np.float64).reshape(3)
    matlab_enu = np.asarray(matlab_enu_m, dtype=np.float64).reshape(1, 3)
    origin = native_xyz.copy() if initial_origin_ecef is None else np.asarray(initial_origin_ecef, dtype=np.float64).reshape(3)
    for _ in range(max_iter):
        projected = enu_to_ecef_relative(matlab_enu, origin)[0]
        ecef_delta = projected - origin
        next_origin = native_xyz - ecef_delta
        if np.linalg.norm(next_origin - origin) < 1.0e-7:
            origin = next_origin
            break
        origin = next_origin
    return origin.astype(np.float64)


def _joined_state_pairs(
    bridge_states: pd.DataFrame,
    matlab_state: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    x_col, y_col, z_col = state_ecef_columns(source)
    missing = sorted({x_col, y_col, z_col} - set(bridge_states.columns))
    if missing:
        raise ValueError(f"bridge state table is missing columns: {missing}")
    native = bridge_states[["UnixTimeMillis", x_col, y_col, z_col]].rename(
        columns={
            x_col: "native_x_ecef_m",
            y_col: "native_y_ecef_m",
            z_col: "native_z_ecef_m",
        },
    )
    matlab_cols = ["utcTimeMillis", "position_x", "position_y", "position_z"]
    if "epoch_index" in matlab_state.columns:
        matlab_cols.insert(0, "epoch_index")
    matlab = matlab_state[matlab_cols].rename(
        columns={
            "position_x": "matlab_east_m",
            "position_y": "matlab_north_m",
            "position_z": "matlab_up_m",
        },
    )
    joined = native.merge(matlab, left_on="UnixTimeMillis", right_on="utcTimeMillis", how="inner")
    joined = joined.sort_values("UnixTimeMillis", kind="mergesort").reset_index(drop=True)
    return joined


def first_finite_pair(joined: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    native_xyz = joined[["native_x_ecef_m", "native_y_ecef_m", "native_z_ecef_m"]].to_numpy(dtype=np.float64)
    matlab_enu = joined[["matlab_east_m", "matlab_north_m", "matlab_up_m"]].to_numpy(dtype=np.float64)
    finite = np.isfinite(native_xyz).all(axis=1) & np.isfinite(matlab_enu).all(axis=1)
    if not finite.any():
        raise ValueError("joined state table has no finite native/MATLAB state pair")
    idx = int(np.flatnonzero(finite)[0])
    return native_xyz[idx], matlab_enu[idx]


def compare_native_to_matlab_state(
    bridge_states: pd.DataFrame,
    matlab_state: pd.DataFrame,
    *,
    origin_ecef: np.ndarray,
    source: str = "fgo",
) -> pd.DataFrame:
    joined = _joined_state_pairs(bridge_states, matlab_state, source=source)
    if joined.empty:
        raise ValueError("native bridge states and MATLAB IMU states have no common utcTimeMillis")
    native_xyz = joined[["native_x_ecef_m", "native_y_ecef_m", "native_z_ecef_m"]].to_numpy(dtype=np.float64)
    native_enu = ecef_to_enu_relative(native_xyz, np.asarray(origin_ecef, dtype=np.float64).reshape(3))
    matlab_enu = joined[["matlab_east_m", "matlab_north_m", "matlab_up_m"]].to_numpy(dtype=np.float64)
    delta = native_enu - matlab_enu
    out = joined.copy()
    out["native_east_m"] = native_enu[:, 0]
    out["native_north_m"] = native_enu[:, 1]
    out["native_up_m"] = native_enu[:, 2]
    out["delta_east_m"] = delta[:, 0]
    out["delta_north_m"] = delta[:, 1]
    out["delta_up_m"] = delta[:, 2]
    out["horizontal_delta_m"] = np.linalg.norm(delta[:, :2], axis=1)
    out["position_delta_3d_m"] = np.linalg.norm(delta, axis=1)
    return out


def finite_delta_summary(comparison: pd.DataFrame) -> dict[str, object]:
    delta = comparison[["delta_east_m", "delta_north_m", "delta_up_m"]].to_numpy(dtype=np.float64)
    horizontal = comparison["horizontal_delta_m"].to_numpy(dtype=np.float64)
    delta_3d = comparison["position_delta_3d_m"].to_numpy(dtype=np.float64)
    finite = np.isfinite(delta).all(axis=1) & np.isfinite(horizontal) & np.isfinite(delta_3d)
    if not finite.any():
        return {
            "matched_rows": int(comparison.shape[0]),
            "finite_rows": 0,
            "mean_abs_east_m": None,
            "mean_abs_north_m": None,
            "mean_abs_up_m": None,
            "mean_horizontal_m": None,
            "p95_horizontal_m": None,
            "max_horizontal_m": None,
            "mean_3d_m": None,
            "p95_3d_m": None,
            "max_3d_m": None,
        }
    finite_delta = np.abs(delta[finite])
    finite_horizontal = horizontal[finite]
    finite_3d = delta_3d[finite]
    return {
        "matched_rows": int(comparison.shape[0]),
        "finite_rows": int(np.count_nonzero(finite)),
        "mean_abs_east_m": float(np.mean(finite_delta[:, 0])),
        "mean_abs_north_m": float(np.mean(finite_delta[:, 1])),
        "mean_abs_up_m": float(np.mean(finite_delta[:, 2])),
        "mean_horizontal_m": float(np.mean(finite_horizontal)),
        "p95_horizontal_m": float(np.percentile(finite_horizontal, 95.0)),
        "max_horizontal_m": float(np.max(finite_horizontal)),
        "mean_3d_m": float(np.mean(finite_3d)),
        "p95_3d_m": float(np.percentile(finite_3d, 95.0)),
        "max_3d_m": float(np.max(finite_3d)),
    }


def compare_taroz_imu_state_tables(
    native_state: pd.DataFrame,
    matlab_state: pd.DataFrame,
) -> pd.DataFrame:
    """Join two Taroz-schema IMU state tables and append per-column deltas."""

    required = {"utcTimeMillis"}
    for label, frame in (("native", native_state), ("matlab", matlab_state)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} state table is missing columns: {missing}")

    native = native_state.copy()
    matlab = matlab_state.copy()
    for frame in (native, matlab):
        frame["utcTimeMillis"] = pd.to_numeric(frame["utcTimeMillis"], errors="coerce").round().astype("Int64")
        frame.dropna(subset=["utcTimeMillis"], inplace=True)
        frame["utcTimeMillis"] = frame["utcTimeMillis"].astype(np.int64)
        frame.sort_values("utcTimeMillis", kind="mergesort", inplace=True)
        frame.drop_duplicates("utcTimeMillis", keep="first", inplace=True)

    common = [
        col
        for group_cols in TAROZ_IMU_STATE_GROUPS.values()
        for col in group_cols
        if col in native.columns and col in matlab.columns
    ]
    if not common:
        raise ValueError("state tables have no common Taroz IMU numeric state columns")

    native_cols = ["utcTimeMillis", *common]
    if "epoch_index" in native.columns:
        native_cols.insert(0, "epoch_index")
    joined = native[native_cols].merge(
        matlab[["utcTimeMillis", *common]],
        on="utcTimeMillis",
        how="inner",
        suffixes=("_native", "_matlab"),
        sort=False,
    )
    joined.sort_values("utcTimeMillis", kind="mergesort", inplace=True)
    joined.reset_index(drop=True, inplace=True)
    for col in common:
        joined[f"delta_{col}"] = (
            pd.to_numeric(joined[f"{col}_native"], errors="coerce").to_numpy(dtype=np.float64)
            - pd.to_numeric(joined[f"{col}_matlab"], errors="coerce").to_numpy(dtype=np.float64)
        )
    return joined


def taroz_imu_state_delta_summary(comparison: pd.DataFrame) -> dict[str, object]:
    """Summarize grouped deltas from :func:`compare_taroz_imu_state_tables`."""

    summary: dict[str, object] = {
        "matched_rows": int(comparison.shape[0]),
        "groups": {},
    }
    groups: dict[str, object] = {}
    for group_name, cols in TAROZ_IMU_STATE_GROUPS.items():
        delta_cols = [f"delta_{col}" for col in cols if f"delta_{col}" in comparison.columns]
        if not delta_cols:
            continue
        delta = comparison[delta_cols].to_numpy(dtype=np.float64)
        finite = np.isfinite(delta).all(axis=1)
        if not finite.any():
            groups[group_name] = {
                "finite_rows": 0,
                "component_rms": None,
                "component_max_abs": None,
                "mean_norm": None,
                "max_norm": None,
            }
            continue
        finite_delta = delta[finite]
        norm = np.linalg.norm(finite_delta, axis=1)
        groups[group_name] = {
            "finite_rows": int(np.count_nonzero(finite)),
            "component_rms": float(np.sqrt(np.mean(finite_delta * finite_delta))),
            "component_max_abs": float(np.max(np.abs(finite_delta))),
            "mean_norm": float(np.mean(norm)),
            "max_norm": float(np.max(norm)),
        }
    summary["groups"] = groups
    return summary


def resolve_origin_ecef(
    args: argparse.Namespace,
    bridge_states: pd.DataFrame,
    matlab_state: pd.DataFrame,
    *,
    source: str,
) -> tuple[np.ndarray, str]:
    if args.origin_ecef is not None:
        return np.asarray(args.origin_ecef, dtype=np.float64).reshape(3), "ecef"
    if args.origin_llh_deg is not None:
        return origin_ecef_from_llh_deg(args.origin_llh_deg), "llh_deg"
    mode = str(args.origin_mode)
    if mode == "taroz_preprocessing":
        trip_dir = Path(args.data_root) / str(args.trip)
        return taroz_preprocessing_origin_ecef(trip_dir), mode
    if mode == "first_pair":
        joined = _joined_state_pairs(bridge_states, matlab_state, source=source)
        native_xyz, matlab_enu = first_finite_pair(joined)
        initial = None
        if args.initial_origin_ecef is not None:
            initial = np.asarray(args.initial_origin_ecef, dtype=np.float64).reshape(3)
        return infer_origin_ecef_from_first_pair(native_xyz, matlab_enu, initial_origin_ecef=initial), mode
    raise ValueError(f"unsupported origin mode: {mode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-states", type=Path, default=None, help="native bridge_states.csv")
    parser.add_argument(
        "--native-imu-state",
        type=Path,
        default=None,
        help="native state already converted into Taroz phone_data_imu_state schema",
    )
    parser.add_argument("--matlab-imu-state", required=True, type=Path, help="Taroz phone_data_imu_state.csv")
    parser.add_argument("--source", default="fgo", help="native state source: fgo, fgo_vd, raw_wls, selected, baseline")
    parser.add_argument(
        "--origin-mode",
        choices=("first_pair", "taroz_preprocessing"),
        default="first_pair",
        help="how to choose the MATLAB ENU origin when --origin-ecef/--origin-llh-deg are not supplied",
    )
    parser.add_argument("--origin-ecef", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--origin-llh-deg", nargs=3, type=float, default=None, metavar=("LAT", "LON", "ALT"))
    parser.add_argument("--initial-origin-ecef", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", default="train/2021-12-08-20-28-us-ca-lax-c/pixel5")
    parser.add_argument("--output", type=Path, default=None, help="write joined state delta CSV")
    parser.add_argument("--summary", type=Path, default=None, help="write JSON summary")
    return parser


def run_comparison(args: argparse.Namespace) -> dict[str, object]:
    if (args.bridge_states is None) == (args.native_imu_state is None):
        raise ValueError("supply exactly one of --bridge-states or --native-imu-state")

    matlab_state = load_matlab_imu_state(args.matlab_imu_state)
    if args.native_imu_state is not None:
        native_state = load_matlab_imu_state(args.native_imu_state)
        comparison = compare_taroz_imu_state_tables(native_state, matlab_state)
        summary = {
            "mode": "taroz_imu_state",
            "native_imu_state_path": str(args.native_imu_state),
            "matlab_imu_state_path": str(args.matlab_imu_state),
            "delta_stats": taroz_imu_state_delta_summary(comparison),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(args.output, index=False)
        summary_path = args.summary
        if summary_path is None and args.output is not None:
            summary_path = args.output.with_suffix(".json")
        if summary_path is not None:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    source = source_prefix(args.source)
    bridge_states = load_bridge_states(args.bridge_states)
    origin_ecef, origin_mode = resolve_origin_ecef(args, bridge_states, matlab_state, source=source)
    comparison = compare_native_to_matlab_state(
        bridge_states,
        matlab_state,
        origin_ecef=origin_ecef,
        source=source,
    )
    summary = {
        "bridge_states_path": str(args.bridge_states),
        "matlab_imu_state_path": str(args.matlab_imu_state),
        "source": source,
        "origin_mode": origin_mode,
        "origin_ecef_m": [float(value) for value in origin_ecef],
        "origin_llh_deg": llh_deg_from_origin_ecef(origin_ecef),
        "delta_stats": finite_delta_summary(comparison),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output, index=False)
    summary_path = args.summary
    if summary_path is None and args.output is not None:
        summary_path = args.output.with_suffix(".json")
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(run_comparison(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
