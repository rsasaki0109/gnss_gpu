#!/usr/bin/env python3
"""Run native fixed-linearized FGO on exported Taroz GNSS graph factors."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_gsdc2023_taroz_linearization import taroz_gtsam_gnss_graph_cost_frame


@dataclass
class NativeFixedGnssArrays:
    state_frame: pd.DataFrame
    state: np.ndarray
    sat_ecef: np.ndarray
    sat_vel: np.ndarray
    pseudorange: np.ndarray
    pseudorange_weights: np.ndarray
    pr_linearization_ref_ecef: np.ndarray
    pr_linearization_los_ecef: np.ndarray
    doppler: np.ndarray
    doppler_weights: np.ndarray
    doppler_linearization_ref_vel: np.ndarray
    doppler_linearization_los_ecef: np.ndarray
    tdcp_meas: np.ndarray
    tdcp_weights: np.ndarray
    tdcp_linearization_ref_ecef: np.ndarray
    sys_kind: np.ndarray
    dt: np.ndarray
    n_clock: int


def _set_epoch_ref(ref: np.ndarray, epoch_idx: int, value: np.ndarray, label: str) -> None:
    value = np.asarray(value, dtype=np.float64).reshape(3)
    if np.isfinite(ref[epoch_idx]).all() and np.linalg.norm(ref[epoch_idx] - value) > 1.0e-9:
        raise ValueError(f"{label} reference mismatch at dense epoch {epoch_idx}")
    ref[epoch_idx] = value


def _resolve_export_csv(export_dir: Path, csv_path: Path | str | None, default_name: str) -> Path:
    if csv_path is None:
        return export_dir / default_name
    path = Path(csv_path)
    if path.is_absolute():
        return path
    return export_dir / path


def _resolve_factor_csv(export_dir: Path, factor_csv: Path | str | None) -> Path:
    return _resolve_export_csv(export_dir, factor_csv, "phone_data_gnss_factor_mask.csv")


def _read_state_frame(state_csv: Path) -> pd.DataFrame:
    state_frame = pd.read_csv(state_csv)
    return state_frame.sort_values("epoch_index").reset_index(drop=True)


def _state_frame_to_native_state(state_frame: pd.DataFrame, *, n_clock: int) -> np.ndarray:
    n_epoch = len(state_frame)
    state_width = 7 + int(n_clock)
    state = np.zeros((n_epoch, state_width), dtype=np.float64)
    state[:, :3] = state_frame[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
    state[:, 3:6] = state_frame[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=np.float64)
    for clock_idx in range(int(n_clock)):
        state[:, 6 + clock_idx] = state_frame[f"clock_bias_m_{clock_idx}"].to_numpy(dtype=np.float64)
    state[:, 6 + int(n_clock)] = state_frame["clock_drift_mps"].to_numpy(dtype=np.float64)
    return state


def align_state_position_origin_to_reference(state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    """Translate state positions so the first epoch shares the reference origin."""

    aligned = np.asarray(state, dtype=np.float64).copy()
    reference = np.asarray(reference_state, dtype=np.float64)
    if aligned.size == 0 or reference.size == 0:
        return aligned
    if aligned.shape[1] < 3 or reference.shape[1] < 3:
        raise ValueError("state arrays must have at least three position columns")
    aligned[:, :3] += reference[0, :3] - aligned[0, :3]
    return aligned


def load_taroz_export_as_native_fixed_arrays(
    export_dir: Path,
    *,
    n_clock: int = 7,
    dummy_sat_range_m: float = 1000.0,
    state_csv: Path | str | None = None,
    factor_csv: Path | str | None = None,
) -> NativeFixedGnssArrays:
    """Convert patched Taroz GNSS export CSVs into native VD solver arrays."""

    export_dir = Path(export_dir)
    factor = pd.read_csv(_resolve_factor_csv(export_dir, factor_csv))
    state_frame = _read_state_frame(_resolve_export_csv(export_dir, state_csv, "phone_data_gnss_graph_state.csv"))
    epochs = state_frame["epoch_index"].astype(int).to_numpy()
    epoch_to_i = {int(epoch): idx for idx, epoch in enumerate(epochs)}
    n_epoch = len(epochs)
    n_clock = int(n_clock)

    field_counts = factor.groupby(["field", "epoch_index"]).size().to_list()
    n_sat = int(max([4, *field_counts]))

    state = _state_frame_to_native_state(state_frame, n_clock=n_clock)

    sat_ecef = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    sat_vel = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, n_sat), dtype=np.float64)
    pseudorange_weights = np.zeros((n_epoch, n_sat), dtype=np.float64)
    pr_ref = np.zeros((n_epoch, 3), dtype=np.float64)
    pr_los = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
    doppler = np.zeros((n_epoch, n_sat), dtype=np.float64)
    doppler_weights = np.zeros((n_epoch, n_sat), dtype=np.float64)
    doppler_ref = np.zeros((n_epoch, 3), dtype=np.float64)
    doppler_los = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
    tdcp_meas = np.zeros((max(0, n_epoch - 1), n_sat), dtype=np.float64)
    tdcp_weights = np.zeros((max(0, n_epoch - 1), n_sat), dtype=np.float64)
    tdcp_ref = np.full((n_epoch, 3), np.nan, dtype=np.float64)
    sys_kind = np.zeros((n_epoch, n_sat), dtype=np.int32)

    for _, group in factor[factor["field"].eq("P")].groupby("epoch_index", sort=True):
        epoch_idx = epoch_to_i[int(group["epoch_index"].iloc[0])]
        pr_ref[epoch_idx] = group[["origin1_e", "origin1_n", "origin1_u"]].iloc[0].to_numpy(dtype=np.float64)
        for slot_idx, (_, row) in enumerate(group.iterrows()):
            pseudorange[epoch_idx, slot_idx] = float(row["measurement"])
            pseudorange_weights[epoch_idx, slot_idx] = 1.0 / float(row["sigma"]) ** 2
            pr_los[epoch_idx, slot_idx] = row[["los_e", "los_n", "los_u"]].to_numpy(dtype=np.float64)
            sys_kind[epoch_idx, slot_idx] = int(row["sigtype"])

    for _, group in factor[factor["field"].eq("D")].groupby("epoch_index", sort=True):
        epoch_idx = epoch_to_i[int(group["epoch_index"].iloc[0])]
        doppler_ref[epoch_idx] = group[["origin1_e", "origin1_n", "origin1_u"]].iloc[0].to_numpy(dtype=np.float64)
        for slot_idx, (_, row) in enumerate(group.iterrows()):
            doppler[epoch_idx, slot_idx] = float(row["measurement"])
            doppler_weights[epoch_idx, slot_idx] = 1.0 / float(row["sigma"]) ** 2
            doppler_los[epoch_idx, slot_idx] = row[["los_e", "los_n", "los_u"]].to_numpy(dtype=np.float64)

    carrier = factor[factor["field"].eq("L")]
    for _, row in carrier.iterrows():
        epoch_idx = epoch_to_i[int(row["epoch_index"])]
        next_idx = epoch_to_i[int(row["next_epoch_index"])]
        _set_epoch_ref(tdcp_ref, epoch_idx, row[["origin1_e", "origin1_n", "origin1_u"]].to_numpy(dtype=np.float64), "TDCP origin1")
        _set_epoch_ref(tdcp_ref, next_idx, row[["origin2_e", "origin2_n", "origin2_u"]].to_numpy(dtype=np.float64), "TDCP origin2")

    for _, group in carrier.groupby("epoch_index", sort=True):
        epoch_idx = epoch_to_i[int(group["epoch_index"].iloc[0])]
        next_idx = epoch_to_i[int(group["next_epoch_index"].iloc[0])]
        for slot_idx, (_, row) in enumerate(group.iterrows()):
            los = row[["los_e", "los_n", "los_u"]].to_numpy(dtype=np.float64)
            tdcp_meas[epoch_idx, slot_idx] = float(row["measurement"])
            tdcp_weights[epoch_idx, slot_idx] = 1.0 / float(row["sigma"]) ** 2
            sat_ecef[next_idx, slot_idx] = tdcp_ref[next_idx] - float(dummy_sat_range_m) * los

    tdcp_ref[~np.isfinite(tdcp_ref).all(axis=1)] = 0.0
    dt = np.zeros(n_epoch, dtype=np.float64)
    if n_epoch > 1:
        dt[:-1] = np.diff(state_frame["utcTimeMillis"].to_numpy(dtype=np.float64)) / 1000.0

    return NativeFixedGnssArrays(
        state_frame=state_frame,
        state=state,
        sat_ecef=sat_ecef,
        sat_vel=sat_vel,
        pseudorange=pseudorange,
        pseudorange_weights=pseudorange_weights,
        pr_linearization_ref_ecef=pr_ref,
        pr_linearization_los_ecef=pr_los,
        doppler=doppler,
        doppler_weights=doppler_weights,
        doppler_linearization_ref_vel=doppler_ref,
        doppler_linearization_los_ecef=doppler_los,
        tdcp_meas=tdcp_meas,
        tdcp_weights=tdcp_weights,
        tdcp_linearization_ref_ecef=tdcp_ref,
        sys_kind=sys_kind,
        dt=dt,
        n_clock=n_clock,
    )


def state_to_taroz_graph_state_frame(template: pd.DataFrame, state: np.ndarray, *, n_clock: int) -> pd.DataFrame:
    out = template.copy()
    out[["position_x", "position_y", "position_z"]] = state[:, :3]
    out[["velocity_x", "velocity_y", "velocity_z"]] = state[:, 3:6]
    for clock_idx in range(int(n_clock)):
        out[f"clock_bias_m_{clock_idx}"] = state[:, 6 + clock_idx]
    out["clock_drift_mps"] = state[:, 6 + int(n_clock)]
    return out


def taroz_graph_cost_for_native_state(
    export_dir: Path,
    template: pd.DataFrame,
    state: np.ndarray,
    *,
    n_clock: int = 7,
    pr_huber_k: float = 0.1,
    doppler_huber_k: float = 0.4,
    carrier_huber_k: float = 0.2,
    motion_sigma_m: float = 0.05,
    clock_sigma_m: float = 0.1,
    factor_csv: Path | str | None = None,
) -> float:
    export_dir = Path(export_dir)
    state_frame = state_to_taroz_graph_state_frame(template, state, n_clock=n_clock)
    with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
        state_frame.to_csv(handle.name, index=False)
        cost_frame = taroz_gtsam_gnss_graph_cost_frame(
            _resolve_factor_csv(export_dir, factor_csv),
            Path(handle.name),
            n_clock=n_clock,
            pr_huber_k=pr_huber_k,
            doppler_huber_k=doppler_huber_k,
            carrier_huber_k=carrier_huber_k,
            motion_sigma_m=motion_sigma_m,
            clock_sigma_m=clock_sigma_m,
        )
    return float(cost_frame["cost"].sum())


def run_native_fixed_objective(
    export_dir: Path,
    *,
    n_clock: int = 7,
    max_iter: int = 5,
    state_csv: Path | str | None = None,
    reference_state_csv: Path | str | None = None,
    pr_huber_k: float = 0.1,
    doppler_huber_k: float = 0.4,
    carrier_huber_k: float = 0.2,
    motion_sigma_m: float = 0.05,
    clock_sigma_m: float = 0.1,
    align_state_origin_to_reference: bool = False,
    factor_csv: Path | str | None = None,
) -> dict[str, float | int]:
    from gnss_gpu.fgo import fgo_gnss_lm_vd

    export_dir = Path(export_dir)
    arrays = load_taroz_export_as_native_fixed_arrays(
        export_dir,
        n_clock=n_clock,
        state_csv=state_csv,
        factor_csv=factor_csv,
    )
    state = arrays.state.copy()
    reference_frame = _read_state_frame(
        _resolve_export_csv(export_dir, reference_state_csv, "phone_data_gnss_graph_state.csv")
    )
    reference_state = _state_frame_to_native_state(reference_frame, n_clock=n_clock)
    if not np.array_equal(
        arrays.state_frame["epoch_index"].to_numpy(dtype=np.int64),
        reference_frame["epoch_index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("state CSV and reference state CSV must contain the same epoch_index sequence")
    if align_state_origin_to_reference:
        state = align_state_position_origin_to_reference(state, reference_state)
    initial_state = state.copy()
    cost_before = taroz_graph_cost_for_native_state(
        export_dir,
        arrays.state_frame,
        state,
        n_clock=n_clock,
        pr_huber_k=pr_huber_k,
        doppler_huber_k=doppler_huber_k,
        carrier_huber_k=carrier_huber_k,
        motion_sigma_m=motion_sigma_m,
        clock_sigma_m=clock_sigma_m,
        factor_csv=factor_csv,
    )
    iters, pr_mse = fgo_gnss_lm_vd(
        arrays.sat_ecef,
        arrays.pseudorange,
        arrays.pseudorange_weights,
        state,
        n_clock=n_clock,
        sys_kind=arrays.sys_kind,
        sat_vel=arrays.sat_vel,
        doppler=arrays.doppler,
        doppler_weights=arrays.doppler_weights,
        dt=arrays.dt,
        motion_sigma_m=motion_sigma_m,
        clock_drift_sigma_m=clock_sigma_m,
        clock_use_average_drift=True,
        max_iter=max_iter,
        tol=1.0e-10,
        huber_k=pr_huber_k,
        line_search=True,
        doppler_huber_k=doppler_huber_k,
        tdcp_huber_k=carrier_huber_k,
        pr_linearization_ref_ecef=arrays.pr_linearization_ref_ecef,
        pr_linearization_los_ecef=arrays.pr_linearization_los_ecef,
        doppler_linearization_ref_vel=arrays.doppler_linearization_ref_vel,
        doppler_linearization_los_ecef=arrays.doppler_linearization_los_ecef,
        tdcp_meas=arrays.tdcp_meas,
        tdcp_weights=arrays.tdcp_weights,
        tdcp_linearization_ref_ecef=arrays.tdcp_linearization_ref_ecef,
    )
    cost_after = taroz_graph_cost_for_native_state(
        export_dir,
        arrays.state_frame,
        state,
        n_clock=n_clock,
        pr_huber_k=pr_huber_k,
        doppler_huber_k=doppler_huber_k,
        carrier_huber_k=carrier_huber_k,
        motion_sigma_m=motion_sigma_m,
        clock_sigma_m=clock_sigma_m,
        factor_csv=factor_csv,
    )
    position_shift = np.linalg.norm(state[:, :3] - initial_state[:, :3], axis=1)
    position_error_to_reference = np.linalg.norm(state[:, :3] - reference_state[:, :3], axis=1)
    isb_delta = np.diff(state[:, 7 : 6 + n_clock], axis=0) if n_clock > 1 else np.zeros((0, 0))
    return {
        "iterations": int(iters),
        "returned_pr_mse": float(pr_mse),
        "cost_before": float(cost_before),
        "cost_after": float(cost_after),
        "cost_delta": float(cost_after - cost_before),
        "position_shift_mean_m": float(position_shift.mean()) if position_shift.size else 0.0,
        "position_shift_max_m": float(position_shift.max()) if position_shift.size else 0.0,
        "position_error_to_reference_mean_m": (
            float(position_error_to_reference.mean()) if position_error_to_reference.size else 0.0
        ),
        "position_error_to_reference_max_m": (
            float(position_error_to_reference.max()) if position_error_to_reference.size else 0.0
        ),
        "state_step_norm": float(np.linalg.norm(state - initial_state)),
        "state_error_to_reference_norm": float(np.linalg.norm(state - reference_state)),
        "max_isb_epoch_delta_m": float(np.max(np.abs(isb_delta))) if isb_delta.size else 0.0,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--n-clock", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=5)
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument("--state-csv", type=Path, default=None)
    state_group.add_argument("--use-initial-state", action="store_true")
    parser.add_argument("--reference-state-csv", type=Path, default=None)
    parser.add_argument("--factor-mask-csv", type=Path, default=None)
    parser.add_argument("--pr-huber-k", type=float, default=0.1)
    parser.add_argument("--doppler-huber-k", type=float, default=0.4)
    parser.add_argument("--carrier-huber-k", type=float, default=0.2)
    parser.add_argument("--motion-sigma-m", type=float, default=0.05)
    parser.add_argument("--clock-sigma-m", type=float, default=0.1)
    parser.add_argument("--align-state-origin-to-reference", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    state_csv = Path("phone_data_gnss_initial_state.csv") if args.use_initial_state else args.state_csv
    result = run_native_fixed_objective(
        args.export_dir,
        n_clock=args.n_clock,
        max_iter=args.max_iter,
        state_csv=state_csv,
        reference_state_csv=args.reference_state_csv,
        pr_huber_k=args.pr_huber_k,
        doppler_huber_k=args.doppler_huber_k,
        carrier_huber_k=args.carrier_huber_k,
        motion_sigma_m=args.motion_sigma_m,
        clock_sigma_m=args.clock_sigma_m,
        align_state_origin_to_reference=bool(args.align_state_origin_to_reference or args.use_initial_state),
        factor_csv=args.factor_mask_csv,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
