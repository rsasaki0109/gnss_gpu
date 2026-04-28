#!/usr/bin/env python3
"""Compare MATLAB factor-mask keys against raw-bridge factor-mask keys."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_M,
    OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    _build_trip_arrays,
)
from experiments.gsdc2023_audit_output import (
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_audit_cli import (
    add_data_root_trip_args as _add_data_root_trip_args,
    add_max_epochs_arg as _add_max_epochs_arg,
    add_multi_gnss_arg as _add_multi_gnss_arg,
    add_output_dir_arg as _add_output_dir_arg,
    nonnegative_max_epochs as _nonnegative_max_epochs,
    resolve_trip_dir as _resolve_trip_dir,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_factor_mask import (
    append_factor_rows as _append_factor_rows,
    factor_mask_side_summary as _factor_mask_side_summary,
    merge_factor_mask_keys as _merge_factor_mask_keys,
    normalize_factor_mask_frame as _normalize_mask_frame,
)
from experiments.gsdc2023_signal_model import (
    factor_frequency_label as _freq_label,
)
from experiments.gsdc2023_trip_window import (
    settings_epoch_window_for_trip as _settings_epoch_window_for_trip,
    trim_epoch_window as _trim_epoch_window,
)


def build_bridge_factor_mask(
    trip_dir: Path,
    *,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M,
    pseudorange_residual_mask_l5_m: float = OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    doppler_residual_mask_mps: float = OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    pseudorange_doppler_mask_m: float = OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    matlab_residual_diagnostics_mask_path: Path | None = None,
) -> pd.DataFrame:
    start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)
    batch = _build_trip_arrays(
        trip_dir,
        max_epochs=bridge_max_epochs,
        start_epoch=start_epoch,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=multi_gnss,
        use_tdcp=True,
        apply_observation_mask=True,
        pseudorange_residual_mask_m=pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=doppler_residual_mask_mps,
        tdcp_consistency_threshold_m=tdcp_consistency_threshold_m,
        pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
        matlab_residual_diagnostics_mask_path=matlab_residual_diagnostics_mask_path,
        dual_frequency=True,
    )
    times_ms = np.asarray(batch.times_ms, dtype=np.float64)
    slot_keys = tuple(batch.slot_keys)
    slot_freq = np.array([_freq_label(key[2]) for key in slot_keys], dtype=object)
    rows: list[dict[str, object]] = []

    for freq in ("L1", "L5"):
        freq_slots = np.flatnonzero(slot_freq == freq)
        if freq_slots.size == 0:
            continue

        p_mask = batch.weights[:, freq_slots] > 0.0
        p_epoch, p_local_slot = np.nonzero(p_mask)
        _append_factor_rows(
            rows,
            field_names=("P", "resPc"),
            freq=freq,
            epoch_indices=p_epoch.astype(np.int64),
            slot_indices=freq_slots[p_local_slot].astype(np.int64),
            times_ms=times_ms,
            slot_keys=slot_keys,
            epoch_offset=start_epoch,
        )

        if batch.doppler_weights is not None:
            d_mask = batch.doppler_weights[:, freq_slots] > 0.0
            d_epoch, d_local_slot = np.nonzero(d_mask)
            _append_factor_rows(
                rows,
                field_names=("D", "resD"),
                freq=freq,
                epoch_indices=d_epoch.astype(np.int64),
                slot_indices=freq_slots[d_local_slot].astype(np.int64),
                times_ms=times_ms,
                slot_keys=slot_keys,
                epoch_offset=start_epoch,
            )

        if batch.tdcp_weights is not None:
            l_mask = batch.tdcp_weights[:, freq_slots] > 0.0
            l_epoch, l_local_slot = np.nonzero(l_mask)
            _append_factor_rows(
                rows,
                field_names=("L", "resL"),
                freq=freq,
                epoch_indices=l_epoch.astype(np.int64),
                slot_indices=freq_slots[l_local_slot].astype(np.int64),
                times_ms=times_ms,
                slot_keys=slot_keys,
                next_epoch_indices=(l_epoch + 1).astype(np.int64),
                epoch_offset=start_epoch,
            )

    return pd.DataFrame(rows)


def compare_factor_masks(
    trip_dir: Path,
    *,
    matlab_mask_path: Path | None = None,
    max_epochs: int = 0,
    multi_gnss: bool = False,
    pseudorange_residual_mask_m: float = OBS_MASK_RESIDUAL_THRESHOLD_M,
    pseudorange_residual_mask_l5_m: float = OBS_MASK_RESIDUAL_THRESHOLD_L5_M,
    doppler_residual_mask_mps: float = OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS,
    tdcp_consistency_threshold_m: float = DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    pseudorange_doppler_mask_m: float = OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M,
    matlab_residual_diagnostics_mask_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if matlab_mask_path is None:
        matlab_mask_path = trip_dir / "phone_data_factor_mask.csv"
    matlab_mask = _normalize_mask_frame(pd.read_csv(matlab_mask_path))
    start_epoch, bridge_max_epochs = _settings_epoch_window_for_trip(trip_dir, max_epochs)
    matlab_mask = _trim_epoch_window(
        matlab_mask,
        start_epoch,
        bridge_max_epochs,
        next_epoch_column="next_epoch_index",
    )
    bridge_mask = _normalize_mask_frame(
        build_bridge_factor_mask(
            trip_dir,
            max_epochs=max_epochs,
            multi_gnss=multi_gnss,
            pseudorange_residual_mask_m=pseudorange_residual_mask_m,
            pseudorange_residual_mask_l5_m=pseudorange_residual_mask_l5_m,
            doppler_residual_mask_mps=doppler_residual_mask_mps,
            tdcp_consistency_threshold_m=tdcp_consistency_threshold_m,
            pseudorange_doppler_mask_m=pseudorange_doppler_mask_m,
            matlab_residual_diagnostics_mask_path=matlab_residual_diagnostics_mask_path,
        ),
    )
    merged = _merge_factor_mask_keys(
        matlab_mask,
        bridge_mask,
        left_only_side="matlab_only",
        right_only_side="bridge_only",
    )
    summary, side_payload = _factor_mask_side_summary(
        merged,
        left_name="matlab",
        right_name="bridge",
        left_only_side="matlab_only",
        right_only_side="bridge_only",
        include_jaccard=True,
    )
    payload = {
        "trip_dir": str(trip_dir),
        "matlab_mask_path": str(matlab_mask_path),
        "multi_gnss": bool(multi_gnss),
        "pseudorange_residual_mask_m": float(pseudorange_residual_mask_m),
        "pseudorange_residual_mask_l5_m": float(pseudorange_residual_mask_l5_m),
        "doppler_residual_mask_mps": float(doppler_residual_mask_mps),
        "tdcp_consistency_threshold_m": float(tdcp_consistency_threshold_m),
        "pseudorange_doppler_mask_m": float(pseudorange_doppler_mask_m),
        "matlab_residual_diagnostics_mask_path": (
            str(matlab_residual_diagnostics_mask_path)
            if matlab_residual_diagnostics_mask_path is not None
            else None
        ),
        "total_matlab_count": side_payload["total_matlab_count"],
        "total_bridge_count": side_payload["total_bridge_count"],
        "total_matched_count": side_payload["total_matched_count"],
        "total_matlab_only": side_payload["total_matlab_only"],
        "total_bridge_only": side_payload["total_bridge_only"],
        "start_epoch": int(start_epoch),
        "max_epochs": int(bridge_max_epochs),
        "jaccard": side_payload["jaccard"],
        "symmetric_parity": side_payload["symmetric_parity"],
    }
    return merged, summary, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_data_root_trip_args(parser, default_root=DEFAULT_ROOT)
    parser.add_argument("--matlab-mask", type=Path, default=None)
    _add_max_epochs_arg(parser)
    parser.add_argument("--pseudorange-residual-mask-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_M)
    parser.add_argument("--pseudorange-residual-mask-l5-m", type=float, default=OBS_MASK_RESIDUAL_THRESHOLD_L5_M)
    parser.add_argument("--doppler-residual-mask-mps", type=float, default=OBS_MASK_DOPPLER_RESIDUAL_THRESHOLD_MPS)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--pseudorange-doppler-mask-m", type=float, default=OBS_MASK_PSEUDORANGE_DOPPLER_THRESHOLD_M)
    parser.add_argument(
        "--matlab-residual-diagnostics-mask",
        type=Path,
        default=None,
        help="optional phone_data_residual_diagnostics.csv used to force bridge P/D/L factor availability",
    )
    _add_multi_gnss_arg(
        parser,
        help_text="match raw bridge multi-GNSS mode; default is GPS-only MATLAB parity",
    )
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    trip_dir = _resolve_trip_dir(args)
    out_dir = _timestamped_output_dir(
        _resolved_output_root(args),
        "gsdc2023_factor_mask_parity",
    )

    merged, summary, payload = compare_factor_masks(
        trip_dir,
        matlab_mask_path=args.matlab_mask,
        max_epochs=_nonnegative_max_epochs(args),
        multi_gnss=args.multi_gnss,
        pseudorange_residual_mask_m=args.pseudorange_residual_mask_m,
        pseudorange_residual_mask_l5_m=args.pseudorange_residual_mask_l5_m,
        doppler_residual_mask_mps=args.doppler_residual_mask_mps,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        pseudorange_doppler_mask_m=args.pseudorange_doppler_mask_m,
        matlab_residual_diagnostics_mask_path=args.matlab_residual_diagnostics_mask,
    )
    merged.to_csv(out_dir / "factor_mask_join.csv", index=False)
    merged[merged["side"] == "matlab_only"].to_csv(out_dir / "matlab_only.csv", index=False)
    merged[merged["side"] == "bridge_only"].to_csv(out_dir / "bridge_only.csv", index=False)
    summary.to_csv(out_dir / "summary_by_field.csv", index=False)
    _write_summary_json(out_dir, payload)
    _print_summary_and_output_dir(payload, out_dir)


if __name__ == "__main__":
    main()
