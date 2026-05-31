#!/usr/bin/env python3
"""Audit GSDC2023 carrier-phase usability for sub-meter candidate work."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.eval_gsdc2023_ct_rbpf_fgo import discover_train_trips  # noqa: E402
from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M,
    DEFAULT_TDCP_GEOMETRY_CORRECTION,
    DEFAULT_TDCP_WEIGHT_SCALE,
    build_trip_arrays,
)
from experiments.gsdc2023_tdcp import (  # noqa: E402
    ADR_STATE_CYCLE_SLIP,
    ADR_STATE_RESET,
    valid_adr_state,
)


DEFAULT_OUTPUT = Path("experiments/results/gsdc2023_carrier_phase_usability_20260519.csv")


def _safe_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def _signal_family_counts(slot_keys: tuple[tuple[int, int, str], ...]) -> dict[str, int]:
    out = {
        "slot_gps": 0,
        "slot_galileo": 0,
        "slot_qzss": 0,
        "slot_l1e1": 0,
        "slot_l5e5": 0,
    }
    for constellation, _svid, signal in slot_keys:
        signal_l = str(signal).lower()
        if int(constellation) == 1:
            out["slot_gps"] += 1
        elif int(constellation) == 6:
            out["slot_galileo"] += 1
        elif int(constellation) == 4:
            out["slot_qzss"] += 1
        if "l5" in signal_l or "e5" in signal_l:
            out["slot_l5e5"] += 1
        else:
            out["slot_l1e1"] += 1
    return out


def carrier_usability_row(trip: str, batch) -> dict[str, object]:
    n_epoch = int(batch.times_ms.size)
    n_slot = int(batch.n_sat_slots)
    adr = batch.adr
    adr_state = batch.adr_state
    adr_unc = batch.adr_uncertainty
    row: dict[str, object] = {
        "trip": trip,
        "phone": Path(trip).name,
        "n_epochs": n_epoch,
        "n_sat_slots": n_slot,
        "dual_frequency": bool(batch.dual_frequency),
        "tdcp_consistency_mask_count": int(batch.tdcp_consistency_mask_count),
        "tdcp_geometry_correction_count": int(batch.tdcp_geometry_correction_count),
    }
    row.update(_signal_family_counts(batch.slot_keys))
    if adr is None or adr_state is None:
        row.update(
            {
                "adr_observations": 0,
                "valid_adr_observations": 0,
                "valid_adr_ratio": 0.0,
                "reset_observations": 0,
                "cycle_slip_observations": 0,
                "valid_adr_mean_per_epoch": 0.0,
                "valid_adr_p50_per_epoch": 0.0,
                "valid_adr_p95_per_epoch": 0.0,
                "continuous_valid_pairs": 0,
                "continuous_valid_mean_per_interval": 0.0,
                "tdcp_pairs": 0,
                "tdcp_mean_pairs_per_interval": 0.0,
                "adr_uncertainty_median_m": float("nan"),
            },
        )
        return row

    finite_adr = np.isfinite(adr) & (adr != 0.0)
    valid_state = np.vectorize(valid_adr_state)(adr_state.astype(np.int64))
    valid_adr = finite_adr & valid_state
    reset = finite_adr & ((adr_state.astype(np.int64) & ADR_STATE_RESET) != 0)
    cycle_slip = finite_adr & ((adr_state.astype(np.int64) & ADR_STATE_CYCLE_SLIP) != 0)
    valid_per_epoch = np.count_nonzero(valid_adr, axis=1) if valid_adr.ndim == 2 else np.zeros(n_epoch)
    continuous = valid_adr[:-1] & valid_adr[1:] if n_epoch > 1 else np.zeros((0, n_slot), dtype=bool)
    continuous_per_interval = (
        np.count_nonzero(continuous, axis=1) if continuous.ndim == 2 else np.zeros(max(n_epoch - 1, 0))
    )
    tdcp_pairs = 0
    tdcp_per_interval = np.zeros(max(n_epoch - 1, 0), dtype=np.int64)
    if batch.tdcp_weights is not None:
        tdcp_valid = np.asarray(batch.tdcp_weights, dtype=np.float64) > 0.0
        tdcp_pairs = int(np.count_nonzero(tdcp_valid))
        if tdcp_valid.ndim == 2:
            tdcp_per_interval = np.count_nonzero(tdcp_valid, axis=1)

    unc = np.asarray(adr_unc, dtype=np.float64) if adr_unc is not None else np.full_like(adr, np.nan)
    row.update(
        {
            "adr_observations": int(np.count_nonzero(finite_adr)),
            "valid_adr_observations": int(np.count_nonzero(valid_adr)),
            "valid_adr_ratio": (
                float(np.count_nonzero(valid_adr) / max(np.count_nonzero(finite_adr), 1))
                if finite_adr.size
                else 0.0
            ),
            "reset_observations": int(np.count_nonzero(reset)),
            "cycle_slip_observations": int(np.count_nonzero(cycle_slip)),
            "valid_adr_mean_per_epoch": float(np.mean(valid_per_epoch)) if valid_per_epoch.size else 0.0,
            "valid_adr_p50_per_epoch": _safe_percentile(valid_per_epoch, 50),
            "valid_adr_p95_per_epoch": _safe_percentile(valid_per_epoch, 95),
            "continuous_valid_pairs": int(np.count_nonzero(continuous)),
            "continuous_valid_mean_per_interval": (
                float(np.mean(continuous_per_interval)) if continuous_per_interval.size else 0.0
            ),
            "tdcp_pairs": int(tdcp_pairs),
            "tdcp_mean_pairs_per_interval": float(np.mean(tdcp_per_interval)) if tdcp_per_interval.size else 0.0,
            "adr_uncertainty_median_m": _safe_percentile(unc[valid_adr], 50),
        },
    )
    return row


def audit_trip(data_root: Path, trip: str, args: argparse.Namespace) -> dict[str, object]:
    batch = build_trip_arrays(
        data_root / trip,
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        multi_gnss=True,
        use_tdcp=True,
        tdcp_consistency_threshold_m=args.tdcp_consistency_threshold_m,
        tdcp_weight_scale=args.tdcp_weight_scale,
        tdcp_geometry_correction=args.tdcp_geometry_correction,
        data_root=data_root,
        trip=trip,
        dual_frequency=args.dual_frequency,
    )
    return carrier_usability_row(trip, batch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trip", action="append", default=[], help="train/.../phone trip; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tdcp-consistency-threshold-m", type=float, default=DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M)
    parser.add_argument("--tdcp-weight-scale", type=float, default=DEFAULT_TDCP_WEIGHT_SCALE)
    parser.add_argument("--tdcp-geometry-correction", action=argparse.BooleanOptionalAction, default=DEFAULT_TDCP_GEOMETRY_CORRECTION)
    args = parser.parse_args()

    trips = discover_train_trips(args.data_root)
    if args.trip:
        wanted = set(args.trip)
        trips = [trip for trip in trips if trip in wanted]
    if args.limit > 0:
        trips = trips[: args.limit]
    if not trips:
        raise RuntimeError("no train trips found")

    rows = []
    for idx, trip in enumerate(trips, start=1):
        rows.append(audit_trip(args.data_root, trip, args))
        print(f"[{idx}/{len(trips)}] {trip}", flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote: {args.output}")
    print(f"mean valid ADR/epoch: {frame['valid_adr_mean_per_epoch'].mean():.2f}")
    print(f"mean TDCP pairs/interval: {frame['tdcp_mean_pairs_per_interval'].mean():.2f}")


if __name__ == "__main__":
    main()
