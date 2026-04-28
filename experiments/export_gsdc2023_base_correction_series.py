#!/usr/bin/env python3
"""Export GSDC2023 base-station pseudorange correction series.

This is a comparison scaffold for MATLAB ``correct_pseudorange.m`` parity. It
uses the raw bridge's current base-correction implementation and writes the
per-epoch, per-slot correction matrix in both wide and long CSV forms. GPS,
Galileo, and QZSS slots are included when matching base RINEX/nav data exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import (  # noqa: E402
    DEFAULT_ROOT,
    build_trip_arrays,
    collect_matlab_parity_audit,
    compute_base_pseudorange_correction_matrix,
)


def _slot_label(key: tuple[int, int, str]) -> str:
    constellation_type, svid, signal_type = key
    return f"c{int(constellation_type)}_s{int(svid):02d}_{str(signal_type)}"


def export_base_correction_series(
    data_root: Path,
    trip: str,
    output_dir: Path,
    *,
    max_epochs: int,
    start_epoch: int,
    constellation_type: int,
    signal_type: str,
    weight_mode: str,
    multi_gnss: bool,
    dual_frequency: bool,
) -> dict[str, object]:
    trip_dir = data_root / trip
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=(max_epochs if max_epochs > 0 else 1_000_000_000),
        start_epoch=start_epoch,
        constellation_type=constellation_type,
        signal_type=signal_type,
        weight_mode=weight_mode,
        multi_gnss=multi_gnss,
        use_tdcp=False,
        apply_base_correction=False,
        data_root=data_root,
        trip=trip,
        dual_frequency=dual_frequency,
    )
    if not batch.slot_keys:
        raise RuntimeError("raw bridge did not expose satellite slot keys")

    correction = compute_base_pseudorange_correction_matrix(
        data_root,
        trip,
        batch.times_ms,
        list(batch.slot_keys),
        signal_type,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    slot_labels = [_slot_label(key) for key in batch.slot_keys]
    wide = pd.DataFrame(correction, columns=slot_labels)
    wide.insert(0, "UnixTimeMillis", batch.times_ms.astype(np.int64))
    wide.to_csv(output_dir / "base_correction_wide.csv", index=False)

    rows: list[dict[str, object]] = []
    for epoch_idx, unix_ms in enumerate(batch.times_ms.astype(np.int64)):
        for slot_idx, key in enumerate(batch.slot_keys):
            constellation_type_i, svid, signal_type_i = key
            value = float(correction[epoch_idx, slot_idx])
            rows.append(
                {
                    "UnixTimeMillis": int(unix_ms),
                    "EpochIndex": int(epoch_idx),
                    "SlotIndex": int(slot_idx),
                    "ConstellationType": int(constellation_type_i),
                    "Svid": int(svid),
                    "SignalType": str(signal_type_i),
                    "CorrectionMeters": value if np.isfinite(value) else np.nan,
                    "ObservationWeightPositive": bool(batch.weights[epoch_idx, slot_idx] > 0.0),
                },
            )
    long_df = pd.DataFrame(rows)
    long_df.to_csv(output_dir / "base_correction_long.csv", index=False)

    finite = np.isfinite(correction)
    applied = finite & (batch.weights > 0.0)
    parity_audit = collect_matlab_parity_audit(data_root, trip)
    summary: dict[str, object] = {
        "trip": trip,
        "data_root": str(data_root),
        "n_epochs": int(batch.times_ms.size),
        "n_slots": int(len(batch.slot_keys)),
        "finite_correction_count": int(np.count_nonzero(finite)),
        "applied_correction_count": int(np.count_nonzero(applied)),
        "finite_slot_count": int(np.count_nonzero(np.any(finite, axis=0))),
        "base_correction_status": parity_audit.get("base_correction_status"),
        "base_correction_ready": bool(parity_audit.get("base_correction_ready", False)),
        "wide_csv": str(output_dir / "base_correction_wide.csv"),
        "long_csv": str(output_dir / "base_correction_long.csv"),
    }
    if finite.any():
        summary.update(
            {
                "correction_mean_m": float(np.nanmean(correction)),
                "correction_std_m": float(np.nanstd(correction)),
                "correction_min_m": float(np.nanmin(correction)),
                "correction_max_m": float(np.nanmax(correction)),
            },
        )
    (output_dir / "base_correction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--trip", required=True, help="relative trip path under data root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--constellation-type", type=int, default=1)
    parser.add_argument("--signal-type", default="GPS_L1_CA")
    parser.add_argument("--weight-mode", choices=("sin2el", "cn0"), default="sin2el")
    parser.add_argument("--multi-gnss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dual-frequency", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    summary = export_base_correction_series(
        args.data_root.resolve(),
        args.trip,
        args.output_dir.resolve(),
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        constellation_type=args.constellation_type,
        signal_type=args.signal_type,
        weight_mode=args.weight_mode,
        multi_gnss=args.multi_gnss,
        dual_frequency=args.dual_frequency,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
