#!/usr/bin/env python3
"""Scan many moving blocks for auto-phase carrier/DDPR evidence in one load."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.dd_carrier import DDCarrierComputer  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex_cache import RinexObservationCache  # noqa: E402
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch  # noqa: E402
from refine_wp31_moving_block_ambiguity import (  # noqa: E402
    _carrier_rows,
    _read_trajectory,
    choose_evidence_phase,
)


def block_spans(start: int, end: int, block_epochs: int) -> list[tuple[int, int]]:
    """Partition ``[start, end)`` without silently padding the final block."""

    if not 0 <= start < end or block_epochs <= 0:
        raise ValueError("moving supply scan bounds are invalid")
    return [
        (block_start, min(block_start + block_epochs, end))
        for block_start in range(start, end, block_epochs)
    ]


def summarize_block(
    *,
    start: int,
    end: int,
    block_epochs: int,
    stride: int,
    epoch_supply: dict[int, dict[str, int]],
    min_evidence_epochs: int,
    min_carrier_rows: int,
    min_ddpr_rows: int,
) -> dict[str, Any]:
    phase_diagnostics = []
    for phase in range(stride):
        rows = [
            epoch_supply[epoch]
            for epoch in range(start, end)
            if epoch % stride == phase and epoch in epoch_supply
        ]
        phase_diagnostics.append(
            {
                "phase": phase,
                "evidence_epochs": sum(row["evidence"] for row in rows),
                "raw_carrier_rows": sum(row["carrier_rows"] for row in rows),
                "ddpr_epochs": sum(row["ddpr_epoch"] for row in rows),
                "raw_ddpr_rows": sum(row["ddpr_rows"] for row in rows),
            }
        )
    selected_phase = choose_evidence_phase(phase_diagnostics)
    selected = phase_diagnostics[selected_phase]
    complete = end - start == block_epochs
    gates = {
        "complete_block": complete,
        "evidence_epochs": selected["evidence_epochs"] >= min_evidence_epochs,
        "raw_carrier_rows": selected["raw_carrier_rows"] >= min_carrier_rows,
        "raw_ddpr_rows": selected["raw_ddpr_rows"] >= min_ddpr_rows,
    }
    return {
        "segment": [start, end],
        "selected_stride_phase": selected_phase,
        "selected_supply": selected,
        "phase_diagnostics": phase_diagnostics,
        "gates": gates,
        "pre_candidate_supply_pass": all(gates.values()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    route = _read_trajectory(args.trajectory, args.start, args.end)
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.end,
        include_sat_velocity=True,
        systems=("G", "R", "E", "C", "J"),
    )
    cache = RinexObservationCache()
    systems = ("G", "E", "J", "C")
    cp_engine = DDCarrierComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"]),
        allowed_systems=systems,
        observation_cache=cache,
    )
    pr_engine = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"]),
        allowed_systems=systems,
        observation_cache=cache,
    )
    families = tuple(value for value in args.carrier_families.split(",") if value)
    epoch_supply: dict[int, dict[str, int]] = {}
    for epoch in range(args.start, args.end):
        approximate = route.get(epoch)
        if approximate is None:
            continue
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch]),
            np.asarray(data["system_ids"][epoch]),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch]),
            approximate,
            systems,
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        cp = cp_engine.compute_dd_families(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=2,
            carrier_families=families,
        )
        pr = pr_engine.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=4,
        )
        carrier_rows = (
            0
            if cp is None
            else len(_carrier_rows(epoch, DDCarrierEpoch.from_result(cp)))
        )
        ddpr_rows = 0 if pr is None else DDPseudorangeEpoch.from_result(pr).n
        epoch_supply[epoch] = {
            "evidence": int(cp is not None or pr is not None),
            "carrier_rows": carrier_rows,
            "ddpr_epoch": int(pr is not None),
            "ddpr_rows": ddpr_rows,
        }
    config = {
        "block_epochs": args.block_epochs,
        "stride": args.stride,
        "stride_phase_mode": "auto_max_available_evidence_then_raw_carrier_then_ddpr",
        "min_evidence_epochs": args.min_evidence_epochs,
        "min_carrier_rows": args.min_carrier_rows,
        "min_ddpr_rows": args.min_ddpr_rows,
    }
    blocks = [
        summarize_block(
            start=start,
            end=end,
            block_epochs=args.block_epochs,
            stride=args.stride,
            epoch_supply=epoch_supply,
            min_evidence_epochs=args.min_evidence_epochs,
            min_carrier_rows=args.min_carrier_rows,
            min_ddpr_rows=args.min_ddpr_rows,
        )
        for start, end in block_spans(args.start, args.end, args.block_epochs)
    ]
    return {
        "schema": "wp46_moving_evidence_supply_scan_v1",
        "production_input_truth": False,
        "truth_usage": "none",
        "range": [args.start, args.end],
        "config": config,
        "blocks": blocks,
        "complete_blocks": sum(block["gates"]["complete_block"] for block in blocks),
        "supply_pass_blocks": sum(
            block["pre_candidate_supply_pass"] for block in blocks
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--block-epochs", type=int, default=55)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--carrier-families", default="L1_E1_B1,L5_E5A_B2A")
    parser.add_argument("--min-evidence-epochs", type=int, default=10)
    parser.add_argument("--min-carrier-rows", type=int, default=24)
    parser.add_argument("--min-ddpr-rows", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
