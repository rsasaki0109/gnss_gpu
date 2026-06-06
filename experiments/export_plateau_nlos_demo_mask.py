#!/usr/bin/env python3
"""Export the PLATEAU NLOS demo ray mask as an experiment CSV.

The first four columns match the existing FGO/SPP/PF mask contract:

    tow,epoch_idx,prn,is_los

where ``is_los=0`` marks a satellite as NLOS. Extra columns are diagnostic and
can be ignored by consumers that only need the minimal contract.

Run from the repo root:

    PYTHONPATH=python:. python3 experiments/export_plateau_nlos_demo_mask.py \
      --out-csv /tmp/plateau_nlos_demo_mask.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_mask.csv"


def _load_demo_module():
    module_path = PROJECT_ROOT / "examples" / "demo_plateau_nlos_simulation.py"
    spec = importlib.util.spec_from_file_location("demo_plateau_nlos_simulation", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prn(prefix: str, sat_idx: int) -> str:
    return f"{prefix}{sat_idx + 1:02d}"


def export_mask_csv(
    out_csv: Path = DEFAULT_OUT_CSV,
    *,
    summary_json: Path | None = None,
    gml_path: Path | None = None,
    start_tow: float = 0.0,
    epoch_dt: float = 1.0,
    prn_prefix: str = "G",
) -> dict[str, object]:
    demo = _load_demo_module()
    result = demo.run_demo(gml_path=gml_path, include_details=True)
    details = result["details"]

    los_mask = np.asarray(details["all_los_mask"], dtype=bool)
    rx_enu = np.asarray(details["all_rx_enu"], dtype=np.float64)
    azel = np.asarray(details["satellite_az_el_deg"], dtype=np.float64)
    epoch_indices = np.asarray(details["all_epoch_index"], dtype=np.int64)
    nlos_count = np.asarray(details["all_nlos_count"], dtype=np.int64)
    expected_bias = np.vstack(
        [demo.nlos_expected_bias_m(azel[:, 1], los_mask[i]) for i in range(los_mask.shape[0])]
    )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_nlos = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "tow",
                "epoch_idx",
                "prn",
                "is_los",
                "system",
                "svid",
                "azimuth_deg",
                "elevation_deg",
                "nlos_expected_bias_m",
                "nlos_count_epoch",
                "receiver_e_m",
                "receiver_n_m",
                "receiver_u_m",
                "ray_source",
            ]
        )
        for row_idx, epoch_idx in enumerate(epoch_indices):
            tow = start_tow + float(epoch_idx) * epoch_dt
            for sat_idx in range(los_mask.shape[1]):
                prn = _prn(prn_prefix, sat_idx)
                is_los = bool(los_mask[row_idx, sat_idx])
                n_rows += 1
                n_nlos += int(not is_los)
                writer.writerow(
                    [
                        f"{tow:.3f}",
                        int(epoch_idx),
                        prn,
                        int(is_los),
                        prn[0],
                        prn[1:],
                        f"{float(azel[sat_idx, 0]):.3f}",
                        f"{float(azel[sat_idx, 1]):.3f}",
                        f"{float(expected_bias[row_idx, sat_idx]):.3f}",
                        int(nlos_count[row_idx]),
                        f"{float(rx_enu[row_idx, 0]):.3f}",
                        f"{float(rx_enu[row_idx, 1]):.3f}",
                        f"{float(rx_enu[row_idx, 2]):.3f}",
                        str(result["ray_source"]),
                    ]
                )

    summary = {
        "out_csv": str(out_csv),
        "gml_path": result["gml_path"],
        "ray_source": result["ray_source"],
        "epochs": int(los_mask.shape[0]),
        "satellites": int(los_mask.shape[1]),
        "rows": int(n_rows),
        "nlos": int(n_nlos),
        "nlos_frac": float(n_nlos / n_rows) if n_rows else 0.0,
        "start_tow": float(start_tow),
        "epoch_dt": float(epoch_dt),
        "prn_prefix": str(prn_prefix),
    }

    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--gml", type=Path, default=None)
    parser.add_argument("--start-tow", type=float, default=0.0)
    parser.add_argument("--epoch-dt", type=float, default=1.0)
    parser.add_argument("--prn-prefix", default="G")
    args = parser.parse_args()
    if args.epoch_dt <= 0.0:
        parser.error("--epoch-dt must be positive")
    if not str(args.prn_prefix):
        parser.error("--prn-prefix must be non-empty")

    summary = export_mask_csv(
        args.out_csv,
        summary_json=args.summary_json,
        gml_path=args.gml,
        start_tow=args.start_tow,
        epoch_dt=args.epoch_dt,
        prn_prefix=args.prn_prefix,
    )
    print(
        f"wrote {summary['out_csv']} rows={summary['rows']} "
        f"nlos={summary['nlos']} nlos_frac={summary['nlos_frac']:.4f}"
    )
    return summary


if __name__ == "__main__":
    main()
