#!/usr/bin/env python3
"""Replay the PLATEAU NLOS demo through an SPP mask-consumer path.

This script consumes the exported mask CSV contract:

    tow,epoch_idx,prn,is_los

It does not ray-trace. The CSV is the only LOS/NLOS input used by the
mask-aware solver. The synthetic pseudoranges are regenerated with the same
deterministic measurement model as the demo so the downstream SPP behavior can
be tested without needing real GNSS logs.

Run from the repo root:

    PYTHONPATH=python:. python3 experiments/replay_plateau_nlos_demo_spp.py \
      --mask-csv experiments/results/plateau_nlos_demo_mask.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from gnss_gpu.robust_spp import robust_spp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_CSV = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_mask.csv"
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_spp_replay_summary.json"
)

PLAIN_LS_THRESHOLD_M = 1.0e12
ROBUST_THRESHOLD_M = 12.0
DEFAULT_NLOS_WEIGHT = 0.12


def _load_demo_module():
    module_path = PROJECT_ROOT / "examples" / "demo_plateau_nlos_simulation.py"
    spec = importlib.util.spec_from_file_location("demo_plateau_nlos_simulation", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sat_index_from_prn(prn: str) -> int:
    digits = "".join(ch for ch in str(prn) if ch.isdigit())
    if not digits:
        raise ValueError(f"PRN has no numeric SVID: {prn!r}")
    idx = int(digits) - 1
    if idx < 0:
        raise ValueError(f"PRN SVID must be positive: {prn!r}")
    return idx


def load_mask_csv(mask_csv: Path, *, n_epochs: int, n_satellites: int) -> dict[str, np.ndarray]:
    los_mask = np.ones((n_epochs, n_satellites), dtype=bool)
    expected_bias = np.zeros((n_epochs, n_satellites), dtype=np.float64)
    present = np.zeros((n_epochs, n_satellites), dtype=bool)
    tow_by_epoch = np.full(n_epochs, np.nan, dtype=np.float64)

    with Path(mask_csv).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"tow", "epoch_idx", "prn", "is_los"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"mask CSV missing required columns: {sorted(missing)}")
        for row in reader:
            epoch_idx = int(row["epoch_idx"])
            sat_idx = _sat_index_from_prn(row["prn"])
            if epoch_idx < 0 or epoch_idx >= n_epochs:
                continue
            if sat_idx < 0 or sat_idx >= n_satellites:
                continue
            is_los = int(row["is_los"]) != 0
            los_mask[epoch_idx, sat_idx] = is_los
            present[epoch_idx, sat_idx] = True
            tow_by_epoch[epoch_idx] = float(row["tow"])
            if row.get("nlos_expected_bias_m", "") != "":
                expected_bias[epoch_idx, sat_idx] = float(row["nlos_expected_bias_m"])

    complete_epoch = np.all(present, axis=1)
    return {
        "los_mask": los_mask,
        "expected_bias_m": expected_bias,
        "complete_epoch": complete_epoch,
        "tow": tow_by_epoch,
        "present": present,
    }


def _p50(values: np.ndarray) -> float:
    return float(np.median(values))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def _summarize_errors(values: np.ndarray) -> dict[str, float]:
    return {
        "p50_m": _p50(values),
        "rms_m": _rms(values),
        "mean_m": float(np.mean(values)),
        "max_m": float(np.max(values)),
    }


def replay_spp(
    mask_csv: Path = DEFAULT_MASK_CSV,
    *,
    summary_json: Path | None = DEFAULT_SUMMARY_JSON,
    gml_path: Path | None = None,
    nlos_weight: float = DEFAULT_NLOS_WEIGHT,
    bias_scale: float = 1.0,
    robust_threshold_m: float = ROBUST_THRESHOLD_M,
) -> dict[str, object]:
    demo = _load_demo_module()
    if gml_path is None:
        gml_path = PROJECT_ROOT / "data" / "sample_plateau.gml"

    rng = np.random.default_rng(20260606)
    triangles = demo.load_plateau_triangles(gml_path)
    origin_ecef, enu_to_ecef, verts_enu = demo.build_local_frame(triangles)
    ecef_to_enu = enu_to_ecef.T
    ground_z_m = float(verts_enu[:, 2].min() + 1.8)

    sats_azel = demo.default_satellite_az_el_deg()
    elevations_deg = np.array([el for _az, el in sats_azel], dtype=np.float64)
    sat_ecef = demo.build_satellites(origin_ecef, enu_to_ecef, sats_azel)

    n_epochs = 70
    rx_enu = np.zeros((n_epochs, 3), dtype=np.float64)
    rx_enu[:, 0] = np.linspace(-55.0, 55.0, n_epochs)
    rx_enu[:, 1] = -10.0
    rx_enu[:, 2] = ground_z_m
    rx_ecef = origin_ecef + rx_enu @ enu_to_ecef.T

    mask = load_mask_csv(mask_csv, n_epochs=n_epochs, n_satellites=len(sats_azel))
    los_mask = np.asarray(mask["los_mask"], dtype=bool)
    expected_bias = np.asarray(mask["expected_bias_m"], dtype=np.float64)
    missing_bias = (~los_mask) & (expected_bias <= 0.0)
    if np.any(missing_bias):
        computed = np.vstack(
            [demo.nlos_expected_bias_m(elevations_deg, los_mask[i]) for i in range(n_epochs)]
        )
        expected_bias[missing_bias] = computed[missing_bias]

    complete_epoch = np.asarray(mask["complete_epoch"], dtype=bool)
    clock_bias_m = 1432.0
    init_guess = origin_ecef + enu_to_ecef @ np.array([15.0, -10.0, 5.0])

    naive_errors: list[float] = []
    robust_errors: list[float] = []
    mask_soft_errors: list[float] = []
    used_epochs: list[int] = []
    n_solver_fail = 0

    for epoch_idx in range(n_epochs):
        if not complete_epoch[epoch_idx]:
            continue
        is_los = los_mask[epoch_idx]
        obs = demo.simulate_pseudorange_epoch(
            rng,
            rx_ecef[epoch_idx],
            sat_ecef,
            elevations_deg,
            is_los,
            clock_bias_m,
        )
        pr = obs["pseudorange_m"]
        correction = bias_scale * expected_bias[epoch_idx]
        weights = np.where(is_los, 1.0, float(nlos_weight))

        naive = robust_spp(
            sat_ecef,
            pr,
            init_pos=init_guess,
            weight_func="huber",
            threshold=PLAIN_LS_THRESHOLD_M,
            min_satellites=5,
        )
        robust = robust_spp(
            sat_ecef,
            pr,
            init_pos=init_guess,
            weight_func="cauchy",
            threshold=robust_threshold_m,
            min_satellites=5,
        )
        mask_soft = robust_spp(
            sat_ecef,
            pr - correction,
            weights=weights,
            init_pos=init_guess,
            weight_func="cauchy",
            threshold=robust_threshold_m,
            min_satellites=5,
        )
        if naive is None or robust is None or mask_soft is None:
            n_solver_fail += 1
            continue

        true_rx = rx_ecef[epoch_idx]
        naive_errors.append(demo.horizontal_error_m(naive, true_rx, ecef_to_enu))
        robust_errors.append(demo.horizontal_error_m(robust, true_rx, ecef_to_enu))
        mask_soft_errors.append(demo.horizontal_error_m(mask_soft, true_rx, ecef_to_enu))
        used_epochs.append(epoch_idx)

    naive_arr = np.asarray(naive_errors, dtype=np.float64)
    robust_arr = np.asarray(robust_errors, dtype=np.float64)
    mask_arr = np.asarray(mask_soft_errors, dtype=np.float64)
    if naive_arr.size == 0:
        raise RuntimeError("SPP replay produced no solved epochs")

    summary = {
        "mask_csv": str(mask_csv),
        "gml_path": str(gml_path),
        "n_epochs": int(n_epochs),
        "n_complete_mask_epochs": int(np.count_nonzero(complete_epoch)),
        "n_solved_epochs": int(naive_arr.size),
        "n_solver_fail": int(n_solver_fail),
        "nlos_weight": float(nlos_weight),
        "bias_scale": float(bias_scale),
        "robust_threshold_m": float(robust_threshold_m),
        "nlos_frac": float(np.mean(~los_mask[complete_epoch])),
        "naive": _summarize_errors(naive_arr),
        "robust": _summarize_errors(robust_arr),
        "mask_soft": _summarize_errors(mask_arr),
        "mask_soft_wins": int(np.sum(mask_arr < naive_arr)),
        "robust_wins": int(np.sum(robust_arr < naive_arr)),
        "rms_gain_vs_naive_pct": float(100.0 * (1.0 - _rms(mask_arr) / _rms(naive_arr))),
        "used_epochs": used_epochs,
    }

    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary["summary_json"] = str(summary_json)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-csv", type=Path, default=DEFAULT_MASK_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--gml", type=Path, default=None)
    parser.add_argument("--nlos-weight", type=float, default=DEFAULT_NLOS_WEIGHT)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument("--robust-threshold-m", type=float, default=ROBUST_THRESHOLD_M)
    args = parser.parse_args()
    if not (0.0 < args.nlos_weight <= 1.0):
        parser.error("--nlos-weight must be in (0, 1]")
    if args.bias_scale < 0.0:
        parser.error("--bias-scale must be non-negative")

    summary = replay_spp(
        args.mask_csv,
        summary_json=args.summary_json,
        gml_path=args.gml,
        nlos_weight=args.nlos_weight,
        bias_scale=args.bias_scale,
        robust_threshold_m=args.robust_threshold_m,
    )
    print("PLATEAU NLOS SPP replay")
    print("=" * 70)
    print(
        f"mask={summary['mask_csv']} solved={summary['n_solved_epochs']}/"
        f"{summary['n_complete_mask_epochs']} nlos_frac={summary['nlos_frac']:.4f}"
    )
    print(f"{'method':<24}{'P50 err':>12}{'RMS err':>12}")
    print("-" * 48)
    for name, label in [
        ("naive", "naive SPP"),
        ("robust", "robust SPP"),
        ("mask_soft", "mask-soft SPP"),
    ]:
        metrics = summary[name]
        print(f"{label:<24}{metrics['p50_m']:>10.2f} m{metrics['rms_m']:>10.2f} m")
    print("-" * 48)
    print(
        f"mask-soft wins {summary['mask_soft_wins']}/{summary['n_solved_epochs']} epochs; "
        f"RMS gain {summary['rms_gain_vs_naive_pct']:.0f}%"
    )
    if summary.get("summary_json"):
        print(f"summary: {summary['summary_json']}")
    return summary


if __name__ == "__main__":
    main()
