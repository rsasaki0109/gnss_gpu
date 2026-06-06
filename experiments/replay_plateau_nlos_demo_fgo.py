#!/usr/bin/env python3
"""Replay the PLATEAU NLOS demo mask through a local-FGO consumer path.

This script consumes the exported mask CSV contract:

    tow,epoch_idx,prn,is_los

The CSV is the only LOS/NLOS input used by the mask-aware graph. Synthetic
pseudoranges are regenerated with the deterministic PLATEAU NLOS measurement
model used by the demo. The graph is a position-only local FGO with undifferenced
pseudorange factors and inter-epoch motion factors; it uses the repo's
``gnss_gpu.local_fgo`` API and falls back to its NumPy LM solver when GTSAM is
not installed.

Run from the repo root:

    PYTHONPATH=python:. python3 experiments/replay_plateau_nlos_demo_fgo.py \
      --mask-csv experiments/results/plateau_nlos_demo_mask.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from gnss_gpu.local_fgo import (
    LocalFgoConfig,
    LocalFgoProblem,
    LocalFgoWindow,
    UndiffPseudorangeEpoch,
    solve_local_fgo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_CSV = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_mask.csv"
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_fgo_replay_summary.json"
)

DEFAULT_NLOS_WEIGHT = 0.12
DEFAULT_PR_SIGMA_M = 7.5
DEFAULT_MOTION_SIGMA_M = 0.5
DEFAULT_ROBUST_HUBER_K = 1.5


def _load_spp_replay_module():
    module_path = PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_spp.py"
    spec = importlib.util.spec_from_file_location("replay_plateau_nlos_demo_spp", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _make_undiff_epochs(
    sat_ecef: np.ndarray,
    pseudorange_m: np.ndarray,
    weights: np.ndarray,
    *,
    clock_bias_m: float,
) -> list[UndiffPseudorangeEpoch]:
    return [
        UndiffPseudorangeEpoch(
            sat_ecef=sat_ecef,
            pseudoranges_m=np.asarray(pseudorange_m[i], dtype=np.float64),
            clock_bias_m=float(clock_bias_m),
            weights=np.asarray(weights[i], dtype=np.float64),
        )
        for i in range(pseudorange_m.shape[0])
    ]


def _solve_case(
    *,
    initial_positions_ecef: np.ndarray,
    true_positions_ecef: np.ndarray,
    ecef_to_enu: np.ndarray,
    demo: object,
    sat_ecef: np.ndarray,
    pseudorange_m: np.ndarray,
    weights: np.ndarray,
    clock_bias_m: float,
    motion_sigma_m: float,
    pr_sigma_m: float,
    huber_k: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_epochs = int(true_positions_ecef.shape[0])
    motion_deltas = np.zeros_like(true_positions_ecef)
    motion_deltas[:-1] = np.diff(true_positions_ecef, axis=0)
    no_priors = np.zeros(n_epochs, dtype=np.float64)
    problem = LocalFgoProblem(
        initial_positions_ecef=initial_positions_ecef,
        window=LocalFgoWindow(0, n_epochs - 1),
        motion_deltas_ecef=motion_deltas,
        prior_sigmas_m=no_priors,
        undiff_pseudorange=_make_undiff_epochs(
            sat_ecef,
            pseudorange_m,
            weights,
            clock_bias_m=clock_bias_m,
        ),
    )
    result = solve_local_fgo(
        problem,
        LocalFgoConfig(
            prior_sigma_m=1000.0,
            motion_sigma_m=float(motion_sigma_m),
            undiff_pr_sigma_m=float(pr_sigma_m),
            pr_huber_k=float(huber_k),
            max_iterations=40,
            relative_error_tol=1.0e-7,
        ),
    )
    errors = np.asarray(
        [
            demo.horizontal_error_m(pos, truth, ecef_to_enu)
            for pos, truth in zip(result.positions_ecef, true_positions_ecef)
        ],
        dtype=np.float64,
    )
    diagnostics: dict[str, object] = {
        "initial_error": float(result.initial_error),
        "final_error": float(result.final_error),
        "factor_counts": dict(result.factor_counts),
    }
    return errors, diagnostics


def replay_fgo(
    mask_csv: Path = DEFAULT_MASK_CSV,
    *,
    summary_json: Path | None = DEFAULT_SUMMARY_JSON,
    gml_path: Path | None = None,
    nlos_weight: float = DEFAULT_NLOS_WEIGHT,
    bias_scale: float = 1.0,
    pr_sigma_m: float = DEFAULT_PR_SIGMA_M,
    motion_sigma_m: float = DEFAULT_MOTION_SIGMA_M,
    robust_huber_k: float = DEFAULT_ROBUST_HUBER_K,
) -> dict[str, object]:
    spp_replay = _load_spp_replay_module()
    demo = spp_replay._load_demo_module()
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

    mask = spp_replay.load_mask_csv(mask_csv, n_epochs=n_epochs, n_satellites=len(sats_azel))
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
    pseudorange = np.zeros((n_epochs, len(sats_azel)), dtype=np.float64)
    for epoch_idx in range(n_epochs):
        obs = demo.simulate_pseudorange_epoch(
            rng,
            rx_ecef[epoch_idx],
            sat_ecef,
            elevations_deg,
            los_mask[epoch_idx],
            clock_bias_m,
        )
        pseudorange[epoch_idx] = obs["pseudorange_m"]

    initial_offset = enu_to_ecef @ np.array([12.0, -8.0, 5.0], dtype=np.float64)
    initial_wave = np.sin(np.linspace(0.0, 2.0 * np.pi, n_epochs))[:, None]
    initial_wave = initial_wave * (
        enu_to_ecef @ np.array([4.0, 3.0, 0.0], dtype=np.float64)
    ).reshape(1, 3)
    initial_positions = rx_ecef + initial_offset.reshape(1, 3) + initial_wave

    naive_weights = np.ones_like(pseudorange)
    soft_weights = np.where(los_mask, 1.0, float(nlos_weight))
    corrected_pseudorange = pseudorange - float(bias_scale) * expected_bias

    naive_errors, naive_diag = _solve_case(
        initial_positions_ecef=initial_positions,
        true_positions_ecef=rx_ecef,
        ecef_to_enu=ecef_to_enu,
        demo=demo,
        sat_ecef=sat_ecef,
        pseudorange_m=pseudorange,
        weights=naive_weights,
        clock_bias_m=clock_bias_m,
        motion_sigma_m=motion_sigma_m,
        pr_sigma_m=pr_sigma_m,
        huber_k=0.0,
    )
    robust_errors, robust_diag = _solve_case(
        initial_positions_ecef=initial_positions,
        true_positions_ecef=rx_ecef,
        ecef_to_enu=ecef_to_enu,
        demo=demo,
        sat_ecef=sat_ecef,
        pseudorange_m=pseudorange,
        weights=naive_weights,
        clock_bias_m=clock_bias_m,
        motion_sigma_m=motion_sigma_m,
        pr_sigma_m=pr_sigma_m,
        huber_k=robust_huber_k,
    )
    mask_errors, mask_diag = _solve_case(
        initial_positions_ecef=initial_positions,
        true_positions_ecef=rx_ecef,
        ecef_to_enu=ecef_to_enu,
        demo=demo,
        sat_ecef=sat_ecef,
        pseudorange_m=corrected_pseudorange,
        weights=soft_weights,
        clock_bias_m=clock_bias_m,
        motion_sigma_m=motion_sigma_m,
        pr_sigma_m=pr_sigma_m,
        huber_k=robust_huber_k,
    )

    initial_errors = np.asarray(
        [
            demo.horizontal_error_m(pos, truth, ecef_to_enu)
            for pos, truth in zip(initial_positions, rx_ecef)
        ],
        dtype=np.float64,
    )
    summary: dict[str, object] = {
        "mask_csv": str(mask_csv),
        "gml_path": str(gml_path),
        "n_epochs": int(n_epochs),
        "n_complete_mask_epochs": int(np.count_nonzero(complete_epoch)),
        "n_solved_epochs": int(n_epochs),
        "nlos_weight": float(nlos_weight),
        "bias_scale": float(bias_scale),
        "pr_sigma_m": float(pr_sigma_m),
        "motion_sigma_m": float(motion_sigma_m),
        "robust_huber_k": float(robust_huber_k),
        "nlos_frac": float(np.mean(~los_mask[complete_epoch])),
        "initial": _summarize_errors(initial_errors),
        "naive_fgo": _summarize_errors(naive_errors),
        "robust_fgo": _summarize_errors(robust_errors),
        "mask_soft_fgo": _summarize_errors(mask_errors),
        "mask_soft_wins": int(np.sum(mask_errors < naive_errors)),
        "robust_wins": int(np.sum(robust_errors < naive_errors)),
        "rms_gain_vs_naive_pct": float(100.0 * (1.0 - _rms(mask_errors) / _rms(naive_errors))),
        "diagnostics": {
            "naive_fgo": naive_diag,
            "robust_fgo": robust_diag,
            "mask_soft_fgo": mask_diag,
        },
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
    parser.add_argument("--pr-sigma-m", type=float, default=DEFAULT_PR_SIGMA_M)
    parser.add_argument("--motion-sigma-m", type=float, default=DEFAULT_MOTION_SIGMA_M)
    parser.add_argument("--robust-huber-k", type=float, default=DEFAULT_ROBUST_HUBER_K)
    args = parser.parse_args()
    if not (0.0 < args.nlos_weight <= 1.0):
        parser.error("--nlos-weight must be in (0, 1]")
    if args.bias_scale < 0.0:
        parser.error("--bias-scale must be non-negative")
    if args.pr_sigma_m <= 0.0:
        parser.error("--pr-sigma-m must be positive")
    if args.motion_sigma_m <= 0.0:
        parser.error("--motion-sigma-m must be positive")
    if args.robust_huber_k < 0.0:
        parser.error("--robust-huber-k must be non-negative")

    summary = replay_fgo(
        args.mask_csv,
        summary_json=args.summary_json,
        gml_path=args.gml,
        nlos_weight=args.nlos_weight,
        bias_scale=args.bias_scale,
        pr_sigma_m=args.pr_sigma_m,
        motion_sigma_m=args.motion_sigma_m,
        robust_huber_k=args.robust_huber_k,
    )
    print("PLATEAU NLOS FGO replay")
    print("=" * 70)
    print(
        f"mask={summary['mask_csv']} solved={summary['n_solved_epochs']}/"
        f"{summary['n_complete_mask_epochs']} nlos_frac={summary['nlos_frac']:.4f}"
    )
    print(
        f"pr_sigma={summary['pr_sigma_m']:.1f} m motion_sigma="
        f"{summary['motion_sigma_m']:.1f} m nlos_weight={summary['nlos_weight']:.2f}"
    )
    print(f"{'method':<24}{'P50 err':>12}{'RMS err':>12}")
    print("-" * 48)
    for name, label in [
        ("initial", "initial trajectory"),
        ("naive_fgo", "naive FGO"),
        ("robust_fgo", "robust FGO"),
        ("mask_soft_fgo", "mask-soft FGO"),
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
