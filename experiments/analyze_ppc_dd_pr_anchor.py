#!/usr/bin/env python3
# ruff: noqa: E402
"""Per-epoch PPC DD-pseudorange anchor diagnostics.

This script is intentionally diagnostic-only: it compares accepted DD-PR LS
anchors with reference.csv to test whether the anchor is a real causal absolute
position cue. Reference error is never used for runtime selection.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _build_dd_measurements,
    _filter_data_by_systems,
    _load_full_reference,
    _load_hybrid_pos_file,
    _reference_position_map,
)
from exp_urbannav_baseline import run_wls  # noqa: E402
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer  # noqa: E402
from gnss_gpu.dd_quality import dd_pseudorange_residuals_m, gate_dd_pseudorange  # noqa: E402
from gnss_gpu.gsdc_dgnss import DDWLSConfig, dd_pseudorange_position_update  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402


_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DEFAULT_RESULTS = _SCRIPT_DIR / "results"


def _parse_systems(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in str(value).split(",") if s.strip())


def _finite_pos(pos: np.ndarray | None) -> bool:
    if pos is None:
        return False
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    return arr.size >= 3 and bool(np.all(np.isfinite(arr[:3]))) and float(np.linalg.norm(arr[:3])) > 1.0


def _sat_counts(sat_ids: tuple[str, ...]) -> str:
    counts = Counter(s[:1] for s in sat_ids if s)
    return ";".join(f"{k}:{counts[k]}" for k in sorted(counts))


def _error_m(pos: np.ndarray | None, ref: np.ndarray | None) -> float:
    if not _finite_pos(pos) or ref is None or not _finite_pos(ref):
        return float("nan")
    return float(np.linalg.norm(np.asarray(pos, dtype=np.float64)[:3] - np.asarray(ref, dtype=np.float64)[:3]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PPC DD-PR LS anchors per epoch")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--run", required=True, help="Run key like nagoya/run2")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--systems", type=str, default="G,R,E,C,J")
    parser.add_argument("--pr-systems", type=str, default="G,E,J")
    parser.add_argument("--dd-systems", type=str, default="G,E,J,C")
    parser.add_argument("--hybrid-pos-dir", type=Path, default=_DEFAULT_RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument("--hybrid-pos-suffix", type=str, default="_full.pos")
    parser.add_argument("--dd-base-interp", action="store_true")
    parser.add_argument("--dd-min-elevation-deg", type=float, default=-90.0)
    parser.add_argument("--dd-min-snr", type=float, default=0.0)
    parser.add_argument("--dd-keep-best", type=int, default=0)
    parser.add_argument("--dd-pr-pair-residual-max-m", type=float, default=5.0)
    parser.add_argument("--dd-pr-epoch-median-residual-max-m", type=float, default=5.0)
    parser.add_argument("--dd-pr-gate-min-pairs", type=int, default=3)
    parser.add_argument("--ls-min-pairs", type=int, default=3)
    parser.add_argument("--ls-dd-sigma-m", type=float, default=2.0)
    parser.add_argument("--ls-prior-sigma-m", type=float, default=100.0)
    parser.add_argument("--ls-max-shift-m", type=float, default=100.0)
    parser.add_argument("--ls-max-postfit-rms-m", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    city, run = args.run.split("/", 1)
    run_dir = args.data_root / city / run
    if not run_dir.is_dir():
        raise SystemExit(f"missing run dir: {run_dir}")

    loader = PPCDatasetLoader(run_dir)
    data = loader.load_experiment_data(
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=_parse_systems(args.systems),
        include_sat_velocity=False,
    )
    ref_map = _reference_position_map(_load_full_reference(run_dir / "reference.csv"))

    hybrid_path = args.hybrid_pos_dir / f"{city}_{run}{args.hybrid_pos_suffix}"
    hybrid_pos: dict[float, np.ndarray] = {}
    if hybrid_path.is_file():
        hybrid_pos, _hybrid_status = _load_hybrid_pos_file(hybrid_path)

    wls_data = _filter_data_by_systems(data, _parse_systems(args.pr_systems))
    wls_positions, _wls_ms = run_wls(wls_data)

    dd_pr = DDPseudorangeComputer(
        run_dir / "base.obs",
        rover_obs_path=run_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=_parse_systems(args.dd_systems),
        interpolate_base_epochs=bool(args.dd_base_interp),
    )

    out_path = args.out
    if out_path is None:
        label = f"{city}_{run}_{args.start_epoch}_{args.max_epochs or 'full'}_ddpr_anchor_diag.csv"
        out_path = _DEFAULT_RESULTS / label
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    times = np.asarray(data["times"], dtype=np.float64)
    for i, tow in enumerate(times):
        t_key = round(float(tow), 1)
        ref = ref_map.get(t_key)
        seed_source = "wls"
        seed = np.asarray(wls_positions[i], dtype=np.float64)[:3]
        hp = hybrid_pos.get(t_key)
        if _finite_pos(hp):
            seed_source = "hybrid"
            seed = np.asarray(hp, dtype=np.float64)[:3]

        row: dict[str, object] = {
            "epoch_index": i,
            "tow": f"{float(tow):.1f}",
            "seed_source": seed_source,
            "seed_error_m": _error_m(seed, ref),
            "status": "init",
            "raw_pairs": 0,
            "kept_pairs": 0,
            "rejected_pairs": 0,
            "gate_median_abs_residual_m": float("nan"),
            "gate_max_abs_residual_m": float("nan"),
            "initial_rms_m": float("nan"),
            "final_rms_m": float("nan"),
            "shift_m": float("nan"),
            "anchor_error_m": float("nan"),
            "improved": 0,
            "pass_05m": 0,
            "pass_5m": 0,
            "sat_counts": "",
            "sat_ids": "",
            "ref_sat_ids": "",
        }
        if not _finite_pos(seed):
            row["status"] = "bad_seed"
            rows.append(row)
            continue

        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data.get("used_prns", [[]])[i]),
            np.asarray(data["weights"][i], dtype=np.float64),
            seed,
            _parse_systems(args.dd_systems),
            min_elevation_deg=float(args.dd_min_elevation_deg),
            min_snr=float(args.dd_min_snr),
            keep_best=int(args.dd_keep_best),
        )
        try:
            dd_raw = dd_pr.compute_dd(
                float(tow),
                measurements,
                rover_position_approx=seed,
                min_common_sats=int(args.ls_min_pairs),
                rover_weights=[float(m.snr) for m in measurements],
            )
        except Exception as exc:
            row["status"] = f"compute_error:{type(exc).__name__}"
            rows.append(row)
            continue

        if dd_raw is None or int(getattr(dd_raw, "n_dd", 0)) <= 0:
            row["status"] = "no_dd"
            rows.append(row)
            continue
        row["raw_pairs"] = int(dd_raw.n_dd)

        raw_abs = np.abs(dd_pseudorange_residuals_m(dd_raw, seed))
        row["raw_median_abs_residual_m"] = float(np.median(raw_abs)) if raw_abs.size else float("nan")
        row["raw_max_abs_residual_m"] = float(np.max(raw_abs)) if raw_abs.size else float("nan")

        dd_gated, gate_stats = gate_dd_pseudorange(
            dd_raw,
            seed,
            pair_residual_max_m=(
                float(args.dd_pr_pair_residual_max_m)
                if float(args.dd_pr_pair_residual_max_m) > 0.0
                else None
            ),
            epoch_median_residual_max_m=(
                float(args.dd_pr_epoch_median_residual_max_m)
                if float(args.dd_pr_epoch_median_residual_max_m) > 0.0
                else None
            ),
            min_pairs=int(args.dd_pr_gate_min_pairs),
        )
        row["kept_pairs"] = int(gate_stats.n_kept_pairs)
        row["rejected_pairs"] = int(gate_stats.n_pair_rejected)
        row["gate_median_abs_residual_m"] = float(gate_stats.metric_median)
        row["gate_max_abs_residual_m"] = float(gate_stats.metric_max)
        if dd_gated is None:
            row["status"] = "gate_reject"
            rows.append(row)
            continue

        anchor, diag = dd_pseudorange_position_update(
            seed,
            dd_gated,
            DDWLSConfig(
                min_dd_pairs=int(args.ls_min_pairs),
                dd_sigma_m=float(args.ls_dd_sigma_m),
                prior_sigma_m=float(args.ls_prior_sigma_m),
                max_shift_m=float(args.ls_max_shift_m),
                max_iter=8,
            ),
        )
        final_rms = float(diag.get("final_rms_m", float("inf")))
        accepted = bool(diag.get("accepted", False)) and final_rms <= float(args.ls_max_postfit_rms_m)
        row["status"] = "accepted" if accepted else "solve_reject"
        row["initial_rms_m"] = float(diag.get("initial_rms_m", float("nan")))
        row["final_rms_m"] = final_rms
        row["shift_m"] = float(diag.get("shift_m", float("nan")))
        row["anchor_error_m"] = _error_m(anchor, ref)
        seed_err = float(row["seed_error_m"])
        anchor_err = float(row["anchor_error_m"])
        if math.isfinite(seed_err) and math.isfinite(anchor_err):
            row["improved"] = int(anchor_err < seed_err)
            row["pass_05m"] = int(anchor_err <= 0.5)
            row["pass_5m"] = int(anchor_err <= 5.0)
        sat_ids = tuple(getattr(dd_gated, "sat_ids", ()) or ())
        ref_sat_ids = tuple(getattr(dd_gated, "ref_sat_ids", ()) or ())
        row["sat_counts"] = _sat_counts(sat_ids)
        row["sat_ids"] = ";".join(sat_ids)
        row["ref_sat_ids"] = ";".join(ref_sat_ids)
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    accepted_rows = [r for r in rows if r["status"] == "accepted"]
    if accepted_rows:
        gt = np.array([float(r["anchor_error_m"]) for r in accepted_rows], dtype=np.float64)
        seed_gt = np.array([float(r["seed_error_m"]) for r in accepted_rows], dtype=np.float64)
        print(
            f"accepted={len(accepted_rows)}/{len(rows)} "
            f"seed_avg={float(np.nanmean(seed_gt)):.2f}m "
            f"anchor_avg={float(np.nanmean(gt)):.2f}m "
            f"anchor_p50={float(np.nanmedian(gt)):.2f}m "
            f"anchor_p90={float(np.nanpercentile(gt, 90)):.2f}m "
            f"improved={100.0 * sum(int(r['improved']) for r in accepted_rows) / len(accepted_rows):.1f}% "
            f"pass5={100.0 * sum(int(r['pass_5m']) for r in accepted_rows) / len(accepted_rows):.1f}%"
        )
    else:
        print(f"accepted=0/{len(rows)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
