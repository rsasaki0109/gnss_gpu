#!/usr/bin/env python3
"""Greedy per-label selector penalty replay for a phase CSV pool.

This is a late-stage diagnostic for Phase 11: candidate additions and hard
blocks are mostly exhausted, so test whether softly penalizing selected labels
can recover pass distance without fully blocking them.
"""

from __future__ import annotations

import argparse
import csv
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
    CTRBPFConfig,
    _apply_rtkdiag_run_index_policy,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _parse_label_list,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from sim_ppc_phase_csv_addcand import _candidate_dir_map, _discover_candidate_dir_map  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DIAG_ROOT = RESULTS_DIR / "libgnss_diag_phase10"

_BASE_MODE_FOR_TEMPORAL = {
    "temporal_n2_v1": "composite_n2_v4",
    "temporal_n2_v2": "composite_n2_v4",
    "temporal_n2_v3": "composite_n2_v4",
    "temporal_n2_v4": "composite_n2_v4",
    "temporal_n2_v5": "composite_n2_v4",
    "temporal_n2_v6": "composite_n2_v4",
    "temporal_n2_v7": "composite_n2_v4",
    "temporal_n2_v8": "composite_n2_v4",
    "temporal_n2_v9": "composite_n2_v4",
    "temporal_n2_v10": "composite_n2_v4",
    "temporal_hybdelta_t3_v1": "composite_t3_v2",
    "temporal_hybdelta_t3_v2": "composite_t3_v2",
    "temporal_hybdelta_t3_v3": "composite_t3_v2",
    "temporal_hybdelta_t3_v4": "composite_t3_v4",
    "temporal_hybdelta_t3_v5": "composite_t3_v4",
    "temporal_hybdelta_t3_v6": "composite_t3_v4",
    "temporal_hybdelta_t3_v7": "composite_t3_v4",
    "temporal_hybdelta_t3_v8": "composite_t3_v4",
    "temporal_hybdelta_n2_v1": "composite_n2_v4",
    "temporal_hybdelta_n3_v1": "composite_n3_v3",
    "temporal_hybdelta_n3_v2": "composite_n3_v4",
    "temporal_hybdelta_n3_v3": "composite_n3_v4",
    "temporal_hybdelta_n3_v4": "composite_n3_v4",
    "temporal_hybdelta_n3_v5": "composite_n3_v4",
    "temporal_hybdelta_n3_v6": "composite_n3_v4",
}
_PREVDIST_ALPHA = {
    "temporal_n2_v1": 0.001,
    "temporal_n2_v2": 0.0006,
    "temporal_n2_v3": 0.00062,
    "temporal_n2_v4": 0.00062,
    "temporal_n2_v5": 0.00062,
    "temporal_n2_v6": 0.00062,
    "temporal_n2_v7": 0.00062,
    "temporal_n2_v8": 0.00062,
    "temporal_n2_v9": 0.00062,
    "temporal_n2_v10": 0.00062,
}
_HYBDELTA_ALPHA = {
    "temporal_hybdelta_t3_v1": 0.0003,
    "temporal_hybdelta_t3_v2": 0.0002,
    "temporal_hybdelta_t3_v3": 0.00022,
    "temporal_hybdelta_t3_v4": 0.0002,
    "temporal_hybdelta_t3_v5": 0.0002,
    "temporal_hybdelta_t3_v6": 0.0002,
    "temporal_hybdelta_t3_v7": 0.0002,
    "temporal_hybdelta_t3_v8": 0.0002,
    "temporal_hybdelta_n2_v1": 0.0003,
    "temporal_hybdelta_n3_v1": 0.0003,
    "temporal_hybdelta_n3_v2": 0.0006,
    "temporal_hybdelta_n3_v3": 0.0006,
    "temporal_hybdelta_n3_v4": 0.0006,
    "temporal_hybdelta_n3_v5": 0.0006,
    "temporal_hybdelta_n3_v6": 0.0006,
}


def _builtin_label_factors(mode: str) -> dict[str, float]:
    if mode == "temporal_hybdelta_t3_v5":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
        }
    if mode == "temporal_hybdelta_t3_v6":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
            "c1p1hr": 1.10,
        }
    if mode == "temporal_hybdelta_t3_v7":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
            "c1p1hr": 1.10,
            "r20ga": 3.00,
            "psig1": 1.50,
            "r15ga": 1.20,
        }
    if mode == "temporal_hybdelta_t3_v8":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
            "c1p1hr": 1.10,
            "r20ga": 3.00,
            "psig1": 1.50,
            "r15ga": 1.20,
            "r25g10": 1.50,
            "r20g10": 1.50,
            "r15g10": 1.10,
        }
    if mode == "temporal_n2_v4":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
        }
    if mode == "temporal_n2_v5":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
        }
    if mode == "temporal_n2_v6":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
        }
    if mode == "temporal_n2_v7":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.01,
        }
    if mode == "temporal_n2_v8":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.01,
            "csig05_psig1": 1.01,
            "rtkout5oG": 1.03,
        }
    if mode == "temporal_n2_v9":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.0403,
            "csig05_psig1": 1.01,
            "rtkout5oG": 1.03,
            "csig05": 1.01,
            "r25g": 1.01,
        }
    if mode == "temporal_n2_v10":
        return {
            "mlc1oGc0001": 1.0706,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.0403,
            "csig05_psig1": 1.01,
            "rtkout5oG": 1.03,
            "csig05": 1.01,
            "r25g": 1.01,
            "n2loose3": 1.06,
            "r25": 1.01,
        }
    if mode == "temporal_hybdelta_n3_v3":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
        }
    if mode == "temporal_hybdelta_n3_v4":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
            "mlc1c005p1": 1.50,
            "n3tight": 1.10,
        }
    if mode == "temporal_hybdelta_n3_v5":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
            "mlc1c005p1": 1.50,
            "n3tight": 1.10,
            "mlc1oGc005p1": 1.03,
            "csig05psh": 1.10,
        }
    if mode == "temporal_hybdelta_n3_v6":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
            "mlc1c005p1": 1.50,
            "n3tight": 1.10,
            "mlc1oGc005p1": 1.03,
            "csig05psh": 1.10,
            "n3tight2": 1.01,
        }
    return {}


def _phase_row(path: Path, city: str, run: str) -> dict[str, str]:
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row["city"]) == city and str(row["run"]) == run:
                return row
    raise SystemExit(f"run not found in {path}: {city}/{run}")


def _load_candidates(city: str, run: str, labels: list[str], label_to_dir: dict[str, str]):
    loaded = []
    for label in labels:
        dir_name = label_to_dir.get(label)
        if dir_name is None and label.startswith("x"):
            dir_name = label_to_dir.get(label[1:])
        if dir_name is None:
            continue
        pos_path = _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        loaded.append((label, pos, diag))
    return loaded


def _collect_options(args: argparse.Namespace):
    row = _phase_row(args.phase_runs_csv, args.city, args.run)
    labels = _parse_label_list(str(row["rtkdiag_candidate_labels"]))
    label_to_dir = _discover_candidate_dir_map(_candidate_dir_map())
    loaded = _load_candidates(args.city, args.run, labels, label_to_dir)
    kept = _filter_rtkdiag_candidates_by_policy(
        loaded,
        city=args.city,
        run=args.run,
        policy=args.policy,
    )
    cfg = _apply_rtkdiag_run_index_policy(
        CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
        city=args.city,
        run=args.run,
        policy=args.policy,
    )
    mode = str(cfg.rtkdiag_candidate_select_mode)
    base_mode = _BASE_MODE_FOR_TEMPORAL.get(mode, mode)
    ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
    rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
    ref = _load_full_reference(args.data_root / args.city / args.run / "reference.csv")
    truth = np.asarray([p for _tow, p in ref], dtype=np.float64)
    hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{args.city}_{args.run}_full.pos")
    epochs = []
    for tow, _true_pos in ref:
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_arr = (
            np.asarray(hp, dtype=np.float64)
            if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0)
            else None
        )
        raw_opts = []
        for label, cand_pos, cand_diag in kept:
            diag_row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(diag_row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            raw_opts.append((label, np.asarray(cand, dtype=np.float64), _rtkdiag_candidate_sort_key(diag_row, mode=base_mode)))
        if raw_opts:
            opt_labels = tuple(label for label, _pos, _key in raw_opts)
            opt_pos = np.vstack([pos for _label, pos, _key in raw_opts]).astype(np.float64, copy=False)
            opt_key0 = np.asarray([float(key[0]) for _label, _pos, key in raw_opts], dtype=np.float64)
            opt_key1 = np.asarray([float(key[1]) for _label, _pos, key in raw_opts], dtype=np.float64)
            opts = (opt_labels, opt_pos, opt_key0, opt_key1)
        else:
            opts = None
        epochs.append((hp_arr, opts))
    return truth, mode, epochs, len(loaded), len(kept)


def _simulate(truth: np.ndarray, mode: str, epochs, factors: dict[str, float]):
    est = np.zeros_like(truth)
    selected = Counter()
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    for i, (hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if opts is None:
            if hp is not None:
                prev = hp
                prev_hybrid = hp
            continue

        labels, pos_arr, base0, base1 = opts
        if factors:
            factor_arr = np.fromiter((float(factors.get(label, 1.0)) for label in labels), dtype=np.float64, count=len(labels))
            key0 = base0 * factor_arr
        else:
            key0 = base0.copy()
        if mode in _PREVDIST_ALPHA and prev is not None:
            key0 += _PREVDIST_ALPHA[mode] * np.linalg.norm(pos_arr - prev, axis=1)
        elif mode in _HYBDELTA_ALPHA and prev is not None and prev_hybrid is not None and hp is not None:
            predicted = prev + (hp - prev_hybrid)
            key0 += _HYBDELTA_ALPHA[mode] * np.linalg.norm(pos_arr - predicted, axis=1)
        idx = int(np.lexsort((base1, key0))[0])
        label = labels[idx]
        pos = pos_arr[idx]
        est[i] = pos
        selected[label] += 1
        prev = pos
        if hp is not None:
            prev_hybrid = hp
    score = score_ppc2024(est, truth)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m), selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    p.add_argument("--phase-runs-csv", type=Path, required=True)
    p.add_argument("--policy", default="phase11eg")
    p.add_argument("--city", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--min-selected", type=int, default=20)
    p.add_argument(
        "--only-labels",
        default="",
        help="Comma-separated labels to consider for penalties after each replay.",
    )
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--factors", default="1.01,1.03,1.06,1.1,1.2,1.5,2,3,5,10")
    args = p.parse_args()

    factors_grid = [float(x) for x in args.factors.split(",") if x.strip()]
    only_labels = set(_parse_label_list(args.only_labels))
    truth, mode, epochs, loaded, kept = _collect_options(args)
    factors: dict[str, float] = _builtin_label_factors(mode)
    base_ppc, base_pass, total_m, selected = _simulate(truth, mode, epochs, factors)
    print(
        f"{args.city}/{args.run} loaded={loaded} kept={kept} mode={mode} "
        f"base={base_ppc:.9f} pass={base_pass:.6f}",
        flush=True,
    )
    cur_pass = base_pass
    cur_ppc = base_ppc
    cur_selected = selected
    for round_idx in range(1, args.rounds + 1):
        best = None
        labels = [
            label
            for label, count in cur_selected.most_common()
            if count >= args.min_selected and (not only_labels or label in only_labels)
        ]
        for label in labels:
            old_factor = factors.get(label, 1.0)
            for f in factors_grid:
                trial = dict(factors)
                trial[label] = old_factor * f
                ppc, pass_m, _total_m, sel = _simulate(truth, mode, epochs, trial)
                gain = pass_m - cur_pass
                if best is None or gain > best[0]:
                    best = (gain, label, old_factor * f, ppc, pass_m, sel)
        if best is None or best[0] <= 1.0e-6:
            print(f"round {round_idx}: no positive label penalty")
            break
        gain, label, factor, ppc, pass_m, sel = best
        factors[label] = factor
        cur_pass = pass_m
        cur_ppc = ppc
        cur_selected = sel
        print(
            f"round {round_idx}: +{label} factor={factor:g} "
            f"score={ppc:.9f} pass={pass_m:.6f} delta_m={pass_m - base_pass:+.3f}",
            flush=True,
        )
    print(f"final score={cur_ppc:.9f} pass={cur_pass:.6f} delta_m={cur_pass - base_pass:+.3f}")
    print("factors=" + ",".join(f"{k}:{v:g}" for k, v in sorted(factors.items())))


if __name__ == "__main__":
    main()
