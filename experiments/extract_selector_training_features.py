#!/usr/bin/env python3
"""Extract per-(epoch, candidate) features for path-weighted supervised selector.

Walks the Phase 19aw candidate pool (phase11fa base + 22 phase19 additions),
joins per-candidate diag CSV with .pos file, computes oracle label (err < 0.5 m)
and OFFICIAL path-weight per epoch (segment length contribution).

Output: experiments/results/selector_training_features.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
REF_BASE = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

RUNS = [
    ("tokyo", "run1"), ("tokyo", "run2"), ("tokyo", "run3"),
    ("nagoya", "run1"), ("nagoya", "run2"), ("nagoya", "run3"),
]

PHASE19_EXTRA_DIRS = [
    "experiments/results/libgnss_diag_phase10/fgo_v2_gap",
    "experiments/results/libgnss_diag_phase10/fgo_v14_snr38",
    "experiments/results/libgnss_diag_phase10/fgo_v17_el25",
    "experiments/results/libgnss_diag_phase19/gici_tc_esdfix",
    "experiments/results/libgnss_diag_phase19/gici_full_zeroarm",
    "experiments/results/libgnss_diag_phase19/gici_full_ratio25",
    "experiments/results/libgnss_diag_phase19/gici_full_loosepr",
    "experiments/results/libgnss_diag_phase19/gici_full_loosephase",
    "experiments/results/libgnss_diag_phase19/gici_full_ratio40",
    "experiments/results/libgnss_diag_phase19/gici_full_combo",
    "experiments/results/libgnss_diag_phase19/gici_full_combo4",
    "experiments/results/libgnss_diag_phase19/gici_full_lprlph",
    "experiments/results/libgnss_diag_phase19/gici_full_zr",
    "experiments/results/libgnss_diag_phase19/gici_full_onarm",
    "experiments/results/libgnss_diag_phase19/gici_full_lowacc",
    "experiments/results/libgnss_diag_phase19/gici_full_hisnr",
    "experiments/results/libgnss_diag_phase19/gici_full_hisnr45",
    "experiments/results/libgnss_diag_phase19/gici_full_hisnr30",
    "experiments/results/libgnss_diag_phase19/gici_full_hielev",
    "experiments/results/libgnss_diag_phase19/gici_full_imurot",
    "experiments/results/libgnss_diag_phase19/gici_full_himuba",
    "experiments/results/libgnss_diag_phase19/gici_full_window5",
]
PHASE19_EXTRA_LABELS = [
    "xd_fgo_v2_gap", "xd_fgo_v14_snr38", "xd_fgo_v17_el25",
    "xd_gici_def", "xd_gici_z", "xd_gici_r", "xd_gici_lp", "xd_gici_lh",
    "xd_gici_r4", "xd_gici_combo", "xd_gici_c4", "xd_gici_lprlph",
    "xd_gici_zr", "xd_gici_oa", "xd_gici_la", "xd_gici_hs", "xd_gici_hs45",
    "xd_gici_hs30", "xd_gici_he", "xd_gici_ir", "xd_gici_mb", "xd_gici_w5",
]


def load_reference(city: str, run: str) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    path = REF_BASE / city / run / "reference.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                tow = round(float(r["GPS TOW (s)"]), 1)
                xyz = np.array([
                    float(r["ECEF X (m)"]),
                    float(r["ECEF Y (m)"]),
                    float(r["ECEF Z (m)"]),
                ])
                out[tow] = xyz
            except (ValueError, KeyError):
                continue
    return out


def compute_path_weights(ref: dict[float, np.ndarray]) -> dict[float, float]:
    """OFFICIAL path-weight per epoch = half of (||ref[t+]-ref[t-]||).

    For path-weighted pass@50cm metric, each epoch contributes a segment of
    half(prev->curr) + half(curr->next). The cumulative path length is the
    sum of all per-epoch contributions.
    """
    tows = sorted(ref.keys())
    weights: dict[float, float] = {}
    for i, t in enumerate(tows):
        w = 0.0
        if i > 0:
            w += 0.5 * float(np.linalg.norm(ref[tows[i]] - ref[tows[i - 1]]))
        if i < len(tows) - 1:
            w += 0.5 * float(np.linalg.norm(ref[tows[i + 1]] - ref[tows[i]]))
        weights[t] = w
    return weights


def load_pos_file(p: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    if not p.exists():
        return out
    with open(p) as f:
        for ln in f:
            if ln.startswith("%") or not ln.strip():
                continue
            pp = ln.split()
            if len(pp) < 5:
                continue
            try:
                tow = round(float(pp[1]), 1)
                xyz = np.array([float(pp[2]), float(pp[3]), float(pp[4])])
                out[tow] = xyz
            except ValueError:
                continue
    return out


def load_diag_csv(p: Path) -> dict[float, dict]:
    out: dict[float, dict] = {}
    if not p.exists():
        return out
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                tow = round(float(r["tow"]), 1)
                out[tow] = {
                    "rms": float(r.get("final_residual_rms", 0) or 0),
                    "ratio": float(r.get("final_ratio", 0) or 0),
                    "abs_max": float(r.get("final_residual_abs_max", 0) or 0),
                    "update_rows": int(r.get("final_update_rows", 0) or 0),
                    "sats": int(r.get("final_sats", 0) or 0),
                    "status": int(r.get("final_status", 0) or 0),
                    "pdop": float(r.get("final_pdop", 0) or 0),
                    "baseline_m": float(r.get("final_baseline_m", 0) or 0),
                    "spp_valid": int(r.get("spp_valid", 0) or 0),
                    "spp_sats": int(r.get("spp_sats", 0) or 0),
                    "spp_pdop": float(r.get("spp_pdop", 0) or 0),
                    "candidate_vs_spp_m": float(r.get("candidate_vs_spp_m", 0) or 0),
                    "candidate_jump_m": float(r.get("candidate_jump_m", 0) or 0),
                    "output_added": int(r.get("output_added", 1) or 1),
                }
            except (ValueError, KeyError):
                continue
    return out


def per_run_dirs_labels(city: str, run: str) -> tuple[list[Path], list[str]]:
    key = f"{city}_{run}"
    base_dirs = (Path("/tmp") / f"{key}_phase11fa_dirs.txt").read_text().strip().split(",")
    base_labels = (Path("/tmp") / f"{key}_phase11fa_labels.txt").read_text().strip().split(",")
    dirs = [REPO / d for d in base_dirs] + [REPO / d for d in PHASE19_EXTRA_DIRS]
    labels = list(base_labels) + list(PHASE19_EXTRA_LABELS)
    return dirs, labels


def extract_one_run(city: str, run: str) -> pd.DataFrame:
    print(f"\n=== {city}/{run} ===")
    ref = load_reference(city, run)
    path_w = compute_path_weights(ref)
    print(f"  reference epochs: {len(ref)}, total path = {sum(path_w.values()):.1f} m")

    dirs, labels = per_run_dirs_labels(city, run)
    print(f"  candidates: {len(dirs)}")

    rows = []
    cand_pos: dict[str, dict[float, np.ndarray]] = {}
    cand_diag: dict[str, dict[float, dict]] = {}
    for d, lbl in zip(dirs, labels):
        pos = load_pos_file(d / f"{city}_{run}_full.pos")
        diag = load_diag_csv(d / f"{city}_{run}_full.csv")
        if pos:
            cand_pos[lbl] = pos
            cand_diag[lbl] = diag

    print(f"  loaded {len(cand_pos)} non-empty candidates")

    # Per-epoch rows
    n_no_ref = 0
    for tow, ref_xyz in ref.items():
        # Gather all candidates present at this epoch
        epoch_cands = []
        for lbl, pos in cand_pos.items():
            if tow not in pos:
                continue
            diag = cand_diag.get(lbl, {}).get(tow, None)
            if diag is None or diag.get("output_added", 1) == 0:
                continue
            if diag.get("rms", 0) <= 0:
                continue
            xyz = pos[tow]
            err = float(np.linalg.norm(xyz - ref_xyz))
            epoch_cands.append((lbl, xyz, diag, err))
        if not epoch_cands:
            continue

        # Cluster sizes (within 50cm)
        xyz_arr = np.array([c[1] for c in epoch_cands])
        # pairwise dist
        diff = xyz_arr[:, None, :] - xyz_arr[None, :, :]
        d2d = np.linalg.norm(diff, axis=-1)
        cluster_size = (d2d < 0.5).sum(axis=1)

        # rms ranking
        rms_arr = np.array([c[2]["rms"] for c in epoch_cands])
        rms_rank = rms_arr.argsort().argsort() + 1  # 1 = smallest rms

        # median position (proxy temporal anchor)
        median_xyz = np.median(xyz_arr, axis=0)

        w = path_w.get(tow, 0.0)

        for i, (lbl, xyz, diag, err) in enumerate(epoch_cands):
            dist_to_median = float(np.linalg.norm(xyz - median_xyz))
            rows.append({
                "run_id": f"{city}_{run}",
                "tow": tow,
                "label": lbl,
                "rms": diag["rms"],
                "ratio": diag["ratio"],
                "abs_max": diag["abs_max"],
                "update_rows": diag["update_rows"],
                "sats": diag["sats"],
                "status": diag["status"],
                "pdop": diag["pdop"],
                "baseline_m": diag["baseline_m"],
                "spp_valid": diag["spp_valid"],
                "spp_sats": diag["spp_sats"],
                "spp_pdop": diag["spp_pdop"],
                "candidate_vs_spp_m": diag["candidate_vs_spp_m"],
                "candidate_jump_m": diag["candidate_jump_m"],
                "cluster_size_50cm": int(cluster_size[i]),
                "rank_by_rms": int(rms_rank[i]),
                "n_candidates_in_epoch": len(epoch_cands),
                "dist_to_median_m": dist_to_median,
                "err_3d_m": err,
                "is_pass_50cm": int(err < 0.5),
                "path_weight": w,
            })

    df = pd.DataFrame(rows)
    print(f"  emitted {len(df)} rows ({df['is_pass_50cm'].mean() * 100:.1f}% pass rate)")
    return df


def main():
    dfs = [extract_one_run(c, r) for c, r in RUNS]
    df = pd.concat(dfs, ignore_index=True)
    out_path = REPO / "experiments/results/selector_training_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== SUMMARY ===")
    print(f"  rows: {len(df)}")
    print(f"  unique (run, tow) epochs: {df.groupby(['run_id', 'tow']).ngroups}")
    print(f"  pass rate overall: {df['is_pass_50cm'].mean() * 100:.2f}%")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
