#!/usr/bin/env python3
"""V3 feature extractor: add cluster-topology + temporal features for wrong-fix
discrimination on n/r2 (and other runs).

Adds:
- cluster_size_25cm, cluster_size_10cm (finer cluster sizes)
- max_cluster_size_50cm (epoch-level max — duplicated)
- is_in_max_cluster_50cm (1 if this candidate's cluster_size == max)
- n_clusters_50cm_ge3 (epoch-level distinct cluster count with >=3 members)
- cluster_min_rms_50cm (per candidate, min rms among 50cm cluster members)
- dist_to_max_cluster_centroid_50cm (per candidate, distance to centroid of
  the largest 50cm cluster)
- delta_pos_norm_m (per candidate, |pos[t] - pos[t-1]| for this label)
- delta_pos_vs_median_m (per candidate, |delta - median delta over labels|)

Rebuilds the entire CSV. Output:
experiments/results/selector_training_features_v3.csv
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
    with open(REF_BASE / city / run / "reference.csv") as f:
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


def compute_path_weights(ref):
    tows = sorted(ref.keys())
    weights = {}
    for i, t in enumerate(tows):
        w = 0.0
        if i > 0:
            w += 0.5 * float(np.linalg.norm(ref[tows[i]] - ref[tows[i - 1]]))
        if i < len(tows) - 1:
            w += 0.5 * float(np.linalg.norm(ref[tows[i + 1]] - ref[tows[i]]))
        weights[t] = w
    return weights


def load_pos_file(p):
    out = {}
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


def load_diag_csv(p):
    out = {}
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


def per_run_dirs_labels(city, run):
    key = f"{city}_{run}"
    base_dirs = (Path("/tmp") / f"{key}_phase11fa_dirs.txt").read_text().strip().split(",")
    base_labels = (Path("/tmp") / f"{key}_phase11fa_labels.txt").read_text().strip().split(",")
    dirs = [REPO / d for d in base_dirs] + [REPO / d for d in PHASE19_EXTRA_DIRS]
    labels = list(base_labels) + list(PHASE19_EXTRA_LABELS)
    return dirs, labels


def assign_clusters(xyz_arr: np.ndarray, radius: float) -> np.ndarray:
    """Greedy clustering: assign each point to a cluster (label) by union of 50cm-radius
    components. Returns cluster ids as ints starting at 0."""
    n = len(xyz_arr)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    diff = xyz_arr[:, None, :] - xyz_arr[None, :, :]
    d2d = np.linalg.norm(diff, axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            if d2d[i, j] < radius:
                union(i, j)
    roots = np.array([find(i) for i in range(n)])
    # Normalize to 0..K-1
    uniq = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
    return np.array([uniq[r] for r in roots], dtype=int)


def extract_one_run(city: str, run: str) -> pd.DataFrame:
    print(f"\n=== {city}/{run} ===", flush=True)
    ref = load_reference(city, run)
    path_w = compute_path_weights(ref)
    print(f"  reference epochs: {len(ref)}, total path = {sum(path_w.values()):.1f} m", flush=True)

    dirs, labels = per_run_dirs_labels(city, run)
    rows = []
    cand_pos = {}
    cand_diag = {}
    for d, lbl in zip(dirs, labels):
        pos = load_pos_file(d / f"{city}_{run}_full.pos")
        diag = load_diag_csv(d / f"{city}_{run}_full.csv")
        if pos:
            cand_pos[lbl] = pos
            cand_diag[lbl] = diag

    print(f"  loaded {len(cand_pos)} non-empty candidates", flush=True)

    # Walk epochs in chronological order to compute temporal features
    sorted_tows = sorted(ref.keys())
    prev_pos_per_label: dict[str, np.ndarray] = {}

    for tow in sorted_tows:
        ref_xyz = ref[tow]
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

        n_cand = len(epoch_cands)
        xyz_arr = np.array([c[1] for c in epoch_cands])
        rms_arr = np.array([c[2]["rms"] for c in epoch_cands])
        diff = xyz_arr[:, None, :] - xyz_arr[None, :, :]
        d2d = np.linalg.norm(diff, axis=-1)

        cluster_size_50cm = (d2d < 0.5).sum(axis=1)
        cluster_size_25cm = (d2d < 0.25).sum(axis=1)
        cluster_size_10cm = (d2d < 0.10).sum(axis=1)

        # Distinct 50cm-radius clusters by union-find
        cluster_ids_50 = assign_clusters(xyz_arr, 0.5)
        n_clusters_50 = len(set(cluster_ids_50.tolist()))
        cluster_ge3_count = 0
        for cid in set(cluster_ids_50.tolist()):
            if (cluster_ids_50 == cid).sum() >= 3:
                cluster_ge3_count += 1

        # Largest 50cm cluster
        # use cluster_size_50cm[i] for membership weight (chains)
        max_cluster_size = int(cluster_size_50cm.max())
        # Per cluster_id: count
        cid_counts = {cid: int((cluster_ids_50 == cid).sum()) for cid in set(cluster_ids_50.tolist())}
        largest_cid = max(cid_counts, key=cid_counts.get)
        largest_cid_count = cid_counts[largest_cid]
        # Centroid of largest cluster
        largest_centroid = xyz_arr[cluster_ids_50 == largest_cid].mean(axis=0)

        # Per cluster_id min rms (across all candidates with same cid)
        cluster_min_rms = {}
        cluster_min_abs_max = {}
        for cid in set(cluster_ids_50.tolist()):
            mask = cluster_ids_50 == cid
            cluster_min_rms[cid] = float(rms_arr[mask].min())
            cluster_min_abs_max[cid] = float(np.array([c[2]["abs_max"] for c in epoch_cands])[mask].min())

        # rms rank
        rms_rank = rms_arr.argsort().argsort() + 1

        # median position
        median_xyz = np.median(xyz_arr, axis=0)

        w = path_w.get(tow, 0.0)

        # delta_pos features: |this_pos - prev_pos_same_label|
        # Then compute median across labels for delta_pos_vs_median
        deltas = []
        for lbl, xyz, _, _ in epoch_cands:
            prev = prev_pos_per_label.get(lbl)
            if prev is not None:
                deltas.append(float(np.linalg.norm(xyz - prev)))
            else:
                deltas.append(np.nan)
        deltas_arr = np.array(deltas)
        median_delta = np.nanmedian(deltas_arr) if not np.all(np.isnan(deltas_arr)) else np.nan

        for i, (lbl, xyz, diag, err) in enumerate(epoch_cands):
            cid = int(cluster_ids_50[i])
            dist_to_largest_centroid = float(np.linalg.norm(xyz - largest_centroid))
            d_norm = deltas[i] if not np.isnan(deltas[i]) else -1.0
            d_vs_med = (abs(deltas[i] - median_delta)
                        if not np.isnan(deltas[i]) and not np.isnan(median_delta)
                        else -1.0)
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
                "cluster_size_50cm": int(cluster_size_50cm[i]),
                # NEW v3 features
                "cluster_size_25cm": int(cluster_size_25cm[i]),
                "cluster_size_10cm": int(cluster_size_10cm[i]),
                "max_cluster_size_50cm": max_cluster_size,
                "is_in_max_cluster_50cm": int(cid_counts[cid] == largest_cid_count),
                "n_clusters_50cm": n_clusters_50,
                "n_clusters_50cm_ge3": cluster_ge3_count,
                "cluster_min_rms_50cm": cluster_min_rms[cid],
                "cluster_min_abs_max_50cm": cluster_min_abs_max[cid],
                "dist_to_max_cluster_centroid_m": dist_to_largest_centroid,
                "delta_pos_norm_m": d_norm,
                "delta_pos_vs_median_m": d_vs_med,
                # legacy
                "rank_by_rms": int(rms_rank[i]),
                "n_candidates_in_epoch": n_cand,
                "dist_to_median_m": float(np.linalg.norm(xyz - median_xyz)),
                "err_3d_m": err,
                "is_pass_50cm": int(err < 0.5),
                "path_weight": w,
            })

        # Update prev_pos for next epoch
        for lbl, xyz, _, _ in epoch_cands:
            prev_pos_per_label[lbl] = xyz

    df = pd.DataFrame(rows)
    print(f"  emitted {len(df)} rows ({df['is_pass_50cm'].mean() * 100:.1f}% pass rate)", flush=True)
    return df


def main():
    dfs = [extract_one_run(c, r) for c, r in RUNS]
    df = pd.concat(dfs, ignore_index=True)
    out_path = REPO / "experiments/results/selector_training_features_v3.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== SUMMARY ===")
    print(f"  rows: {len(df)}")
    print(f"  unique (run, tow): {df.groupby(['run_id', 'tow']).ngroups}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
