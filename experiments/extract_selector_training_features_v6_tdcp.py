#!/usr/bin/env python3
"""V6 feature extractor: v3 + TDCP-style multi-step / vertical / acceleration
features derived from per-candidate position history.

New columns (5):
- delta_pos_2step_m: |xyz[t] - xyz[t-2]| per label, NaN→-1
- delta_pos_3step_m: |xyz[t] - xyz[t-3]| per label, NaN→-1
- delta_pos_vertical_m: |z[t] - z[t-1]| per label (vertical-only, NLOS glitch indicator)
- delta_pos_accel_m: |v[t] - v[t-1]| per label (= 2nd derivative, smoothness)
- delta_pos_horizontal_m: |xy[t] - xy[t-1]| per label (horizontal-only, often less affected by NLOS)

Output: experiments/results/selector_training_features_v6_tdcp.csv
(later merged with NLOS via augment_selector_training_features_with_nlos.py)
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

    sorted_tows = sorted(ref.keys())
    # Track position history per label (lookup last 3 epochs)
    history_per_label: dict[str, list[tuple[float, np.ndarray]]] = {}

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

        cluster_ids_50 = assign_clusters(xyz_arr, 0.5)
        n_clusters_50 = len(set(cluster_ids_50.tolist()))
        cluster_ge3_count = 0
        for cid in set(cluster_ids_50.tolist()):
            if (cluster_ids_50 == cid).sum() >= 3:
                cluster_ge3_count += 1

        max_cluster_size = int(cluster_size_50cm.max())
        cid_counts = {cid: int((cluster_ids_50 == cid).sum()) for cid in set(cluster_ids_50.tolist())}
        largest_cid = max(cid_counts, key=cid_counts.get)
        largest_cid_count = cid_counts[largest_cid]
        largest_centroid = xyz_arr[cluster_ids_50 == largest_cid].mean(axis=0)

        cluster_min_rms = {}
        cluster_min_abs_max = {}
        for cid in set(cluster_ids_50.tolist()):
            mask = cluster_ids_50 == cid
            cluster_min_rms[cid] = float(rms_arr[mask].min())
            cluster_min_abs_max[cid] = float(np.array([c[2]["abs_max"] for c in epoch_cands])[mask].min())

        rms_rank = rms_arr.argsort().argsort() + 1
        median_xyz = np.median(xyz_arr, axis=0)
        w = path_w.get(tow, 0.0)

        # Compute v3 1-step deltas + v6 multi-step / vertical / accel deltas
        deltas_1step = []
        deltas_2step = []
        deltas_3step = []
        deltas_vertical = []
        deltas_horizontal = []
        deltas_accel = []
        for lbl, xyz, _, _ in epoch_cands:
            hist = history_per_label.get(lbl, [])
            # 1-step
            d1 = float(np.linalg.norm(xyz - hist[-1][1])) if hist else np.nan
            deltas_1step.append(d1)
            # 2-step
            d2 = float(np.linalg.norm(xyz - hist[-2][1])) if len(hist) >= 2 else np.nan
            deltas_2step.append(d2)
            # 3-step
            d3 = float(np.linalg.norm(xyz - hist[-3][1])) if len(hist) >= 3 else np.nan
            deltas_3step.append(d3)
            # Vertical-only (z change in ECEF — approximate, will be normalized later)
            dz = float(abs(xyz[2] - hist[-1][1][2])) if hist else np.nan
            deltas_vertical.append(dz)
            # Horizontal-only (xy change)
            dh = float(np.linalg.norm(xyz[:2] - hist[-1][1][:2])) if hist else np.nan
            deltas_horizontal.append(dh)
            # Acceleration = |v[t] - v[t-1]|
            if len(hist) >= 2:
                v_now = xyz - hist[-1][1]
                v_prev = hist[-1][1] - hist[-2][1]
                deltas_accel.append(float(np.linalg.norm(v_now - v_prev)))
            else:
                deltas_accel.append(np.nan)

        deltas_arr = np.array(deltas_1step)
        median_delta = np.nanmedian(deltas_arr) if not np.all(np.isnan(deltas_arr)) else np.nan

        for i, (lbl, xyz, diag, err) in enumerate(epoch_cands):
            cid = int(cluster_ids_50[i])
            dist_to_largest_centroid = float(np.linalg.norm(xyz - largest_centroid))

            d1 = deltas_1step[i] if not np.isnan(deltas_1step[i]) else -1.0
            d2 = deltas_2step[i] if not np.isnan(deltas_2step[i]) else -1.0
            d3 = deltas_3step[i] if not np.isnan(deltas_3step[i]) else -1.0
            dv = deltas_vertical[i] if not np.isnan(deltas_vertical[i]) else -1.0
            dh = deltas_horizontal[i] if not np.isnan(deltas_horizontal[i]) else -1.0
            da = deltas_accel[i] if not np.isnan(deltas_accel[i]) else -1.0
            d_vs_med = (abs(deltas_1step[i] - median_delta)
                        if not np.isnan(deltas_1step[i]) and not np.isnan(median_delta)
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
                "cluster_size_25cm": int(cluster_size_25cm[i]),
                "cluster_size_10cm": int(cluster_size_10cm[i]),
                "max_cluster_size_50cm": max_cluster_size,
                "is_in_max_cluster_50cm": int(cid_counts[cid] == largest_cid_count),
                "n_clusters_50cm": n_clusters_50,
                "n_clusters_50cm_ge3": cluster_ge3_count,
                "cluster_min_rms_50cm": cluster_min_rms[cid],
                "cluster_min_abs_max_50cm": cluster_min_abs_max[cid],
                "dist_to_max_cluster_centroid_m": dist_to_largest_centroid,
                "delta_pos_norm_m": d1,
                "delta_pos_vs_median_m": d_vs_med,
                # v6 NEW TDCP-style features
                "delta_pos_2step_m": d2,
                "delta_pos_3step_m": d3,
                "delta_pos_vertical_m": dv,
                "delta_pos_horizontal_m": dh,
                "delta_pos_accel_m": da,
                "rank_by_rms": int(rms_rank[i]),
                "n_candidates_in_epoch": n_cand,
                "dist_to_median_m": float(np.linalg.norm(xyz - median_xyz)),
                "err_3d_m": err,
                "is_pass_50cm": int(err < 0.5),
                "path_weight": w,
            })

        # Update history (keep last 3 epochs per label)
        for lbl, xyz, _, _ in epoch_cands:
            hist = history_per_label.setdefault(lbl, [])
            hist.append((tow, xyz))
            if len(hist) > 3:
                hist.pop(0)

    df = pd.DataFrame(rows)
    print(f"  emitted {len(df)} rows ({df['is_pass_50cm'].mean() * 100:.1f}% pass rate)", flush=True)
    return df


def main():
    dfs = [extract_one_run(c, r) for c, r in RUNS]
    df = pd.concat(dfs, ignore_index=True)
    out_path = REPO / "experiments/results/selector_training_features_v6_tdcp.csv"
    df.to_csv(out_path, index=False)
    print(f"\n=== SUMMARY ===")
    print(f"  rows: {len(df)}")
    print(f"  unique (run, tow): {df.groupby(['run_id', 'tow']).ngroups}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
