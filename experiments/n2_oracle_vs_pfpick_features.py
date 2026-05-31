#!/usr/bin/env python3
"""For n/r2 K=3 mistake epochs, compare oracle vs PF-picked candidate features.

Goal: find a discriminator that lets the selector pick oracle when rms alone says
PF-pick (which is wrong cluster).
"""
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
REF_BASE = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
CITY, RUN = "nagoya", "run2"
PF_TAG = "phase19aw_rmsfilter3"


def load_ref():
    out = {}
    with open(REF_BASE / CITY / RUN / "reference.csv") as f:
        for r in csv.DictReader(f):
            t = round(float(r["GPS TOW (s)"]), 1)
            out[t] = np.array([float(r["ECEF X (m)"]), float(r["ECEF Y (m)"]), float(r["ECEF Z (m)"])])
    return out


def load_pf():
    p = Path(f"/tmp/ppc_{PF_TAG}_{CITY}_{RUN}/{CITY}_{RUN}_RBPF-velKF+DD+gate+hybrid+rtkdiag_pf.pos")
    out = {}
    with open(p) as f:
        for ln in f:
            if ln.startswith("%") or not ln.strip(): continue
            pp = ln.split()
            try:
                t = round(float(pp[1]), 1)
                out[t] = np.array([float(pp[2]), float(pp[3]), float(pp[4])])
            except: pass
    return out


def get_dirs():
    base_file = f"/tmp/{CITY}_{RUN}_phase11fa_dirs.txt"
    with open(base_file) as f:
        base = f.read().strip().split(",")
    extras = []
    extras.extend([
        "experiments/results/libgnss_diag_phase10/fgo_v2_gap",
        "experiments/results/libgnss_diag_phase10/fgo_v14_snr38",
        "experiments/results/libgnss_diag_phase10/fgo_v17_el25",
    ])
    for nm in ["gici_tc_esdfix", "gici_full_zeroarm", "gici_full_ratio25", "gici_full_loosepr",
               "gici_full_loosephase", "gici_full_ratio40", "gici_full_combo", "gici_full_combo4",
               "gici_full_lprlph", "gici_full_zr", "gici_full_onarm", "gici_full_lowacc",
               "gici_full_hisnr", "gici_full_hisnr45", "gici_full_hisnr30", "gici_full_hielev",
               "gici_full_imurot", "gici_full_himuba", "gici_full_window5"]:
        extras.append(f"experiments/results/libgnss_diag_phase19/{nm}")
    return [REPO / d for d in base + extras]


def load_cand_with_features(d, city, run):
    pos_file = d / f"{city}_{run}_full.pos"
    csv_file = d / f"{city}_{run}_full.csv"
    if not pos_file.exists():
        return {}
    out = {}
    with open(pos_file) as f:
        for ln in f:
            if ln.startswith("%") or not ln.strip(): continue
            pp = ln.split()
            try:
                t = round(float(pp[1]), 1)
                out[t] = {
                    "xyz": np.array([float(pp[2]), float(pp[3]), float(pp[4])]),
                }
            except: pass
    if csv_file.exists():
        with open(csv_file) as f:
            for r in csv.DictReader(f):
                try:
                    t = round(float(r["tow"]), 1)
                    if t not in out: continue
                    out[t]["rms"] = float(r.get("final_residual_rms", 0) or 0)
                    out[t]["ratio"] = float(r.get("final_ratio", 0) or 0)
                    out[t]["abs_max"] = float(r.get("final_residual_abs_max", 0) or 0)
                    out[t]["update_rows"] = int(r.get("final_update_rows", 0) or 0)
                    out[t]["sats"] = int(r.get("final_sats", 0) or 0)
                    out[t]["status"] = int(r.get("final_status", 0) or 0)
                    out[t]["output_added"] = int(r.get("output_added", 1) or 1)
                except: pass
    return out


def main():
    ref = load_ref()
    pf = load_pf()
    cands = {}
    for d in get_dirs():
        c = load_cand_with_features(d, CITY, RUN)
        if c: cands[d.name] = c

    # For each mistake epoch, gather oracle and PF-pick features
    mistakes = []
    for tow, ref_pos in ref.items():
        if tow not in pf: continue
        pf_err = float(np.linalg.norm(pf[tow] - ref_pos))
        if pf_err <= 0.5: continue
        # Find oracle
        best_err = float("inf"); best_v = None; best_feat = None
        for v, data in cands.items():
            if tow not in data: continue
            c = data[tow]
            if c.get("output_added", 1) == 0: continue
            if "rms" not in c: continue
            err = float(np.linalg.norm(c["xyz"] - ref_pos))
            if err < best_err:
                best_err = err; best_v = v; best_feat = c
        if best_feat is None or best_err > 0.5: continue  # unreach or no candidate
        # Find PF-nearest (proxy for PF-picked)
        pf_pos = pf[tow]
        nearest_err = float("inf"); nearest_v = None; nearest_feat = None
        for v, data in cands.items():
            if tow not in data: continue
            c = data[tow]
            if c.get("output_added", 1) == 0: continue
            if "rms" not in c: continue
            err = float(np.linalg.norm(c["xyz"] - pf_pos))
            if err < nearest_err:
                nearest_err = err; nearest_v = v; nearest_feat = c
        if nearest_feat is None: continue
        mistakes.append({
            "tow": tow,
            "oracle_v": best_v, "oracle_feat": best_feat, "oracle_err": best_err,
            "pf_v": nearest_v, "pf_feat": nearest_feat, "pf_err": pf_err,
        })

    print(f"\nn/r2 mistake epochs collected: {len(mistakes)}")
    # Per-feature: compare oracle vs PF-pick
    print(f"\n{'feature':>15s} {'oracle_p50':>12s} {'pfpick_p50':>12s} {'oracle_p10':>12s} {'pfpick_p10':>12s} {'oracle_p90':>12s} {'pfpick_p90':>12s}")
    for feat in ["rms", "ratio", "abs_max", "update_rows", "sats"]:
        o_vals = np.array([m["oracle_feat"].get(feat, 0) for m in mistakes if feat in m["oracle_feat"]])
        p_vals = np.array([m["pf_feat"].get(feat, 0) for m in mistakes if feat in m["pf_feat"]])
        if len(o_vals) == 0 or len(p_vals) == 0: continue
        print(f"{feat:>15s} {np.percentile(o_vals,50):>12.4f} {np.percentile(p_vals,50):>12.4f} "
              f"{np.percentile(o_vals,10):>12.4f} {np.percentile(p_vals,10):>12.4f} "
              f"{np.percentile(o_vals,90):>12.4f} {np.percentile(p_vals,90):>12.4f}")

    # Status histograms
    print(f"\nOracle status hist:  {Counter(m['oracle_feat'].get('status', 0) for m in mistakes)}")
    print(f"PF-pick status hist: {Counter(m['pf_feat'].get('status', 0) for m in mistakes)}")

    # How often does oracle have HIGHER rms than PF-pick? (rms-trap)
    rms_trap = sum(1 for m in mistakes if m['oracle_feat'].get('rms', 0) > m['pf_feat'].get('rms', 0))
    print(f"\nrms-trap (oracle rms > pf-pick rms): {rms_trap}/{len(mistakes)} = {100*rms_trap/len(mistakes):.1f}%")

    # Discriminative features: oracle higher than pf in...
    higher = {"rms": 0, "ratio": 0, "abs_max": 0, "update_rows": 0, "sats": 0}
    for m in mistakes:
        for feat in higher:
            ov = m["oracle_feat"].get(feat, 0)
            pv = m["pf_feat"].get(feat, 0)
            if ov > pv: higher[feat] += 1
    print(f"\nOracle > PF-pick by feature:")
    for feat, cnt in higher.items():
        print(f"  {feat}: {cnt}/{len(mistakes)} = {100*cnt/len(mistakes):.1f}%")

    # Variant of oracle and PF-pick
    print(f"\nOracle variant frequency (top 10):")
    o_var = Counter(m['oracle_v'] for m in mistakes)
    for v, c in o_var.most_common(10):
        print(f"  {v}: {c}")
    print(f"\nPF-pick variant frequency (top 10):")
    p_var = Counter(m['pf_v'] for m in mistakes)
    for v, c in p_var.most_common(10):
        print(f"  {v}: {c}")

    # Save full data
    out_path = Path("/tmp/n2_oracle_vs_pfpick_features.json")
    with open(out_path, "w") as f:
        json.dump([{
            "tow": m["tow"], "oracle_err": m["oracle_err"], "pf_err": m["pf_err"],
            "oracle_v": m["oracle_v"], "pf_v": m["pf_v"],
            "oracle_feat": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                          for k, v in m["oracle_feat"].items() if k != "xyz"},
            "pf_feat": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                      for k, v in m["pf_feat"].items() if k != "xyz"},
        } for m in mistakes[:500]], f, indent=2)
    print(f"\nFirst 500 saved to {out_path}")


if __name__ == "__main__":
    main()
