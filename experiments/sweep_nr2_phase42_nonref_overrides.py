#!/usr/bin/env python3
"""Sweep deployable n/r2 overrides inspired by Phase 42 diagnostics.

The Phase 42 oracle showed that many remaining n/r2 misses are solved by
switching from the ranker pick to another candidate in the same GICI family.
This script tests truth-free override rules:

- start from a baseline ranker prediction CSV
- reconstruct the gated candidate options for every epoch
- optionally override the ranker pick using only diagnostics available at
  runtime: residual-RMS rank, p_pass delta, family, cluster size, and distance
  from the original pick
- score the resulting path against reference for offline validation

No TOW span or truth-derived label is used by the rule itself.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnose_nr2_ranker_with_extra_candidate import (  # noqa: E402
    DATA_ROOT,
    RESULTS,
    _candidate_options,
    _default_candidates,
    _effective_config,
    _load_candidates,
)
from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_ranker_predictions,
)
from gnss_gpu.ppc_score import ppc_segment_distances  # noqa: E402

PICK_SETS: dict[str, set[str] | None] = {
    "any_gici": None,
    "c4": {"xd_gici_c4"},
    "c4_oa_combo": {"xd_gici_c4", "xd_gici_oa", "xd_gici_combo"},
    "c4_oa_combo_z_hs": {
        "xd_gici_c4",
        "xd_gici_oa",
        "xd_gici_combo",
        "xd_gici_z",
        "xd_gici_hs",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v5_nlos.csv",
    )
    parser.add_argument("--policy", default="phase11ep")
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--rms-prefilter-k", type=int, default=99)
    parser.add_argument(
        "--base-labels-file",
        type=Path,
        default=Path("/tmp/nagoya_run2_phase11fa_labels.txt"),
    )
    parser.add_argument(
        "--base-dirs-file",
        type=Path,
        default=Path("/tmp/nagoya_run2_phase11fa_dirs.txt"),
    )
    parser.add_argument("--extra-label", action="append", default=["xd_nr2_hs_pb40"])
    parser.add_argument(
        "--extra-dir",
        action="append",
        type=Path,
        default=[
            RESULTS / "libgnss_diag_phase34/nr2_hs_piecebias40_oracle_556184_556337"
        ],
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS / "nr2_phase43_nonref_override_sweep.csv",
    )
    parser.add_argument(
        "--out-picks",
        type=Path,
        default=RESULTS / "nr2_phase43_nonref_override_best_picks.csv",
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def _family(label: str) -> str:
    if label.startswith("xd_gici_"):
        return "xd_gici"
    if label.startswith("xd_fgo_"):
        return "xd_fgo"
    if label.startswith("xd_nr2_"):
        return "nr2_oracle"
    if label.startswith("rtkout"):
        return "rtkout"
    if label.startswith("mlc"):
        return "mlc"
    return label.split("_", 1)[0]


def _cluster_counts(positions: np.ndarray, radius_m: float) -> np.ndarray:
    if positions.size == 0:
        return np.zeros(0, dtype=np.int32)
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return (dist <= float(radius_m)).sum(axis=1).astype(np.int32)


def _score_positions(
    selected: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, int]:
    err = np.linalg.norm(selected - truth, axis=1)
    passed = np.isfinite(err) & (err < 0.5)
    pass_w = float(weights[passed].sum())
    total_w = float(weights.sum())
    return (100.0 * pass_w / total_w if total_w else 0.0, pass_w, int(passed.sum()))


def _build_epochs(args: argparse.Namespace) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    city = str(args.city)
    run = str(args.run)
    run_id = f"{city}_{run}"
    labels, dirs = _default_candidates(args)
    candidates_all = _load_candidates(labels, dirs, city=city, run=run)
    candidates = _filter_rtkdiag_candidates_by_policy(
        candidates_all,
        city=city,
        run=run,
        policy=str(args.policy),
    )
    cfg = _effective_config(args)
    pred = _load_ranker_predictions(str(args.predictions))
    pred_run = {(tow, label): p for (rid, tow, label), p in pred.items() if rid == run_id}

    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([ecef for _tow, ecef in ref], dtype=np.float64)
    weights = ppc_segment_distances(truth)
    hybrid_pos, _status = _load_hybrid_pos_file(
        args.hybrid_pos_dir / f"{city}_{run}_full.pos"
    )

    epochs: list[dict[str, object]] = []
    baseline = np.zeros_like(truth)
    fallback = np.zeros_like(truth)
    for i, (tow_raw, true_pos) in enumerate(ref):
        tow = round(float(tow_raw), 1)
        hp = hybrid_pos.get(tow)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp) == 0.0):
            fallback[i] = np.asarray(hp, dtype=np.float64)
        options_raw = _candidate_options(candidates, tow=tow, cfg=cfg)
        if not options_raw:
            baseline[i] = fallback[i]
            epochs.append({"tow": tow, "options": [], "pick_idx": -1, "fallback_pos": fallback[i]})
            continue

        labels_i = [label for label, _pos, _diag in options_raw]
        pos_i = np.stack([pos for _label, pos, _diag in options_raw], axis=0).astype(np.float64)
        rms_i = np.asarray(
            [_diag_float(diag, "final_residual_rms") for _label, _pos, diag in options_raw],
            dtype=np.float64,
        )
        p_i = np.asarray(
            [float(pred_run.get((tow, label), -np.inf)) for label in labels_i],
            dtype=np.float64,
        )
        finite_p = np.isfinite(p_i)
        if np.any(finite_p):
            pick_idx = int(np.nanargmax(p_i))
        else:
            pick_idx = int(np.nanargmin(rms_i))
        baseline[i] = pos_i[pick_idx]
        rms_order = np.argsort(rms_i, kind="stable")
        rms_rank = np.empty_like(rms_order)
        rms_rank[rms_order] = np.arange(1, len(rms_i) + 1)
        p_order = np.argsort(-p_i, kind="stable")
        p_rank = np.empty_like(p_order)
        p_rank[p_order] = np.arange(1, len(p_i) + 1)
        clusters50 = _cluster_counts(pos_i, 0.50)
        clusters25 = _cluster_counts(pos_i, 0.25)
        epochs.append(
            {
                "tow": tow,
                "options": labels_i,
                "pos": pos_i,
                "rms": rms_i,
                "p": p_i,
                "rms_rank": rms_rank.astype(np.int32),
                "p_rank": p_rank.astype(np.int32),
                "cluster50": clusters50,
                "cluster25": clusters25,
                "pick_idx": pick_idx,
                "fallback_pos": fallback[i],
                "truth": np.asarray(true_pos, dtype=np.float64),
            }
        )
    print(f"loaded_candidates={len(candidates_all)} after_policy={len(candidates)}")
    return epochs, baseline, truth, weights


def _apply_rule(
    epochs: list[dict[str, object]],
    baseline: np.ndarray,
    *,
    pick_set_name: str,
    pool: str,
    selector: str,
    max_rms_rank: int,
    min_cluster50: int,
    max_dist_to_pick_m: float,
    min_dist_to_pick_m: float,
    min_p_delta: float,
    max_p_rank: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    selected = baseline.copy()
    overrides: list[dict[str, object]] = []
    pick_allow = PICK_SETS[pick_set_name]
    for i, ep in enumerate(epochs):
        labels = ep.get("options")
        if not labels:
            continue
        labels = list(labels)
        pick_idx = int(ep["pick_idx"])
        if pick_idx < 0:
            continue
        pick_label = labels[pick_idx]
        if not pick_label.startswith("xd_gici_"):
            continue
        if pick_allow is not None and pick_label not in pick_allow:
            continue
        pos = np.asarray(ep["pos"], dtype=np.float64)
        rms = np.asarray(ep["rms"], dtype=np.float64)
        p = np.asarray(ep["p"], dtype=np.float64)
        rms_rank = np.asarray(ep["rms_rank"], dtype=np.int32)
        p_rank = np.asarray(ep["p_rank"], dtype=np.int32)
        cluster50 = np.asarray(ep["cluster50"], dtype=np.int32)
        pick_pos = pos[pick_idx]
        dist_to_pick = np.linalg.norm(pos - pick_pos[None, :], axis=1)
        pick_family = _family(pick_label)

        mask = np.ones(len(labels), dtype=bool)
        mask[pick_idx] = False
        if pool == "same_family":
            mask &= np.asarray([_family(label) == pick_family for label in labels], dtype=bool)
        elif pool == "all_gici":
            mask &= np.asarray([label.startswith("xd_gici_") for label in labels], dtype=bool)
        else:
            raise ValueError(pool)
        mask &= rms_rank <= int(max_rms_rank)
        mask &= cluster50 >= int(min_cluster50)
        mask &= dist_to_pick <= float(max_dist_to_pick_m)
        mask &= dist_to_pick >= float(min_dist_to_pick_m)
        if np.isfinite(min_p_delta):
            mask &= (p - p[pick_idx]) >= float(min_p_delta)
        if int(max_p_rank) > 0:
            mask &= p_rank <= int(max_p_rank)
        cand_idx = np.flatnonzero(mask)
        if cand_idx.size == 0:
            continue
        if selector == "best_rms":
            chosen = int(cand_idx[np.argmin(rms[cand_idx])])
        elif selector == "best_cluster_rms":
            best_cluster = int(cluster50[cand_idx].max())
            cluster_idx = cand_idx[cluster50[cand_idx] == best_cluster]
            chosen = int(cluster_idx[np.argmin(rms[cluster_idx])])
        elif selector == "best_p":
            chosen = int(cand_idx[np.argmax(p[cand_idx])])
        else:
            raise ValueError(selector)
        selected[i] = pos[chosen]
        overrides.append(
            {
                "tow": float(ep["tow"]),
                "pick_label": pick_label,
                "chosen_label": labels[chosen],
                "pick_p": float(p[pick_idx]),
                "chosen_p": float(p[chosen]),
                "pick_rms": float(rms[pick_idx]),
                "chosen_rms": float(rms[chosen]),
                "chosen_rms_rank": int(rms_rank[chosen]),
                "chosen_cluster50": int(cluster50[chosen]),
                "dist_to_pick_m": float(dist_to_pick[chosen]),
            }
        )
    return selected, overrides


def main() -> None:
    args = _parse_args()
    epochs, baseline, truth, weights = _build_epochs(args)
    base_score, base_pass_w, base_pass_n = _score_positions(baseline, truth, weights)
    print(
        f"baseline score={base_score:.6f} pass={base_pass_w:.3f}/"
        f"{float(weights.sum()):.3f} epochs={base_pass_n}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    best_selected: np.ndarray | None = None
    best_overrides: list[dict[str, object]] = []
    best_score = -float("inf")
    # Focused grid from Phase 42 diagnostics.  The broad Cartesian product is
    # expensive and mostly tests regimes the oracle never used: forced labels
    # usually sit around RMS rank 5-12, cluster50 3-8, and p-rank 8-16.
    grids = itertools.product(
        ["c4_oa_combo_z_hs"],
        ["same_family"],
        ["best_rms", "best_cluster_rms"],
        [8, 12],
        [0, 6],
        [0.0],
        [0.8, 1.2],
        [-float("inf"), -1.0],
        [0],
    )
    n = 0
    for (
        pick_set_name,
        pool,
        selector,
        max_rms_rank,
        min_cluster50,
        min_dist,
        max_dist,
        min_p_delta,
        max_p_rank,
    ) in grids:
        n += 1
        selected, overrides = _apply_rule(
            epochs,
            baseline,
            pick_set_name=pick_set_name,
            pool=pool,
            selector=selector,
            max_rms_rank=max_rms_rank,
            min_cluster50=min_cluster50,
            max_dist_to_pick_m=max_dist,
            min_dist_to_pick_m=min_dist,
            min_p_delta=min_p_delta,
            max_p_rank=max_p_rank,
        )
        if not overrides:
            continue
        score, pass_w, pass_n = _score_positions(selected, truth, weights)
        row = {
            "score": score,
            "delta_score": score - base_score,
            "pass_w": pass_w,
            "delta_pass_w": pass_w - base_pass_w,
            "pass_epochs": pass_n,
            "n_overrides": len(overrides),
            "pick_set": pick_set_name,
            "pool": pool,
            "selector": selector,
            "max_rms_rank": max_rms_rank,
            "min_cluster50": min_cluster50,
            "min_dist_to_pick_m": min_dist,
            "max_dist_to_pick_m": max_dist,
            "min_p_delta": min_p_delta,
            "max_p_rank": max_p_rank,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_selected = selected
            best_overrides = overrides
    out = pd.DataFrame(rows).sort_values(["score", "delta_pass_w"], ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    if best_selected is not None:
        pick_df = pd.DataFrame(best_overrides)
        truth_pass_before = []
        truth_pass_after = []
        for row in pick_df.itertuples(index=False):
            tow = round(float(row.tow), 1)
            idx = next(i for i, ep in enumerate(epochs) if round(float(ep["tow"]), 1) == tow)
            before_err = float(np.linalg.norm(baseline[idx] - truth[idx]))
            after_err = float(np.linalg.norm(best_selected[idx] - truth[idx]))
            truth_pass_before.append(int(before_err < 0.5))
            truth_pass_after.append(int(after_err < 0.5))
        pick_df["pass_before"] = truth_pass_before
        pick_df["pass_after"] = truth_pass_after
        pick_df.to_csv(args.out_picks, index=False)

    print(f"tested_rules={n} nonempty={len(out)}")
    print("top rules:")
    cols = [
        "score",
        "delta_score",
        "delta_pass_w",
        "n_overrides",
        "pick_set",
        "pool",
        "selector",
        "max_rms_rank",
        "min_cluster50",
        "min_dist_to_pick_m",
        "max_dist_to_pick_m",
        "min_p_delta",
        "max_p_rank",
    ]
    print(out[cols].head(int(args.top_n)).to_string(index=False))
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_picks}")


if __name__ == "__main__":
    main()
