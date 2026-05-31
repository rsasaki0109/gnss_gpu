#!/usr/bin/env python3
"""Targeted selector probes for one PPC run.

This is intentionally exploratory: it replays the authoritative candidate
pool from a phase runs CSV, reproduces the current temporal selector for the
chosen run, then tests simple spatial-cluster and cross-run learned-ranker
alternatives. Positive probes can later be promoted into a named phase.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    CTRBPFConfig,
    _apply_rtkdiag_run_index_policy,
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from sim_ppc_phase_csv_addcand import (  # noqa: E402
    _candidate_dir_map,
    _discover_candidate_dir_map,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DIAG_ROOT = RESULTS_DIR / "libgnss_diag_phase10"
_FULL_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)
_BASE_MODE_FOR_TEMPORAL = {
    "temporal_n2_v1": "composite_n2_v4",
    "temporal_n2_v2": "composite_n2_v4",
    "temporal_n2_v3": "composite_n2_v4",
    "temporal_hybdelta_t3_v1": "composite_t3_v2",
    "temporal_hybdelta_t3_v2": "composite_t3_v2",
    "temporal_hybdelta_t3_v3": "composite_t3_v2",
    "temporal_hybdelta_t3_v4": "composite_t3_v4",
    "temporal_hybdelta_n2_v1": "composite_n2_v4",
    "temporal_hybdelta_n3_v1": "composite_n3_v3",
    "temporal_hybdelta_n3_v2": "composite_n3_v4",
}
_PREVDIST_ALPHA = {
    "temporal_n2_v1": 0.001,
    "temporal_n2_v2": 0.0006,
    "temporal_n2_v3": 0.00062,
}
_HYBDELTA_ALPHA = {
    "temporal_hybdelta_t3_v1": 0.0003,
    "temporal_hybdelta_t3_v2": 0.0002,
    "temporal_hybdelta_t3_v3": 0.00022,
    "temporal_hybdelta_t3_v4": 0.0002,
    "temporal_hybdelta_n2_v1": 0.0003,
    "temporal_hybdelta_n3_v1": 0.0003,
    "temporal_hybdelta_n3_v2": 0.0006,
}
_FEATURE_KEYS = (
    "final_residual_rms",
    "final_ratio",
    "final_residual_abs_max",
    "final_update_rows",
    "final_pdop",
    "final_sats",
    "final_baseline_m",
    "final_suppressed_outliers",
    "candidate_jump_m",
    "candidate_vs_spp_m",
    "fixed_drift_guard_m",
    "fixed_height_guard_m",
)


def _load_anchor_pos(args: argparse.Namespace, city: str, run: str) -> dict[float, np.ndarray] | None:
    path: Path | None = None
    if args.anchor_pos_file is not None and city == args.city and run == args.run:
        path = args.anchor_pos_file
    elif args.anchor_pos_dir is not None:
        name = str(args.anchor_pos_pattern).format(city=city, run=run)
        path = args.anchor_pos_dir / name
    if path is None or not path.is_file():
        return None
    pos, _ = _load_hybrid_pos_file(path)
    if pos:
        return pos
    # CTRBPF diagnostic .pos files in experiments/results/libgnss_ctrbpf_pos use
    # columns: week tow x y z Q ...; libgnss++ parser expects Q at col 8.
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                tow = round(float(parts[1]), 1)
                xyz = np.asarray([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
                status = int(float(parts[5]))
            except (ValueError, IndexError):
                continue
            if status == 0 or not np.all(np.isfinite(xyz)) or np.all(xyz == 0.0):
                continue
            pos[tow] = xyz
    return pos


def _phase_labels(path: Path) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[(str(row["city"]), str(row["run"]))] = [
                x.strip()
                for x in str(row.get("rtkdiag_candidate_labels", "")).split(",")
                if x.strip()
            ]
    return out


def _load_run(args: argparse.Namespace, city: str, run: str, label_to_dir: dict[str, str], labels_by_run: dict[tuple[str, str], list[str]]):
    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([p for _, p in ref], dtype=np.float64)
    hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
    anchor_pos = _load_anchor_pos(args, city, run)
    cfg = _apply_rtkdiag_run_index_policy(
        CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
        city=city,
        run=run,
        policy=str(args.policy),
    )
    loaded = []
    for label in labels_by_run.get((city, run), []):
        dir_name = label_to_dir.get(label)
        if dir_name is None:
            continue
        pos_path = _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        loaded.append((label, pos, diag))
    kept = _filter_rtkdiag_candidates_by_policy(
        loaded,
        city=city,
        run=run,
        policy=str(args.policy),
    )
    select_mode = str(cfg.rtkdiag_candidate_select_mode)
    base_mode = _BASE_MODE_FOR_TEMPORAL.get(select_mode, select_mode)
    epochs = []
    for tow, true_pos in ref:
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_arr = (
            np.asarray(hp, dtype=np.float64)
            if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0)
            else None
        )
        ap = anchor_pos.get(t_key) if anchor_pos is not None else None
        ap_arr = (
            np.asarray(ap, dtype=np.float64)
            if ap is not None and np.all(np.isfinite(ap)) and not np.all(ap == 0.0)
            else None
        )
        raw = []
        for label, cand_pos, cand_diag in kept:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(
                row,
                ratio_min=float(cfg.rtkdiag_candidate_ratio_min),
                residual_rms_max=float(cfg.rtkdiag_candidate_residual_rms_max),
            ):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            pos = np.asarray(cand, dtype=np.float64)
            base_key = _rtkdiag_candidate_sort_key(row, mode=base_mode)
            features = np.asarray(
                [_diag_float(row, k) for k in _FEATURE_KEYS] + [base_key[0], base_key[1]],
                dtype=np.float64,
            )
            raw.append(
                {
                    "label": label,
                    "pos": pos,
                    "row": row,
                    "base": base_key,
                    "feat0": features,
                    "target": min(float(np.linalg.norm(pos - np.asarray(true_pos, dtype=np.float64))), 6.0),
                }
            )
        if raw:
            pts = np.stack([o["pos"] for o in raw], axis=0)
            dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
            median = np.median(pts, axis=0)
            for idx, opt in enumerate(raw):
                extras = np.asarray(
                    [
                        float((dists[idx] <= 2.0).sum()),
                        float((dists[idx] <= 5.0).sum()),
                        float((dists[idx] <= 10.0).sum()),
                        float(np.linalg.norm(opt["pos"] - median)),
                        float(len(raw)),
                        float(np.linalg.norm(opt["pos"] - hp_arr)) if hp_arr is not None else 1.0e6,
                        float(np.linalg.norm(opt["pos"] - ap_arr)) if ap_arr is not None else 1.0e6,
                    ],
                    dtype=np.float64,
                )
                opt["feat"] = np.concatenate([opt["feat0"], extras])
                opt["anchor_dist"] = float(extras[-1])
        epochs.append((t_key, hp_arr, raw))
    return {
        "city": city,
        "run": run,
        "ref": ref,
        "truth": truth,
        "select_mode": select_mode,
        "epochs": epochs,
        "loaded": len(loaded),
        "kept": len(kept),
        "ratio_min": float(cfg.rtkdiag_candidate_ratio_min),
        "rms_max": float(cfg.rtkdiag_candidate_residual_rms_max),
    }


def _score(run_data: dict[str, object], est: np.ndarray):
    score = score_ppc2024(est, run_data["truth"])  # type: ignore[arg-type]
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m)


def _baseline_est(run_data: dict[str, object]) -> np.ndarray:
    epochs = run_data["epochs"]  # type: ignore[assignment]
    truth = run_data["truth"]  # type: ignore[assignment]
    select_mode = str(run_data["select_mode"])
    est = np.zeros_like(truth)
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    for i, (_t_key, hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if not opts:
            if hp is not None:
                prev = hp
                prev_hybrid = hp
            continue
        if select_mode in _PREVDIST_ALPHA and prev is not None:
            alpha = _PREVDIST_ALPHA[select_mode]
            best = min(opts, key=lambda o: (o["base"][0] + alpha * float(np.linalg.norm(o["pos"] - prev)), o["base"][1]))
        elif select_mode in _HYBDELTA_ALPHA and prev is not None and prev_hybrid is not None and hp is not None:
            alpha = _HYBDELTA_ALPHA[select_mode]
            predicted = prev + (hp - prev_hybrid)
            best = min(opts, key=lambda o: (o["base"][0] + alpha * float(np.linalg.norm(o["pos"] - predicted)), o["base"][1]))
        else:
            best = min(opts, key=lambda o: o["base"])
        est[i] = best["pos"]
        prev = best["pos"]
        if hp is not None:
            prev_hybrid = hp
    return est


def _cluster_est(run_data: dict[str, object], mode: str, threshold_m: float, temporal_weight: float) -> np.ndarray:
    epochs = run_data["epochs"]  # type: ignore[assignment]
    truth = run_data["truth"]  # type: ignore[assignment]
    est = np.zeros_like(truth)
    prev: np.ndarray | None = None
    for i, (_t_key, hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if not opts:
            if hp is not None:
                prev = hp
            continue
        pts = np.stack([o["pos"] for o in opts], axis=0)
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        counts = (dists <= threshold_m).sum(axis=1)
        if mode == "count":
            pos = pts[int(np.argmax(counts))]
        elif mode == "median":
            idx = int(np.argmax(counts))
            pos = np.median(pts[dists[idx] <= threshold_m], axis=0)
        elif mode == "blend":
            ranks = np.argsort(np.argsort([o["base"][0] for o in opts]))
            idx = int(np.argmin(ranks + (len(opts) - counts)))
            pos = pts[idx]
        elif mode == "temporal_blend":
            vals = []
            for idx, opt in enumerate(opts):
                temporal = temporal_weight * float(np.linalg.norm(opt["pos"] - prev)) if prev is not None else 0.0
                vals.append(float(opt["base"][0]) + temporal + 0.02 * float(len(opts) - counts[idx]))
            pos = pts[int(np.argmin(vals))]
        else:
            raise ValueError(mode)
        est[i] = pos
        prev = pos
    return est


def _anchor_est(run_data: dict[str, object], beta: float, *, multiply: bool) -> np.ndarray:
    epochs = run_data["epochs"]  # type: ignore[assignment]
    truth = run_data["truth"]  # type: ignore[assignment]
    select_mode = str(run_data["select_mode"])
    est = np.zeros_like(truth)
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    for i, (_t_key, hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if not opts:
            if hp is not None:
                prev = hp
                prev_hybrid = hp
            continue

        def current_key(opt: dict[str, object]) -> tuple[float, float]:
            base = opt["base"]
            if select_mode in _PREVDIST_ALPHA and prev is not None:
                return (
                    base[0] + _PREVDIST_ALPHA[select_mode] * float(np.linalg.norm(opt["pos"] - prev)),
                    base[1],
                )
            if select_mode in _HYBDELTA_ALPHA and prev is not None and prev_hybrid is not None and hp is not None:
                predicted = prev + (hp - prev_hybrid)
                return (
                    base[0] + _HYBDELTA_ALPHA[select_mode] * float(np.linalg.norm(opt["pos"] - predicted)),
                    base[1],
                )
            return base

        if multiply:
            best = min(
                opts,
                key=lambda o: (
                    current_key(o)[0] * (1.0 + beta * min(float(o.get("anchor_dist", 1.0e6)), 1.0e5) ** 2),
                    current_key(o)[1],
                ),
            )
        else:
            best = min(
                opts,
                key=lambda o: (
                    current_key(o)[0] + beta * min(float(o.get("anchor_dist", 1.0e6)), 1.0e5),
                    current_key(o)[1],
                ),
            )
        est[i] = best["pos"]
        prev = best["pos"]
        if hp is not None:
            prev_hybrid = hp
    return est


def _learned_est(run_data: dict[str, object], train_runs: list[dict[str, object]], max_iter: int, lr: float) -> np.ndarray:
    x_train = []
    y_train = []
    for data in train_runs:
        for _t_key, _hp, opts in data["epochs"]:  # type: ignore[index]
            for opt in opts:
                x_train.append(np.where(np.isfinite(opt["feat"]), opt["feat"], 0.0))
                y_train.append(float(opt["target"]))
    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=lr,
        max_depth=4,
        min_samples_leaf=40,
        random_state=42,
    )
    model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64))
    epochs = run_data["epochs"]  # type: ignore[assignment]
    truth = run_data["truth"]  # type: ignore[assignment]
    est = np.zeros_like(truth)
    for i, (_t_key, hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if not opts:
            continue
        feats = np.stack([np.where(np.isfinite(o["feat"]), o["feat"], 0.0) for o in opts], axis=0)
        est[i] = opts[int(np.argmin(model.predict(feats)))]["pos"]
    return est


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--phase-runs-csv", type=Path, required=True)
    parser.add_argument("--policy", type=str, default="phase11ec")
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--anchor-pos-file", type=Path)
    parser.add_argument("--anchor-pos-dir", type=Path)
    parser.add_argument("--anchor-pos-pattern", type=str, default="{city}_{run}_full.pos")
    parser.add_argument("--anchor-betas", type=str, default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--learned", action="store_true")
    parser.add_argument("--scope", choices=("all", "city"), default="all")
    args = parser.parse_args()

    label_to_dir = _discover_candidate_dir_map(_candidate_dir_map())
    labels_by_run = _phase_labels(args.phase_runs_csv)
    target = _load_run(args, args.city, args.run, label_to_dir, labels_by_run)
    print(
        f"{args.city}/{args.run} loaded={target['loaded']} kept={target['kept']} "
        f"mode={target['select_mode']} ratio={target['ratio_min']} rms={target['rms_max']}",
        flush=True,
    )
    base_score = _score(target, _baseline_est(target))
    print(f"baseline score={base_score[0]:.9f} pass={base_score[1]:.6f}", flush=True)

    for mode in ("count", "median", "blend", "temporal_blend"):
        for threshold in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0):
            est = _cluster_est(target, mode, threshold, temporal_weight=0.00062)
            score = _score(target, est)
            print(
                f"{mode:14s} th={threshold:4.1f} score={score[0]:.9f} "
                f"pass={score[1]:.6f} delta_m={score[1] - base_score[1]:+.3f}",
                flush=True,
            )

    betas = [float(x) for x in str(args.anchor_betas).split(",") if x.strip()]
    has_anchor = any(
        float(opt.get("anchor_dist", 1.0e6)) < 1.0e5
        for _t_key, _hp, opts in target["epochs"]  # type: ignore[index]
        for opt in opts
    )
    if has_anchor:
        for multiply in (False, True):
            name = "anchor_mul" if multiply else "anchor_add"
            for beta in betas:
                est = _anchor_est(target, beta, multiply=multiply)
                score = _score(target, est)
                print(
                    f"{name:14s} beta={beta:<8g} score={score[0]:.9f} "
                    f"pass={score[1]:.6f} delta_m={score[1] - base_score[1]:+.3f}",
                    flush=True,
                )

    if args.learned:
        train_runs = []
        for city, run in _FULL_RUNS:
            if city == args.city and run == args.run:
                continue
            if args.scope == "city" and city != args.city:
                continue
            train_runs.append(_load_run(args, city, run, label_to_dir, labels_by_run))
        n_train = sum(len(opts) for data in train_runs for _t_key, _hp, opts in data["epochs"])  # type: ignore[index]
        print(f"learned train_runs={len(train_runs)} train_pairs={n_train}", flush=True)
        for max_iter, lr in ((200, 0.04), (400, 0.03), (800, 0.02)):
            est = _learned_est(target, train_runs, max_iter=max_iter, lr=lr)
            score = _score(target, est)
            print(
                f"learned_hgb it={max_iter} lr={lr} score={score[0]:.9f} "
                f"pass={score[1]:.6f} delta_m={score[1] - base_score[1]:+.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
