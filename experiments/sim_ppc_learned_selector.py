#!/usr/bin/env python3
"""Learned-ranker selector on Phase 11z candidate pool.

Phase 11z gated_oracle = 63.84%, phase = 62.29%, selector_gap +1.55pp.
Trains a HistGradientBoostingRegressor to predict per-candidate
truth-distance from RTK diagnostic features, then selects the candidate
with the smallest predicted distance per epoch.

Cross-validation: leave-one-run-out across the 6 PPC full runs. The model
sees 5 runs of (label, features, truth_distance_m) and predicts the held
out run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

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
    _parse_label_list,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402

from sim_ppc_phase_csv_addcand import (  # noqa: E402
    _candidate_dir_map,
    _discover_candidate_dir_map,
)
from sim_ppc_selector_sweep import (  # noqa: E402
    _CANDIDATES_PHASE11V,
    _DIAG_ROOT,
    _eligible_for_run,
    _FULL_RUNS,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

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
# Extra position-based features computed at sim time (length must match _EXTRA_FEATURE_KEYS).
_EXTRA_FEATURE_KEYS = (
    "agreement_count_5m",      # neighbours within 5m
    "agreement_count_2m",      # neighbours within 2m
    "dist_to_hybrid_m",        # distance to hybrid floor
    "dist_to_pos_median_m",    # distance to median of all gated candidate positions
    "n_gated_options",         # # of gated candidates this epoch
)
_ALL_FEATURE_KEYS = _FEATURE_KEYS + _EXTRA_FEATURE_KEYS
_LABEL_TRUNC = 6.0  # cap truth-distance regression target at 6m to focus on fine-grained selection


@dataclass
class FeatureRow:
    city: str
    run: str
    tow: float
    label: str
    truth_distance_m: float
    features: np.ndarray


def _load_candidates_for_run(city, run):
    out = []
    for label, dir_name, restrict in _CANDIDATES_PHASE11V:
        if not _eligible_for_run(city, run, restrict):
            continue
        pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        out.append((label, pos, diag))
    return out


def _load_candidates_for_labels(city: str, run: str, labels: list[str], label_to_dir: dict[str, str]):
    out = []
    for label in labels:
        dir_name = label_to_dir.get(label)
        if dir_name is None:
            continue
        pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        out.append((label, pos, diag))
    return out


def _build_features(row: dict[str, str]) -> np.ndarray:
    return np.asarray([_diag_float(row, k) for k in _FEATURE_KEYS], dtype=np.float64)


def _augment_features(positions: list[np.ndarray], base_feats: list[np.ndarray],
                       hybrid_pos_arr: np.ndarray | None) -> list[np.ndarray]:
    """Return per-candidate augmented feature vectors (base + extras)."""
    n = len(positions)
    pts = np.asarray(positions, dtype=np.float64)
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    cnt5 = (dists <= 5.0).sum(axis=1).astype(np.float64)
    cnt2 = (dists <= 2.0).sum(axis=1).astype(np.float64)
    median = np.median(pts, axis=0)
    dist_med = np.linalg.norm(pts - median, axis=1)
    if hybrid_pos_arr is not None and np.all(np.isfinite(hybrid_pos_arr)):
        dist_hyb = np.linalg.norm(pts - hybrid_pos_arr, axis=1)
    else:
        dist_hyb = np.full(n, 1e6, dtype=np.float64)
    out: list[np.ndarray] = []
    for i in range(n):
        extras = np.asarray([
            float(cnt5[i]), float(cnt2[i]),
            float(dist_hyb[i]), float(dist_med[i]),
            float(n),
        ], dtype=np.float64)
        out.append(np.concatenate([base_feats[i], extras]))
    return out


def _build_dataset(
    data_root: Path,
    hybrid_pos_dir: Path,
    policy: str,
    *,
    phase_rows: dict[tuple[str, str], list[str]] | None = None,
    label_to_dir: dict[str, str] | None = None,
):
    """Yield (city, run, ref, hybrid_pos_dict, gate_pass_records).

    Each gate_pass_record is a list of (tow, [(label, pos_arr, feat_arr, truth_dist), ...]).
    """
    out = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(data_root / city / run / "reference.csv")
        truth_by_t = {round(float(t), 1): pos for t, pos in ref}
        hybrid_pos, _ = _load_hybrid_pos_file(hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=policy, city=city,
        )
        ratio_min = float(variant.rtkdiag_candidate_ratio_min)
        rms_max = float(variant.rtkdiag_candidate_residual_rms_max)
        if phase_rows is not None:
            labels = phase_rows.get((city, run), [])
            loaded = _load_candidates_for_labels(city, run, labels, label_to_dir or {})
        else:
            loaded = _load_candidates_for_run(city, run)
        kept = _filter_rtkdiag_candidates_by_policy(
            loaded,
            city=city, run=run, policy=policy,
        )
        gate_records: list[tuple[float, list[tuple[str, np.ndarray, np.ndarray, float]]]] = []
        for tow, _ in ref:
            t_key = round(float(tow), 1)
            raw_options: list[tuple[str, np.ndarray, np.ndarray, float]] = []
            for label, cand_pos, cand_diag in kept:
                row = cand_diag.get(t_key)
                if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                    continue
                cand = cand_pos.get(t_key)
                if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                    continue
                truth = truth_by_t.get(t_key)
                if truth is None:
                    continue
                truth_dist = float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - np.asarray(truth, dtype=np.float64)))
                feats = _build_features(row)
                raw_options.append((label, np.asarray(cand, dtype=np.float64), feats, truth_dist))
            if raw_options:
                positions = [o[1] for o in raw_options]
                base_feats = [o[2] for o in raw_options]
                hyb = hybrid_pos.get(t_key)
                hyb_arr = np.asarray(hyb, dtype=np.float64) if hyb is not None else None
                aug = _augment_features(positions, base_feats, hyb_arr)
                options = [(o[0], o[1], aug[i], o[3]) for i, o in enumerate(raw_options)]
                gate_records.append((t_key, options))
        out.append((city, run, ref, hybrid_pos, gate_records))
    return out


def _train_and_predict(data, *, holdout_idx: int, max_iter: int, lr: float,
                        model_kind: str = "hgb", scope: str = "all"):
    """Train on training runs, predict for holdout_idx.

    scope:
      - "all": use all runs except holdout (cross-run).
      - "city": use only the same-city runs except holdout (per-city).
    """
    X_train_list: list[np.ndarray] = []
    y_train_list: list[float] = []
    group_train: list[int] = []
    holdout_city = data[holdout_idx][0]
    for i, (city, run, _ref, _hp, recs) in enumerate(data):
        if i == holdout_idx:
            continue
        if scope == "city" and city != holdout_city:
            continue
        for tow, options in recs:
            n = len(options)
            for _label, _pos, feats, truth_dist in options:
                X_train_list.append(feats)
                y_train_list.append(min(truth_dist, _LABEL_TRUNC))
            group_train.append(n)
    X_train = np.asarray(X_train_list, dtype=np.float64)
    y_train = np.asarray(y_train_list, dtype=np.float64)
    X_train = np.where(np.isfinite(X_train), X_train, 0.0)
    if model_kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=lr,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return ("hgb", model)
    if model_kind == "lgb_rank":
        if not _HAS_LGB:
            raise RuntimeError("lightgbm not installed")
        # LambdaRank: relevance must be a non-negative integer; smaller truth
        # distance => higher relevance. Bin truth distance.
        # Use 5 bins: <0.1m=4, <0.5m=3, <1m=2, <3m=1, else=0.
        rel = np.where(y_train < 0.1, 4,
              np.where(y_train < 0.5, 3,
              np.where(y_train < 1.0, 2,
              np.where(y_train < 3.0, 1, 0)))).astype(np.int32)
        model = lgb.LGBMRanker(
            n_estimators=max_iter,
            learning_rate=lr,
            max_depth=6,
            min_child_samples=20,
            objective="lambdarank",
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, rel, group=np.asarray(group_train, dtype=np.int64))
        return ("lgb_rank", model)
    raise ValueError(f"unknown model_kind {model_kind}")


def _select_with_model(data, holdout_idx: int, model_pair):
    kind, model = model_pair
    city, run, ref, hybrid_pos, recs = data[holdout_idx]
    truth_arr = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
    rec_lookup = {tow: options for tow, options in recs}
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        opts = rec_lookup.get(t_key)
        if opts is None:
            continue
        feats = np.stack([np.where(np.isfinite(o[2]), o[2], 0.0) for o in opts])
        preds = model.predict(feats)
        if kind == "hgb":
            # Smaller predicted distance is better.
            idx = int(np.argmin(preds))
        else:
            # Higher rank score is better (LambdaRank).
            idx = int(np.argmax(preds))
        est[i] = opts[idx][1]
    score = score_ppc2024(est, truth_arr)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m)


def _baseline_score_phase(data, holdout_idx: int, policy: str) -> tuple[float, float, float]:
    """Compute the same selector as phase11z (sort_key) for comparison."""
    from exp_ppc_ctrbpf_fgo import _rtkdiag_candidate_sort_key
    city, run, ref, hybrid_pos, recs = data[holdout_idx]
    variant = _apply_rtkdiag_run_index_policy(
        CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
        run=run, policy=policy, city=city,
    )
    sort_mode = str(variant.rtkdiag_candidate_select_mode)
    truth_arr = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
    rec_lookup = {tow: options for tow, options in recs}
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        opts = rec_lookup.get(t_key)
        if opts is None:
            continue
        best_i = 0
        best_key = None
        for j, o in enumerate(opts):
            # opts[j][2] is augmented feature vector; use the first len(_FEATURE_KEYS) entries.
            feat_dict = {k: str(o[2][idx]) for idx, k in enumerate(_FEATURE_KEYS)}
            key = _rtkdiag_candidate_sort_key(feat_dict, mode=sort_mode)
            if best_key is None or key < best_key:
                best_key = key
                best_i = j
        est[i] = opts[best_i][1]
    score = score_ppc2024(est, truth_arr)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m)


def _oracle_score(data, holdout_idx: int) -> tuple[float, float, float]:
    city, run, ref, hybrid_pos, recs = data[holdout_idx]
    truth_arr = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
    rec_lookup = {tow: options for tow, options in recs}
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        opts = rec_lookup.get(t_key)
        if opts is None:
            continue
        best_i = min(range(len(opts)), key=lambda j: opts[j][3])
        est[i] = opts[best_i][1]
    score = score_ppc2024(est, truth_arr)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11z")
    parser.add_argument("--phase-runs-csv", type=Path, default=None,
                        help="Use rtkdiag_candidate_labels from this runs CSV as the authoritative pool.")
    parser.add_argument("--discover-diag-dirs", action="store_true",
                        help="Include unmapped libgnss_diag_phase10 dirs in the label map.")
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--model-kind", choices=("hgb", "lgb_rank"), default="hgb")
    parser.add_argument("--scope", choices=("all", "city"), default="all",
                        help="Training scope per holdout: all=cross-run, city=same-city only")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_learned_selector_phase11z.csv")
    args = parser.parse_args()

    phase_rows = None
    label_to_dir = None
    if args.phase_runs_csv is not None:
        label_to_dir = _candidate_dir_map()
        if args.discover_diag_dirs:
            label_to_dir = _discover_candidate_dir_map(label_to_dir)
        with args.phase_runs_csv.open(newline="") as fh:
            phase_rows = {
                (str(row["city"]), str(row["run"])): _parse_label_list(str(row["rtkdiag_candidate_labels"]))
                for row in csv.DictReader(fh)
            }
    print(f"Loading data...", flush=True)
    data = _build_dataset(
        args.data_root,
        args.hybrid_pos_dir,
        str(args.policy),
        phase_rows=phase_rows,
        label_to_dir=label_to_dir,
    )
    n_records = sum(len(recs) for _, _, _, _, recs in data)
    n_options = sum(len(opts) for _, _, _, _, recs in data for _, opts in recs)
    print(f"Loaded {len(data)} runs, {n_records} gate-pass epochs, {n_options} (epoch, candidate) pairs", flush=True)

    rows = []
    learned_pass_sum = 0.0
    base_pass_sum = 0.0
    total_sum = 0.0
    for i, (city, run, _ref, _hp, _recs) in enumerate(data):
        print(f"\nHoldout {city}/{run}, training on others...", flush=True)
        model = _train_and_predict(data, holdout_idx=i, max_iter=args.max_iter, lr=args.lr,
                                    model_kind=str(args.model_kind), scope=str(args.scope))
        learned_ppc, learned_pm, total_m = _select_with_model(data, i, model)
        base_ppc, base_pm, _tm = _baseline_score_phase(data, i, str(args.policy))
        oracle_ppc, oracle_pm, _otm = _oracle_score(data, i)
        delta_pp = learned_ppc - base_ppc
        delta_m = learned_pm - base_pm
        learned_pass_sum += learned_pm
        base_pass_sum += base_pm
        total_sum += total_m
        rows.append({
            "city": city, "run": run,
            "phase_baseline_ppc": base_ppc, "phase_baseline_pass_m": base_pm,
            "oracle_ppc": oracle_ppc, "oracle_pass_m": oracle_pm,
            "learned_ppc": learned_ppc, "learned_pass_m": learned_pm,
            "delta_pp": delta_pp, "delta_m": delta_m,
            "oracle_gap_pp": oracle_ppc - base_ppc,
            "oracle_gap_m": oracle_pm - base_pm,
            "total_m": total_m,
        })
        print(f"  baseline (sort_key): ppc={base_ppc:.4f}% pass={base_pm:.1f}", flush=True)
        print(f"  oracle:              ppc={oracle_ppc:.4f}% pass={oracle_pm:.1f}", flush=True)
        print(f"  learned (HGB): ppc={learned_ppc:.4f}% pass={learned_pm:.1f}", flush=True)
        print(f"  delta: {delta_pp:+.4f}pp ({delta_m:+.1f}m)", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n========== Aggregate ==========")
    print(f"Phase baseline: {100*base_pass_sum/total_sum:.4f}% (pass {base_pass_sum:.1f}/{total_sum:.1f})")
    print(f"Learned ranker: {100*learned_pass_sum/total_sum:.4f}% (pass {learned_pass_sum:.1f}/{total_sum:.1f})")
    print(f"Delta: {100*(learned_pass_sum-base_pass_sum)/total_sum:+.4f}pp ({learned_pass_sum-base_pass_sum:+.1f}m)")


if __name__ == "__main__":
    main()
