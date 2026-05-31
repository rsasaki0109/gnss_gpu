#!/usr/bin/env python3
"""Diagnose PPC loss that cannot be fixed by the current candidate selector.

Late Phase 11 selector tweaks can only approach the per-epoch oracle of the
current RTKDiag candidate pool.  This script splits the remaining loss into:

* selector miss: oracle candidate passes 0.5 m, current selector does not
* pool miss: even the truth-closest gated candidate misses 0.5 m
* no gated candidate: the policy gate yields no usable candidate

The output is distance-weighted using the PPC2024 segment weights, so the
numbers map directly to pass-distance headroom.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
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
    _parse_label_list,
    _rtkdiag_candidate_gate,
    _rtkdiag_fixed_output_ok,
    _rtkdiag_local_ungate_labels,
    _rtkdiag_local_ungate_labels_for_tow,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import ppc_3d_errors, ppc_segment_distances, score_ppc2024  # noqa: E402
from sim_ppc_phase_csv_addcand import _temporal_select_params, _valid_hybrid  # noqa: E402
from sim_ppc_trap_diagnosis import _label_dir_map, _load_candidate, _label_penalty_factors  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


@dataclass(frozen=True)
class EpochDiag:
    city: str
    run: str
    idx: int
    tow: float
    weight_m: float
    current_error_m: float
    oracle_error_m: float
    hybrid_error_m: float
    n_gated: int
    current_label: str
    oracle_label: str
    kind: str


def _run_phase_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get("method", "").endswith("rtkdiag_pf")]


def _parse_run_filter(spec: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "/" not in chunk:
            raise ValueError(f"bad run filter spec: {chunk!r}")
        city, run = chunk.split("/", 1)
        out.add((city, run))
    return out


def _load_candidates(city: str, run: str, labels: list[str]):
    label_to_dir = _label_dir_map()
    loaded = []
    missing = []
    for label in labels:
        dir_name = label_to_dir.get(label)
        if dir_name is None and label.startswith("x"):
            dir_name = label_to_dir.get(label[1:])
        if dir_name is None:
            missing.append(label)
            continue
        cand = _load_candidate(city, run, label, dir_name)
        if cand is None:
            missing.append(label)
            continue
        loaded.append((cand.label, cand.pos, cand.diag))
    return loaded, missing


def _collect_epoch_options(
    *,
    city: str,
    run: str,
    row: dict[str, str],
    policy: str,
    data_root: Path,
    hybrid_pos_dir: Path,
):
    labels = _parse_label_list(str(row["rtkdiag_candidate_labels"]))
    loaded, missing = _load_candidates(city, run, labels)
    kept = _filter_rtkdiag_candidates_by_policy(loaded, city=city, run=run, policy=policy)
    cfg = _apply_rtkdiag_run_index_policy(
        CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
        city=city,
        run=run,
        policy=policy,
    )
    mode = str(cfg.rtkdiag_candidate_select_mode)
    base_mode, temporal_kind, temporal_alpha = _temporal_select_params(mode)
    label_penalties = _label_penalty_factors(mode)
    if cfg.rtkdiag_candidate_label_factors:
        label_penalties = dict(label_penalties)
        label_penalties.update(
            {
                str(label): float(factor)
                for label, factor in cfg.rtkdiag_candidate_label_factors
            }
        )
    ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
    rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
    local_ungate_windows = tuple(cfg.rtkdiag_candidate_local_ungate_windows)
    local_ungate_tow_windows = tuple(cfg.rtkdiag_candidate_local_ungate_tow_windows)

    ref = _load_full_reference(data_root / city / run / "reference.csv")
    truth = np.asarray([p for _tow, p in ref], dtype=np.float64)
    weights = ppc_segment_distances(truth)
    hybrid_pos, _ = _load_hybrid_pos_file(hybrid_pos_dir / f"{city}_{run}_full.pos")

    epochs = []
    for tow, _true_pos in ref:
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_arr = np.asarray(hp, dtype=np.float64) if _valid_hybrid(hp) else None
        local_ungate_labels = _rtkdiag_local_ungate_labels(local_ungate_windows, len(epochs))
        if local_ungate_labels is None:
            local_ungate_labels = _rtkdiag_local_ungate_labels_for_tow(
                local_ungate_tow_windows,
                float(t_key),
            )
        opts = []
        for label, cand_pos, cand_diag in kept:
            diag_row = cand_diag.get(t_key)
            gate_ok = _rtkdiag_candidate_gate(diag_row, ratio_min=ratio_min, residual_rms_max=rms_max)
            local_ungate_ok = (
                local_ungate_labels is not None
                and _rtkdiag_fixed_output_ok(diag_row)
                and (not local_ungate_labels or label in local_ungate_labels)
            )
            if not gate_ok and not local_ungate_ok:
                continue
            cand = cand_pos.get(t_key)
            if not _valid_hybrid(cand):
                continue
            key = _rtkdiag_candidate_sort_key(diag_row, mode=base_mode)
            key0 = float(key[0]) * float(label_penalties.get(label, 1.0))
            key1 = float(key[1])
            opts.append((label, np.asarray(cand, dtype=np.float64), key0, key1))
        epochs.append((float(tow), hp_arr, opts))
    return truth, weights, mode, epochs, len(loaded), len(kept), missing


def _simulate_current(truth: np.ndarray, mode: str, epochs):
    est = np.zeros_like(truth)
    labels = [""] * len(epochs)
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    _base_mode, temporal_kind, temporal_alpha = _temporal_select_params(mode)
    for i, (_tow, hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if not opts:
            if hp is not None:
                prev = hp
                prev_hybrid = hp
            continue
        key0 = np.asarray([o[2] for o in opts], dtype=np.float64)
        key1 = np.asarray([o[3] for o in opts], dtype=np.float64)
        pos_arr = np.vstack([o[1] for o in opts]).astype(np.float64, copy=False)
        if temporal_kind == "prevdist" and prev is not None:
            key0 += float(temporal_alpha) * np.linalg.norm(pos_arr - prev, axis=1)
        elif temporal_kind == "hybdelta" and prev is not None and prev_hybrid is not None and hp is not None:
            predicted = prev + (hp - prev_hybrid)
            key0 += float(temporal_alpha) * np.linalg.norm(pos_arr - predicted, axis=1)
        idx = int(np.lexsort((key1, key0))[0])
        labels[i] = str(opts[idx][0])
        est[i] = pos_arr[idx]
        prev = pos_arr[idx]
        if hp is not None:
            prev_hybrid = hp
    return est, labels


def _simulate_oracle(truth: np.ndarray, epochs):
    est = np.zeros_like(truth)
    labels = [""] * len(epochs)
    n_gated = np.zeros(len(epochs), dtype=np.int32)
    for i, (_tow, hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        n_gated[i] = len(opts)
        if not opts:
            continue
        dists = np.asarray([float(np.linalg.norm(o[1] - truth[i])) for o in opts], dtype=np.float64)
        idx = int(np.argmin(dists))
        labels[i] = str(opts[idx][0])
        est[i] = np.asarray(opts[idx][1], dtype=np.float64)
    return est, labels, n_gated


def _simulate_hybrid(truth: np.ndarray, epochs):
    est = np.zeros_like(truth)
    for i, (_tow, hp, _opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
    return est


def _classify_epochs(
    *,
    city: str,
    run: str,
    threshold_m: float,
    truth: np.ndarray,
    weights: np.ndarray,
    epochs,
    current_est: np.ndarray,
    current_labels: list[str],
    oracle_est: np.ndarray,
    oracle_labels: list[str],
    hybrid_est: np.ndarray,
    n_gated: np.ndarray,
) -> list[EpochDiag]:
    current_err = ppc_3d_errors(current_est, truth)
    oracle_err = ppc_3d_errors(oracle_est, truth)
    hybrid_err = ppc_3d_errors(hybrid_est, truth)
    out = []
    for i, (tow, _hp, _opts) in enumerate(epochs):
        cur_pass = bool(current_err[i] <= threshold_m)
        ora_pass = bool(oracle_err[i] <= threshold_m)
        if cur_pass:
            kind = "current_pass"
        elif n_gated[i] <= 0:
            kind = "no_gated_candidate"
        elif ora_pass:
            kind = "selector_miss"
        else:
            kind = "pool_miss"
        out.append(EpochDiag(
            city=city,
            run=run,
            idx=i,
            tow=float(tow),
            weight_m=float(weights[i]),
            current_error_m=float(current_err[i]),
            oracle_error_m=float(oracle_err[i]),
            hybrid_error_m=float(hybrid_err[i]),
            n_gated=int(n_gated[i]),
            current_label=current_labels[i],
            oracle_label=oracle_labels[i],
            kind=kind,
        ))
    return out


def _segments(rows: list[EpochDiag]) -> list[dict[str, object]]:
    segs: list[dict[str, object]] = []
    cur: list[EpochDiag] = []
    for row in rows:
        if row.kind in {"selector_miss", "pool_miss", "no_gated_candidate"}:
            if cur and (cur[-1].kind != row.kind or cur[-1].idx + 1 != row.idx):
                segs.append(_segment_row(cur))
                cur = []
            cur.append(row)
        elif cur:
            segs.append(_segment_row(cur))
            cur = []
    if cur:
        segs.append(_segment_row(cur))
    segs.sort(key=lambda r: float(r["weight_m"]), reverse=True)
    return segs


def _segment_row(rows: list[EpochDiag]) -> dict[str, object]:
    oracle_labels = Counter(r.oracle_label for r in rows if r.oracle_label)
    current_labels = Counter(r.current_label for r in rows if r.current_label)
    return {
        "city": rows[0].city,
        "run": rows[0].run,
        "kind": rows[0].kind,
        "start_idx": rows[0].idx,
        "end_idx": rows[-1].idx,
        "start_tow": rows[0].tow,
        "end_tow": rows[-1].tow,
        "epochs": len(rows),
        "weight_m": sum(r.weight_m for r in rows),
        "mean_current_error_m": sum(r.current_error_m for r in rows) / max(len(rows), 1),
        "mean_oracle_error_m": sum(r.oracle_error_m for r in rows) / max(len(rows), 1),
        "mean_hybrid_error_m": sum(r.hybrid_error_m for r in rows) / max(len(rows), 1),
        "mean_gated": sum(r.n_gated for r in rows) / max(len(rows), 1),
        "top_oracle_labels": ",".join(f"{k}:{v}" for k, v in oracle_labels.most_common(5)),
        "top_current_labels": ",".join(f"{k}:{v}" for k, v in current_labels.most_common(5)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    p.add_argument("--phase-runs-csv", type=Path, required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--only-runs", default="")
    p.add_argument("--threshold-m", type=float, default=0.5)
    p.add_argument("--out-runs-csv", type=Path, default=RESULTS_DIR / "ppc_oracle_miss_runs.csv")
    p.add_argument("--out-segments-csv", type=Path, default=RESULTS_DIR / "ppc_oracle_miss_segments.csv")
    p.add_argument("--out-epochs-csv", type=Path, default=None)
    args = p.parse_args()

    phase_rows = _run_phase_rows(args.phase_runs_csv)
    only_runs = _parse_run_filter(str(args.only_runs))
    if only_runs:
        phase_rows = [r for r in phase_rows if (str(r["city"]), str(r["run"])) in only_runs]
    if not phase_rows:
        raise SystemExit(f"no matching rtkdiag rows in {args.phase_runs_csv}")

    run_out: list[dict[str, object]] = []
    all_epochs: list[EpochDiag] = []
    all_segments: list[dict[str, object]] = []
    agg_current_pass = 0.0
    agg_oracle_pass = 0.0
    agg_hybrid_pass = 0.0
    agg_total = 0.0

    for row in phase_rows:
        city = str(row["city"])
        run = str(row["run"])
        truth, weights, mode, epochs, loaded, kept, missing = _collect_epoch_options(
            city=city,
            run=run,
            row=row,
            policy=str(args.policy),
            data_root=args.data_root,
            hybrid_pos_dir=args.hybrid_pos_dir,
        )
        current_est, current_labels = _simulate_current(truth, mode, epochs)
        oracle_est, oracle_labels, n_gated = _simulate_oracle(truth, epochs)
        hybrid_est = _simulate_hybrid(truth, epochs)
        current_score = score_ppc2024(current_est, truth, threshold_m=float(args.threshold_m), segment_distances_m=weights)
        oracle_score = score_ppc2024(oracle_est, truth, threshold_m=float(args.threshold_m), segment_distances_m=weights)
        hybrid_score = score_ppc2024(hybrid_est, truth, threshold_m=float(args.threshold_m), segment_distances_m=weights)
        epoch_rows = _classify_epochs(
            city=city,
            run=run,
            threshold_m=float(args.threshold_m),
            truth=truth,
            weights=weights,
            epochs=epochs,
            current_est=current_est,
            current_labels=current_labels,
            oracle_est=oracle_est,
            oracle_labels=oracle_labels,
            hybrid_est=hybrid_est,
            n_gated=n_gated,
        )
        selector_miss_m = sum(e.weight_m for e in epoch_rows if e.kind == "selector_miss")
        pool_miss_m = sum(e.weight_m for e in epoch_rows if e.kind == "pool_miss")
        no_gated_m = sum(e.weight_m for e in epoch_rows if e.kind == "no_gated_candidate")
        total_m = float(current_score.total_distance_m)
        run_out.append({
            "city": city,
            "run": run,
            "mode": mode,
            "loaded_candidates": loaded,
            "kept_candidates": kept,
            "missing_labels": ",".join(missing),
            "current_ppc": current_score.score_pct,
            "oracle_ppc": oracle_score.score_pct,
            "hybrid_ppc": hybrid_score.score_pct,
            "current_pass_m": current_score.pass_distance_m,
            "oracle_pass_m": oracle_score.pass_distance_m,
            "hybrid_pass_m": hybrid_score.pass_distance_m,
            "total_m": total_m,
            "selector_headroom_m": oracle_score.pass_distance_m - current_score.pass_distance_m,
            "pool_miss_m": pool_miss_m,
            "selector_miss_m": selector_miss_m,
            "no_gated_m": no_gated_m,
            "pool_miss_pct_total": 100.0 * pool_miss_m / max(total_m, 1.0e-12),
            "selector_miss_pct_total": 100.0 * selector_miss_m / max(total_m, 1.0e-12),
            "no_gated_pct_total": 100.0 * no_gated_m / max(total_m, 1.0e-12),
        })
        all_epochs.extend(epoch_rows)
        all_segments.extend(_segments(epoch_rows))
        agg_current_pass += float(current_score.pass_distance_m)
        agg_oracle_pass += float(oracle_score.pass_distance_m)
        agg_hybrid_pass += float(hybrid_score.pass_distance_m)
        agg_total += total_m
        print(
            f"{city}/{run}: current={current_score.score_pct:.6f}% "
            f"oracle={oracle_score.score_pct:.6f}% "
            f"selector_headroom={oracle_score.pass_distance_m - current_score.pass_distance_m:.3f}m "
            f"pool_miss={pool_miss_m:.3f}m no_gated={no_gated_m:.3f}m",
            flush=True,
        )

    all_segments.sort(key=lambda r: float(r["weight_m"]), reverse=True)
    _write_csv(args.out_runs_csv, run_out)
    _write_csv(args.out_segments_csv, all_segments)
    if args.out_epochs_csv is not None:
        _write_csv(args.out_epochs_csv, [e.__dict__ for e in all_epochs])

    print("\n========== aggregate ==========")
    print(f"current={100.0 * agg_current_pass / agg_total:.9f}% pass={agg_current_pass:.6f}/{agg_total:.6f}")
    print(f"oracle={100.0 * agg_oracle_pass / agg_total:.9f}% pass={agg_oracle_pass:.6f}/{agg_total:.6f}")
    print(f"hybrid={100.0 * agg_hybrid_pass / agg_total:.9f}% pass={agg_hybrid_pass:.6f}/{agg_total:.6f}")
    print(f"selector_headroom_m={agg_oracle_pass - agg_current_pass:.6f}")
    print(f"saved runs:    {args.out_runs_csv}")
    print(f"saved segments:{args.out_segments_csv}")
    if args.out_epochs_csv is not None:
        print(f"saved epochs:  {args.out_epochs_csv}")
    print("\nTop pool-miss/no-gated segments:")
    shown = 0
    for seg in all_segments:
        if str(seg["kind"]) not in {"pool_miss", "no_gated_candidate"}:
            continue
        print(
            f"  {seg['city']}/{seg['run']} {seg['kind']} "
            f"tow={float(seg['start_tow']):.1f}-{float(seg['end_tow']):.1f} "
            f"epochs={seg['epochs']} weight={float(seg['weight_m']):.1f}m "
            f"oracle_err={float(seg['mean_oracle_error_m']):.1f}m "
            f"hyb_err={float(seg['mean_hybrid_error_m']):.1f}m "
            f"oracle_labels={seg['top_oracle_labels']}"
        )
        shown += 1
        if shown >= 12:
            break


if __name__ == "__main__":
    main()
