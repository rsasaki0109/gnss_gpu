#!/usr/bin/env python3
"""Diagnose why the runtime selector misses truth-closest candidates.

For each gated epoch, compare:
  - oracle pick (truth-closest among gated candidates)
  - score pick (selector's actual choice)

Dump per-epoch feature differences (ratio/residual/nrows/score) so we can
see if any feature correlates with truth-distance better than score does.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _apply_rtkdiag_run_index_policy,
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_rtk_diag_file,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
    CTRBPFConfig,
)


def _load_pos_file(path: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                tow = round(float(parts[1]), 1)
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError:
                continue
            out[tow] = np.array([x, y, z], dtype=np.float64)
    return out


_FEATURE_KEYS = (
    "final_ratio",
    "final_residual_rms",
    "final_residual_abs_max",
    "final_update_rows",
    "final_status",
    "output_added",
)


def _row_features(row: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in _FEATURE_KEYS:
        v = _diag_float(row, key)
        if np.isfinite(v):
            out[key] = v
    # derived
    if "final_residual_rms" in out and "final_ratio" in out:
        out["score_metric"] = out["final_residual_rms"] / max(out["final_ratio"], 1.0e-6)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"))
    p.add_argument("--candidate-dirs", required=True)
    p.add_argument("--candidate-labels", required=True)
    p.add_argument("--policy", default="phase11bw")
    p.add_argument("--city", default="nagoya")
    p.add_argument("--run", default="run2")
    p.add_argument("--max-print", type=int, default=20,
                   help="Max per-epoch divergent rows to print")
    p.add_argument("--dump-csv", type=Path, default=None,
                   help="Optional CSV dump for downstream analysis")
    args = p.parse_args()

    cand_dirs = [Path(s.strip()) for s in args.candidate_dirs.split(",") if s.strip()]
    cand_labels = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]
    if len(cand_dirs) != len(cand_labels):
        raise SystemExit(f"dirs={len(cand_dirs)} != labels={len(cand_labels)}")

    city = args.city
    run = args.run
    pos_filename = f"{city}_{run}_full.pos"

    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    n_epochs = len(ref)

    candidates: list[tuple[str, dict, dict]] = []
    for d, lbl in zip(cand_dirs, cand_labels):
        pos_path = d / pos_filename
        diag_path = d / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        cand_pos = _load_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        candidates.append((lbl, cand_pos, diag))

    filtered = _filter_rtkdiag_candidates_by_policy(
        candidates, city=city, run=run, policy=args.policy,
    )

    base_cfg = CTRBPFConfig(
        enable_rtkdiag_pf_rescue=True,
        rtkdiag_candidate_select_mode="score",
        rtkdiag_candidate_ratio_min=1.0,
        rtkdiag_candidate_residual_rms_max=50.0,
        rtkdiag_candidate_emit_mode="candidate",
    )
    cfg = _apply_rtkdiag_run_index_policy(base_cfg, run=run, policy=args.policy, city=city)
    ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
    rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
    select_mode = str(cfg.rtkdiag_candidate_select_mode)
    print(f"{city}/{run} policy={args.policy} mode={select_mode} "
          f"ratio_min={ratio_min} rms_max={rms_max} cands={len(filtered)}")

    n_with_gated = 0
    n_score_match_oracle = 0
    n_score_within_5m = 0
    score_dists: list[float] = []
    oracle_dists: list[float] = []
    score_minus_oracle: list[float] = []
    oracle_pick_label_counter: Counter = Counter()
    score_pick_label_counter: Counter = Counter()
    diff_rows = []
    feature_corr_truth: dict[str, list[tuple[float, float]]] = defaultdict(list)
    csv_rows: list[str] = []
    if args.dump_csv:
        csv_rows.append("tow,oracle_label,oracle_dist,score_label,score_dist,n_gated,"
                        "oracle_ratio,oracle_resrms,oracle_nrows,oracle_score,"
                        "score_ratio,score_resrms,score_nrows,score_score")

    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        true_pos = truth[i]
        gated = []  # list of (label, pos, row_features, dist_to_truth, score_key)
        for label, cand_pos, cand_diag in filtered:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            cand = np.asarray(cand, dtype=np.float64)
            dist = float(np.linalg.norm(cand - true_pos))
            feats = _row_features(row)
            mode_for_key = select_mode if select_mode in {"ratio", "score", "maxabs", "nrows", "residual"} else "score"
            score_key = _rtkdiag_candidate_sort_key(row, mode=mode_for_key)
            gated.append((label, cand, feats, dist, score_key))
        if not gated:
            continue
        n_with_gated += 1
        # Oracle pick
        oracle = min(gated, key=lambda c: c[3])
        # Score pick (replicate selector tie-break: smaller key wins)
        score_pick = min(gated, key=lambda c: c[4] if c[4] is not None else (float("inf"),))
        oracle_dists.append(oracle[3])
        score_dists.append(score_pick[3])
        score_minus_oracle.append(score_pick[3] - oracle[3])
        oracle_pick_label_counter[oracle[0]] += 1
        score_pick_label_counter[score_pick[0]] += 1
        if oracle[0] == score_pick[0]:
            n_score_match_oracle += 1
        if abs(score_pick[3] - oracle[3]) < 5.0:
            n_score_within_5m += 1
        # Feature correlation
        for c in gated:
            for k, v in c[2].items():
                feature_corr_truth[k].append((v, c[3]))
        if oracle[0] != score_pick[0] and len(diff_rows) < args.max_print:
            diff_rows.append((tow, oracle, score_pick, len(gated)))
        if args.dump_csv:
            of = oracle[2]
            sf = score_pick[2]
            csv_rows.append(
                f"{tow:.1f},{oracle[0]},{oracle[3]:.2f},{score_pick[0]},{score_pick[3]:.2f},{len(gated)},"
                f"{of.get('final_ratio', 'NaN')},{of.get('final_residual_rms', 'NaN')},{of.get('final_update_rows', 'NaN')},{of.get('score_metric', 'NaN')},"
                f"{sf.get('final_ratio', 'NaN')},{sf.get('final_residual_rms', 'NaN')},{sf.get('final_update_rows', 'NaN')},{sf.get('score_metric', 'NaN')}"
            )

    print(f"\nGated epochs: {n_with_gated} / {n_epochs}")
    print(f"  score selector matches oracle: {n_score_match_oracle} ({100.0*n_score_match_oracle/max(n_with_gated,1):.1f}%)")
    print(f"  score within 5m of oracle:    {n_score_within_5m} ({100.0*n_score_within_5m/max(n_with_gated,1):.1f}%)")
    print(f"  median oracle dist: {median(oracle_dists):.2f}m")
    print(f"  median score dist:  {median(score_dists):.2f}m")
    print(f"  mean (score - oracle) dist: {mean(score_minus_oracle):.2f}m")

    print("\nOracle pick label distribution (top 10):")
    for lbl, n in oracle_pick_label_counter.most_common(10):
        print(f"  {n:5d}  {lbl}")
    print("\nScore  pick label distribution (top 10):")
    for lbl, n in score_pick_label_counter.most_common(10):
        print(f"  {n:5d}  {lbl}")

    print("\nFeature ↔ truth-distance correlation (Spearman approx via Pearson on ranks):")
    for key, pairs in sorted(feature_corr_truth.items()):
        if len(pairs) < 50:
            continue
        arr = np.asarray(pairs, dtype=np.float64)
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        if len(arr) < 50:
            continue
        rx = arr[:, 0].argsort().argsort().astype(np.float64)
        ry = arr[:, 1].argsort().argsort().astype(np.float64)
        if rx.std() == 0 or ry.std() == 0:
            continue
        rho = float(np.corrcoef(rx, ry)[0, 1])
        print(f"  {key:15s}  rho={rho:+.3f}  n={len(arr)}")

    if diff_rows:
        print(f"\nDivergent epochs (oracle != score), showing first {len(diff_rows)}:")
        print(f"  {'tow':>10s}  {'oracle_label':24s} {'o_dist':>7s} {'score_label':24s} {'s_dist':>7s} {'n_gate':>6s}")
        for tow, oracle, sp, ng in diff_rows:
            print(f"  {tow:10.1f}  {oracle[0]:24s} {oracle[3]:7.1f} {sp[0]:24s} {sp[3]:7.1f} {ng:6d}")

    if args.dump_csv and csv_rows:
        args.dump_csv.parent.mkdir(parents=True, exist_ok=True)
        args.dump_csv.write_text("\n".join(csv_rows) + "\n")
        print(f"\nWrote {len(csv_rows)-1} rows -> {args.dump_csv}")


if __name__ == "__main__":
    main()
