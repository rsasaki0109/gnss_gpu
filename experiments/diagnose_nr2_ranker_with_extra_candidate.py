#!/usr/bin/env python3
"""Diagnose nagoya/run2 ranker picks with extra candidates outside features.

This rebuilds the per-epoch ranker choice from candidate ``.pos``/diagnostic
CSV files, so forced candidates such as the phase35 pb40 block are included
even when they are absent from ``selector_training_features_*``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

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
    _load_ranker_predictions,
    _load_rtk_diag_file,
    _rtkdiag_candidate_diag_policy_gate,
    _rtkdiag_candidate_gate,
)
from gnss_gpu.ppc_score import ppc_segment_distances, score_ppc2024  # noqa: E402

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
RESULTS = REPO / "experiments/results"

PHASE19_EXTRAS: tuple[tuple[str, Path], ...] = (
    ("xd_fgo_v2_gap", RESULTS / "libgnss_diag_phase10/fgo_v2_gap"),
    ("xd_fgo_v14_snr38", RESULTS / "libgnss_diag_phase10/fgo_v14_snr38"),
    ("xd_fgo_v17_el25", RESULTS / "libgnss_diag_phase10/fgo_v17_el25"),
    ("xd_gici_def", RESULTS / "libgnss_diag_phase19/gici_tc_esdfix"),
    ("xd_gici_z", RESULTS / "libgnss_diag_phase19/gici_full_zeroarm"),
    ("xd_gici_r", RESULTS / "libgnss_diag_phase19/gici_full_ratio25"),
    ("xd_gici_lp", RESULTS / "libgnss_diag_phase19/gici_full_loosepr"),
    ("xd_gici_lh", RESULTS / "libgnss_diag_phase19/gici_full_loosephase"),
    ("xd_gici_r4", RESULTS / "libgnss_diag_phase19/gici_full_ratio40"),
    ("xd_gici_combo", RESULTS / "libgnss_diag_phase19/gici_full_combo"),
    ("xd_gici_c4", RESULTS / "libgnss_diag_phase19/gici_full_combo4"),
    ("xd_gici_lprlph", RESULTS / "libgnss_diag_phase19/gici_full_lprlph"),
    ("xd_gici_zr", RESULTS / "libgnss_diag_phase19/gici_full_zr"),
    ("xd_gici_oa", RESULTS / "libgnss_diag_phase19/gici_full_onarm"),
    ("xd_gici_la", RESULTS / "libgnss_diag_phase19/gici_full_lowacc"),
    ("xd_gici_hs", RESULTS / "libgnss_diag_phase19/gici_full_hisnr"),
    ("xd_gici_hs45", RESULTS / "libgnss_diag_phase19/gici_full_hisnr45"),
    ("xd_gici_hs30", RESULTS / "libgnss_diag_phase19/gici_full_hisnr30"),
    ("xd_gici_he", RESULTS / "libgnss_diag_phase19/gici_full_hielev"),
    ("xd_gici_ir", RESULTS / "libgnss_diag_phase19/gici_full_imurot"),
    ("xd_gici_mb", RESULTS / "libgnss_diag_phase19/gici_full_himuba"),
    ("xd_gici_w5", RESULTS / "libgnss_diag_phase19/gici_full_window5"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40.csv",
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
    parser.add_argument(
        "--extra-label",
        action="append",
        default=["xd_nr2_hs_pb40"],
        help="Extra label to append; repeat with matching --extra-dir.",
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        type=Path,
        default=[
            RESULTS / "libgnss_diag_phase34/nr2_hs_piecebias40_oracle_556184_556337"
        ],
        help="Extra candidate directory to append; repeat with matching --extra-label.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS / "nr2_phase35_pb40_ranker_wrongpicks.csv",
    )
    parser.add_argument(
        "--span-csv",
        type=Path,
        default=RESULTS / "nr2_phase35_pb40_ranker_wrongpick_spans.csv",
    )
    return parser.parse_args()


def _split_csv_text(path: Path) -> list[str]:
    text = path.read_text().strip()
    return [item.strip() for item in text.split(",") if item.strip()]


def _default_candidates(args: argparse.Namespace) -> tuple[list[str], list[Path]]:
    labels = _split_csv_text(args.base_labels_file)
    dirs = [Path(item) for item in _split_csv_text(args.base_dirs_file)]
    for label, directory in PHASE19_EXTRAS:
        labels.append(label)
        dirs.append(directory)
    if len(args.extra_label) != len(args.extra_dir):
        raise SystemExit("--extra-label count must match --extra-dir count")
    for label, directory in zip(args.extra_label, args.extra_dir, strict=True):
        labels.append(str(label))
        dirs.append(Path(directory))
    if len(labels) != len(dirs):
        raise SystemExit(f"labels={len(labels)} dirs={len(dirs)}")
    return labels, dirs


def _load_candidates(
    labels: list[str],
    dirs: list[Path],
    *,
    city: str,
    run: str,
) -> list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]]:
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]] = []
    pos_name = f"{city}_{run}_full.pos"
    csv_name = f"{city}_{run}_full.csv"
    for label, directory in zip(labels, dirs, strict=True):
        pos_path = directory / pos_name
        diag_path = directory / csv_name
        if not pos_path.is_file() or not diag_path.is_file():
            print(f"skip {label}: missing {pos_path} or {diag_path}", flush=True)
            continue
        pos, _status = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        candidates.append((label, pos, diag))
    return candidates


def _effective_config(args: argparse.Namespace) -> CTRBPFConfig:
    base = CTRBPFConfig(
        enable_rtkdiag_pf_rescue=True,
        rtkdiag_candidate_select_mode="ranker",
        rtkdiag_candidate_ratio_min=float(args.ratio_min),
        rtkdiag_candidate_residual_rms_max=float(args.residual_rms_max),
        rtkdiag_candidate_emit_mode="candidate",
        rtkdiag_candidate_fallback_mode="hybrid",
        rtkdiag_candidate_rms_prefilter_k=int(args.rms_prefilter_k),
        rtkdiag_candidate_max_to_hybrid_m=0.0,
        rtkdiag_candidate_recenter_max_shift_m=10000.0,
        rtkdiag_candidate_emit_max_diff_m=0.4,
        rtkdiag_candidate_bridge_enable=True,
        rtkdiag_candidate_bridge_max_s=6.0,
        rtkdiag_candidate_bridge_residual_rms_m=0.2,
    )
    cfg = _apply_rtkdiag_run_index_policy(
        base,
        run=str(args.run),
        policy=str(args.policy),
        city=str(args.city),
    )
    return CTRBPFConfig(
        **{
            **cfg.__dict__,
            "rtkdiag_candidate_select_mode": "ranker",
            "rtkdiag_candidate_rms_prefilter_k": int(args.rms_prefilter_k),
            "rtkdiag_candidate_ratio_min": float(args.ratio_min),
            "rtkdiag_candidate_residual_rms_max": float(args.residual_rms_max),
            "rtkdiag_candidate_max_to_hybrid_m": 0.0,
        }
    )


def _candidate_options(
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]],
    *,
    tow: float,
    cfg: CTRBPFConfig,
) -> list[tuple[str, np.ndarray, dict[str, str]]]:
    out: list[tuple[str, np.ndarray, dict[str, str]]] = []
    for label, pos_lookup, diag_lookup in candidates:
        row = diag_lookup.get(tow)
        if not _rtkdiag_candidate_gate(
            row,
            ratio_min=float(cfg.rtkdiag_candidate_ratio_min),
            residual_rms_max=float(cfg.rtkdiag_candidate_residual_rms_max),
            status5_residual_rms_max=float(cfg.rtkdiag_candidate_main_status5_residual_rms_max),
        ):
            continue
        if not _rtkdiag_candidate_diag_policy_gate(
            row,
            require_any_fields=tuple(cfg.rtkdiag_candidate_require_any_diag_fields),
            require_all_fields=tuple(cfg.rtkdiag_candidate_require_all_diag_fields),
            min_fields=tuple(cfg.rtkdiag_candidate_min_diag_fields),
            max_fields=tuple(cfg.rtkdiag_candidate_max_diag_fields),
        ):
            continue
        pos = pos_lookup.get(tow)
        if pos is None or not np.all(np.isfinite(pos)) or np.all(np.asarray(pos) == 0.0):
            continue
        out.append((label, np.asarray(pos, dtype=np.float64), row))
    k = int(cfg.rtkdiag_candidate_rms_prefilter_k)
    if k > 0 and len(out) > k:
        out = sorted(out, key=lambda item: _diag_float(item[2], "final_residual_rms"))[:k]
    return out


def _make_spans(wrongs: pd.DataFrame, gap_s: float = 0.21) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if wrongs.empty:
        return pd.DataFrame(rows)
    ordered = wrongs.sort_values("tow").reset_index(drop=True)
    start = 0
    for i in range(1, len(ordered) + 1):
        if i < len(ordered) and float(ordered.loc[i, "tow"] - ordered.loc[i - 1, "tow"]) <= gap_s:
            continue
        span = ordered.iloc[start:i]
        rows.append(
            {
                "start_tow": float(span["tow"].iloc[0]),
                "end_tow": float(span["tow"].iloc[-1]),
                "n": int(len(span)),
                "path_w": float(span["path_weight"].sum()),
                "pick_label_mode": str(span["pick_label"].mode().iat[0]),
                "best_label_mode": str(span["best_label"].mode().iat[0]),
                "median_pick_err": float(span["pick_err_m"].median()),
                "median_best_err": float(span["best_err_m"].median()),
            }
        )
        start = i
    return pd.DataFrame(rows).sort_values("path_w", ascending=False)


def main() -> None:
    args = _parse_args()
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
    truth = np.asarray([ecef for _, ecef in ref], dtype=np.float64)
    weights = ppc_segment_distances(truth)
    hybrid_pos, _hybrid_status = _load_hybrid_pos_file(
        args.hybrid_pos_dir / f"{city}_{run}_full.pos"
    )

    selected = np.zeros_like(truth)
    oracle = np.zeros_like(truth)
    wrong_rows: list[dict[str, object]] = []
    no_pass_rows: list[dict[str, object]] = []
    selected_labels: Counter[str] = Counter()
    oracle_labels: Counter[str] = Counter()
    gated_epochs = 0

    for i, (tow_raw, true_pos) in enumerate(ref):
        tow = round(float(tow_raw), 1)
        hp = hybrid_pos.get(tow)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp) == 0.0):
            selected[i] = np.asarray(hp, dtype=np.float64)
            oracle[i] = np.asarray(hp, dtype=np.float64)

        options = _candidate_options(candidates, tow=tow, cfg=cfg)
        if not options:
            selected_labels["hybrid"] += 1
            oracle_labels["hybrid"] += 1
            continue
        gated_epochs += 1

        scored = [
            (float(pred_run.get((tow, label), -np.inf)), label, pos, row)
            for label, pos, row in options
        ]
        with_pred = [item for item in scored if np.isfinite(item[0])]
        if with_pred:
            pick_score, pick_label, pick_pos, pick_diag = max(with_pred, key=lambda item: item[0])
        else:
            pick_score, pick_label, pick_pos, pick_diag = min(
                scored,
                key=lambda item: _diag_float(item[3], "final_residual_rms"),
            )
        selected[i] = pick_pos
        selected_labels[pick_label] += 1

        best_score, best_label, best_pos, best_diag = min(
            scored,
            key=lambda item: float(np.linalg.norm(item[2] - true_pos)),
        )
        oracle[i] = best_pos
        oracle_labels[best_label] += 1

        pick_err = float(np.linalg.norm(pick_pos - true_pos))
        best_err = float(np.linalg.norm(best_pos - true_pos))
        if best_err >= 0.5:
            no_pass_rows.append(
                {
                    "tow": tow,
                    "path_weight": float(weights[i]),
                    "best_label": best_label,
                    "best_err_m": best_err,
                }
            )
            continue
        if pick_err >= 0.5:
            wrong_rows.append(
                {
                    "tow": tow,
                    "path_weight": float(weights[i]),
                    "pick_label": pick_label,
                    "best_label": best_label,
                    "pick_err_m": pick_err,
                    "best_err_m": best_err,
                    "pick_p_pass": float(pick_score) if np.isfinite(pick_score) else np.nan,
                    "best_p_pass": float(best_score) if np.isfinite(best_score) else np.nan,
                    "pick_rms": _diag_float(pick_diag, "final_residual_rms"),
                    "best_rms": _diag_float(best_diag, "final_residual_rms"),
                    "pick_status": _diag_float(pick_diag, "final_status"),
                    "best_status": _diag_float(best_diag, "final_status"),
                    "n_options": len(options),
                }
            )

    selected_score = score_ppc2024(selected, truth)
    oracle_score = score_ppc2024(oracle, truth)
    wrongs = pd.DataFrame(wrong_rows)
    no_pass = pd.DataFrame(no_pass_rows)
    spans = _make_spans(wrongs)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    wrongs.to_csv(args.out_csv, index=False)
    spans.to_csv(args.span_csv, index=False)

    print(f"loaded_candidates={len(candidates_all)} after_policy={len(candidates)}")
    print(f"ranker_entries={len(pred_run)} gated_epochs={gated_epochs} epochs={len(ref)}")
    print(
        "selected_ppc="
        f"{selected_score.score_pct:.6f} pass={selected_score.pass_distance_m:.3f}/"
        f"{selected_score.total_distance_m:.3f}"
    )
    print(
        "oracle_ppc="
        f"{oracle_score.score_pct:.6f} pass={oracle_score.pass_distance_m:.3f}/"
        f"{oracle_score.total_distance_m:.3f}"
    )
    if not wrongs.empty:
        print(
            "recoverable_wrong="
            f"{len(wrongs)} path_w={wrongs['path_weight'].sum():.3f} "
            f"pct={100.0 * wrongs['path_weight'].sum() / selected_score.total_distance_m:.4f}"
        )
        print("top wrong pick labels:")
        for label, count in Counter(wrongs["pick_label"]).most_common(15):
            path_w = float(wrongs.loc[wrongs["pick_label"] == label, "path_weight"].sum())
            print(f"  {label:28s} n={count:5d} path_w={path_w:9.3f}")
        print("top missed pass labels:")
        for label, count in Counter(wrongs["best_label"]).most_common(15):
            path_w = float(wrongs.loc[wrongs["best_label"] == label, "path_weight"].sum())
            print(f"  {label:28s} n={count:5d} path_w={path_w:9.3f}")
        print("top wrong-pick spans:")
        print(spans.head(20).to_string(index=False))
    if not no_pass.empty:
        print(
            "pool_no_pass="
            f"{len(no_pass)} path_w={no_pass['path_weight'].sum():.3f} "
            f"pct={100.0 * no_pass['path_weight'].sum() / selected_score.total_distance_m:.4f}"
        )
    print("top selected labels:")
    for label, count in selected_labels.most_common(12):
        print(f"  {label:28s} n={count:5d}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.span_csv}")


if __name__ == "__main__":
    main()
