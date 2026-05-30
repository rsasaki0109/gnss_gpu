#!/usr/bin/env python3
"""Mine non-reference signals behind Phase 42 n/r2 span-local rules.

Phase 42 is an oracle diagnostic: it forces labels on short TOW spans using
truth.  This script keeps truth only for measuring pass/fail and gain, then
reports features that would have been available without truth:

- ranker score/rank and residual-RMS rank of the forced label
- diagnostic deltas versus the ranker pick
- candidate-family agreement and local position-cluster support

The output is meant to answer whether the Phase 42 rules share a deployable
trigger, not to create another truth-derived selector.
"""

from __future__ import annotations

import argparse
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
    _load_ranker_predictions,
)
from gnss_gpu.ppc_score import ppc_segment_distances  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--rules",
        type=Path,
        default=RESULTS / "nr2_phase42_span20_rules.csv",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=RESULTS
        / "selector_ranker_predictions_v3_nr2_labelboost_pb40_motionguard_span2_tf.csv",
        help="Baseline predictions before applying the Phase 42 span20 rules.",
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
        "--out-epochs",
        type=Path,
        default=RESULTS / "nr2_phase42_nonref_trigger_epochs.csv",
    )
    parser.add_argument(
        "--out-spans",
        type=Path,
        default=RESULTS / "nr2_phase42_nonref_trigger_spans.csv",
    )
    return parser.parse_args()


def _family(label: str) -> str:
    if label.startswith("xd_gici_"):
        return "xd_gici"
    if label.startswith("xd_fgo_"):
        return "xd_fgo"
    if label.startswith("xd_fixedicb_"):
        return "fixedicb"
    if label.startswith("xd_tdcp_"):
        return "tdcp"
    if label.startswith("xd_nr2_"):
        return "nr2_oracle"
    if label.startswith("rtkout"):
        return "rtkout"
    if label.startswith("mlc"):
        return "mlc"
    if label.startswith("c"):
        return "csig"
    return label.split("_", 1)[0]


def _rank(labels_scores: list[tuple[str, float]], target: str, *, reverse: bool) -> float:
    ordered = sorted(labels_scores, key=lambda item: item[1], reverse=reverse)
    for i, (label, _score) in enumerate(ordered, start=1):
        if label == target:
            return float(i)
    return float("nan")


def _cluster_count(
    positions: dict[str, np.ndarray],
    center_label: str,
    radius_m: float,
    *,
    family: str | None = None,
) -> int:
    center = positions.get(center_label)
    if center is None:
        return 0
    count = 0
    for label, pos in positions.items():
        if family is not None and _family(label) != family:
            continue
        if float(np.linalg.norm(pos - center)) <= float(radius_m):
            count += 1
    return count


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
    truth_by_tow = {round(float(tow), 1): np.asarray(pos, dtype=np.float64) for tow, pos in ref}
    truth_arr = np.asarray([pos for _, pos in ref], dtype=np.float64)
    weight_by_tow = {
        round(float(tow), 1): float(w)
        for (tow, _pos), w in zip(ref, ppc_segment_distances(truth_arr), strict=True)
    }

    rules = pd.read_csv(args.rules)
    epoch_rows: list[dict[str, object]] = []

    for rule in rules.itertuples(index=False):
        start_tow = round(float(rule.start_tow), 1)
        end_tow = round(float(rule.end_tow), 1)
        forced_label = str(rule.forced_label)
        span_id = f"{start_tow:.1f}-{end_tow:.1f}"
        tows = sorted(
            tow for tow in truth_by_tow.keys() if start_tow - 0.05 <= tow <= end_tow + 0.05
        )
        for tow in tows:
            options = _candidate_options(candidates, tow=tow, cfg=cfg)
            if not options:
                continue
            pos_by_label = {label: pos for label, pos, _diag in options}
            diag_by_label = {label: diag for label, _pos, diag in options}
            scored = [
                (label, float(pred_run.get((tow, label), -np.inf)))
                for label, _pos, _diag in options
            ]
            with_pred = [(label, score) for label, score in scored if np.isfinite(score)]
            if with_pred:
                pick_label, pick_p = max(with_pred, key=lambda item: item[1])
            else:
                pick_label, pick_p = min(
                    (
                        (label, _diag_float(diag, "final_residual_rms"))
                        for label, _pos, diag in options
                    ),
                    key=lambda item: item[1],
                )
            if forced_label not in pos_by_label:
                epoch_rows.append(
                    {
                        "span": span_id,
                        "tow": tow,
                        "path_weight": weight_by_tow.get(tow, 0.0),
                        "forced_label": forced_label,
                        "forced_present": 0,
                        "pick_label": pick_label,
                        "n_options": len(options),
                    }
                )
                continue

            truth = truth_by_tow[tow]
            forced_pos = pos_by_label[forced_label]
            pick_pos = pos_by_label[pick_label]
            forced_diag = diag_by_label[forced_label]
            pick_diag = diag_by_label[pick_label]
            forced_p = float(pred_run.get((tow, forced_label), np.nan))
            forced_rms = _diag_float(forced_diag, "final_residual_rms")
            pick_rms = _diag_float(pick_diag, "final_residual_rms")
            rms_scores = [
                (label, _diag_float(diag, "final_residual_rms"))
                for label, _pos, diag in options
            ]
            median_pos = np.median(np.stack(list(pos_by_label.values()), axis=0), axis=0)
            forced_family = _family(forced_label)
            pick_family = _family(pick_label)
            epoch_rows.append(
                {
                    "span": span_id,
                    "tow": tow,
                    "path_weight": weight_by_tow.get(tow, 0.0),
                    "forced_label": forced_label,
                    "forced_family": forced_family,
                    "forced_present": 1,
                    "pick_label": pick_label,
                    "pick_family": pick_family,
                    "same_family": int(forced_family == pick_family),
                    "n_options": len(options),
                    "pick_err_m": float(np.linalg.norm(pick_pos - truth)),
                    "forced_err_m": float(np.linalg.norm(forced_pos - truth)),
                    "pick_pass": int(float(np.linalg.norm(pick_pos - truth)) < 0.5),
                    "forced_pass": int(float(np.linalg.norm(forced_pos - truth)) < 0.5),
                    "pick_p_pass": pick_p if np.isfinite(pick_p) else np.nan,
                    "forced_p_pass": forced_p,
                    "p_delta_forced_minus_pick": forced_p - pick_p
                    if np.isfinite(forced_p) and np.isfinite(pick_p)
                    else np.nan,
                    "forced_p_rank": _rank(scored, forced_label, reverse=True),
                    "pick_rms": pick_rms,
                    "forced_rms": forced_rms,
                    "rms_delta_forced_minus_pick": forced_rms - pick_rms,
                    "forced_rms_rank": _rank(rms_scores, forced_label, reverse=False),
                    "pick_status": _diag_float(pick_diag, "final_status"),
                    "forced_status": _diag_float(forced_diag, "final_status"),
                    "pick_ratio": _diag_float(pick_diag, "final_ratio"),
                    "forced_ratio": _diag_float(forced_diag, "final_ratio"),
                    "dist_forced_to_pick_m": float(np.linalg.norm(forced_pos - pick_pos)),
                    "dist_forced_to_median_m": float(np.linalg.norm(forced_pos - median_pos)),
                    "dist_pick_to_median_m": float(np.linalg.norm(pick_pos - median_pos)),
                    "forced_cluster_025": _cluster_count(pos_by_label, forced_label, 0.25),
                    "forced_cluster_050": _cluster_count(pos_by_label, forced_label, 0.50),
                    "pick_cluster_025": _cluster_count(pos_by_label, pick_label, 0.25),
                    "pick_cluster_050": _cluster_count(pos_by_label, pick_label, 0.50),
                    "forced_family_cluster_050": _cluster_count(
                        pos_by_label, forced_label, 0.50, family=forced_family
                    ),
                    "pick_family_cluster_050": _cluster_count(
                        pos_by_label, pick_label, 0.50, family=pick_family
                    ),
                }
            )

    epochs = pd.DataFrame(epoch_rows)
    if epochs.empty:
        raise SystemExit("no epoch rows produced")

    args.out_epochs.parent.mkdir(parents=True, exist_ok=True)
    epochs.to_csv(args.out_epochs, index=False)

    present = epochs.loc[epochs["forced_present"] == 1].copy()
    span_rows: list[dict[str, object]] = []
    for span, group in present.groupby("span", sort=False):
        path = float(group["path_weight"].sum())
        forced_gain_path = float(
            group.loc[(group["forced_pass"] == 1) & (group["pick_pass"] == 0), "path_weight"].sum()
        )
        forced_loss_path = float(
            group.loc[(group["forced_pass"] == 0) & (group["pick_pass"] == 1), "path_weight"].sum()
        )
        span_rows.append(
            {
                "span": span,
                "n_epochs": int(len(group)),
                "path_weight": path,
                "forced_label_mode": str(group["forced_label"].mode().iat[0]),
                "pick_label_mode": str(group["pick_label"].mode().iat[0]),
                "same_family_frac": float(group["same_family"].mean()),
                "forced_pass_path": float(group.loc[group["forced_pass"] == 1, "path_weight"].sum()),
                "pick_pass_path": float(group.loc[group["pick_pass"] == 1, "path_weight"].sum()),
                "net_gain_path": forced_gain_path - forced_loss_path,
                "median_p_delta": float(group["p_delta_forced_minus_pick"].median()),
                "median_forced_p_rank": float(group["forced_p_rank"].median()),
                "median_rms_delta": float(group["rms_delta_forced_minus_pick"].median()),
                "median_forced_rms_rank": float(group["forced_rms_rank"].median()),
                "median_dist_forced_to_pick_m": float(group["dist_forced_to_pick_m"].median()),
                "median_dist_forced_to_median_m": float(group["dist_forced_to_median_m"].median()),
                "median_forced_cluster_050": float(group["forced_cluster_050"].median()),
                "median_pick_cluster_050": float(group["pick_cluster_050"].median()),
            }
        )
    spans = pd.DataFrame(span_rows).sort_values("net_gain_path", ascending=False)
    spans.to_csv(args.out_spans, index=False)

    total_gain = float(spans["net_gain_path"].sum())
    total_path = float(present["path_weight"].sum())
    print(f"loaded_candidates={len(candidates_all)} after_policy={len(candidates)}")
    print(f"rules={len(rules)} epoch_rows={len(epochs)} present={len(present)}")
    print(f"net_gain_path={total_gain:.3f}m over_rule_path={total_path:.3f}m")
    print("top span non-reference summary:")
    cols = [
        "span",
        "forced_label_mode",
        "pick_label_mode",
        "net_gain_path",
        "median_p_delta",
        "median_forced_p_rank",
        "median_rms_delta",
        "median_forced_rms_rank",
        "median_dist_forced_to_pick_m",
        "median_forced_cluster_050",
    ]
    print(spans[cols].head(15).to_string(index=False))
    print(f"wrote {args.out_epochs}")
    print(f"wrote {args.out_spans}")


if __name__ == "__main__":
    main()
