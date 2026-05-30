#!/usr/bin/env python3
"""Probe Phase77 label-pair detector rules on full n/r2."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from audit_nr2_profile_span_oracle import (  # noqa: E402
    _discover_candidates,
    _distance_weights,
    _float,
    _read_csv,
    _truth_xyz,
    _write_csv,
)
from exp_ppc_ctrbpf_fgo import _diag_float, _rtkdiag_candidate_gate  # noqa: E402


def _base_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    if label:
        return label
    label = row.get("rtkdiag_selected_label", "").removesuffix("+rnk")
    return label or row.get("emitted_source", "")


def _xyz_from_row(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [
            _float(row, f"{prefix}_x"),
            _float(row, f"{prefix}_y"),
            _float(row, f"{prefix}_z"),
        ],
        dtype=np.float64,
    )


def _quantile_grid(values: list[float], *, precision: int = 3) -> list[float]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return []
    qs = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    out = {round(finite[min(len(finite) - 1, int(q * (len(finite) - 1)))], precision) for q in qs}
    return sorted(out)


def _top_counts(counter: Counter[str], limit: int = 8) -> str:
    return ",".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def _read_pairs(path: Path, top: int) -> list[tuple[str, str, float]]:
    pairs: list[tuple[str, str, float]] = []
    for row in _read_csv(path):
        gain = _float(row, "oracle_recoverable_gain_m")
        if not math.isfinite(gain) or gain <= 0.0:
            continue
        pairs.append((row["selected_label"], row["oracle_label"], gain))
        if len(pairs) >= top:
            break
    return pairs


def _prepare_opportunities(
    rows: list[dict[str, str]],
    weights: list[float],
    candidates: list[dict[str, Any]],
    pairs: list[tuple[str, str, float]],
    *,
    pass_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    by_label = {str(cand["label"]): cand for cand in candidates}
    pair_keys = [(selected, target) for selected, target, _gain in pairs]
    row_records: list[dict[str, Any]] = []
    opportunities: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row_index, row in enumerate(rows):
        epoch = int(float(row["epoch"]))
        tow = round(float(row["tow"]), 1)
        truth = _truth_xyz(row)
        current_pos = _xyz_from_row(row, "pf_before_emit")
        current_error = _float(row, "emit_to_ref_m")
        current_pass = math.isfinite(current_error) and current_error <= pass_m
        selected = _base_label(row)
        row_record = {
            "row_index": row_index,
            "epoch": epoch,
            "tow": tow,
            "weight_m": weights[row_index],
            "selected_label": selected,
            "current_error_m": current_error,
            "current_pass": current_pass,
            "family_span_m": _float(row, "rtkdiag_candidate_family_span_m"),
            "agreement_1m": _float(row, "rtkdiag_candidate_agreement_count_1m"),
            "selected_rms": _float(row, "rtkdiag_selected_diag_rms"),
            "selected_absmax": _float(row, "rtkdiag_selected_diag_abs_max"),
            "selected_sats": _float(row, "rtkdiag_selected_diag_sats"),
            "selected_ratio": _float(row, "rtkdiag_selected_diag_ratio"),
        }
        row_records.append(row_record)

        for selected_label, target_label in pair_keys:
            if selected != selected_label:
                continue
            cand = by_label.get(target_label)
            if cand is None:
                continue
            pos = cand["pos"].get(tow)
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            if not np.all(np.isfinite(pos)):
                continue
            diag = cand["diag"].get(tow)
            if not _rtkdiag_candidate_gate(
                diag,
                ratio_min=ratio_min,
                residual_rms_max=residual_rms_max,
                status5_residual_rms_max=status5_residual_rms_max,
            ):
                continue
            target_error = float(np.linalg.norm(pos - truth))
            target_pass = target_error <= pass_m
            delta_m = 0.0
            if target_pass and not current_pass:
                delta_m = weights[row_index]
            elif current_pass and not target_pass:
                delta_m = -weights[row_index]
            offset_m = (
                float(np.linalg.norm(pos - current_pos))
                if np.all(np.isfinite(current_pos))
                else float("nan")
            )
            opportunities[(selected_label, target_label)].append(
                {
                    **row_record,
                    "target_label": target_label,
                    "target_error_m": target_error,
                    "target_pass": target_pass,
                    "delta_m": delta_m,
                    "offset_m": offset_m,
                    "target_status": _diag_float(diag, "final_status") if diag else float("nan"),
                    "target_sats": _diag_float(diag, "final_sats") if diag else float("nan"),
                    "target_ratio": _diag_float(diag, "final_ratio") if diag else float("nan"),
                    "target_rms": _diag_float(diag, "final_residual_rms") if diag else float("nan"),
                    "target_absmax": _diag_float(diag, "final_residual_abs_max") if diag else float("nan"),
                    "target_updates": _diag_float(diag, "final_update_rows") if diag else float("nan"),
                },
            )
    return row_records, opportunities


def _passes_rule(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    offset = float(row["offset_m"])
    if not math.isfinite(offset) or offset < float(rule["offset_min_m"]) or offset > float(rule["offset_max_m"]):
        return False
    family_span = float(row["family_span_m"])
    if not math.isfinite(family_span) or family_span < float(rule["family_span_min_m"]):
        return False
    agreement = float(row["agreement_1m"])
    if math.isfinite(agreement) and agreement > float(rule["agreement_1m_max"]):
        return False
    target_rms = float(row["target_rms"])
    if math.isfinite(target_rms) and target_rms > float(rule["target_rms_max"]):
        return False
    return True


def _score_rows(rows: list[dict[str, Any]], rule: dict[str, Any] | None = None) -> dict[str, Any]:
    gain_m = 0.0
    overrides = 0
    good_epochs = 0
    bad_epochs = 0
    target_pass_m = 0.0
    target_total_m = 0.0
    for row in rows:
        if rule is not None and not _passes_rule(row, rule):
            continue
        overrides += 1
        gain_m += float(row["delta_m"])
        target_total_m += float(row["weight_m"])
        if bool(row["target_pass"]):
            target_pass_m += float(row["weight_m"])
        if bool(row["target_pass"]) and not bool(row["current_pass"]):
            good_epochs += 1
        elif bool(row["current_pass"]) and not bool(row["target_pass"]):
            bad_epochs += 1
    return {
        "gain_m": gain_m,
        "overrides": overrides,
        "good_epochs": good_epochs,
        "bad_epochs": bad_epochs,
        "target_pass_m": target_pass_m,
        "target_total_m": target_total_m,
    }


def _candidate_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    offsets = _quantile_grid([float(row["offset_m"]) for row in rows])
    target_rms = _quantile_grid([float(row["target_rms"]) for row in rows])
    offset_min_values = sorted({0.0, *offsets[:4]})
    offset_max_values = sorted({*offsets[3:], 999.0})
    family_span_min_values = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0)
    agreement_max_values = (4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 99.0)
    target_rms_max_values = sorted({0.1, 0.13, 0.15, 0.2, 0.25, 0.3, 0.5, 50.0, *target_rms})

    rules: list[dict[str, Any]] = []
    for offset_min in offset_min_values:
        for offset_max in offset_max_values:
            if offset_min > offset_max:
                continue
            for family_span_min in family_span_min_values:
                for agreement_max in agreement_max_values:
                    for target_rms_max in target_rms_max_values:
                        rules.append(
                            {
                                "offset_min_m": offset_min,
                                "offset_max_m": offset_max,
                                "family_span_min_m": family_span_min,
                                "agreement_1m_max": agreement_max,
                                "target_rms_max": target_rms_max,
                            },
                        )
    return rules


def _best_pair_rules(
    pairs: list[tuple[str, str, float]],
    opportunities: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    full_total_m: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for selected_label, target_label, oracle_gain in pairs:
        key = (selected_label, target_label)
        rows = opportunities.get(key, [])
        raw = _score_rows(rows)
        best_rule: dict[str, Any] | None = None
        best = {"gain_m": float("-inf")}
        for rule in _candidate_rules(rows):
            scored = _score_rows(rows, rule)
            if float(scored["gain_m"]) > float(best["gain_m"]):
                best = scored
                best_rule = rule
        if best_rule is None:
            best_rule = {
                "offset_min_m": "",
                "offset_max_m": "",
                "family_span_min_m": "",
                "agreement_1m_max": "",
                "target_rms_max": "",
            }
            best = _score_rows([])
        out.append(
            {
                "selected_label": selected_label,
                "target_label": target_label,
                "phase77_oracle_gain_m": oracle_gain,
                "opportunity_epochs": len(rows),
                "raw_gain_m": raw["gain_m"],
                "raw_overrides": raw["overrides"],
                "raw_good_epochs": raw["good_epochs"],
                "raw_bad_epochs": raw["bad_epochs"],
                "best_gain_m": best["gain_m"],
                "best_n2_delta_pp": 100.0 * float(best["gain_m"]) / full_total_m if full_total_m > 0.0 else "",
                "best_official_delta_pp": 100.0 * float(best["gain_m"]) / full_total_m / 6.0
                if full_total_m > 0.0
                else "",
                "best_overrides": best["overrides"],
                "best_good_epochs": best["good_epochs"],
                "best_bad_epochs": best["bad_epochs"],
                "best_target_pass_m": best["target_pass_m"],
                **best_rule,
            },
        )
    out.sort(key=lambda row: float(row["best_gain_m"]), reverse=True)
    return out


def _rule_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "offset_min_m": float(row["offset_min_m"]),
        "offset_max_m": float(row["offset_max_m"]),
        "family_span_min_m": float(row["family_span_min_m"]),
        "agreement_1m_max": float(row["agreement_1m_max"]),
        "target_rms_max": float(row["target_rms_max"]),
    }


def _combo_replay(
    pair_rules: list[dict[str, Any]],
    opportunities: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    base_pass_m: float,
    full_total_m: float,
    max_rules_values: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rule_row in pair_rules:
        if float(rule_row["best_gain_m"]) <= 0.0:
            continue
        pair = (str(rule_row["selected_label"]), str(rule_row["target_label"]))
        rule = _rule_from_row(rule_row)
        for opp in opportunities.get(pair, []):
            if _passes_rule(opp, rule):
                by_row[int(opp["row_index"])].append(
                    {
                        **opp,
                        "rule_selected_label": pair[0],
                        "rule_target_label": pair[1],
                        "rule_gain_rank": 0,
                    },
                )

    combo_rows: list[dict[str, Any]] = []
    chosen_by_combo: dict[str, dict[str, Any]] = {}
    positive_rules = [row for row in pair_rules if float(row["best_gain_m"]) > 0.0]
    for max_rules in max_rules_values:
        active = positive_rules[:max_rules]
        active_pairs = {(str(row["selected_label"]), str(row["target_label"])): rank for rank, row in enumerate(active)}
        replay_pass_m = base_pass_m
        overrides = 0
        good_epochs = 0
        bad_epochs = 0
        chosen_labels: Counter[str] = Counter()
        chosen_rows: list[dict[str, Any]] = []
        for row_index, choices in by_row.items():
            eligible = [
                choice
                for choice in choices
                if (str(choice["rule_selected_label"]), str(choice["rule_target_label"])) in active_pairs
            ]
            if not eligible:
                continue
            eligible.sort(
                key=lambda choice: active_pairs[
                    (str(choice["rule_selected_label"]), str(choice["rule_target_label"]))
                ],
            )
            chosen = dict(eligible[0])
            chosen["rule_gain_rank"] = active_pairs[
                (str(chosen["rule_selected_label"]), str(chosen["rule_target_label"]))
            ]
            replay_pass_m += float(chosen["delta_m"])
            overrides += 1
            chosen_labels[str(chosen["rule_target_label"])] += 1
            if bool(chosen["target_pass"]) and not bool(chosen["current_pass"]):
                good_epochs += 1
            elif bool(chosen["current_pass"]) and not bool(chosen["target_pass"]):
                bad_epochs += 1
            chosen_rows.append(chosen)
        gain_m = replay_pass_m - base_pass_m
        key = f"top{max_rules}"
        combo_rows.append(
            {
                "combo": key,
                "rules_used": len(active),
                "base_pass_m": base_pass_m,
                "replay_pass_m": replay_pass_m,
                "gain_m": gain_m,
                "n2_delta_pp": 100.0 * gain_m / full_total_m if full_total_m > 0.0 else "",
                "official_delta_pp": 100.0 * gain_m / full_total_m / 6.0 if full_total_m > 0.0 else "",
                "overrides": overrides,
                "good_epochs": good_epochs,
                "bad_epochs": bad_epochs,
                "chosen_labels": _top_counts(chosen_labels),
            },
        )
        chosen_by_combo[key] = {str(row["row_index"]): row for row in chosen_rows}
    combo_rows.sort(key=lambda row: float(row["gain_m"]), reverse=True)
    best_key = str(combo_rows[0]["combo"]) if combo_rows else ""
    return combo_rows, chosen_by_combo.get(best_key, {}), positive_rules


def _epoch_output(chosen: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(chosen.values(), key=lambda item: int(item["epoch"])):
        rows.append(
            {
                "epoch": row["epoch"],
                "tow": row["tow"],
                "weight_m": row["weight_m"],
                "selected_label": row["selected_label"],
                "target_label": row["rule_target_label"],
                "current_error_m": row["current_error_m"],
                "target_error_m": row["target_error_m"],
                "delta_m": row["delta_m"],
                "current_pass": row["current_pass"],
                "target_pass": row["target_pass"],
                "offset_m": row["offset_m"],
                "family_span_m": row["family_span_m"],
                "agreement_1m": row["agreement_1m"],
                "target_rms": row["target_rms"],
                "rule_gain_rank": row["rule_gain_rank"],
            },
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-epochs-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv"),
    )
    parser.add_argument(
        "--phase77-label-pairs-csv",
        type=Path,
        default=Path("experiments/results/phase77_nr2_selector_recovery_bucket_label_pairs.csv"),
    )
    parser.add_argument("--search-root", action="append", type=Path, default=[Path("experiments/results")])
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--top-pairs", type=int, default=30)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument(
        "--exclude-label-contains",
        action="append",
        default=["oracle", "anchortruth", "phase74"],
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/phase78_nr2_label_pair_detector"),
    )
    args = parser.parse_args(argv)

    rows = _read_csv(args.internal_epochs_csv)
    weights = _distance_weights(rows)
    full_total_m = sum(weights)
    base_pass_m = sum(
        weight
        for weight, row in zip(weights, rows)
        if math.isfinite(_float(row, "emit_to_ref_m")) and _float(row, "emit_to_ref_m") <= float(args.pass_m)
    )
    pairs = _read_pairs(args.phase77_label_pairs_csv, int(args.top_pairs))
    candidates, excluded = _discover_candidates(
        args.search_root,
        city=args.city,
        run=args.run,
        exclude_label_contains=list(args.exclude_label_contains),
    )
    print(f"loaded profile candidates: {len(candidates)}")
    if excluded:
        print(f"excluded profile candidates: {excluded}")
    print(f"loaded pairs: {len(pairs)}")

    _row_records, opportunities = _prepare_opportunities(
        rows,
        weights,
        candidates,
        pairs,
        pass_m=float(args.pass_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    pair_rows = _best_pair_rules(pairs, opportunities, full_total_m=full_total_m)
    combo_rows, best_chosen, positive_rules = _combo_replay(
        pair_rows,
        opportunities,
        base_pass_m=base_pass_m,
        full_total_m=full_total_m,
        max_rules_values=(1, 2, 3, 5, 10, 15, 20, 30),
    )

    best_pair = pair_rows[0] if pair_rows else {}
    best_combo = combo_rows[0] if combo_rows else {}
    summary_rows = [
        {
            "scope": "phase78_base",
            "base_pass_m": base_pass_m,
            "full_total_m": full_total_m,
            "base_score_pct": 100.0 * base_pass_m / full_total_m if full_total_m > 0.0 else "",
            "top_pairs": len(pairs),
            "positive_pair_rules": len(positive_rules),
        },
        {
            "scope": "best_single_pair_rule",
            "selected_label": best_pair.get("selected_label", ""),
            "target_label": best_pair.get("target_label", ""),
            "gain_m": best_pair.get("best_gain_m", ""),
            "n2_delta_pp": best_pair.get("best_n2_delta_pp", ""),
            "official_delta_pp": best_pair.get("best_official_delta_pp", ""),
            "overrides": best_pair.get("best_overrides", ""),
            "good_epochs": best_pair.get("best_good_epochs", ""),
            "bad_epochs": best_pair.get("best_bad_epochs", ""),
            "rule": ",".join(
                f"{key}={best_pair.get(key, '')}"
                for key in (
                    "offset_min_m",
                    "offset_max_m",
                    "family_span_min_m",
                    "agreement_1m_max",
                    "target_rms_max",
                )
            ),
        },
        {
            "scope": "best_greedy_combo",
            **best_combo,
        },
    ]

    _write_csv(Path(f"{args.out_prefix}_summary.csv"), summary_rows)
    _write_csv(Path(f"{args.out_prefix}_pair_sweep.csv"), pair_rows)
    _write_csv(Path(f"{args.out_prefix}_combo_sweep.csv"), combo_rows)
    _write_csv(Path(f"{args.out_prefix}_best_combo_epochs.csv"), _epoch_output(best_chosen))

    if best_pair:
        print("best pair:", best_pair)
    if best_combo:
        print("best combo:", best_combo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
