#!/usr/bin/env python3
"""Analyze Phase73 n/r2 selector-recovery bucket with deployable signals."""

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


STRATEGIES = (
    "min_rms",
    "min_absmax",
    "max_ratio",
    "max_sats",
    "max_update_rows",
    "nearest_current",
    "nearest_pf_init",
)

PHASE59_LABELS = {
    "xd_gici_w5",
    "xd_gici_ir",
    "xd_gici_z",
    "xd_gici_zr",
    "xd_gici_mb",
    "xd_gici_r4",
    "xd_gici_combo",
    "xd_gici_he",
    "xd_gici_la",
}

RECOVERABLE_LABELS = PHASE59_LABELS | {
    "xd_gici_r",
    "xd_gici_def",
    "xd_gici_lprlph",
    "xd_fgo_v14_snr38",
}


def _finite(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


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


def _top_counts(counter: Counter[str], limit: int = 8) -> str:
    return ",".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def _update_best(
    best: dict[str, dict[str, Any] | None],
    option: dict[str, Any],
    *,
    current_pos: np.ndarray,
    pf_init_pos: np.ndarray,
) -> None:
    label = str(option["label"])

    def replace(strategy: str, key: tuple[float, ...]) -> None:
        current = best.get(strategy)
        if current is None or key < current["_rank_key"]:
            best[strategy] = {**option, "_rank_key": key}

    rms = _finite(float(option["rms"]), 1.0e9)
    absmax = _finite(float(option["absmax"]), 1.0e9)
    ratio = _finite(float(option["ratio"]), -1.0e9)
    sats = _finite(float(option["sats"]), -1.0e9)
    updates = _finite(float(option["updates"]), -1.0e9)
    current_dist = (
        float(np.linalg.norm(option["pos"] - current_pos))
        if np.all(np.isfinite(current_pos))
        else 1.0e9
    )
    init_dist = (
        float(np.linalg.norm(option["pos"] - pf_init_pos))
        if np.all(np.isfinite(pf_init_pos))
        else 1.0e9
    )

    replace("min_rms", (rms, absmax, label))
    replace("min_absmax", (absmax, rms, label))
    replace("max_ratio", (-ratio, rms, label))
    replace("max_sats", (-sats, rms, label))
    replace("max_update_rows", (-updates, rms, label))
    replace("nearest_current", (current_dist, rms, label))
    replace("nearest_pf_init", (init_dist, rms, label))


def _public_choice(choice: dict[str, Any] | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    out = dict(choice)
    out.pop("pos", None)
    out.pop("_rank_key", None)
    return out


def _prepare_records(
    rows: list[dict[str, str]],
    weights: list[float],
    candidates: list[dict[str, Any]],
    *,
    pass_m: float,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        epoch = int(float(row["epoch"]))
        tow = round(float(row["tow"]), 1)
        truth = _truth_xyz(row)
        current_pos = _xyz_from_row(row, "pf_before_emit")
        pf_init_pos = _xyz_from_row(row, "pf_init")
        current_error = _float(row, "emit_to_ref_m")
        selected = _base_label(row)
        best_oracle: dict[str, Any] | None = None
        best: dict[str, dict[str, Any] | None] = {strategy: None for strategy in STRATEGIES}
        gated_count = 0
        available_count = 0

        for cand in candidates:
            pos = cand["pos"].get(tow)
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=np.float64)
            if not np.all(np.isfinite(pos)):
                continue
            available_count += 1
            diag = cand["diag"].get(tow)
            if not _rtkdiag_candidate_gate(
                diag,
                ratio_min=ratio_min,
                residual_rms_max=residual_rms_max,
                status5_residual_rms_max=status5_residual_rms_max,
            ):
                continue
            gated_count += 1
            error_m = float(np.linalg.norm(pos - truth))
            option = {
                "label": str(cand["label"]),
                "pos": pos,
                "error_m": error_m,
                "pass": error_m <= pass_m,
                "status": _diag_float(diag, "final_status") if diag else float("nan"),
                "sats": _diag_float(diag, "final_sats") if diag else float("nan"),
                "ratio": _diag_float(diag, "final_ratio") if diag else float("nan"),
                "rms": _diag_float(diag, "final_residual_rms") if diag else float("nan"),
                "absmax": _diag_float(diag, "final_residual_abs_max") if diag else float("nan"),
                "updates": _diag_float(diag, "final_update_rows") if diag else float("nan"),
            }
            if best_oracle is None or error_m < float(best_oracle["error_m"]):
                best_oracle = dict(option)
            _update_best(best, option, current_pos=current_pos, pf_init_pos=pf_init_pos)

        records.append(
            {
                "epoch": epoch,
                "tow": tow,
                "weight_m": weights[row_index],
                "selected": selected,
                "current_error_m": current_error,
                "current_pass": math.isfinite(current_error) and current_error <= pass_m,
                "family_span_m": _float(row, "rtkdiag_candidate_family_span_m"),
                "agreement_1m": _float(row, "rtkdiag_candidate_agreement_count_1m"),
                "selected_rms": _float(row, "rtkdiag_selected_diag_rms"),
                "selected_ratio": _float(row, "rtkdiag_selected_diag_ratio"),
                "selected_sats": _float(row, "rtkdiag_selected_diag_sats"),
                "available_count": available_count,
                "gated_count": gated_count,
                "oracle": _public_choice(best_oracle),
                "choices": {strategy: _public_choice(best[strategy]) for strategy in STRATEGIES},
            },
        )
    return records


def _span_index(row: dict[str, str], fallback: int) -> int:
    raw = row.get("span_index", "")
    return int(float(raw)) if raw else fallback


def _positive_recoverable_spans(
    span_rows: list[dict[str, str]],
    *,
    skip_first_spans: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fallback, row in enumerate(span_rows, start=1):
        span_index = _span_index(row, fallback)
        gated_gain = _float(row, "gated_profile_oracle_gain_m")
        if span_index <= skip_first_spans or not math.isfinite(gated_gain) or gated_gain <= 0.0:
            continue
        out.append(
            {
                **row,
                "span_index": span_index,
                "start_epoch_i": int(float(row["start_epoch"])),
                "end_epoch_i": int(float(row["end_epoch"])),
                "gated_gain_m": gated_gain,
            },
        )
    return out


def _records_for_span(span: dict[str, Any], records_by_epoch: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        records_by_epoch[epoch]
        for epoch in range(int(span["start_epoch_i"]), int(span["end_epoch_i"]) + 1)
        if epoch in records_by_epoch
    ]


def _summarize_strategy(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    full_total_m: float,
    pass_m: float,
) -> dict[str, Any]:
    current_pass_m = 0.0
    replay_pass_m = 0.0
    total_m = 0.0
    available_epochs = 0
    good_epochs = 0
    bad_epochs = 0
    chosen_labels: Counter[str] = Counter()

    for rec in records:
        weight = float(rec["weight_m"])
        total_m += weight
        current_pass = bool(rec["current_pass"])
        if current_pass:
            current_pass_m += weight
        choice = rec["oracle"] if strategy == "gated_oracle" else rec["choices"].get(strategy)
        if choice is None:
            chosen_pass = current_pass
        else:
            available_epochs += 1
            chosen_labels[str(choice["label"])] += 1
            chosen_pass = float(choice["error_m"]) <= pass_m
            if chosen_pass and not current_pass:
                good_epochs += 1
            elif current_pass and not chosen_pass:
                bad_epochs += 1
        if chosen_pass:
            replay_pass_m += weight

    gain_m = replay_pass_m - current_pass_m
    n2_delta_pp = 100.0 * gain_m / full_total_m if full_total_m > 0.0 else float("nan")
    return {
        "strategy": strategy,
        "n_epochs": len(records),
        "total_m": total_m,
        "current_pass_m": current_pass_m,
        "replay_pass_m": replay_pass_m,
        "gain_m": gain_m,
        "score_pct_in_scope": 100.0 * replay_pass_m / total_m if total_m > 0.0 else "",
        "n2_delta_pp": n2_delta_pp,
        "official_delta_pp": n2_delta_pp / 6.0 if math.isfinite(n2_delta_pp) else "",
        "available_epochs": available_epochs,
        "good_epochs": good_epochs,
        "bad_epochs": bad_epochs,
        "chosen_labels": _top_counts(chosen_labels),
    }


def _span_rows(
    spans: list[dict[str, Any]],
    records_by_epoch: dict[int, dict[str, Any]],
    *,
    full_total_m: float,
    pass_m: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for span in spans:
        records = _records_for_span(span, records_by_epoch)
        summary = {
            strategy: _summarize_strategy(records, strategy=strategy, full_total_m=full_total_m, pass_m=pass_m)
            for strategy in ("gated_oracle", "min_rms", "min_absmax", "max_sats", "nearest_current")
        }
        oracle_labels = Counter(
            str(rec["oracle"]["label"])
            for rec in records
            if rec.get("oracle") is not None
        )
        selected_labels = Counter(str(rec["selected"]) for rec in records)
        out.append(
            {
                "span_index": span["span_index"],
                "label": span.get("label", ""),
                "family": span.get("family", ""),
                "start_epoch": span["start_epoch_i"],
                "end_epoch": span["end_epoch_i"],
                "n_epochs": len(records),
                "total_m": summary["gated_oracle"]["total_m"],
                "current_pass_m": summary["gated_oracle"]["current_pass_m"],
                "gated_oracle_gain_m": summary["gated_oracle"]["gain_m"],
                "min_rms_gain_m": summary["min_rms"]["gain_m"],
                "min_absmax_gain_m": summary["min_absmax"]["gain_m"],
                "max_sats_gain_m": summary["max_sats"]["gain_m"],
                "nearest_current_gain_m": summary["nearest_current"]["gain_m"],
                "selected_labels": _top_counts(selected_labels),
                "oracle_labels": _top_counts(oracle_labels),
                "phase73_gated_best_labels": span.get("gated_best_labels", ""),
            },
        )
    out.sort(key=lambda row: float(row["gated_oracle_gain_m"]), reverse=True)
    return out


def _epoch_rows(
    spans: list[dict[str, Any]],
    records_by_epoch: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for span in spans:
        for rec in _records_for_span(span, records_by_epoch):
            row: dict[str, Any] = {
                "span_index": span["span_index"],
                "span_label": span.get("label", ""),
                "epoch": rec["epoch"],
                "tow": rec["tow"],
                "weight_m": rec["weight_m"],
                "selected": rec["selected"],
                "current_error_m": rec["current_error_m"],
                "current_pass": rec["current_pass"],
                "family_span_m": rec["family_span_m"],
                "agreement_1m": rec["agreement_1m"],
                "gated_count": rec["gated_count"],
            }
            oracle = rec.get("oracle")
            row["oracle_label"] = oracle.get("label") if oracle else ""
            row["oracle_error_m"] = oracle.get("error_m") if oracle else ""
            for strategy in ("min_rms", "min_absmax", "max_sats", "nearest_current"):
                choice = rec["choices"].get(strategy)
                row[f"{strategy}_label"] = choice.get("label") if choice else ""
                row[f"{strategy}_error_m"] = choice.get("error_m") if choice else ""
            out.append(row)
    return out


def _label_pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    recoverable_gain: defaultdict[tuple[str, str], float] = defaultdict(float)
    total_weight: defaultdict[tuple[str, str], float] = defaultdict(float)
    for rec in records:
        oracle = rec.get("oracle")
        if oracle is None:
            continue
        pair = (str(rec["selected"]), str(oracle["label"]))
        counts[pair] += 1
        total_weight[pair] += float(rec["weight_m"])
        if bool(oracle["pass"]) and not bool(rec["current_pass"]):
            recoverable_gain[pair] += float(rec["weight_m"])
    rows = [
        {
            "selected_label": selected,
            "oracle_label": oracle,
            "epochs": counts[(selected, oracle)],
            "total_weight_m": total_weight[(selected, oracle)],
            "oracle_recoverable_gain_m": recoverable_gain[(selected, oracle)],
        }
        for selected, oracle in counts
    ]
    rows.sort(key=lambda row: float(row["oracle_recoverable_gain_m"]), reverse=True)
    return rows


def _selected_set_match(name: str, selected: str) -> bool:
    if name == "phase59_labels":
        return selected in PHASE59_LABELS
    if name == "all_recoverable_labels":
        return selected in RECOVERABLE_LABELS
    if name == "no_hs_c4_gici":
        return selected.startswith("xd_gici_") and selected not in {"xd_gici_hs", "xd_gici_c4"}
    raise ValueError(f"unknown selected set: {name}")


def _guard_sweep(
    records: list[dict[str, Any]],
    *,
    full_total_m: float,
    pass_m: float,
) -> list[dict[str, Any]]:
    selected_sets = ("phase59_labels", "all_recoverable_labels", "no_hs_c4_gici")
    strategies = ("min_rms", "min_absmax", "max_sats")
    family_span_min_values = (0, 5, 10, 20, 30, 40, 50, 75, 100, 150)
    agreement_max_values = (0, 2, 4, 6, 8, 10, 12, 16, 20, 99)

    base_pass_m = sum(float(rec["weight_m"]) for rec in records if bool(rec["current_pass"]))
    out: list[dict[str, Any]] = []
    for selected_set in selected_sets:
        for strategy in strategies:
            for family_span_min in family_span_min_values:
                for agreement_max in agreement_max_values:
                    replay_pass_m = 0.0
                    overrides = 0
                    good_epochs = 0
                    bad_epochs = 0
                    chosen_labels: Counter[str] = Counter()
                    for rec in records:
                        selected = str(rec["selected"])
                        family_span = float(rec["family_span_m"])
                        agreement = float(rec["agreement_1m"])
                        eligible = (
                            _selected_set_match(selected_set, selected)
                            and math.isfinite(family_span)
                            and family_span >= family_span_min
                            and (not math.isfinite(agreement) or agreement <= agreement_max)
                        )
                        choice = rec["choices"].get(strategy) if eligible else None
                        if choice is None:
                            if bool(rec["current_pass"]):
                                replay_pass_m += float(rec["weight_m"])
                            continue
                        overrides += 1
                        chosen_labels[str(choice["label"])] += 1
                        chosen_pass = float(choice["error_m"]) <= pass_m
                        if chosen_pass:
                            replay_pass_m += float(rec["weight_m"])
                        if chosen_pass and not bool(rec["current_pass"]):
                            good_epochs += 1
                        elif bool(rec["current_pass"]) and not chosen_pass:
                            bad_epochs += 1
                    gain_m = replay_pass_m - base_pass_m
                    n2_delta_pp = 100.0 * gain_m / full_total_m if full_total_m > 0.0 else float("nan")
                    out.append(
                        {
                            "selected_set": selected_set,
                            "strategy": strategy,
                            "family_span_min_m": family_span_min,
                            "agreement_1m_max": agreement_max,
                            "base_pass_m": base_pass_m,
                            "replay_pass_m": replay_pass_m,
                            "gain_m": gain_m,
                            "n2_delta_pp": n2_delta_pp,
                            "official_delta_pp": n2_delta_pp / 6.0 if math.isfinite(n2_delta_pp) else "",
                            "overrides": overrides,
                            "good_epochs": good_epochs,
                            "bad_epochs": bad_epochs,
                            "chosen_labels": _top_counts(chosen_labels),
                        },
                    )
    out.sort(key=lambda row: float(row["gain_m"]), reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-epochs-csv",
        type=Path,
        default=Path("experiments/results/ppc_phase70_osmroad_overlay_nagoya_run2_full_internal_epochs.csv"),
    )
    parser.add_argument(
        "--span-oracle-csv",
        type=Path,
        default=Path("experiments/results/phase73_nr2_deployable_profile_span_oracle_top999_spans.csv"),
    )
    parser.add_argument("--search-root", action="append", type=Path, default=[Path("experiments/results")])
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--skip-first-spans", type=int, default=3)
    parser.add_argument(
        "--exclude-label-contains",
        action="append",
        default=["oracle", "anchortruth", "phase74"],
        help="Skip profile candidates whose label/path contains this token.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("experiments/results/phase77_nr2_selector_recovery_bucket"),
    )
    args = parser.parse_args(argv)

    rows = _read_csv(args.internal_epochs_csv)
    weights = _distance_weights(rows)
    full_total_m = sum(weights)
    span_rows = _read_csv(args.span_oracle_csv)
    spans = _positive_recoverable_spans(span_rows, skip_first_spans=int(args.skip_first_spans))
    bucket_epochs = {
        epoch
        for span in spans
        for epoch in range(int(span["start_epoch_i"]), int(span["end_epoch_i"]) + 1)
    }

    candidates, excluded = _discover_candidates(
        args.search_root,
        city=args.city,
        run=args.run,
        exclude_label_contains=list(args.exclude_label_contains),
    )
    if not candidates:
        raise SystemExit("no profile candidates found")
    print(f"loaded profile candidates: {len(candidates)}")
    if excluded:
        print(f"excluded profile candidates: {excluded}")

    records = _prepare_records(
        rows,
        weights,
        candidates,
        pass_m=float(args.pass_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    records_by_epoch = {int(rec["epoch"]): rec for rec in records}
    bucket_records = [rec for rec in records if int(rec["epoch"]) in bucket_epochs]

    summary_rows = []
    for strategy in ("gated_oracle", *STRATEGIES):
        summary_rows.append(
            {
                "scope": f"phase73_positive_recoverable_after_top{int(args.skip_first_spans)}",
                **_summarize_strategy(
                    bucket_records,
                    strategy=strategy,
                    full_total_m=full_total_m,
                    pass_m=float(args.pass_m),
                ),
            },
        )
    guard_rows = _guard_sweep(records, full_total_m=full_total_m, pass_m=float(args.pass_m))
    if guard_rows:
        best_guard = guard_rows[0]
        summary_rows.append(
            {
                "scope": "full_nr2_best_observable_guard",
                "strategy": f"{best_guard['selected_set']}:{best_guard['strategy']}:family_span>={best_guard['family_span_min_m']}:agreement<={best_guard['agreement_1m_max']}",
                "n_epochs": len(records),
                "total_m": full_total_m,
                "current_pass_m": best_guard["base_pass_m"],
                "replay_pass_m": best_guard["replay_pass_m"],
                "gain_m": best_guard["gain_m"],
                "score_pct_in_scope": 100.0 * float(best_guard["replay_pass_m"]) / full_total_m,
                "n2_delta_pp": best_guard["n2_delta_pp"],
                "official_delta_pp": best_guard["official_delta_pp"],
                "available_epochs": best_guard["overrides"],
                "good_epochs": best_guard["good_epochs"],
                "bad_epochs": best_guard["bad_epochs"],
                "chosen_labels": best_guard["chosen_labels"],
            },
        )

    _write_csv(Path(f"{args.out_prefix}_summary.csv"), summary_rows)
    _write_csv(
        Path(f"{args.out_prefix}_spans.csv"),
        _span_rows(spans, records_by_epoch, full_total_m=full_total_m, pass_m=float(args.pass_m)),
    )
    _write_csv(Path(f"{args.out_prefix}_epochs.csv"), _epoch_rows(spans, records_by_epoch))
    _write_csv(Path(f"{args.out_prefix}_label_pairs.csv"), _label_pair_rows(bucket_records))
    _write_csv(Path(f"{args.out_prefix}_guard_sweep.csv"), guard_rows)

    print(f"positive recoverable spans after top{int(args.skip_first_spans)}: {len(spans)}")
    if summary_rows:
        print("bucket oracle:", summary_rows[0])
    if guard_rows:
        print("best full guard:", guard_rows[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
