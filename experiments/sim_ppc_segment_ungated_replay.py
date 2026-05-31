#!/usr/bin/env python3
"""Replay segment-local ungated RTKDiag rescue inside the CT-RBPF-FGO pool.

This is a no-PF diagnostic.  It keeps the Phase policy, candidate labels,
temporal selector, and label priors, but for audited gate-too-strict segments it
can admit already-generated RTKDiag positions even when they fail the normal
ratio/RMS gate.  The goal is to measure whether a local gate rescue is worth
turning into a truth-free CT-RBPF policy.
"""

from __future__ import annotations

import argparse
import csv
import sys
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
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from sim_ppc_oracle_miss_diagnosis import _load_candidates, _run_phase_rows  # noqa: E402
from sim_ppc_phase_csv_addcand import _temporal_select_params, _valid_hybrid  # noqa: E402
from sim_ppc_trap_diagnosis import _label_penalty_factors  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


@dataclass(frozen=True)
class SegmentSpec:
    city: str
    run: str
    start_idx: int
    end_idx: int
    labels: frozenset[str]
    weight_m: float
    extra_m: float


@dataclass(frozen=True)
class ReplayResult:
    city: str
    run: str
    mode: str
    segment_count: int
    label_source: str
    min_extra_m: float
    ungated_label_penalties: str
    ungated_label_blocks: str
    current_ppc_pct: float
    current_pass_m: float
    replay_ppc_pct: float
    replay_pass_m: float
    delta_pass_m: float
    total_m: float
    selected_ungated_epochs: int
    selected_ungated_labels: str


def _parse_run_filter(spec: str) -> set[tuple[str, str]] | None:
    spec = spec.strip()
    if not spec or spec == "all":
        return None
    out = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        city, run = chunk.split("/", 1)
        out.add((city, run))
    return out


def _parse_label_counts(spec: str) -> frozenset[str]:
    labels = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        labels.append(chunk.split(":", 1)[0])
    return frozenset(labels)


def _parse_scoped_label_factors(specs: list[str]) -> dict[tuple[str, str] | None, dict[str, float]]:
    out: dict[tuple[str, str] | None, dict[str, float]] = {}
    for spec in specs:
        for chunk in str(spec).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            scope_text, rules_text = chunk.split(":", 1)
            scope = None
            if scope_text.strip() not in {"*", "all"}:
                city, run = scope_text.strip().split("/", 1)
                scope = (city, run)
            rules = out.setdefault(scope, {})
            for rule in rules_text.split(","):
                rule = rule.strip()
                if not rule:
                    continue
                label, factor = rule.split("=", 1)
                rules[label.strip()] = float(factor)
    return out


def _parse_scoped_label_blocks(specs: list[str]) -> dict[tuple[str, str] | None, set[str]]:
    out: dict[tuple[str, str] | None, set[str]] = {}
    for spec in specs:
        for chunk in str(spec).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            scope_text, labels_text = chunk.split(":", 1)
            scope = None
            if scope_text.strip() not in {"*", "all"}:
                city, run = scope_text.strip().split("/", 1)
                scope = (city, run)
            labels = out.setdefault(scope, set())
            for label in labels_text.split(","):
                label = label.strip()
                if label:
                    labels.add(label)
    return out


def _scoped_label_factor(
    rules: dict[tuple[str, str] | None, dict[str, float]],
    city: str,
    run: str,
    label: str,
) -> float:
    factor = rules.get(None, {}).get(label, 1.0)
    factor = rules.get((city, run), {}).get(label, factor)
    return float(factor)


def _scoped_label_blocked(
    rules: dict[tuple[str, str] | None, set[str]],
    city: str,
    run: str,
    label: str,
) -> bool:
    return label in rules.get(None, set()) or label in rules.get((city, run), set())


def _load_segments(args: argparse.Namespace) -> dict[tuple[str, str], list[SegmentSpec]]:
    run_filter = _parse_run_filter(str(args.runs))
    out: dict[tuple[str, str], list[SegmentSpec]] = {}
    with args.audit_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if str(r.get("diagnosis", "")) == str(args.diagnosis)]
    rows = [r for r in rows if float(r.get("ungated_extra_pass_weight_m", "0") or 0.0) >= float(args.min_extra_m)]
    rows.sort(key=lambda r: float(r.get("ungated_extra_pass_weight_m", "0") or 0.0), reverse=True)
    if int(args.top) > 0:
        rows = rows[: int(args.top)]
    for row in rows:
        city = str(row["city"])
        run = str(row["run"])
        if run_filter is not None and (city, run) not in run_filter:
            continue
        if str(args.label_source) == "top_best_all":
            labels = _parse_label_counts(str(row.get("top_best_all_labels", "")))
        else:
            labels = frozenset()
        spec = SegmentSpec(
            city=city,
            run=run,
            start_idx=int(row["start_idx"]),
            end_idx=int(row["end_idx"]),
            labels=labels,
            weight_m=float(row["weight_m"]),
            extra_m=float(row["ungated_extra_pass_weight_m"]),
        )
        out.setdefault((city, run), []).append(spec)
    return out


def _phase_by_run(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {(str(r["city"]), str(r["run"])): r for r in _run_phase_rows(path)}


def _active_segment(segments: list[SegmentSpec], idx: int) -> SegmentSpec | None:
    for seg in segments:
        if int(seg.start_idx) <= idx <= int(seg.end_idx):
            return seg
    return None


def _fixed_output_ok(row: dict[str, str]) -> bool:
    try:
        return int(row.get("output_added", "0")) == 1 and int(row.get("final_status", "0")) == 4
    except ValueError:
        return False


def _diag_float(row: dict[str, str], key: str, default: float = float("inf")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _collect_options(
    *,
    city: str,
    run: str,
    row: dict[str, str],
    policy: str,
    data_root: Path,
    hybrid_pos_dir: Path,
    segments: list[SegmentSpec],
    require_fixed_ungated: bool,
    ungated_ratio_min: float,
    ungated_rms_max: float,
    ungated_label_penalties: dict[tuple[str, str] | None, dict[str, float]],
    ungated_label_blocks: dict[tuple[str, str] | None, set[str]],
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
    base_mode, _temporal_kind, _temporal_alpha = _temporal_select_params(mode)
    label_penalties = _label_penalty_factors(mode)
    ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
    rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)

    ref = _load_full_reference(data_root / city / run / "reference.csv")
    truth = np.asarray([p for _tow, p in ref], dtype=np.float64)
    hybrid_pos, _ = _load_hybrid_pos_file(hybrid_pos_dir / f"{city}_{run}_full.pos")
    epochs = []
    for idx, (tow, _true_pos) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_arr = np.asarray(hp, dtype=np.float64) if _valid_hybrid(hp) else None
        seg = _active_segment(segments, idx)
        opts = []
        for label, cand_pos, cand_diag in kept:
            diag_row = cand_diag.get(t_key)
            cand = cand_pos.get(t_key)
            if not _valid_hybrid(cand) or diag_row is None:
                continue
            gated = _rtkdiag_candidate_gate(diag_row, ratio_min=ratio_min, residual_rms_max=rms_max)
            ungated_allowed = False
            if seg is not None and not gated:
                ungated_allowed = (
                    (not require_fixed_ungated or _fixed_output_ok(diag_row))
                    and _diag_float(diag_row, "final_ratio", default=-float("inf")) >= float(ungated_ratio_min)
                    and _diag_float(diag_row, "final_residual_rms", default=float("inf")) <= float(ungated_rms_max)
                    and (not seg.labels or label in seg.labels)
                )
            if not gated and not ungated_allowed:
                continue
            if ungated_allowed and _scoped_label_blocked(ungated_label_blocks, city, run, label):
                continue
            key = _rtkdiag_candidate_sort_key(diag_row, mode=base_mode)
            key0 = float(key[0]) * float(label_penalties.get(label, 1.0))
            if ungated_allowed:
                key0 *= _scoped_label_factor(ungated_label_penalties, city, run, label)
            key1 = float(key[1])
            opts.append((label, np.asarray(cand, dtype=np.float64), key0, key1, bool(ungated_allowed)))
        epochs.append((float(tow), hp_arr, opts))
    return truth, mode, epochs, len(loaded), len(kept), missing


def _simulate(truth: np.ndarray, mode: str, epochs):
    est = np.zeros_like(truth)
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    _base_mode, temporal_kind, temporal_alpha = _temporal_select_params(mode)
    selected_ungated = []
    selected_rows: list[dict[str, object]] = []
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
        pick = int(np.lexsort((key1, key0))[0])
        est[i] = pos_arr[pick]
        prev = pos_arr[pick]
        if hp is not None:
            prev_hybrid = hp
        if bool(opts[pick][4]):
            label = str(opts[pick][0])
            selected_ungated.append(label)
            selected_rows.append({
                "idx": i,
                "tow": float(_tow),
                "label": label,
                "key0": float(key0[pick]),
                "key1": float(key1[pick]),
            })
    score = score_ppc2024(est, truth)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m), selected_ungated, selected_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    p.add_argument("--phase-runs-csv", type=Path, required=True)
    p.add_argument("--audit-csv", type=Path, required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--runs", default="all")
    p.add_argument("--diagnosis", default="gate_too_strict")
    p.add_argument("--min-extra-m", type=float, default=1.0)
    p.add_argument("--top", type=int, default=0)
    p.add_argument("--label-source", choices=("top_best_all", "all"), default="top_best_all")
    p.add_argument("--require-fixed-ungated", action="store_true")
    p.add_argument("--ungated-ratio-min", type=float, default=-float("inf"))
    p.add_argument("--ungated-rms-max", type=float, default=float("inf"))
    p.add_argument(
        "--ungated-label-penalty",
        action="append",
        default=[],
        help=(
            "Multiply ungated-only candidate key0 for scoped labels. "
            "Format: city/run:label=factor,label=factor;all:label=factor"
        ),
    )
    p.add_argument(
        "--ungated-label-block",
        action="append",
        default=[],
        help="Drop ungated-only candidates for scoped labels. Format: city/run:label,label;all:label",
    )
    p.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_segment_ungated_replay.csv")
    p.add_argument("--out-epochs-csv", type=Path, default=None)
    args = p.parse_args()

    segments_by_run = _load_segments(args)
    phase = _phase_by_run(args.phase_runs_csv)
    ungated_label_penalties = _parse_scoped_label_factors(list(args.ungated_label_penalty))
    ungated_label_blocks = _parse_scoped_label_blocks(list(args.ungated_label_block))
    penalty_spec = ";".join(str(x) for x in args.ungated_label_penalty)
    block_spec = ";".join(str(x) for x in args.ungated_label_block)
    rows: list[ReplayResult] = []
    epoch_rows: list[dict[str, object]] = []
    for key in sorted(segments_by_run):
        city, run = key
        if key not in phase:
            raise SystemExit(f"missing phase row for {city}/{run}")
        truth, mode, epochs, loaded, kept, missing = _collect_options(
            city=city,
            run=run,
            row=phase[key],
            policy=str(args.policy),
            data_root=args.data_root,
            hybrid_pos_dir=args.hybrid_pos_dir,
            segments=segments_by_run[key],
            require_fixed_ungated=bool(args.require_fixed_ungated),
            ungated_ratio_min=float(args.ungated_ratio_min),
            ungated_rms_max=float(args.ungated_rms_max),
            ungated_label_penalties=ungated_label_penalties,
            ungated_label_blocks=ungated_label_blocks,
        )
        no_segments_truth, _mode, no_segments_epochs, _loaded, _kept, _missing = _collect_options(
            city=city,
            run=run,
            row=phase[key],
            policy=str(args.policy),
            data_root=args.data_root,
            hybrid_pos_dir=args.hybrid_pos_dir,
            segments=[],
            require_fixed_ungated=bool(args.require_fixed_ungated),
            ungated_ratio_min=float(args.ungated_ratio_min),
            ungated_rms_max=float(args.ungated_rms_max),
            ungated_label_penalties=ungated_label_penalties,
            ungated_label_blocks=ungated_label_blocks,
        )
        cur_ppc, cur_pass, total_m, _cur_ungated, _cur_epoch_rows = _simulate(
            no_segments_truth,
            mode,
            no_segments_epochs,
        )
        ppc, pass_m, total_m, selected_ungated, selected_epoch_rows = _simulate(truth, mode, epochs)
        for erow in selected_epoch_rows:
            erow["city"] = city
            erow["run"] = run
            epoch_rows.append(erow)
        counts = {}
        for label in selected_ungated:
            counts[label] = counts.get(label, 0) + 1
        label_summary = ",".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10])
        row = ReplayResult(
            city=city,
            run=run,
            mode=mode,
            segment_count=len(segments_by_run[key]),
            label_source=str(args.label_source),
            min_extra_m=float(args.min_extra_m),
            ungated_label_penalties=penalty_spec,
            ungated_label_blocks=block_spec,
            current_ppc_pct=cur_ppc,
            current_pass_m=cur_pass,
            replay_ppc_pct=ppc,
            replay_pass_m=pass_m,
            delta_pass_m=pass_m - cur_pass,
            total_m=total_m,
            selected_ungated_epochs=len(selected_ungated),
            selected_ungated_labels=label_summary,
        )
        rows.append(row)
        print(
            f"{city}/{run} mode={mode} segs={len(segments_by_run[key])} "
            f"loaded={loaded} kept={kept} missing={len(missing)} "
            f"ppc={ppc:.6f}% delta={pass_m - cur_pass:+.3f}m "
            f"selected_ungated={len(selected_ungated)}",
            flush=True,
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ReplayResult.__dataclass_fields__.keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    pass_sum = sum(r.replay_pass_m for r in rows)
    cur_sum = sum(r.current_pass_m for r in rows)
    total_sum = sum(r.total_m for r in rows)
    print(f"saved: {args.out_csv}")
    if args.out_epochs_csv is not None:
        args.out_epochs_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_epochs_csv.open("w", newline="") as fh:
            fieldnames = ["city", "run", "idx", "tow", "label", "key0", "key1"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(epoch_rows)
        print(f"saved epochs: {args.out_epochs_csv}")
    if total_sum > 0:
        print(
            f"aggregate={100.0 * pass_sum / total_sum:.9f}% "
            f"delta={pass_sum - cur_sum:+.6f}m pass={pass_sum:.6f}/{total_sum:.6f}"
        )


if __name__ == "__main__":
    main()
