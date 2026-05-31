#!/usr/bin/env python3
"""Oracle diagnostic for constant-bias RTK relative trajectory rescue.

This is not a deployable policy. It answers one question: if an already
generated RTKDiag candidate has a good local trajectory shape but a constant
ECEF offset, how much PPC pass distance is recoverable by a segment-local
median truth bias? The result is a target for CT-RBPF/FGO bias-observation
factors, not a replacement for them.
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
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _parse_label_list,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from sim_ppc_oracle_miss_diagnosis import _load_candidates, _run_phase_rows  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


@dataclass(frozen=True)
class RelativeBiasRow:
    city: str
    run: str
    start_idx: int
    end_idx: int
    diagnosis: str
    segment_weight_m: float
    label: str
    n_candidate_epochs: int
    current_ppc_pct: float
    current_pass_m: float
    rescued_ppc_pct: float
    rescued_pass_m: float
    delta_pass_m: float
    raw_p50_2d_m: float
    raw_p95_2d_m: float
    debiased_p50_2d_m: float
    debiased_p95_2d_m: float
    debiased_epoch_pass_pct: float
    oracle_shift_x_m: float
    oracle_shift_y_m: float
    oracle_shift_z_m: float


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


def _load_current(path: Path, ref: list[tuple[float, np.ndarray]]) -> np.ndarray:
    pos: dict[float, np.ndarray] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                tow = round(float(parts[1]), 1)
                arr = np.asarray([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
            except (ValueError, IndexError):
                continue
            if np.all(np.isfinite(arr)) and not np.all(arr == 0.0):
                pos[tow] = arr
    out = np.zeros((len(ref), 3), dtype=np.float64)
    for i, (tow, _truth) in enumerate(ref):
        p = pos.get(round(float(tow), 1))
        if p is not None and np.all(np.isfinite(p)) and not np.all(np.asarray(p) == 0.0):
            out[i] = np.asarray(p, dtype=np.float64)
    return out


def _phase_by_run(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {(str(r["city"]), str(r["run"])): r for r in _run_phase_rows(path)}


def _load_segments(args: argparse.Namespace) -> dict[tuple[str, str], list[dict[str, str]]]:
    run_filter = _parse_run_filter(str(args.runs))
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    with args.audit_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        city = str(row["city"])
        run = str(row["run"])
        if run_filter is not None and (city, run) not in run_filter:
            continue
        if str(row.get("diagnosis", "")) != str(args.diagnosis):
            continue
        if float(row.get("weight_m", "0") or 0.0) < float(args.min_weight_m):
            continue
        out.setdefault((city, run), []).append(row)
    for rows_for_run in out.values():
        rows_for_run.sort(key=lambda r: float(r.get("weight_m", "0") or 0.0), reverse=True)
        if int(args.top) > 0:
            del rows_for_run[int(args.top):]
    return out


def _candidate_segment_indices(
    *,
    ref_tows: np.ndarray,
    pos: dict[float, np.ndarray],
    start_idx: int,
    end_idx: int,
) -> list[int]:
    idx = []
    for i in range(max(0, int(start_idx)), min(len(ref_tows), int(end_idx) + 1)):
        p = pos.get(float(ref_tows[i]))
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float64)
        if np.all(np.isfinite(arr)) and not np.all(arr == 0.0):
            idx.append(i)
    return idx


def _evaluate_run(
    *,
    city: str,
    run: str,
    phase_row: dict[str, str],
    segments: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[RelativeBiasRow]:
    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    ref_tows = np.asarray([round(float(tow), 1) for tow, _pos in ref], dtype=np.float64)
    truth = np.asarray([pos for _tow, pos in ref], dtype=np.float64)
    current = _load_current(args.current_pos_dir / f"{city}_{run}_{args.current_pos_suffix}", ref)
    current_score = score_ppc2024(current, truth)

    labels = _parse_label_list(str(phase_row["rtkdiag_candidate_labels"]))
    loaded, missing = _load_candidates(city, run, labels)
    if missing:
        print(f"{city}/{run}: missing labels {','.join(missing)}", flush=True)
    candidates = _filter_rtkdiag_candidates_by_policy(
        loaded,
        city=city,
        run=run,
        policy=str(args.policy),
    )
    rows: list[RelativeBiasRow] = []
    for seg in segments:
        start_idx = int(seg["start_idx"])
        end_idx = int(seg["end_idx"])
        best: RelativeBiasRow | None = None
        for label, cand_pos, _cand_diag in candidates:
            idx = _candidate_segment_indices(
                ref_tows=ref_tows,
                pos=cand_pos,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            if len(idx) < int(args.min_candidate_epochs):
                continue
            cand = np.asarray([cand_pos[float(ref_tows[i])] for i in idx], dtype=np.float64)
            err = cand - truth[idx]
            shift = np.median(err, axis=0)
            debiased = cand - shift
            raw_2d = np.linalg.norm(err[:, :2], axis=1)
            deb_2d = np.linalg.norm((debiased - truth[idx])[:, :2], axis=1)
            trial = current.copy()
            trial[idx] = debiased
            rescued_score = score_ppc2024(trial, truth)
            row = RelativeBiasRow(
                city=city,
                run=run,
                start_idx=start_idx,
                end_idx=end_idx,
                diagnosis=str(seg.get("diagnosis", "")),
                segment_weight_m=float(seg.get("weight_m", "0") or 0.0),
                label=label,
                n_candidate_epochs=len(idx),
                current_ppc_pct=float(current_score.score_pct),
                current_pass_m=float(current_score.pass_distance_m),
                rescued_ppc_pct=float(rescued_score.score_pct),
                rescued_pass_m=float(rescued_score.pass_distance_m),
                delta_pass_m=float(rescued_score.pass_distance_m - current_score.pass_distance_m),
                raw_p50_2d_m=float(np.median(raw_2d)),
                raw_p95_2d_m=float(np.percentile(raw_2d, 95.0)),
                debiased_p50_2d_m=float(np.median(deb_2d)),
                debiased_p95_2d_m=float(np.percentile(deb_2d, 95.0)),
                debiased_epoch_pass_pct=float(100.0 * np.mean(deb_2d < 1.0)),
                oracle_shift_x_m=float(shift[0]),
                oracle_shift_y_m=float(shift[1]),
                oracle_shift_z_m=float(shift[2]),
            )
            if best is None or row.delta_pass_m > best.delta_pass_m:
                best = row
        if best is not None:
            rows.append(best)
            print(
                f"{city}/{run} {start_idx}-{end_idx} best={best.label} "
                f"delta={best.delta_pass_m:+.3f}m raw_p50={best.raw_p50_2d_m:.2f} "
                f"deb_p50={best.debiased_p50_2d_m:.2f} pass_ep={best.debiased_epoch_pass_pct:.1f}%",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--phase-runs-csv", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--current-pos-dir", type=Path, required=True)
    parser.add_argument(
        "--current-pos-suffix",
        default="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf.pos",
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--runs", default="all")
    parser.add_argument("--diagnosis", default="candidate_generation_needed")
    parser.add_argument("--min-weight-m", type=float, default=1.0)
    parser.add_argument("--min-candidate-epochs", type=int, default=5)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_relative_bias_oracle.csv")
    args = parser.parse_args()

    phase = _phase_by_run(args.phase_runs_csv)
    segments = _load_segments(args)
    rows: list[RelativeBiasRow] = []
    for key in sorted(segments):
        if key not in phase:
            raise SystemExit(f"missing phase row for {key[0]}/{key[1]}")
        rows.extend(
            _evaluate_run(
                city=key[0],
                run=key[1],
                phase_row=phase[key],
                segments=segments[key],
                args=args,
            )
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(RelativeBiasRow.__dataclass_fields__.keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    print(f"saved: {args.out_csv}")
    if rows:
        positive = sum(max(0.0, row.delta_pass_m) for row in rows)
        print(f"sum_positive_individual_delta={positive:+.6f}m")


if __name__ == "__main__":
    main()
