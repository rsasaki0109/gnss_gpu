#!/usr/bin/env python3
"""Replay per-run single-label RTKDiag blocks from selected-loss diagnosis."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _parse_label_list,
)
from sim_ppc_trap_diagnosis import (  # noqa: E402
    Candidate,
    LabelStats,
    _DEFAULT_DATA_ROOT,
    _label_dir_map,
    _load_candidate,
    _run_phase_rows,
    _score_run,
)

RESULTS_DIR = _SCRIPT_DIR / "results"


def _load_base_runs(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as fh:
        return {(r["city"], r["run"]): r for r in csv.DictReader(fh)}


def _load_top_labels(path: Path, *, top_per_run: int) -> dict[tuple[str, str], list[dict[str, str]]]:
    by_run: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["selected_count"]) <= 0:
                continue
            by_run[(row["city"], row["run"])].append(row)
    return {key: rows[:top_per_run] for key, rows in by_run.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-runs-csv", type=Path, required=True)
    parser.add_argument("--diagnosis-labels-csv", type=Path, required=True)
    parser.add_argument("--diagnosis-runs-csv", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--top-per-run", type=int, default=25)
    args = parser.parse_args()

    base_runs = _load_base_runs(args.diagnosis_runs_csv)
    top_labels = _load_top_labels(args.diagnosis_labels_csv, top_per_run=int(args.top_per_run))
    label_to_dir = _label_dir_map()
    loaded_by_run: dict[tuple[str, str], tuple[list[tuple[float, object]], dict[float, object], list[Candidate]]] = {}

    for phase_row in _run_phase_rows(args.phase_runs_csv):
        city = str(phase_row["city"])
        run = str(phase_row["run"])
        labels = _parse_label_list(str(phase_row["rtkdiag_candidate_labels"]))
        loaded: list[Candidate] = []
        for label in labels:
            dir_name = label_to_dir.get(label)
            if dir_name is None and label.startswith("x"):
                dir_name = label_to_dir.get(label[1:])
            if dir_name is None:
                continue
            candidate = _load_candidate(city, run, label, dir_name)
            if candidate is not None:
                loaded.append(candidate)

        filtered = _filter_rtkdiag_candidates_by_policy(
            [(c.label, c.pos, c.diag) for c in loaded],
            city=city,
            run=run,
            policy=str(args.policy),
        )
        candidates = [Candidate(label=label, pos=pos, diag=diag) for label, pos, diag in filtered]
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        loaded_by_run[(city, run)] = (ref, hybrid_pos, candidates)
        print(
            f"loaded {city}/{run}: candidates={len(candidates)} "
            f"test_labels={len(top_labels.get((city, run), []))}",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    for city, run in sorted(top_labels):
        ref, hybrid_pos, candidates = loaded_by_run[(city, run)]
        base = base_runs[(city, run)]
        base_ppc = float(base["ppc_pct"])
        base_pass = float(base["pass_m"])
        for label_row in top_labels[(city, run)]:
            label = label_row["label"]
            stats: dict[tuple[str, str, str], LabelStats] = defaultdict(LabelStats)
            ppc, pass_m, total_m, _selected_epochs = _score_run(
                city=city,
                run=run,
                ref=ref,
                hybrid_pos=hybrid_pos,
                candidates=candidates,
                policy=str(args.policy),
                blocked_labels=set(),
                blocked_pairs={(city, run, label)},
                stats=stats,
                bad_dist_m=5.0,
                bad_loss_m=2.0,
            )
            gain_m = pass_m - base_pass
            row = {
                "city": city,
                "run": run,
                "label": label,
                "base_ppc_pct": base_ppc,
                "block_ppc_pct": ppc,
                "base_pass_m": base_pass,
                "block_pass_m": pass_m,
                "total_m": total_m,
                "gain_m": gain_m,
                "gain_pp": ppc - base_ppc,
                "selected_loss_sum_m": float(label_row["selected_loss_sum_m"]),
                "selected_count": int(label_row["selected_count"]),
            }
            rows.append(row)
            if gain_m > 0.0:
                print(f"POS {city}/{run}:{label} gain={gain_m:.6f}m ppc={ppc:.9f}", flush=True)

    rows.sort(key=lambda r: (float(r["gain_m"]), float(r["selected_loss_sum_m"])), reverse=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "city",
        "run",
        "label",
        "base_ppc_pct",
        "block_ppc_pct",
        "base_pass_m",
        "block_pass_m",
        "total_m",
        "gain_m",
        "gain_pp",
        "selected_loss_sum_m",
        "selected_count",
    ]
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved {args.out_csv} rows={len(rows)}", flush=True)
    print("Top gains:", flush=True)
    for row in rows[:20]:
        print(
            f"{row['city']}/{row['run']} {row['label']} "
            f"gain={float(row['gain_m']):.6f}m "
            f"ppc={float(row['block_ppc_pct']):.9f} "
            f"loss={float(row['selected_loss_sum_m']):.1f} "
            f"sel={int(row['selected_count'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
