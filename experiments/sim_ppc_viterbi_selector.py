#!/usr/bin/env python3
"""Viterbi / shortest-path trajectory selector over RTKDiag candidates.

This is a no-PF replay experiment.  It keeps the current Phase policy's
candidate pool and label-penalty priors, but replaces greedy per-epoch
selection with a global path through the top-K candidates at each epoch.
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

from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from sim_ppc_label_penalty_sweep import (  # noqa: E402
    _HYBDELTA_ALPHA,
    _PREVDIST_ALPHA,
    _builtin_label_factors,
    _collect_options,
    _simulate,
)

RESULTS_DIR = _SCRIPT_DIR / "results"


@dataclass(frozen=True)
class ViterbiResult:
    city: str
    run: str
    mode: str
    top_k: int
    alpha: float
    transition: str
    local_weight: float
    current_ppc_pct: float
    current_pass_m: float
    ppc_pct: float
    pass_m: float
    delta_pass_m: float
    total_m: float
    selected_epochs: int


def _rank_costs(keys0: np.ndarray, keys1: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((keys1, keys0))
    keep = order[: min(int(top_k), len(order))]
    if len(keep) <= 1:
        ranks = np.zeros(len(keep), dtype=np.float64)
    else:
        ranks = np.arange(len(keep), dtype=np.float64) / float(len(keep) - 1)
    return keep, ranks


def _greedy_indices(mode: str, epochs, factors: dict[str, float]) -> list[int | None]:
    out: list[int | None] = []
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    for hp, opts in epochs:
        if opts is None:
            out.append(None)
            if hp is not None:
                prev = np.asarray(hp, dtype=np.float64)
                prev_hybrid = np.asarray(hp, dtype=np.float64)
            continue

        labels, pos_arr, base0, base1 = opts
        if factors:
            factor_arr = np.fromiter(
                (float(factors.get(label, 1.0)) for label in labels),
                dtype=np.float64,
                count=len(labels),
            )
            key0 = np.asarray(base0, dtype=np.float64) * factor_arr
        else:
            key0 = np.asarray(base0, dtype=np.float64).copy()
        if mode in _PREVDIST_ALPHA and prev is not None:
            key0 += _PREVDIST_ALPHA[mode] * np.linalg.norm(pos_arr - prev, axis=1)
        elif mode in _HYBDELTA_ALPHA and prev is not None and prev_hybrid is not None and hp is not None:
            predicted = prev + (np.asarray(hp, dtype=np.float64) - prev_hybrid)
            key0 += _HYBDELTA_ALPHA[mode] * np.linalg.norm(pos_arr - predicted, axis=1)
        idx = int(np.lexsort((base1, key0))[0])
        out.append(idx)
        prev = np.asarray(pos_arr[idx], dtype=np.float64)
        if hp is not None:
            prev_hybrid = np.asarray(hp, dtype=np.float64)
    return out


def _transition_matrix(
    prev_pos: np.ndarray,
    cur_pos: np.ndarray,
    *,
    transition: str,
    prev_hybrid: np.ndarray | None,
    cur_hybrid: np.ndarray | None,
    cap_m: float,
) -> np.ndarray:
    if transition == "prevdist":
        ref = prev_pos
    elif transition == "hybdelta" and prev_hybrid is not None and cur_hybrid is not None:
        delta = cur_hybrid - prev_hybrid
        ref = prev_pos + delta[None, :]
    elif transition == "hybdist" and cur_hybrid is not None:
        d = np.linalg.norm(cur_pos - cur_hybrid[None, :], axis=1)
        return np.tile(np.minimum(d, cap_m) / cap_m, (prev_pos.shape[0], 1))
    else:
        ref = prev_pos
    d = np.linalg.norm(cur_pos[None, :, :] - ref[:, None, :], axis=2)
    return np.minimum(d, cap_m) / cap_m


def _viterbi_segment(
    states: list[dict[str, object]],
    *,
    alpha: float,
    transition: str,
    local_weight: float,
    cap_m: float,
) -> list[int]:
    if not states:
        return []
    costs = np.asarray(states[0]["local_cost"], dtype=np.float64) * float(local_weight)
    back_ptrs: list[np.ndarray] = []
    for i in range(1, len(states)):
        prev_pos = np.asarray(states[i - 1]["pos"], dtype=np.float64)
        cur_pos = np.asarray(states[i]["pos"], dtype=np.float64)
        trans = _transition_matrix(
            prev_pos,
            cur_pos,
            transition=transition,
            prev_hybrid=states[i - 1]["hybrid"],
            cur_hybrid=states[i]["hybrid"],
            cap_m=float(cap_m),
        )
        cand_cost = costs[:, None] + float(alpha) * trans
        back = np.argmin(cand_cost, axis=0).astype(np.int32)
        costs = cand_cost[back, np.arange(cur_pos.shape[0])] + (
            np.asarray(states[i]["local_cost"], dtype=np.float64) * float(local_weight)
        )
        back_ptrs.append(back)
    path = [int(np.argmin(costs))]
    for back in reversed(back_ptrs):
        path.append(int(back[path[-1]]))
    path.reverse()
    return path


def _simulate_viterbi(
    truth: np.ndarray,
    mode: str,
    epochs,
    factors: dict[str, float],
    *,
    top_k: int,
    alpha: float,
    transition: str,
    local_weight: float,
    cap_m: float,
    greedy_anchor: bool,
) -> tuple[float, float, float, int]:
    est = np.zeros_like(truth)
    states: list[dict[str, object]] = []
    state_epoch_indices: list[int] = []
    selected_epochs = 0
    greedy_idxs = _greedy_indices(mode, epochs, factors) if greedy_anchor else [None] * len(epochs)

    def flush_segment() -> None:
        nonlocal selected_epochs
        if not states:
            return
        path = _viterbi_segment(
            states,
            alpha=float(alpha),
            transition=str(transition),
            local_weight=float(local_weight),
            cap_m=float(cap_m),
        )
        for state, epoch_idx, state_idx in zip(states, state_epoch_indices, path, strict=True):
            pos = np.asarray(state["pos"], dtype=np.float64)
            est[epoch_idx] = pos[int(state_idx)]
            selected_epochs += 1
        states.clear()
        state_epoch_indices.clear()

    for i, (hp, opts) in enumerate(epochs):
        if hp is not None:
            est[i] = hp
        if opts is None:
            flush_segment()
            continue
        labels, pos_arr, base0, base1 = opts
        factor_arr = np.fromiter(
            (float(factors.get(label, 1.0)) for label in labels),
            dtype=np.float64,
            count=len(labels),
        )
        keys0 = np.asarray(base0, dtype=np.float64) * factor_arr
        keys1 = np.asarray(base1, dtype=np.float64)
        keep, local_cost = _rank_costs(keys0, keys1, top_k)
        greedy_idx = greedy_idxs[i]
        if greedy_idx is not None:
            greedy_idx = int(greedy_idx)
            hit = np.flatnonzero(keep == greedy_idx)
            if len(hit):
                local_cost[int(hit[0])] = 0.0
            else:
                keep = np.concatenate([np.asarray([greedy_idx], dtype=keep.dtype), keep])
                local_cost = np.concatenate([np.asarray([0.0], dtype=np.float64), local_cost])
        if len(keep) == 0:
            flush_segment()
            continue
        states.append({
            "labels": tuple(labels[int(j)] for j in keep),
            "pos": np.asarray(pos_arr[keep], dtype=np.float64),
            "hybrid": None if hp is None else np.asarray(hp, dtype=np.float64),
            "local_cost": local_cost,
        })
        state_epoch_indices.append(i)
    flush_segment()
    score = score_ppc2024(est, truth)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m), selected_epochs


def _parse_run_filter(spec: str) -> list[tuple[str, str]]:
    if not spec.strip() or spec.strip() == "all":
        return [
            ("tokyo", "run1"),
            ("tokyo", "run2"),
            ("tokyo", "run3"),
            ("nagoya", "run1"),
            ("nagoya", "run2"),
            ("nagoya", "run3"),
        ]
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "/" not in chunk:
            raise ValueError(f"bad run spec: {chunk!r}")
        out.append(tuple(chunk.split("/", 1)))  # type: ignore[arg-type]
    return out


def _float_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-runs-csv", type=Path, required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--runs", default="tokyo/run3,nagoya/run2,nagoya/run3")
    p.add_argument("--top-k", default="4,8,12,16")
    p.add_argument("--alphas", default="0,0.05,0.1,0.2,0.5,1,2,5")
    p.add_argument("--transitions", default="hybdelta,prevdist,hybdist")
    p.add_argument("--local-weights", default="0.25,0.5,1,2")
    p.add_argument("--cap-m", type=float, default=25.0)
    p.add_argument("--no-greedy-anchor", action="store_true")
    p.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_viterbi_selector.csv")
    args = p.parse_args()

    rows: list[ViterbiResult] = []
    for city, run in _parse_run_filter(str(args.runs)):
        ns = argparse.Namespace(
            phase_runs_csv=args.phase_runs_csv,
            policy=str(args.policy),
            city=city,
            run=run,
            data_root=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"),
            hybrid_pos_dir=RESULTS_DIR / "libgnss_rtk_pos_v5",
        )
        truth, mode, epochs, loaded, kept = _collect_options(ns)
        factors = _builtin_label_factors(mode)
        cur_ppc, cur_pass, total_m, _sel = _simulate(truth, mode, epochs, factors)
        print(f"\n{city}/{run} mode={mode} loaded={loaded} kept={kept}")
        print(f"  current replay: ppc={cur_ppc:.6f}% pass={cur_pass:.3f}/{total_m:.3f}")
        best: ViterbiResult | None = None
        for transition in [x.strip() for x in str(args.transitions).split(",") if x.strip()]:
            for local_weight in _float_list(str(args.local_weights)):
                for top_k in _int_list(str(args.top_k)):
                    for alpha in _float_list(str(args.alphas)):
                        ppc, pass_m, total_m, selected = _simulate_viterbi(
                            truth,
                            mode,
                            epochs,
                            factors,
                            top_k=top_k,
                            alpha=alpha,
                            transition=transition,
                            local_weight=local_weight,
                            cap_m=float(args.cap_m),
                            greedy_anchor=not bool(args.no_greedy_anchor),
                        )
                        row = ViterbiResult(
                            city=city,
                            run=run,
                            mode=mode,
                            top_k=top_k,
                            alpha=alpha,
                            transition=transition,
                            local_weight=local_weight,
                            current_ppc_pct=cur_ppc,
                            current_pass_m=cur_pass,
                            ppc_pct=ppc,
                            pass_m=pass_m,
                            delta_pass_m=pass_m - cur_pass,
                            total_m=total_m,
                            selected_epochs=selected,
                        )
                        rows.append(row)
                        if best is None or row.pass_m > best.pass_m:
                            best = row
        if best is not None:
            print(
                f"  best: ppc={best.ppc_pct:.6f}% pass={best.pass_m:.3f} "
                f"delta={best.pass_m - cur_pass:+.3f} "
                f"trans={best.transition} K={best.top_k} alpha={best.alpha:g} "
                f"local_w={best.local_weight:g}",
                flush=True,
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ViterbiResult.__dataclass_fields__.keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"\nsaved: {args.out_csv}")
    by_run: dict[tuple[str, str], ViterbiResult] = {}
    for row in rows:
        key = (row.city, row.run)
        if key not in by_run or row.pass_m > by_run[key].pass_m:
            by_run[key] = row
    pass_sum = sum(r.pass_m for r in by_run.values())
    safe_pass_sum = sum(max(r.pass_m, r.current_pass_m) for r in by_run.values())
    total_sum = sum(r.total_m for r in by_run.values())
    print("per-run best aggregate:")
    for key, row in sorted(by_run.items()):
        print(
            f"  {row.city}/{row.run}: {row.ppc_pct:.6f}% pass={row.pass_m:.3f} "
            f"delta={row.delta_pass_m:+.3f} "
            f"{row.transition} K={row.top_k} alpha={row.alpha:g} local_w={row.local_weight:g}"
        )
    if total_sum > 0.0:
        print(f"  aggregate={100.0 * pass_sum / total_sum:.9f}% pass={pass_sum:.6f}/{total_sum:.6f}")
        print(
            f"  safe_aggregate={100.0 * safe_pass_sum / total_sum:.9f}% "
            f"pass={safe_pass_sum:.6f}/{total_sum:.6f}"
        )


if __name__ == "__main__":
    main()
