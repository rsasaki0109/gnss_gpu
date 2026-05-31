#!/usr/bin/env python3
"""Diagnose expanded-pool trap candidates.

Phase 11de showed that blindly adding many RTKDiag candidates can destroy the
selector: some candidates have excellent diagnostic sort keys but are far from
truth.  This script replays a phase runs CSV, computes per-label selected/oracle
statistics, and optionally re-scores after blocking suspect labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_rtk_diag_file,
    _parse_label_list,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from sim_ppc_phase_csv_addcand import (  # noqa: E402
    _candidate_dir_map,
    _discover_candidate_dir_map,
    _temporal_select_params,
    _valid_hybrid,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DIAG_ROOT = Path("experiments/results/libgnss_diag_phase10")


@dataclass(frozen=True)
class Candidate:
    label: str
    pos: dict[float, np.ndarray]
    diag: dict[float, dict[str, str]]


@dataclass
class LabelStats:
    gate_count: int = 0
    selected_count: int = 0
    oracle_count: int = 0
    selected_dist_sum: float = 0.0
    selected_loss_sum: float = 0.0
    selected_bad_count: int = 0
    selected_key_sum: float = 0.0


def _label_dir_map() -> dict[str, str]:
    out = _discover_candidate_dir_map(_candidate_dir_map())
    for label, dir_name in list(out.items()):
        out.setdefault(f"x{label}", dir_name)

    # Labels introduced by expanded-pool phase runs but not present in the
    # smaller Phase-CSV add-candidate map.
    out.update({
        "xcsig5": "full_ratio15_lock3_trustedseed_csig5",
        "xar5": "full_ratio15_lock3_trustedseed_ar5",
        "xarfilt": "full_ratio15_lock3_trustedseed_arfilt",
        "xarfm05": "full_ratio15_lock3_trustedseed_arfm05",
        "xglonassar_ionoest": "full_ratio15_lock3_trustedseed_glonassar_ionoest",
        "xholdvrlx": "full_ratio15_lock3_trustedseed_holdvrlx",
        "xionest": "full_ratio15_lock3_trustedseed_ionest",
        "xionoest": "full_ratio15_lock3_trustedseed_ionoest",
        "xmodeauto": "full_ratio15_lock3_trustedseed_modeauto",
        "xmodestatic": "full_ratio15_lock3_trustedseed_modestatic",
        "xnoarfilt": "full_ratio15_lock3_trustedseed_noarfilt",
        "xnoarfilter": "full_ratio15_lock3_trustedseed_noarfilter",
        "xnopostflt": "full_ratio15_lock3_trustedseed_nopostflt",
        "xonlyGE": "full_ratio15_lock3_trustedseed_onlyGE",
        "xonlyGEJ": "full_ratio15_lock3_trustedseed_onlyGEJ",
        "xr25_nohold": "full_ratio25_lock3_trustedseed_nohold",
        "xr2_nohold": "full_ratio2_lock3_trustedseed_nohold",
        "xr2_g20_m5": "full_ratio2_lock3_trustedseed_gate20_min5",
        "xr2_g20_m8": "full_ratio2_lock3_trustedseed_gate20_min8",
        "xr3_g10_m6": "full_ratio3_lock3_trustedseed_gate10_min6",
        "xr3_g15_m6": "full_ratio3_lock3_trustedseed_gate15_min6",
        "xr3_g20_m6": "full_ratio3_lock3_trustedseed_gate20_min6",
        "xr35": "full_ratio35_lock3_trustedseed",
        "xr4": "full_ratio4_lock3_trustedseed",
        "xt1_tight3_r60_g3": "t1_tight3_ratio60_gate3",
        "xt2_tight_r40_g5_m8": "t2_tight_ratio40_gate5_min8",
        "xt3_tight2_r50_g4": "t3_tight2_ratio50_gate4",
        "xn1_tight_r40_g5_m8": "n1_tight_ratio40_gate5_min8",
        "xn2_tight_r40_g5": "n2_tight_ratio40_gate5",
        "xn3_loose_hold4_r15_g10_m6": "n3_loose_hold4_ratio15_gate10_min6",
        "xn3_loose_hold5_r20_g10_m6": "n3_loose_hold5_ratio20_gate10_min6",
        "xn3_loose_hold5_r20_g8_m6": "n3_loose_hold5_ratio20_gate8_min6",
        "xn3_tight3_r60_g3": "n3_tight3_ratio60_gate3",
    })

    diag_root = _PROJECT_ROOT / _DIAG_ROOT
    for label in list(out):
        if label.startswith("x"):
            stripped = label[1:]
            generic = f"full_ratio15_lock3_trustedseed_{stripped}"
            if (diag_root / generic).is_dir():
                out.setdefault(label, generic)
    return out


def _load_candidate(city: str, run: str, label: str, dir_name: str) -> Candidate | None:
    pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
    diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
    if not pos_path.is_file() or not diag_path.is_file():
        return None
    pos, _ = _load_hybrid_pos_file(pos_path)
    diag = _load_rtk_diag_file(diag_path)
    return Candidate(label=label, pos=pos, diag=diag)


def _run_phase_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get("method", "").endswith("rtkdiag_pf")]


def _parse_blocked_pairs(spec: str) -> set[tuple[str, str, str]]:
    """Parse 'city/run:label,label;city/run:label' into blocked triples."""
    out: set[tuple[str, str, str]] = set()
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk or "/" not in chunk.split(":", 1)[0]:
            raise ValueError(f"bad blocked pair spec: {chunk!r}")
        run_part, labels_part = chunk.split(":", 1)
        city, run = run_part.split("/", 1)
        for label in _parse_label_list(labels_part):
            out.add((city, run, label))
    return out


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


def _label_penalty_factors(mode: str) -> dict[str, float]:
    if mode == "temporal_hybdelta_t3_v5":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
        }
    if mode == "temporal_hybdelta_t3_v6":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
            "c1p1hr": 1.10,
        }
    if mode == "temporal_hybdelta_t3_v7":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
            "c1p1hr": 1.10,
            "r20ga": 3.00,
            "psig1": 1.50,
            "r15ga": 1.20,
        }
    if mode == "temporal_hybdelta_t3_v8":
        return {
            "rtkout5minobs3": 1.06,
            "mlc1r10": 1.03,
            "c1p1hr": 1.10,
            "r20ga": 3.00,
            "psig1": 1.50,
            "r15ga": 1.20,
            "r25g10": 1.50,
            "r20g10": 1.50,
            "r15g10": 1.10,
        }
    if mode == "temporal_n2_v4":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
        }
    if mode == "temporal_n2_v5":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
        }
    if mode == "temporal_n2_v6":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
        }
    if mode == "temporal_n2_v7":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.01,
        }
    if mode == "temporal_n2_v8":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.01,
            "csig05_psig1": 1.01,
            "rtkout5oG": 1.03,
        }
    if mode == "temporal_n2_v9":
        return {
            "mlc1oGc0001": 1.06,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.0403,
            "csig05_psig1": 1.01,
            "rtkout5oG": 1.03,
            "csig05": 1.01,
            "r25g": 1.01,
        }
    if mode == "temporal_n2_v10":
        return {
            "mlc1oGc0001": 1.0706,
            "mlc1r10oG": 1.10,
            "rtkout3": 1.06,
            "csig005_em10": 1.06,
            "mlc1oG": 1.06,
            "oGc005": 1.10,
            "psig3": 1.20,
            "r15": 1.06,
            "r15g": 1.0403,
            "csig05_psig1": 1.01,
            "rtkout5oG": 1.03,
            "csig05": 1.01,
            "r25g": 1.01,
            "n2loose3": 1.06,
            "r25": 1.01,
        }
    if mode == "temporal_hybdelta_n3_v3":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
        }
    if mode == "temporal_hybdelta_n3_v4":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
            "mlc1c005p1": 1.50,
            "n3tight": 1.10,
        }
    if mode == "temporal_hybdelta_n3_v5":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
            "mlc1c005p1": 1.50,
            "n3tight": 1.10,
            "mlc1oGc005p1": 1.03,
            "csig05psh": 1.10,
        }
    if mode == "temporal_hybdelta_n3_v6":
        return {
            "rtkout5c005em3": 1.06,
            "mlc2nobds": 1.50,
            "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
            "mlc1c005p1": 1.50,
            "n3tight": 1.10,
            "mlc1oGc005p1": 1.03,
            "csig05psh": 1.10,
            "n3tight2": 1.01,
        }
    return {}


def _score_run(
    *,
    city: str,
    run: str,
    ref: list[tuple[float, np.ndarray]],
    hybrid_pos: dict[float, np.ndarray],
    candidates: list[Candidate],
    policy: str,
    blocked_labels: set[str],
    blocked_pairs: set[tuple[str, str, str]],
    stats: dict[tuple[str, str, str], LabelStats],
    bad_dist_m: float,
    bad_loss_m: float,
) -> tuple[float, float, float, int]:
    cfg = _apply_rtkdiag_run_index_policy(
        CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
        run=run,
        policy=policy,
        city=city,
    )
    ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
    rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
    mode = str(cfg.rtkdiag_candidate_select_mode)
    base_mode, temporal_kind, temporal_alpha = _temporal_select_params(mode)
    label_penalties = _label_penalty_factors(mode)
    truth = np.asarray([p for _, p in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    selected_epochs = 0
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None

    for i, (tow, true_pos) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_valid = _valid_hybrid(hp)
        if hp_valid:
            est[i] = np.asarray(hp, dtype=np.float64)

        predicted: np.ndarray | None = None
        if temporal_kind == "hybdelta" and prev is not None and prev_hybrid is not None and hp_valid:
            predicted = prev + (np.asarray(hp, dtype=np.float64) - prev_hybrid)

        options: list[tuple[str, np.ndarray, dict[str, str], tuple[float, float], float]] = []
        for cand_obj in candidates:
            if cand_obj.label in blocked_labels or (city, run, cand_obj.label) in blocked_pairs:
                continue
            row = cand_obj.diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_obj.pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            raw_key = _rtkdiag_candidate_sort_key(row, mode=base_mode)
            factor = float(label_penalties.get(cand_obj.label, 1.0))
            key_base = (float(raw_key[0]) * factor, float(raw_key[1]))
            if temporal_kind == "prevdist" and prev is not None:
                key = (
                    key_base[0] + temporal_alpha * float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - prev)),
                    key_base[1],
                )
            elif temporal_kind == "hybdelta" and predicted is not None:
                key = (
                    key_base[0] + temporal_alpha * float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - predicted)),
                    key_base[1],
                )
            else:
                key = key_base
            dist = float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - np.asarray(true_pos, dtype=np.float64)))
            options.append((cand_obj.label, np.asarray(cand, dtype=np.float64), row, key, dist))
            stats[(city, run, cand_obj.label)].gate_count += 1
        if not options:
            if temporal_kind != "none" and hp_valid:
                prev = np.asarray(hp, dtype=np.float64)
                prev_hybrid = np.asarray(hp, dtype=np.float64)
            continue

        selected = min(options, key=lambda item: item[3])
        oracle = min(options, key=lambda item: item[4])
        oracle_label = oracle[0]
        selected_label, selected_pos, _selected_row, selected_key, selected_dist = selected
        oracle_dist = oracle[4]
        loss = max(0.0, selected_dist - oracle_dist)

        stats[(city, run, oracle_label)].oracle_count += 1
        st = stats[(city, run, selected_label)]
        st.selected_count += 1
        st.selected_dist_sum += selected_dist
        st.selected_loss_sum += loss
        st.selected_key_sum += float(selected_key[0])
        if selected_dist >= bad_dist_m and loss >= bad_loss_m:
            st.selected_bad_count += 1

        est[i] = selected_pos
        selected_epochs += 1
        if temporal_kind != "none":
            prev = np.asarray(selected_pos, dtype=np.float64)
            if hp_valid:
                prev_hybrid = np.asarray(hp, dtype=np.float64)

    score = score_ppc2024(est, truth)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m), selected_epochs


def _write_label_stats(path: Path, stats: dict[tuple[str, str, str], LabelStats]) -> None:
    rows: list[dict[str, object]] = []
    for (city, run, label), st in stats.items():
        if st.gate_count == 0 and st.selected_count == 0 and st.oracle_count == 0:
            continue
        rows.append({
            "city": city,
            "run": run,
            "label": label,
            "gate_count": st.gate_count,
            "selected_count": st.selected_count,
            "oracle_count": st.oracle_count,
            "selected_bad_count": st.selected_bad_count,
            "oracle_rate_when_gated": st.oracle_count / max(st.gate_count, 1),
            "bad_rate_when_selected": st.selected_bad_count / max(st.selected_count, 1),
            "mean_selected_dist_m": st.selected_dist_sum / max(st.selected_count, 1),
            "mean_selected_loss_m": st.selected_loss_sum / max(st.selected_count, 1),
            "selected_loss_sum_m": st.selected_loss_sum,
            "mean_selected_key": st.selected_key_sum / max(st.selected_count, 1),
        })
    rows.sort(key=lambda r: (float(r["selected_loss_sum_m"]), int(r["selected_count"])), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        fieldnames = [
            "city", "run", "label", "gate_count", "selected_count", "oracle_count",
            "selected_bad_count", "oracle_rate_when_gated", "bad_rate_when_selected",
            "mean_selected_dist_m", "mean_selected_loss_m", "selected_loss_sum_m",
            "mean_selected_key",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    p.add_argument("--phase-runs-csv", type=Path,
                   default=RESULTS_DIR / "ppc_ctrbpf_fgo_phase11de_expanded_full_p2k_runs.csv")
    p.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    p.add_argument("--policy", default="phase11dd")
    p.add_argument("--blocked-labels", default="",
                   help="Comma-separated labels to block before replay.")
    p.add_argument("--blocked-pairs", default="",
                   help="Semicolon-separated per-run blocks: city/run:label,label;city/run:label")
    p.add_argument("--only-runs", default="",
                   help="Comma-separated run filters like city/run,city/run.")
    p.add_argument("--out-csv", type=Path,
                   default=RESULTS_DIR / "ppc_trap_diagnosis_phase11de_labels.csv")
    p.add_argument("--out-runs-csv", type=Path,
                   default=RESULTS_DIR / "ppc_trap_diagnosis_phase11de_runs.csv")
    p.add_argument("--bad-dist-m", type=float, default=5.0)
    p.add_argument("--bad-loss-m", type=float, default=2.0)
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args()

    label_to_dir = _label_dir_map()
    blocked = set(_parse_label_list(args.blocked_labels))
    blocked_pairs = _parse_blocked_pairs(str(args.blocked_pairs))
    only_runs = _parse_run_filter(str(args.only_runs))
    phase_rows = _run_phase_rows(args.phase_runs_csv)
    if only_runs:
        phase_rows = [r for r in phase_rows if (str(r["city"]), str(r["run"])) in only_runs]
    if not phase_rows:
        raise SystemExit(f"no rtkdiag_pf rows found in {args.phase_runs_csv}")

    stats: dict[tuple[str, str, str], LabelStats] = defaultdict(LabelStats)
    run_rows: list[dict[str, object]] = []
    pass_sum = 0.0
    total_sum = 0.0
    missing: set[str] = set()

    for row in phase_rows:
        city = str(row["city"])
        run = str(row["run"])
        labels = _parse_label_list(str(row["rtkdiag_candidate_labels"]))
        loaded: list[Candidate] = []
        for label in labels:
            dir_name = label_to_dir.get(label)
            if dir_name is None and label.startswith("x"):
                dir_name = label_to_dir.get(label[1:])
            if dir_name is None:
                missing.add(label)
                continue
            cand = _load_candidate(city, run, label, dir_name)
            if cand is None:
                missing.add(label)
                continue
            loaded.append(cand)
        filtered_raw = _filter_rtkdiag_candidates_by_policy(
            [(c.label, c.pos, c.diag) for c in loaded],
            city=city,
            run=run,
            policy=str(args.policy),
        )
        candidates = [Candidate(label=l, pos=p0, diag=d0) for l, p0, d0 in filtered_raw]
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        ppc, pass_m, total_m, selected_epochs = _score_run(
            city=city,
            run=run,
            ref=ref,
            hybrid_pos=hybrid_pos,
            candidates=candidates,
            policy=str(args.policy),
            blocked_labels=blocked,
            blocked_pairs=blocked_pairs,
            stats=stats,
            bad_dist_m=float(args.bad_dist_m),
            bad_loss_m=float(args.bad_loss_m),
        )
        pass_sum += pass_m
        total_sum += total_m
        run_rows.append({
            "city": city,
            "run": run,
            "ppc_pct": ppc,
            "pass_m": pass_m,
            "total_m": total_m,
            "selected_epochs": selected_epochs,
            "n_candidates": len(candidates),
        })
        print(f"{city}/{run}: ppc={ppc:.6f}% pass={pass_m:.3f}/{total_m:.3f} selected={selected_epochs} cand={len(candidates)}")

    _write_label_stats(args.out_csv, stats)
    args.out_runs_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_runs_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()))
        w.writeheader()
        w.writerows(run_rows)

    print("\n========== aggregate ==========")
    print(f"blocked_labels={','.join(sorted(blocked)) if blocked else '(none)'}")
    if blocked_pairs:
        pair_text = ",".join(f"{c}/{r}:{l}" for c, r, l in sorted(blocked_pairs))
        print(f"blocked_pairs={pair_text}")
    print(f"aggregate={100.0 * pass_sum / total_sum:.9f}% pass={pass_sum:.6f}/{total_sum:.6f}")
    if missing:
        print(f"missing_labels={','.join(sorted(missing))}")
    print(f"saved labels: {args.out_csv}")
    print(f"saved runs:   {args.out_runs_csv}")

    rows = []
    with args.out_csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    print("\nTop selected-loss labels:")
    for r in rows[: max(0, int(args.top))]:
        print(
            f"  {r['city']}/{r['run']} {r['label']:<28s} "
            f"sel={int(r['selected_count']):5d} oracle={int(r['oracle_count']):5d} "
            f"loss={float(r['selected_loss_sum_m']):8.1f}m "
            f"bad={100.0 * float(r['bad_rate_when_selected']):5.1f}% "
            f"mean_dist={float(r['mean_selected_dist_m']):6.2f}m"
        )


if __name__ == "__main__":
    main()
