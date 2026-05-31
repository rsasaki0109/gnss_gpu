#!/usr/bin/env python3
"""Offline add-candidate sweep using a phase runs CSV as the base pool.

The older add-candidate sweep uses a static Python candidate list as the base.
For late Phase 11 experiments that is easy to make stale, because the actual
PF runs carry the authoritative per-run pool in the
``rtkdiag_candidate_labels`` column.  This script reads that column from a
phase runs CSV, replays the current per-run selector/gate, and tests extra
candidates one at a time.
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
    _load_rtk_diag_file,
    _parse_label_list,
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402
from materialize_ppc_tdcp_height_prior_candidate import (  # noqa: E402
    _estimate_tdcp_height_series,
    _solve_z_for_height,
)
from sim_ppc_addcand_sweep import _VARIANTS_INDIV  # noqa: E402
from sim_ppc_selector_sweep import _CANDIDATES_PHASE11V, _DIAG_ROOT  # noqa: E402

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


@dataclass(frozen=True)
class Candidate:
    label: str
    pos: dict[float, np.ndarray]
    diag: dict[float, dict[str, str]]


def _candidate_dir_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for label, dir_name, _restrict in _CANDIDATES_PHASE11V:
        out.setdefault(label, dir_name)
    for _name, entries in _VARIANTS_INDIV:
        for label, dir_name in entries:
            out.setdefault(label, dir_name)
    out.update({
        "xr25_glonassar": "full_ratio25_lock3_trustedseed_glonassar",
        "xr17_glonassar": "full_ratio17_lock3_trustedseed_glonassar",
        "xmlc1psig005": "full_ratio15_lock3_trustedseed_mlc1psig005",
        "xcsig005_em10": "full_ratio15_lock3_trustedseed_csig005_em10",
        "xpsig05": "full_ratio15_lock3_trustedseed_psig05",
        "xnobds_holdrlx": "full_ratio15_lock3_trustedseed_nobds_holdrlx",
    })
    return out


def _label_for_discovered_dir(dir_name: str) -> str:
    stem = dir_name
    replacements = (
        ("full_ratio15_lock3_trustedseed_", ""),
        ("full_ratio15_lock3_trustedseed", "r15base"),
        ("full_ratio2_lock3_trustedseed_", "r2_"),
        ("full_ratio2_lock3_trustedseed", "r2"),
        ("full_ratio25_lock3_trustedseed_", "r25_"),
        ("full_ratio25_lock3_trustedseed", "r25"),
        ("full_ratio", "ratio"),
        ("_lock3_trustedseed", ""),
    )
    for old, new in replacements:
        stem = stem.replace(old, new)
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return f"xd_{safe}"


def _discover_candidate_dir_map(base: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    known_dirs = set(base.values())
    diag_root = _PROJECT_ROOT / _DIAG_ROOT
    if not diag_root.is_dir():
        return out
    for path in sorted(diag_root.iterdir()):
        if not path.is_dir() or path.name in known_dirs:
            continue
        if not any(path.glob("*_full.pos")) or not any(path.glob("*_full.csv")):
            continue
        label = _label_for_discovered_dir(path.name)
        while label in out:
            label = f"{label}_x"
        out[label] = path.name
    return out


def _load_candidate(
    *,
    city: str,
    run: str,
    label: str,
    dir_name: str,
) -> Candidate | None:
    pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
    diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
    if not pos_path.is_file() or not diag_path.is_file():
        return None
    pos, _ = _load_hybrid_pos_file(pos_path)
    diag = _load_rtk_diag_file(diag_path)
    return Candidate(label=label, pos=pos, diag=diag)


def _policy_filter(
    candidates: list[Candidate],
    *,
    city: str,
    run: str,
    policy: str,
) -> list[Candidate]:
    raw = [(c.label, c.pos, c.diag) for c in candidates]
    kept = _filter_rtkdiag_candidates_by_policy(raw, city=city, run=run, policy=policy)
    return [Candidate(label=label, pos=pos, diag=diag) for label, pos, diag in kept]


def _parse_allowed_pairs(spec: str) -> dict[str, set[tuple[str, str]]]:
    """Parse 'city/run:label,label;city/run:label' into label -> allowed runs."""
    out: dict[str, set[tuple[str, str]]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk or "/" not in chunk.split(":", 1)[0]:
            raise ValueError(f"bad allowed pair spec: {chunk!r}")
        run_part, labels_part = chunk.split(":", 1)
        city, run = run_part.split("/", 1)
        for label in _parse_label_list(labels_part):
            out.setdefault(label, set()).add((city, run))
    return out


def _parse_label_factors(spec: str) -> dict[str, float]:
    """Parse 'label=factor,label=factor' into sort-key multipliers."""
    out: dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad label factor spec: {chunk!r}")
        label, value = chunk.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"bad label factor spec: {chunk!r}")
        out[label] = float(value)
    return out


def _parse_project_labels(spec: str) -> set[str] | None:
    labels = set(_parse_label_list(spec))
    if not labels:
        return None
    if "all" in labels or "*" in labels:
        return set()
    return labels


def _project_candidate_height(
    label: str,
    pos: np.ndarray,
    tow_key: float,
    height_series: dict[float, float] | None,
    project_labels: set[str] | None,
) -> np.ndarray:
    if height_series is None or project_labels is None:
        return pos
    if project_labels and label not in project_labels:
        return pos
    height = height_series.get(tow_key)
    if height is None or not np.isfinite(float(height)):
        return pos
    out = np.asarray(pos, dtype=np.float64).copy()
    out[2] = _solve_z_for_height(
        float(out[0]),
        float(out[1]),
        float(out[2]),
        float(height),
    )
    return out


def _temporal_select_params(mode: str) -> tuple[str, str, float]:
    if mode == "temporal_n2_v1":
        return "composite_n2_v4", "prevdist", 0.001
    if mode == "temporal_n2_v2":
        return "composite_n2_v4", "prevdist", 0.0006
    if mode == "temporal_n2_v3":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v4":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v5":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v6":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v7":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v8":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v9":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_n2_v10":
        return "composite_n2_v4", "prevdist", 0.00062
    if mode == "temporal_hybdelta_t3_v1":
        return "composite_t3_v2", "hybdelta", 0.0003
    if mode == "temporal_hybdelta_t3_v2":
        return "composite_t3_v2", "hybdelta", 0.0002
    if mode == "temporal_hybdelta_t3_v3":
        return "composite_t3_v2", "hybdelta", 0.00022
    if mode == "temporal_hybdelta_t3_v4":
        return "composite_t3_v4", "hybdelta", 0.0002
    if mode == "temporal_hybdelta_t3_v5":
        return "composite_t3_v4", "hybdelta", 0.0002
    if mode == "temporal_hybdelta_t3_v6":
        return "composite_t3_v4", "hybdelta", 0.0002
    if mode == "temporal_hybdelta_t3_v7":
        return "composite_t3_v4", "hybdelta", 0.0002
    if mode == "temporal_hybdelta_t3_v8":
        return "composite_t3_v4", "hybdelta", 0.0002
    if mode == "temporal_hybdelta_n2_v1":
        return "composite_n2_v4", "hybdelta", 0.0003
    if mode == "temporal_hybdelta_n3_v1":
        return "composite_n3_v3", "hybdelta", 0.0003
    if mode == "temporal_hybdelta_n3_v2":
        return "composite_n3_v4", "hybdelta", 0.0006
    if mode == "temporal_hybdelta_n3_v3":
        return "composite_n3_v4", "hybdelta", 0.0006
    if mode == "temporal_hybdelta_n3_v4":
        return "composite_n3_v4", "hybdelta", 0.0006
    if mode == "temporal_hybdelta_n3_v5":
        return "composite_n3_v4", "hybdelta", 0.0006
    if mode == "temporal_hybdelta_n3_v6":
        return "composite_n3_v4", "hybdelta", 0.0006
    return mode, "none", 0.0


def _valid_hybrid(hp: np.ndarray | None) -> bool:
    return hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp, dtype=np.float64) == 0.0)


def _simulate(
    *,
    ref: list[tuple[float, np.ndarray]],
    hybrid_pos: dict[float, np.ndarray],
    candidates: list[Candidate],
    mode: str,
    ratio_min: float,
    rms_max: float,
    label_factors: dict[str, float] | None = None,
    tdcp_height_series: dict[float, float] | None = None,
    tdcp_height_project_labels: set[str] | None = None,
) -> tuple[float, float, float, int]:
    base_mode, temporal_kind, temporal_alpha = _temporal_select_params(mode)
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    selected = 0
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_valid = _valid_hybrid(hp)
        if hp_valid:
            est[i] = np.asarray(hp, dtype=np.float64)
        predicted: np.ndarray | None = None
        if temporal_kind == "hybdelta" and prev is not None and prev_hybrid is not None and hp_valid:
            predicted = prev + (np.asarray(hp, dtype=np.float64) - prev_hybrid)
        best_key: tuple[float, float] | None = None
        best_pos: np.ndarray | None = None
        best_label = ""
        for cand_obj in candidates:
            row = cand_obj.diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_obj.pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            base_key = _rtkdiag_candidate_sort_key(row, mode=base_mode)
            if temporal_kind == "prevdist" and prev is not None:
                sort_key = (base_key[0] + temporal_alpha * float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - prev)), base_key[1])
            elif temporal_kind == "hybdelta" and predicted is not None:
                sort_key = (base_key[0] + temporal_alpha * float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - predicted)), base_key[1])
            else:
                sort_key = base_key
            if label_factors and cand_obj.label in label_factors:
                sort_key = (sort_key[0] * float(label_factors[cand_obj.label]), sort_key[1])
            if best_key is None or sort_key < best_key:
                best_key = sort_key
                best_pos = np.asarray(cand, dtype=np.float64)
                best_label = cand_obj.label
        if best_pos is not None:
            best_pos = _project_candidate_height(
                best_label,
                best_pos,
                t_key,
                tdcp_height_series,
                tdcp_height_project_labels,
            )
            est[i] = best_pos
            selected += 1
            if temporal_kind != "none":
                prev = best_pos
                if hp_valid:
                    prev_hybrid = np.asarray(hp, dtype=np.float64)
        elif temporal_kind != "none" and hp_valid:
            prev = np.asarray(hp, dtype=np.float64)
            prev_hybrid = np.asarray(hp, dtype=np.float64)
    score = score_ppc2024(est, truth)
    return (
        float(score.score_pct),
        float(score.pass_distance_m),
        float(score.total_distance_m),
        selected,
    )


def _base_replay(
    *,
    ref: list[tuple[float, np.ndarray]],
    hybrid_pos: dict[float, np.ndarray],
    candidates: list[Candidate],
    mode: str,
    ratio_min: float,
    rms_max: float,
    label_factors: dict[str, float] | None = None,
    tdcp_height_series: dict[float, float] | None = None,
    tdcp_height_project_labels: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float] | None], int]:
    base_mode, temporal_kind, temporal_alpha = _temporal_select_params(mode)
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    best_keys: list[tuple[float, float] | None] = [None] * len(ref)
    selected = 0
    prev: np.ndarray | None = None
    prev_hybrid: np.ndarray | None = None
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        hp_valid = _valid_hybrid(hp)
        if hp_valid:
            est[i] = np.asarray(hp, dtype=np.float64)
        predicted: np.ndarray | None = None
        if temporal_kind == "hybdelta" and prev is not None and prev_hybrid is not None and hp_valid:
            predicted = prev + (np.asarray(hp, dtype=np.float64) - prev_hybrid)
        best_key: tuple[float, float] | None = None
        best_pos: np.ndarray | None = None
        best_label = ""
        for cand_obj in candidates:
            row = cand_obj.diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            cand = cand_obj.pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            base_key = _rtkdiag_candidate_sort_key(row, mode=base_mode)
            if temporal_kind == "prevdist" and prev is not None:
                sort_key = (base_key[0] + temporal_alpha * float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - prev)), base_key[1])
            elif temporal_kind == "hybdelta" and predicted is not None:
                sort_key = (base_key[0] + temporal_alpha * float(np.linalg.norm(np.asarray(cand, dtype=np.float64) - predicted)), base_key[1])
            else:
                sort_key = base_key
            if label_factors and cand_obj.label in label_factors:
                sort_key = (sort_key[0] * float(label_factors[cand_obj.label]), sort_key[1])
            if best_key is None or sort_key < best_key:
                best_key = sort_key
                best_pos = np.asarray(cand, dtype=np.float64)
                best_label = cand_obj.label
        if best_pos is not None:
            best_pos = _project_candidate_height(
                best_label,
                best_pos,
                t_key,
                tdcp_height_series,
                tdcp_height_project_labels,
            )
            est[i] = best_pos
            best_keys[i] = best_key
            selected += 1
            if temporal_kind != "none":
                prev = best_pos
                if hp_valid:
                    prev_hybrid = np.asarray(hp, dtype=np.float64)
        elif temporal_kind != "none" and hp_valid:
            prev = np.asarray(hp, dtype=np.float64)
            prev_hybrid = np.asarray(hp, dtype=np.float64)
    return truth, est, best_keys, selected


def _simulate_extra(
    *,
    ref: list[tuple[float, np.ndarray]],
    truth: np.ndarray,
    base_est: np.ndarray,
    base_keys: list[tuple[float, float] | None],
    extra: Candidate,
    mode: str,
    ratio_min: float,
    rms_max: float,
    label_factors: dict[str, float] | None = None,
    tdcp_height_series: dict[float, float] | None = None,
    tdcp_height_project_labels: set[str] | None = None,
) -> tuple[float, float, float, int]:
    est = base_est.copy()
    selected = sum(1 for key in base_keys if key is not None)
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        row = extra.diag.get(t_key)
        if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
            continue
        cand = extra.pos.get(t_key)
        if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
            continue
        sort_key = _rtkdiag_candidate_sort_key(row, mode=mode)
        if label_factors and extra.label in label_factors:
            sort_key = (sort_key[0] * float(label_factors[extra.label]), sort_key[1])
        base_key = base_keys[i]
        if base_key is None or sort_key < base_key:
            est[i] = _project_candidate_height(
                extra.label,
                np.asarray(cand, dtype=np.float64),
                t_key,
                tdcp_height_series,
                tdcp_height_project_labels,
            )
            if base_key is None:
                selected += 1
    score = score_ppc2024(est, truth)
    return (
        float(score.score_pct),
        float(score.pass_distance_m),
        float(score.total_distance_m),
        selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--phase-runs-csv",
        type=Path,
        default=RESULTS_DIR / "ppc_ctrbpf_fgo_phase11dd_full_p2k_runs.csv",
    )
    parser.add_argument("--policy", default="phase11dd")
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_phase_csv_addcand_phase11dd.csv")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--only-labels",
        default="",
        help="Comma-separated candidate labels to test; empty means all known labels.",
    )
    parser.add_argument(
        "--discover-diag-dirs",
        action="store_true",
        help="Also test diag directories under libgnss_diag_phase10 that are not in the static label map.",
    )
    parser.add_argument(
        "--allowed-pairs",
        default="",
        help="Semicolon-separated per-run extra allow list: city/run:label,label;city/run:label. "
             "When set, an extra label is only tested on its listed runs.",
    )
    parser.add_argument(
        "--only-runs",
        default="",
        help="Comma-separated city/run list to replay, e.g. nagoya/run2. Empty means all phase CSV rows.",
    )
    parser.add_argument(
        "--extra-label-factors",
        default="",
        help="Comma-separated label=factor multipliers applied to extra candidate sort keys only.",
    )
    parser.add_argument(
        "--tdcp-height-project-labels",
        default="",
        help=(
            "Comma-separated selected candidate labels whose ECEF Z should be "
            "projected to the GPS L1 TDCP height series before scoring. Use "
            "'all' to project every selected candidate. Empty disables it."
        ),
    )
    parser.add_argument("--tdcp-height-max-epochs", type=int, default=10000)
    parser.add_argument("--tdcp-height-postfit-max-m", type=float, default=2.0)
    args = parser.parse_args()

    label_to_dir = _candidate_dir_map()
    if args.discover_diag_dirs:
        label_to_dir = _discover_candidate_dir_map(label_to_dir)
    with args.phase_runs_csv.open(newline="") as fh:
        phase_rows = list(csv.DictReader(fh))
    if args.only_runs.strip():
        keep_runs = {
            tuple(chunk.strip().split("/", 1))
            for chunk in args.only_runs.split(",")
            if "/" in chunk.strip()
        }
        phase_rows = [
            row for row in phase_rows
            if (str(row["city"]), str(row["run"])) in keep_runs
        ]
        if not phase_rows:
            raise SystemExit(f"--only-runs matched no rows: {args.only_runs}")

    only_labels = set(_parse_label_list(args.only_labels)) if args.only_labels else None
    allowed_pairs = _parse_allowed_pairs(str(args.allowed_pairs))
    extra_label_factors = _parse_label_factors(str(args.extra_label_factors))
    tdcp_height_project_labels = _parse_project_labels(
        str(args.tdcp_height_project_labels)
    )
    if allowed_pairs:
        only_labels = set(allowed_pairs) if only_labels is None else set(only_labels) | set(allowed_pairs)
    rows: list[dict[str, object]] = []
    aggregate_pass: dict[str, float] = {}
    aggregate_total: dict[str, float] = {}
    aggregate_labels = sorted(only_labels) if only_labels is not None else sorted(label_to_dir)

    for phase_row in phase_rows:
        city = str(phase_row["city"])
        run = str(phase_row["run"])
        base_labels = _parse_label_list(str(phase_row["rtkdiag_candidate_labels"]))
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        tdcp_height_series: dict[float, float] | None = None
        if tdcp_height_project_labels is not None:
            tdcp_height_series, accepted = _estimate_tdcp_height_series(
                args.data_root / city / run,
                int(args.tdcp_height_max_epochs),
                float(args.tdcp_height_postfit_max_m),
            )
            print(
                f"[{city}/{run}] TDCP height series: "
                f"n={len(tdcp_height_series)} accepted={accepted}"
            )
        cfg = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run,
            policy=str(args.policy),
            city=city,
        )
        mode = str(cfg.rtkdiag_candidate_select_mode)
        _base_mode, temporal_kind, _temporal_alpha = _temporal_select_params(mode)
        ratio_min = float(cfg.rtkdiag_candidate_ratio_min)
        rms_max = float(cfg.rtkdiag_candidate_residual_rms_max)
        label_factors = {
            str(label): float(factor)
            for label, factor in cfg.rtkdiag_candidate_label_factors
        }
        label_factors.update(extra_label_factors)

        base_candidates: list[Candidate] = []
        for label in base_labels:
            dir_name = label_to_dir.get(label)
            if dir_name is None:
                continue
            cand = _load_candidate(city=city, run=run, label=label, dir_name=dir_name)
            if cand is not None:
                base_candidates.append(cand)

        truth, base_est, base_keys, base_selected = _base_replay(
            ref=ref,
            hybrid_pos=hybrid_pos,
            candidates=base_candidates,
            mode=mode,
            ratio_min=ratio_min,
            rms_max=rms_max,
            label_factors=label_factors,
            tdcp_height_series=tdcp_height_series,
            tdcp_height_project_labels=tdcp_height_project_labels,
        )
        base_score = score_ppc2024(base_est, truth)
        base_ppc = float(base_score.score_pct)
        base_pass = float(base_score.pass_distance_m)
        total_m = float(base_score.total_distance_m)
        aggregate_pass["base"] = aggregate_pass.get("base", 0.0) + base_pass
        aggregate_total["base"] = aggregate_total.get("base", 0.0) + total_m
        for label in aggregate_labels:
            variant = f"+{label}"
            aggregate_pass[variant] = aggregate_pass.get(variant, 0.0) + base_pass
            aggregate_total[variant] = aggregate_total.get(variant, 0.0) + total_m
        rows.append({
            "city": city,
            "run": run,
            "variant": "base",
            "ppc_pct": base_ppc,
            "pass_m": base_pass,
            "total_m": total_m,
            "delta_pass_m": 0.0,
            "selected": base_selected,
            "base_labels_loaded": len(base_candidates),
            "base_labels_csv": len(base_labels),
        })

        for label, dir_name in sorted(label_to_dir.items()):
            if only_labels is not None and label not in only_labels:
                continue
            if allowed_pairs and (city, run) not in allowed_pairs.get(label, set()):
                continue
            if label in base_labels:
                continue
            cand = _load_candidate(city=city, run=run, label=label, dir_name=dir_name)
            if cand is None:
                continue
            filtered_with_extra = _policy_filter(
                [*base_candidates, cand],
                city=city,
                run=run,
                policy=str(args.policy),
            )
            base_label_set = {base.label for base in base_candidates}
            filtered_label_set = {filtered.label for filtered in filtered_with_extra}
            if filtered_label_set == base_label_set:
                ppc, pass_m, total_m2, selected = base_ppc, base_pass, total_m, base_selected
            elif temporal_kind != "none" or not base_label_set.issubset(filtered_label_set) or label not in filtered_label_set:
                ppc, pass_m, total_m2, selected = _simulate(
                    ref=ref,
                    hybrid_pos=hybrid_pos,
                    candidates=filtered_with_extra,
                    mode=mode,
                    ratio_min=ratio_min,
                    rms_max=rms_max,
                    label_factors=label_factors,
                    tdcp_height_series=tdcp_height_series,
                    tdcp_height_project_labels=tdcp_height_project_labels,
                )
            else:
                # Fast path is valid when policy kept the extra and did not
                # alter the already materialized phase CSV base pool. Temporal
                # selectors must replay fully because an extra pick can change
                # the state used at later epochs.
                ppc, pass_m, total_m2, selected = _simulate_extra(
                    ref=ref,
                    truth=truth,
                    base_est=base_est,
                    base_keys=base_keys,
                    extra=cand,
                    mode=mode,
                    ratio_min=ratio_min,
                    rms_max=rms_max,
                    label_factors=label_factors,
                    tdcp_height_series=tdcp_height_series,
                    tdcp_height_project_labels=tdcp_height_project_labels,
                )
            variant = f"+{label}"
            aggregate_pass[variant] += pass_m - base_pass
            rows.append({
                "city": city,
                "run": run,
                "variant": variant,
                "ppc_pct": ppc,
                "pass_m": pass_m,
                "total_m": total_m2,
                "delta_pass_m": pass_m - base_pass,
                "selected": selected,
                "base_labels_loaded": len(base_candidates),
                "base_labels_csv": len(base_labels),
            })

        if only_labels:
            combo: list[Candidate] = []
            for label in sorted(only_labels):
                if allowed_pairs and (city, run) not in allowed_pairs.get(label, set()):
                    continue
                if label in base_labels:
                    continue
                dir_name = label_to_dir.get(label)
                if dir_name is None:
                    continue
                cand = _load_candidate(city=city, run=run, label=label, dir_name=dir_name)
                if cand is not None:
                    combo.append(cand)
            if combo:
                combo_candidates = _policy_filter(
                    [*base_candidates, *combo],
                    city=city,
                    run=run,
                    policy=str(args.policy),
                )
                ppc, pass_m, total_m2, selected = _simulate(
                    ref=ref,
                    hybrid_pos=hybrid_pos,
                    candidates=combo_candidates,
                    mode=mode,
                    ratio_min=ratio_min,
                    rms_max=rms_max,
                    label_factors=label_factors,
                    tdcp_height_series=tdcp_height_series,
                    tdcp_height_project_labels=tdcp_height_project_labels,
                )
                variant = "+combo"
                aggregate_pass[variant] = aggregate_pass.get(variant, 0.0) + pass_m
                aggregate_total[variant] = aggregate_total.get(variant, 0.0) + total_m2
                rows.append({
                    "city": city,
                    "run": run,
                    "variant": variant,
                    "ppc_pct": ppc,
                    "pass_m": pass_m,
                    "total_m": total_m2,
                    "delta_pass_m": pass_m - base_pass,
                    "selected": selected,
                    "base_labels_loaded": len(base_candidates),
                    "base_labels_csv": len(base_labels),
                })
            else:
                aggregate_pass["+combo"] = aggregate_pass.get("+combo", 0.0) + base_pass
                aggregate_total["+combo"] = aggregate_total.get("+combo", 0.0) + total_m

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    base_pass = aggregate_pass["base"]
    base_total = aggregate_total["base"]
    print(f"base aggregate {100.0 * base_pass / base_total:.6f}% pass={base_pass:.3f}/{base_total:.3f}")
    ranked = []
    for variant, pass_m in aggregate_pass.items():
        if variant == "base":
            continue
        total = aggregate_total[variant]
        ranked.append((100.0 * (pass_m - base_pass) / base_total, variant, 100.0 * pass_m / total, pass_m))
    for delta_pp, variant, ppc, pass_m in sorted(ranked, reverse=True)[: int(args.top)]:
        print(f"{variant:24s} aggregate={ppc:.6f}% delta={delta_pp:+.6f}pp pass={pass_m:.3f}")


if __name__ == "__main__":
    main()
