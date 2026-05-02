#!/usr/bin/env python3
"""Offline test: does adding new RTK config variants help Phase 11aa pool?

Iterates over a list of `_EXTRA` candidate variants. For each variant
(or combination), runs the gated/sort selection over the union of
Phase 11v candidates + that variant, and reports the ppc delta vs the
base (Phase 11v alone).
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
    _rtkdiag_candidate_gate,
    _rtkdiag_candidate_sort_key,
)
from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402

from sim_ppc_selector_sweep import (  # noqa: E402
    _CANDIDATES_PHASE11V,
    _DIAG_ROOT,
    _eligible_for_run,
    _FULL_RUNS,
)

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

# Candidate variants to test individually (each adds ONE extra to Phase 11v).
_VARIANTS_INDIV: list[tuple[str, list[tuple[str, str]]]] = [
    ("r15ga", [("r15ga", "full_ratio15_lock3_trustedseed_glonassar")]),
    ("r20ga", [("r20ga", "full_ratio2_lock3_trustedseed_glonassar")]),
    ("psig1", [("psig1", "full_ratio15_lock3_trustedseed_psig1")]),
    ("psig2", [("psig2", "full_ratio15_lock3_trustedseed_psig2")]),
    ("psig05", [("psig05", "full_ratio15_lock3_trustedseed_psig05")]),
    ("psig3", [("psig3", "full_ratio15_lock3_trustedseed_psig3")]),
    ("holdrlx", [("holdrlx", "full_ratio15_lock3_trustedseed_holdrlx")]),
    ("holdvrlx", [("holdvrlx", "full_ratio15_lock3_trustedseed_holdvrlx")]),
    ("psig1_holdrlx", [("psig1_holdrlx", "full_ratio15_lock3_trustedseed_psig1_holdrlx")]),
    ("csig1", [("csig1", "full_ratio15_lock3_trustedseed_csig1")]),
    ("minobs6", [("minobs6", "full_ratio15_lock3_trustedseed_minobs6")]),
    ("nobds", [("nobds", "full_ratio15_lock3_trustedseed_nobds")]),
    ("em10", [("em10", "full_ratio15_lock3_trustedseed_elevmask10")]),
    ("r12ga", [("r12ga", "full_ratio12_lock3_trustedseed_glonassar")]),
    ("noglo", [("noglo", "full_ratio15_lock3_trustedseed_noglo")]),
    ("csig05", [("csig05", "full_ratio15_lock3_trustedseed_csig05")]),
    ("arfilt", [("arfilt", "full_ratio15_lock3_trustedseed_arfilt")]),
    ("rout30", [("rout30", "full_ratio15_lock3_trustedseed_rout30")]),
    ("rout20", [("rout20", "full_ratio15_lock3_trustedseed_rout20")]),
    ("rout100", [("rout100", "full_ratio15_lock3_trustedseed_rout100")]),
    ("mb5k", [("mb5k", "full_ratio15_lock3_trustedseed_mb5k")]),
    ("minar5", [("minar5", "full_ratio15_lock3_trustedseed_minar5")]),
    ("csig05_psig1", [("csig05_psig1", "full_ratio15_lock3_trustedseed_csig05_psig1")]),
    ("csig05_holdrlx", [("csig05_holdrlx", "full_ratio15_lock3_trustedseed_csig05_holdrlx")]),
    ("csig01", [("csig01", "full_ratio15_lock3_trustedseed_csig01")]),
    ("csig05_nobds", [("csig05_nobds", "full_ratio15_lock3_trustedseed_csig05_nobds")]),
    ("csig05_psig1_holdrlx", [("csig05_psig1_holdrlx", "full_ratio15_lock3_trustedseed_csig05_psig1_holdrlx")]),
    ("csig05_holdvrlx", [("csig05_holdvrlx", "full_ratio15_lock3_trustedseed_csig05_holdvrlx")]),
    ("csig05_em10", [("csig05_em10", "full_ratio15_lock3_trustedseed_csig05_em10")]),
    ("csig01_holdrlx", [("csig01_holdrlx", "full_ratio15_lock3_trustedseed_csig01_holdrlx")]),
    ("csig05_psig1_holdvrlx", [("csig05_psig1_holdvrlx", "full_ratio15_lock3_trustedseed_csig05_psig1_holdvrlx")]),
    ("csig01_psig1_holdrlx", [("csig01_psig1_holdrlx", "full_ratio15_lock3_trustedseed_csig01_psig1_holdrlx")]),
    ("csig05_holdrlx_em10", [("csig05_holdrlx_em10", "full_ratio15_lock3_trustedseed_csig05_holdrlx_em10")]),
    ("psig1_holdvrlx", [("psig1_holdvrlx", "full_ratio15_lock3_trustedseed_psig1_holdvrlx")]),
    ("csig05_nobds_holdrlx", [("csig05_nobds_holdrlx", "full_ratio15_lock3_trustedseed_csig05_nobds_holdrlx")]),
    ("csig01_holdvrlx", [("csig01_holdvrlx", "full_ratio15_lock3_trustedseed_csig01_holdvrlx")]),
    ("csig005", [("csig005", "full_ratio15_lock3_trustedseed_csig005")]),
    ("csig01_nobds", [("csig01_nobds", "full_ratio15_lock3_trustedseed_csig01_nobds")]),
    ("nobds_holdrlx", [("nobds_holdrlx", "full_ratio15_lock3_trustedseed_nobds_holdrlx")]),
    ("csig01_psig1", [("csig01_psig1", "full_ratio15_lock3_trustedseed_csig01_psig1")]),
    ("csig005_holdrlx", [("csig005_holdrlx", "full_ratio15_lock3_trustedseed_csig005_holdrlx")]),
    ("csig005_psig1", [("csig005_psig1", "full_ratio15_lock3_trustedseed_csig005_psig1")]),
    ("csig005_nobds", [("csig005_nobds", "full_ratio15_lock3_trustedseed_csig005_nobds")]),
    ("csig005_em10", [("csig005_em10", "full_ratio15_lock3_trustedseed_csig005_em10")]),
    ("csig0001", [("csig0001", "full_ratio15_lock3_trustedseed_csig0001")]),
    ("csig00005", [("csig00005", "full_ratio15_lock3_trustedseed_csig00005")]),
    ("nopostflt", [("nopostflt", "full_ratio15_lock3_trustedseed_nopostflt")]),
    ("csig05_nopostflt", [("csig05_nopostflt", "full_ratio15_lock3_trustedseed_csig05_nopostflt")]),
    ("csig005_nopostflt", [("csig005_nopostflt", "full_ratio15_lock3_trustedseed_csig005_nopostflt")]),
    ("iono_iflc", [("iono_iflc", "full_ratio15_lock3_trustedseed_iono_iflc")]),
    ("csig005_holdvrlx", [("csig005_holdvrlx", "full_ratio15_lock3_trustedseed_csig005_holdvrlx")]),
    ("csig005_psig1_holdrlx", [("csig005_psig1_holdrlx", "full_ratio15_lock3_trustedseed_csig005_psig1_holdrlx")]),
    ("csig005_glonassar", [("csig005_glonassar", "full_ratio15_lock3_trustedseed_csig005_glonassar")]),
    ("csig005_psig1_holdvrlx", [("csig005_psig1_holdvrlx", "full_ratio15_lock3_trustedseed_csig005_psig1_holdvrlx")]),
    ("noarfilt", [("noarfilt", "full_ratio15_lock3_trustedseed_noarfilt")]),
    ("csig05_noarfilt", [("csig05_noarfilt", "full_ratio15_lock3_trustedseed_csig05_noarfilt")]),
    ("r05", [("r05", "full_ratio15_lock3_trustedseed_r05")]),
    ("r08", [("r08", "full_ratio15_lock3_trustedseed_r08")]),
    ("r10", [("r10", "full_ratio15_lock3_trustedseed_r10")]),
    ("csig05_r10", [("csig05_r10", "full_ratio15_lock3_trustedseed_csig05_r10")]),
    ("csig005_r10", [("csig005_r10", "full_ratio15_lock3_trustedseed_csig005_r10")]),
    ("csig005_minar3", [("csig005_minar3", "full_ratio15_lock3_trustedseed_csig005_minar3")]),
    ("mb1k", [("mb1k", "full_ratio15_lock3_trustedseed_mb1k")]),
    ("mb50k", [("mb50k", "full_ratio15_lock3_trustedseed_mb50k")]),
    ("modeauto", [("modeauto", "full_ratio15_lock3_trustedseed_modeauto")]),
    ("modestatic", [("modestatic", "full_ratio15_lock3_trustedseed_modestatic")]),
    ("onlyG", [("onlyG", "full_ratio15_lock3_trustedseed_onlyG")]),
    ("psig01", [("psig01", "full_ratio15_lock3_trustedseed_psig01")]),
    ("psig005", [("psig005", "full_ratio15_lock3_trustedseed_psig005")]),
    ("csig005_psig01", [("csig005_psig01", "full_ratio15_lock3_trustedseed_csig005_psig01")]),
    ("onlyG_holdrlx", [("onlyG_holdrlx", "full_ratio15_lock3_trustedseed_onlyG_holdrlx")]),
    ("onlyG_csig05", [("onlyG_csig05", "full_ratio15_lock3_trustedseed_onlyG_csig05")]),
    ("onlyG_csig005", [("onlyG_csig005", "full_ratio15_lock3_trustedseed_onlyG_csig005")]),
    ("onlyG_psig1", [("onlyG_psig1", "full_ratio15_lock3_trustedseed_onlyG_psig1")]),
    ("onlyG_holdvrlx", [("onlyG_holdvrlx", "full_ratio15_lock3_trustedseed_onlyG_holdvrlx")]),
    ("onlyG_r05", [("onlyG_r05", "full_ratio15_lock3_trustedseed_onlyG_r05")]),
    ("onlyG_csig01", [("onlyG_csig01", "full_ratio15_lock3_trustedseed_onlyG_csig01")]),
    ("onlyG_em10", [("onlyG_em10", "full_ratio15_lock3_trustedseed_onlyG_em10")]),
    ("onlyG_psig1_csig05", [("onlyG_psig1_csig05", "full_ratio15_lock3_trustedseed_onlyG_psig1_csig05")]),
    ("onlyG_psig1_holdrlx", [("onlyG_psig1_holdrlx", "full_ratio15_lock3_trustedseed_onlyG_psig1_holdrlx")]),
    ("oGc005hr", [("oGc005hr", "full_ratio15_lock3_trustedseed_oGc005hr")]),
    ("oGc005p1", [("oGc005p1", "full_ratio15_lock3_trustedseed_oGc005p1")]),
    ("oGc005p1hr", [("oGc005p1hr", "full_ratio15_lock3_trustedseed_oGc005p1hr")]),
    ("oGc01p1", [("oGc01p1", "full_ratio15_lock3_trustedseed_oGc01p1")]),
    ("c005p1", [("c005p1", "full_ratio15_lock3_trustedseed_c005p1")]),
    ("c005p1hr", [("c005p1hr", "full_ratio15_lock3_trustedseed_c005p1hr")]),
    ("arfm05", [("arfm05", "full_ratio15_lock3_trustedseed_arfm05")]),
    ("prsurv", [("prsurv", "full_ratio15_lock3_trustedseed_prsurv")]),
    ("oGc0001p1", [("oGc0001p1", "full_ratio15_lock3_trustedseed_oGc0001p1")]),
    ("oGc00005p1", [("oGc00005p1", "full_ratio15_lock3_trustedseed_oGc00005p1")]),
    ("oGc005p05", [("oGc005p05", "full_ratio15_lock3_trustedseed_oGc005p05")]),
    ("oGc005p2", [("oGc005p2", "full_ratio15_lock3_trustedseed_oGc005p2")]),
    ("oGc005p1ar3", [("oGc005p1ar3", "full_ratio15_lock3_trustedseed_oGc005p1ar3")]),
    ("nobdsc005p1", [("nobdsc005p1", "full_ratio15_lock3_trustedseed_nobdsc005p1")]),
    ("ratio12", [("ratio12", "full_ratio15_lock3_trustedseed_ratio12")]),
    ("ratio12oG", [("ratio12oG", "full_ratio15_lock3_trustedseed_ratio12oG")]),
    ("mlc2", [("mlc2", "full_ratio15_lock3_trustedseed_mlc2")]),
    ("mlc2oG", [("mlc2oG", "full_ratio15_lock3_trustedseed_mlc2oG")]),
    ("em5", [("em5", "full_ratio15_lock3_trustedseed_em5")]),
    ("ionest", [("ionest", "full_ratio15_lock3_trustedseed_ionest")]),
    ("em3", [("em3", "full_ratio15_lock3_trustedseed_em3")]),
    ("em7", [("em7", "full_ratio15_lock3_trustedseed_em7")]),
    ("em5oG", [("em5oG", "full_ratio15_lock3_trustedseed_em5oG")]),
    ("em5c005p1", [("em5c005p1", "full_ratio15_lock3_trustedseed_em5c005p1")]),
    ("mlc1oG", [("mlc1oG", "full_ratio15_lock3_trustedseed_mlc1oG")]),
    ("em5mlc2oG", [("em5mlc2oG", "full_ratio15_lock3_trustedseed_em5mlc2oG")]),
    ("mlc1", [("mlc1", "full_ratio15_lock3_trustedseed_mlc1")]),
    ("mlc1oGc005", [("mlc1oGc005", "full_ratio15_lock3_trustedseed_mlc1oGc005")]),
    ("mlc1oGp1", [("mlc1oGp1", "full_ratio15_lock3_trustedseed_mlc1oGp1")]),
    ("mlc1oGc005p1", [("mlc1oGc005p1", "full_ratio15_lock3_trustedseed_mlc1oGc005p1")]),
    ("em3oG", [("em3oG", "full_ratio15_lock3_trustedseed_em3oG")]),
    ("em3mlc1oG", [("em3mlc1oG", "full_ratio15_lock3_trustedseed_em3mlc1oG")]),
    ("mlc1c005p1", [("mlc1c005p1", "full_ratio15_lock3_trustedseed_mlc1c005p1")]),
    ("mlc1oGc0001", [("mlc1oGc0001", "full_ratio15_lock3_trustedseed_mlc1oGc0001")]),
    ("mlc1oGc005em3", [("mlc1oGc005em3", "full_ratio15_lock3_trustedseed_mlc1oGc005em3")]),
    ("mlc1oGc005r12", [("mlc1oGc005r12", "full_ratio15_lock3_trustedseed_mlc1oGc005r12")]),
    ("mlc0oG", [("mlc0oG", "full_ratio15_lock3_trustedseed_mlc0oG")]),
    ("mlc1nobds", [("mlc1nobds", "full_ratio15_lock3_trustedseed_mlc1nobds")]),
    ("mlc0", [("mlc0", "full_ratio15_lock3_trustedseed_mlc0")]),
    ("mlc1c005", [("mlc1c005", "full_ratio15_lock3_trustedseed_mlc1c005")]),
    ("mlc1r10", [("mlc1r10", "full_ratio15_lock3_trustedseed_mlc1r10")]),
    ("mlc1psig005", [("mlc1psig005", "full_ratio15_lock3_trustedseed_mlc1psig005")]),
    ("mlc1c005r10", [("mlc1c005r10", "full_ratio15_lock3_trustedseed_mlc1c005r10")]),
    ("mlc2nobds", [("mlc2nobds", "full_ratio15_lock3_trustedseed_mlc2nobds")]),
    ("mlc1r10c005p1", [("mlc1r10c005p1", "full_ratio15_lock3_trustedseed_mlc1r10c005p1")]),
    ("mlc1r10oG", [("mlc1r10oG", "full_ratio15_lock3_trustedseed_mlc1r10oG")]),
    ("mlc1r10oGc005", [("mlc1r10oGc005", "full_ratio15_lock3_trustedseed_mlc1r10oGc005")]),
    ("mlc1r08", [("mlc1r08", "full_ratio15_lock3_trustedseed_mlc1r08")]),
    ("mlc1r08c005", [("mlc1r08c005", "full_ratio15_lock3_trustedseed_mlc1r08c005")]),
    ("r10c005p1", [("r10c005p1", "full_ratio15_lock3_trustedseed_r10c005p1")]),
]

# Combination tests (union of multiple).
_VARIANTS_COMBO: list[tuple[str, list[tuple[str, str]]]] = [
    ("all_psig", [
        ("psig05", "full_ratio15_lock3_trustedseed_psig05"),
        ("psig1", "full_ratio15_lock3_trustedseed_psig1"),
        ("psig2", "full_ratio15_lock3_trustedseed_psig2"),
        ("psig3", "full_ratio15_lock3_trustedseed_psig3"),
    ]),
    ("all_hold", [
        ("holdrlx", "full_ratio15_lock3_trustedseed_holdrlx"),
        ("holdvrlx", "full_ratio15_lock3_trustedseed_holdvrlx"),
    ]),
    ("all_new", [
        ("psig05", "full_ratio15_lock3_trustedseed_psig05"),
        ("psig3", "full_ratio15_lock3_trustedseed_psig3"),
        ("holdvrlx", "full_ratio15_lock3_trustedseed_holdvrlx"),
        ("psig1_holdrlx", "full_ratio15_lock3_trustedseed_psig1_holdrlx"),
        ("csig1", "full_ratio15_lock3_trustedseed_csig1"),
        ("minobs6", "full_ratio15_lock3_trustedseed_minobs6"),
        ("nobds", "full_ratio15_lock3_trustedseed_nobds"),
    ]),
]


@dataclass
class Result:
    city: str
    run: str
    variant: str
    ppc_pct: float
    pass_m: float
    total_m: float
    n_gated: int


def _load_one(city, run, label, dir_name):
    pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
    diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
    if not pos_path.is_file() or not diag_path.is_file():
        return None
    pos, _ = _load_hybrid_pos_file(pos_path)
    diag = _load_rtk_diag_file(diag_path)
    return (label, pos, diag)


def _load_phase11v_run(city, run):
    out = []
    for label, dir_name, restrict in _CANDIDATES_PHASE11V:
        if not _eligible_for_run(city, run, restrict):
            continue
        c = _load_one(city, run, label, dir_name)
        if c is not None:
            out.append(c)
    return out


def _simulate(city, run, mode, hybrid_pos, candidates, ref, ratio_min, rms_max):
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    n_gated = 0
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
        best_key = None
        best_pos = None
        any_gated = False
        for _, cand_pos, cand_diag in candidates:
            row = cand_diag.get(t_key)
            if not _rtkdiag_candidate_gate(row, ratio_min=ratio_min, residual_rms_max=rms_max):
                continue
            any_gated = True
            cand = cand_pos.get(t_key)
            if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                continue
            sort_key = _rtkdiag_candidate_sort_key(row, mode=mode)
            if best_key is None or sort_key < best_key:
                best_key = sort_key
                best_pos = np.asarray(cand, dtype=np.float64)
        if any_gated:
            n_gated += 1
        if best_pos is not None:
            est[i] = best_pos
    score = score_ppc2024(est, truth)
    return float(score.score_pct), float(score.pass_distance_m), float(score.total_distance_m), n_gated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11aa")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_addcand_sweep_phase11aa.csv")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip variants with missing files (default: include partial loads)")
    args = parser.parse_args()

    all_variants = [("base", [])] + _VARIANTS_INDIV + _VARIANTS_COMBO

    # Per-run/per-variant aggregate.
    rows: list[Result] = []
    pass_by_variant: dict[str, float] = {}
    total_by_variant: dict[str, float] = {}

    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        variant_cfg = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run, policy=str(args.policy), city=city,
        )
        ratio_min = float(variant_cfg.rtkdiag_candidate_ratio_min)
        rms_max = float(variant_cfg.rtkdiag_candidate_residual_rms_max)
        mode = str(variant_cfg.rtkdiag_candidate_select_mode)
        base_cands = _filter_rtkdiag_candidates_by_policy(
            _load_phase11v_run(city, run),
            city=city, run=run, policy=str(args.policy),
        )

        print(f"\n=== {city}/{run}: ratio>={ratio_min} rms<={rms_max} mode={mode} ===")
        base_ppc = base_pm = base_tm = 0.0
        base_ng = 0
        for variant_name, extras in all_variants:
            extra_loaded: list = []
            missing = False
            for label, dir_name in extras:
                c = _load_one(city, run, label, dir_name)
                if c is None:
                    missing = True
                    if args.skip_missing:
                        break
                else:
                    extra_loaded.append(c)
            if missing and args.skip_missing:
                print(f"  {variant_name:<16s}: SKIP (missing files)")
                continue
            cands = base_cands + extra_loaded
            ppc, pm, tm, ng = _simulate(city, run, mode, hybrid_pos, cands, ref, ratio_min, rms_max)
            rows.append(Result(city, run, variant_name, ppc, pm, tm, ng))
            pass_by_variant[variant_name] = pass_by_variant.get(variant_name, 0.0) + pm
            total_by_variant[variant_name] = total_by_variant.get(variant_name, 0.0) + tm
            if variant_name == "base":
                base_ppc, base_pm, base_tm, base_ng = ppc, pm, tm, ng
                print(f"  base            : ppc={ppc:.4f}% (gated {ng})")
            else:
                d_pm = pm - base_pm
                d_ng = ng - base_ng
                print(f"  +{variant_name:<14s}: ppc={ppc:.4f}% (gated {ng}, {d_ng:+d} epochs); dpass={d_pm:+.1f}m")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "variant", "ppc_pct", "pass_m", "total_m", "n_gated"])
        for r in rows:
            w.writerow([r.city, r.run, r.variant, f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}", r.n_gated])

    base_pass = pass_by_variant.get("base", 0.0)
    base_total = total_by_variant.get("base", 0.0)
    print(f"\nAggregate (uniform across all 6 runs):")
    print(f"  base            : ppc={100*base_pass/base_total:.4f}% (pass {base_pass:.1f}/{base_total:.1f})")
    for variant_name, _ in _VARIANTS_INDIV + _VARIANTS_COMBO:
        if variant_name not in pass_by_variant:
            continue
        ps = pass_by_variant[variant_name]
        ts = total_by_variant[variant_name]
        delta = 100 * (ps - base_pass) / base_total
        print(f"  +{variant_name:<14s}: ppc={100*ps/ts:.4f}% (pass {ps:.1f}/{ts:.1f}); delta={delta:+.4f}pp")


if __name__ == "__main__":
    main()
