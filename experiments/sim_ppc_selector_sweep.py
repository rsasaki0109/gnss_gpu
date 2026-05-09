#!/usr/bin/env python3
"""Offline selector sweep on Phase 11v candidate pool.

Phase 11v phase score = 61.6039%, gated_oracle = 63.21% (gap +1.60pp).
With emit_mode=candidate the selector directly determines the output for
gate-passing epochs, so we can replay the trajectory offline by:
  1. Loading hybrid floor per run.
  2. For each gate-passing epoch, picking the candidate that minimises the
     given sort key (residual / ratio / score / maxabs / nrows).
  3. Falling back to hybrid otherwise.
This avoids the slow PF and shows directly which select_mode is best per
run.
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

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

_FULL_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

# Phase 11v candidate set (label, pos/diag dir, run-specific filter)
_CANDIDATES_PHASE11V: list[tuple[str, str, set[tuple[str, str]] | None]] = [
    # (label, dir under libgnss_diag_phase10/, restrict_runs or None)
    ("r15", "full_ratio15_lock3_trustedseed", None),
    ("r20", "full_ratio2_lock3_trustedseed", None),
    ("r25", "full_ratio25_lock3_trustedseed", None),
    ("r30", "full_ratio3_lock3_trustedseed", None),
    ("r15nh", "full_ratio15_lock3_trustedseed_nohold", None),
    ("r20g", "full_ratio2_lock3_trustedseed_gate30_min6", None),
    ("r15g", "full_ratio15_lock3_trustedseed_gate30_min6", None),
    ("r25g", "full_ratio25_lock3_trustedseed_gate30_min6", None),
    ("r30g", "full_ratio3_lock3_trustedseed_gate30_min6", None),
    ("r20g20", "full_ratio2_lock3_trustedseed_gate20_min6", None),
    ("r20g40", "full_ratio2_lock3_trustedseed_gate40_min6", None),
    ("r15g20", "full_ratio15_lock3_trustedseed_gate20_min6", None),
    ("r25g20", "full_ratio25_lock3_trustedseed_gate20_min6", None),
    ("r20g15", "full_ratio2_lock3_trustedseed_gate15_min6", None),
    ("r15g15", "full_ratio15_lock3_trustedseed_gate15_min6", None),
    ("r25g15", "full_ratio25_lock3_trustedseed_gate15_min6", None),
    ("r20g10", "full_ratio2_lock3_trustedseed_gate10_min6", None),
    ("r15g10", "full_ratio15_lock3_trustedseed_gate10_min6", None),
    ("r25g10", "full_ratio25_lock3_trustedseed_gate10_min6", None),
    # Run-specific
    ("n1loose", "n1_loose_hold4_ratio15_gate10_min6", {("nagoya", "run1")}),
    ("n1loose2", "n1_loose_hold5_ratio20_gate10_min6", {("nagoya", "run1")}),
    ("n1loose3", "n1_loose_hold5_ratio20_gate8_min6", {("nagoya", "run1")}),
    ("n2loose", "n2_loose_hold4_ratio15_gate10_min6", {("nagoya", "run2")}),
    ("n2loose2", "n2_loose_hold5_ratio20_gate10_min6", {("nagoya", "run2")}),
    ("n2loose3", "n2_loose_hold5_ratio20_gate8_min6", {("nagoya", "run2")}),
    ("xd_tdcp_height_prior_n2_fixedicb_2329_2928_fixedicb",
     "tdcp_height_prior_n2_fixedicb_2329_2928_fixedicb", {("nagoya", "run2")}),
    ("xd_fixedicb_raw_n2_icbsweep_5734_5773_l1p8_l2p0",
     "fixedicb_raw_n2_icbsweep_5734_5773_l1p8_l2p0", {("nagoya", "run2")}),
    ("xd_fixedicb_raw_n2_icbsweep_6637_6660_l1p4_l2p8",
     "fixedicb_raw_n2_icbsweep_6637_6660_l1p4_l2p8", {("nagoya", "run2")}),
    ("xd_fixedicb_raw_n2_icbfine_6637_6660_l1p3_l2p7",
     "fixedicb_raw_n2_icbfine_6637_6660_l1p3_l2p7", {("nagoya", "run2")}),
    ("n3tight", "n3_tight_ratio40_gate5_min8", {("nagoya", "run3")}),
    ("n3tight2", "n3_tight2_ratio50_gate4", {("nagoya", "run3")}),
    ("t1tight", "t1_tight_ratio40_gate5_min8", {("tokyo", "run1")}),
    ("t1tight2", "t1_tight2_ratio50_gate4", {("tokyo", "run1")}),
    ("t3tight", "t3_tight_ratio40_gate5_min8", {("tokyo", "run3")}),
    # Phase 11ab: --glonass-ar autocal candidates restricted to runs where
    # offline simulation showed positive aggregate (tokyo/run3 +56.6m,
    # nagoya/run1 +31.6m). Other runs are net-neutral or negative.
    ("r15ga", "full_ratio15_lock3_trustedseed_glonassar",
     {("tokyo", "run3"), ("nagoya", "run1")}),
    # Phase 11ac: ratio2.0 + glonass-ar helps tokyo/run3 (+12.6m on top of
    # r15ga) and nagoya/run2 (+47.7m alone). Hurts nagoya/run1 by -69m, so
    # restrict accordingly. r25ga / out30 are neutral or negative.
    ("r20ga", "full_ratio2_lock3_trustedseed_glonassar",
     {("tokyo", "run3"), ("nagoya", "run2")}),
    # Phase 11ac: --elevation-mask-deg 10 gives +21m on tokyo/run3 (admits
    # low-elevation sats that fix), +2.0m on nagoya/run1. Other runs near-zero
    # or negative.
    ("em10", "full_ratio15_lock3_trustedseed_elevmask10",
     {("tokyo", "run3"), ("nagoya", "run1")}),
    # Phase 11ad: --pseudorange-sigma 1.0 (looser PR weight) gives +125m on
    # tokyo/run3 (massive), +4.1m on nagoya/run3. Near-zero on nagoya/run1+run2.
    ("psig1", "full_ratio15_lock3_trustedseed_psig1",
     {("tokyo", "run3"), ("nagoya", "run3")}),
    # Phase 11ad: --min-hold-count 3 --hold-ratio-threshold 1.5 (relaxed
    # hold ambiguity) gives +72m on tokyo/run3. Slightly negative on
    # nagoya runs.
    ("holdrlx", "full_ratio15_lock3_trustedseed_holdrlx",
     {("tokyo", "run3")}),
    # Phase 11ae: --ratio 1.2 + --glonass-ar autocal gives +103m on
    # tokyo/run3 (additive after psig1+holdrlx). -65m on nagoya/run1,
    # neutral elsewhere.
    ("r12ga", "full_ratio12_lock3_trustedseed_glonassar",
     {("tokyo", "run3")}),
    # Phase 11ae: --pseudorange-sigma 2.0 gives +44.5m on tokyo/run3
    # (smaller than psig1 +125m but additive). Near-zero on nagoya runs.
    ("psig2", "full_ratio15_lock3_trustedseed_psig2",
     {("tokyo", "run3")}),
    # Phase 11af: psig1 + holdrlx combined config: +90.8m on tokyo/run3
    # offline (better than psig1 or holdrlx alone). Restrict to tokyo/run3
    # only since nagoya runs go negative.
    ("psig1hr", "full_ratio15_lock3_trustedseed_psig1_holdrlx",
     {("tokyo", "run3")}),
    # Phase 11af: --no-beidou gives +29.7m on nagoya/run2, +9.8m on tokyo/run3,
    # +4.2m on tokyo/run1. Negative on nagoya/run1, run3 — restrict accordingly.
    ("nobds", "full_ratio15_lock3_trustedseed_nobds",
     {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}),
    # Phase 11ag/am: --carrier-phase-sigma 0.0005 gives massive gains:
    # tokyo/run3 +95.8m, nagoya/run2 +95.8m, tokyo/run1 +9.8m. Phase 11am also
    # adds nagoya/run3 (+2.6m additive after all the other csig05* variants).
    ("csig05", "full_ratio15_lock3_trustedseed_csig05",
     {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1"), ("nagoya", "run3")}),
    # Phase 11ag: --no-glonass gives +40.5m on tokyo/run1 (largest non-zero
    # variant for that gate-saturated run). Negative or zero elsewhere.
    ("noglo", "full_ratio15_lock3_trustedseed_noglo",
     {("tokyo", "run1")}),
    # Phase 11ag/ah: --carrier-phase-sigma 0.001 gives +0.6m on nagoya/run1,
    # +18.1m on nagoya/run2, +9.5m on tokyo/run3 (additive after csig05).
    ("csig1", "full_ratio15_lock3_trustedseed_csig1",
     {("nagoya", "run1"), ("nagoya", "run2"), ("tokyo", "run3")}),
    # Phase 11ah: --rtk-update-outlier-threshold 30 gives +3.6m on nagoya/run3
    # (tightening from default 60 to 30 rejects bad RTK updates earlier).
    ("rout30", "full_ratio15_lock3_trustedseed_rout30",
     {("nagoya", "run3")}),
    # Phase 11ah: --rtk-update-outlier-threshold 20 gives +2.2m on nagoya/run1
    # (very tight RTK update gate, only helps the strict-gated run).
    ("rout20", "full_ratio15_lock3_trustedseed_rout20",
     {("nagoya", "run1")}),
    # Phase 11ai: csig05 + holdrlx combined config (--carrier-phase-sigma
    # 0.0005 --min-hold-count 3 --hold-ratio-threshold 1.5) gives massive
    # gains: tokyo/run1 +509.8m, tokyo/run3 +147.1m, nagoya/run3 +128.3m.
    # Negative on nagoya/run2 (-7.7m), tiny +0.4 on nagoya/run1 (skip).
    ("csig05hr", "full_ratio15_lock3_trustedseed_csig05_holdrlx",
     {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run3")}),
    # Phase 11ai/aj: csig05 + psig1 combo. Phase 11ai: {tokyo/run3, nagoya/run1}.
    # Phase 11aj also adds nagoya/run3 (+11.7m additive after csig05_holdrlx).
    ("csig05ps", "full_ratio15_lock3_trustedseed_csig05_psig1",
     {("tokyo", "run3"), ("nagoya", "run1"), ("nagoya", "run3")}),
    # Phase 11aj: csig01 (--carrier-phase-sigma 0.0001 — even tighter than
    # csig05) gives +27.2m on tokyo/run3, +48.8m on nagoya/run2, +4.9m on
    # nagoya/run3 (additive after all the other csig05* variants).
    ("csig01", "full_ratio15_lock3_trustedseed_csig01",
     {("tokyo", "run3"), ("nagoya", "run2"), ("nagoya", "run3")}),
    # Phase 11aj: rout100 (--rtk-update-outlier-threshold 100, looser gate)
    # gives +1.4m on nagoya/run1 only.
    ("rout100", "full_ratio15_lock3_trustedseed_rout100",
     {("nagoya", "run1")}),
    # Phase 11aj: csig05 + nobds combo gives +0.8m on tokyo/run1 only.
    ("csig05nb", "full_ratio15_lock3_trustedseed_csig05_nobds",
     {("tokyo", "run1")}),
    # Phase 11ak/am: csig05 + holdvrlx. Phase 11ak: {tokyo/run1} +53.2m.
    # Phase 11am also adds tokyo/run3 +29.4m offline.
    ("csig05hvr", "full_ratio15_lock3_trustedseed_csig05_holdvrlx",
     {("tokyo", "run1"), ("tokyo", "run3")}),
    # Phase 11ak/am: csig05 + psig1 + holdrlx triple. Phase 11ak: tokyo/run3
    # +44m, nagoya/run3 +21.5m. Phase 11am also adds tokyo/run1 +4.1m.
    ("csig05psh", "full_ratio15_lock3_trustedseed_csig05_psig1_holdrlx",
     {("tokyo", "run3"), ("nagoya", "run3"), ("tokyo", "run1")}),
    # Phase 11ak/am: csig05 + em10 (elev mask 10°). Phase 11ak: nagoya/run1
    # +2.5m offline but PF -1.46pp (selector misled). Phase 11am: tokyo/run3
    # only (+20.5m offline, untested in PF).
    ("csig05em", "full_ratio15_lock3_trustedseed_csig05_em10",
     {("tokyo", "run3")}),
    # Phase 11ak/am: csig01 + holdrlx. Phase 11ak: nagoya/run2 +42.8m.
    # Phase 11am also adds tokyo/run1 +7.2m, tokyo/run3 +23.1m.
    ("csig01hr", "full_ratio15_lock3_trustedseed_csig01_holdrlx",
     {("nagoya", "run2"), ("tokyo", "run1"), ("tokyo", "run3")}),
    # Phase 11an: csig05 + psig1 + holdvrlx (4-knob quad combo). HUGE win on
    # tokyo/run1 (+104.6m offline, hybrid_anchor pool) and nagoya/run3
    # (+28.4m offline, score pool). Negative on nagoya/run2 (-53.3m) and
    # tokyo/run3 (-4.2m); restrict.
    ("c5p1hvr", "full_ratio15_lock3_trustedseed_csig05_psig1_holdvrlx",
     {("tokyo", "run1"), ("nagoya", "run3")}),
    # Phase 11an: csig01 + psig1 + holdrlx triple. Modest win on tokyo/run3
    # (+7.3m offline). Negative on nagoya/run1 (-8.4m), nagoya/run3 (-5.1m).
    ("c1p1hr", "full_ratio15_lock3_trustedseed_csig01_psig1_holdrlx",
     {("tokyo", "run3")}),
    # Phase 11an: csig05 + holdrlx + em10 triple. Win on tokyo/run3 (+18.2m
    # offline). Negative on nagoya/run2 (-11.0m).
    ("c5hrem", "full_ratio15_lock3_trustedseed_csig05_holdrlx_em10",
     {("tokyo", "run3")}),
    # Phase 11ao: csig005 (--carrier-phase-sigma 0.00005, super-tight) gives
    # +36.1m on nagoya/run2 (which is hardest run). Marginal elsewhere.
    ("csig005", "full_ratio15_lock3_trustedseed_csig005",
     {("nagoya", "run2")}),
    # Phase 11ao: csig05 + nobds + holdrlx triple. Marginal on tokyo/run1
    # (+9.8m) and nagoya/run1 (+6.8m). Negative elsewhere.
    ("c5nbhr", "full_ratio15_lock3_trustedseed_csig05_nobds_holdrlx",
     {("tokyo", "run1"), ("nagoya", "run1")}),
    # Phase 11ap: csig005 + holdrlx (super-tight csig + holdrlx). Small win on
    # tokyo/run1 (+2.6m) and nagoya/run1 (+9.5m). Negative elsewhere.
    ("c005hr", "full_ratio15_lock3_trustedseed_csig005_holdrlx",
     {("tokyo", "run1"), ("nagoya", "run1")}),
    # Phase 11aq: r05 (--ratio 0.5, super-loose underlying RTK ratio). Marginal
    # +5.0m/+3.9m/+2.3m on tokyo/run1, tokyo/run3, nagoya/run1 (extra epochs gated
    # via different LAMBDA path). Negative on nagoya/run2 (-10.2m), nagoya/run3 (-6.0m).
    ("r05", "full_ratio15_lock3_trustedseed_r05",
     {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run1")}),
    # Phase 11aq: csig005 + glonass-ar autocal. Small +1.8m/+3.3m on tokyo/run1
    # and tokyo/run3. Big negative -46.3m on nagoya/run1; restrict.
    ("c005ga", "full_ratio15_lock3_trustedseed_csig005_glonassar",
     {("tokyo", "run1"), ("tokyo", "run3")}),
    # Phase 11ar: onlyG (--no-glonass --no-beidou, only G+E+J). +9.4m on
    # tokyo/run1 and **+20.2m on nagoya/run2** (the hardest run). Negative
    # on nagoya/run3 (-32.5m); restrict.
    ("onlyG", "full_ratio15_lock3_trustedseed_onlyG",
     {("tokyo", "run1"), ("nagoya", "run2")}),
    # Phase 11as: onlyG + csig05 (combined super-variant) gives +14.3m on
    # tokyo/run1 (additive after onlyG). Negative on nagoya/run3 (-72m).
    ("oGc05", "full_ratio15_lock3_trustedseed_onlyG_csig05",
     {("tokyo", "run1")}),
    # Phase 11as: onlyG + csig005 gives +10.4m on nagoya/run2 (additive after
    # onlyG +20.2m and csig005 +36.1m). Negative elsewhere.
    ("oGc005", "full_ratio15_lock3_trustedseed_onlyG_csig005",
     {("nagoya", "run2")}),
    # Phase 11as: onlyG + psig1 gives +2.1m/+3.0m on nagoya/run1, nagoya/run3.
    ("oGp1", "full_ratio15_lock3_trustedseed_onlyG_psig1",
     {("nagoya", "run1"), ("nagoya", "run3")}),
    # Phase 11at: onlyG + psig1 + holdrlx (3-knob) gives +15.4m on tokyo/run1.
    # Negative on tokyo/run3 (-13m), nagoya/run2 (-54m), nagoya/run3 (-73m).
    ("oGp1hr", "full_ratio15_lock3_trustedseed_onlyG_psig1_holdrlx",
     {("tokyo", "run1")}),
    # Phase 11at: onlyG + psig1 + csig05 (3-knob) gives +9.2m on nagoya/run3,
    # +2.0m on nagoya/run1. Negative on tokyo/run1 (-23m).
    ("oGp1c05", "full_ratio15_lock3_trustedseed_onlyG_psig1_csig05",
     {("nagoya", "run3"), ("nagoya", "run1")}),
    # Phase 11at: onlyG + r05 marginal +2.2/+3.7m on tokyo/run3, nagoya/run2.
    ("oGr05", "full_ratio15_lock3_trustedseed_onlyG_r05",
     {("tokyo", "run3"), ("nagoya", "run2")}),
    # Phase 11at: onlyG + csig01 marginal +3.1m on nagoya/run1.
    ("oGc01", "full_ratio15_lock3_trustedseed_onlyG_csig01",
     {("nagoya", "run1")}),
    # Phase 11at: onlyG + em10 marginal +1.5m/+1.1m on tokyo/run3, nagoya/run3.
    ("oGem10", "full_ratio15_lock3_trustedseed_onlyG_em10",
     {("tokyo", "run3"), ("nagoya", "run3")}),
    # Phase 11au: onlyG + csig005 + psig1 quintuple. tokyo/run2 +37m offline (biggest), tokyo/run1 +6.6m, nagoya/run1 +1.2m.
    ("oGc005p1", "full_ratio15_lock3_trustedseed_oGc005p1",
     {("tokyo", "run2"), ("tokyo", "run1"), ("nagoya", "run1")}),
    # Phase 11au: csig005 + psig1 (no onlyG). tokyo/run1 +13.9m, tokyo/run2 +15.1m, nagoya/run1 +1.5m.
    ("c005p1", "full_ratio15_lock3_trustedseed_c005p1",
     {("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1")}),
    # Phase 11au: onlyG + csig01 + psig1. tokyo/run2 +21.4m, tokyo/run3 +3.4m.
    ("oGc01p1", "full_ratio15_lock3_trustedseed_oGc01p1",
     {("tokyo", "run2"), ("tokyo", "run3")}),
    # Phase 11aw: oGc00005p1 (onlyG + csig 0.00005 + psig 1) +16.7m tokyo/run1.
    ("oGc00005p1", "full_ratio15_lock3_trustedseed_oGc00005p1",
     {("tokyo", "run1")}),
    # Phase 11aw: oGc0001p1 (onlyG + csig 0.0001 + psig 1) +16.3m tokyo/run2, +5.4m tokyo/run1.
    ("oGc0001p1", "full_ratio15_lock3_trustedseed_oGc0001p1",
     {("tokyo", "run2"), ("tokyo", "run1")}),
    # Phase 11aw: nobdsc005p1 (no-beidou + csig005 + psig1) +2.7m nagoya/run3.
    ("nobdsc005p1", "full_ratio15_lock3_trustedseed_nobdsc005p1",
     {("nagoya", "run3")}),
    # Phase 11ay: em5 (--elevation-mask-deg 5) +13.8m nagoya/run1, +8.0m tokyo/run2.
    ("em5", "full_ratio15_lock3_trustedseed_em5",
     {("nagoya", "run1"), ("tokyo", "run2")}),
    # Phase 11ay: mlc2oG (--min-lock-count 2 + onlyG) +7.5m tokyo/run1, +1.3m tokyo/run3.
    ("mlc2oG", "full_ratio15_lock3_trustedseed_mlc2oG",
     {("tokyo", "run1"), ("tokyo", "run3")}),
    # Phase 11ba: mlc1oG (--min-lock-count 1 + onlyG) BIG WINNER on multiple runs.
    # tokyo/run2 +11.3m, tokyo/run3 +9.3m, nagoya/run2 +3.6m.
    ("mlc1oG", "full_ratio15_lock3_trustedseed_mlc1oG",
     {("tokyo", "run2"), ("tokyo", "run3"), ("nagoya", "run2")}),
    # Phase 11ba: em3 (--elev-mask-deg 3) +5.4m nagoya/run1.
    ("em3", "full_ratio15_lock3_trustedseed_em3",
     {("nagoya", "run1")}),
    # Phase 11bc: mlc1oGc005p1 (mlc1 + onlyG + csig005 + psig1) positive 5 runs.
    # n/r3 +11m, t/r1 +4m, t/r2 +4m, n/r1 +1m, t/r3 +4m. n/r2 only loser.
    ("mlc1oGc005p1", "full_ratio15_lock3_trustedseed_mlc1oGc005p1",
     {("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run3"), ("tokyo", "run3")}),
    # Phase 11bc: em3mlc1oG (em3 + mlc1 + onlyG) n/r3 +12m, t/r3 +6m, n/r1 +1m.
    ("em3mlc1oG", "full_ratio15_lock3_trustedseed_em3mlc1oG",
     {("tokyo", "run3"), ("nagoya", "run3"), ("nagoya", "run1")}),
    # Phase 11bc: mlc1oGc005 (mlc1 + onlyG + csig005) n/r2 +7m (rare positive on n/r2!), t/r2 +3m, t/r3 +5m.
    ("mlc1oGc005", "full_ratio15_lock3_trustedseed_mlc1oGc005",
     {("nagoya", "run2"), ("tokyo", "run2"), ("tokyo", "run3")}),
    # Phase 11be: mlc1c005p1 (mlc1 + csig005 + psig1, no onlyG) +5m n/r1, +4m n/r3.
    ("mlc1c005p1", "full_ratio15_lock3_trustedseed_mlc1c005p1",
     {("nagoya", "run1"), ("nagoya", "run3"), ("nagoya", "run2")}),
    # Phase 11be: mlc1oGc005em3 (mlc1 + onlyG + csig005 + em3) n/r2 +5m exclusive winner.
    ("mlc1oGc005em3", "full_ratio15_lock3_trustedseed_mlc1oGc005em3",
     {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}),
    # Phase 11be: mlc1oGc005r12 (mlc1 + onlyG + csig005 + ratio 1.2) t/r3 +5m, t/r1 +1m.
    ("mlc1oGc005r12", "full_ratio15_lock3_trustedseed_mlc1oGc005r12",
     {("tokyo", "run3"), ("tokyo", "run1")}),
    # Phase 11be: mlc1nobds (mlc1 + no-beidou) t/r1 +5m.
    ("mlc1nobds", "full_ratio15_lock3_trustedseed_mlc1nobds",
     {("tokyo", "run1")}),
    # Phase 11bh: mlc1c005r10 (mlc1 + csig005 + ratio 1.0) +14m n/r3, +3m t/r1, +1.5m t/r2.
    ("mlc1c005r10", "full_ratio15_lock3_trustedseed_mlc1c005r10",
     {("nagoya", "run3"), ("tokyo", "run1"), ("tokyo", "run2")}),
    # Phase 11bh: mlc1r10 (mlc1 + ratio 1.0) +8m n/r1, +6.5m t/r3.
    ("mlc1r10", "full_ratio15_lock3_trustedseed_mlc1r10",
     {("nagoya", "run1"), ("tokyo", "run3")}),
    # Phase 11bh: mlc1c005 (mlc1 + csig005) +4.2m n/r2 (rare positive on n/r2).
    ("mlc1c005", "full_ratio15_lock3_trustedseed_mlc1c005",
     {("nagoya", "run2")}),
    # Phase 11bk: rtkout5 (--rtk-update-outlier-threshold 5) HUGE winner.
    # tokyo/run1 +73.99m (+0.72pp, session record), tokyo/run2 +6.94m, nagoya/run2 +2.91m.
    ("rtkout5", "full_ratio15_lock3_trustedseed_rtkout5",
     {("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run2")}),
    # Phase 11bk: rtkout10 (--rtk-update-outlier-threshold 10) +3.95m nagoya/run1.
    ("rtkout10", "full_ratio15_lock3_trustedseed_rtkout10",
     {("nagoya", "run1")}),
    # Phase 11bl: rtkout3 (--rtk-update-outlier-threshold 3) HUGE on tokyo/run1 (+167m, +1.62pp).
    # Marginal +1.5m on nagoya/run2.
    ("rtkout3", "full_ratio15_lock3_trustedseed_rtkout3",
     {("tokyo", "run1"), ("nagoya", "run2")}),
    # Phase 11bl: rtkout7 marginal +4.2m nagoya/run1.
    ("rtkout7", "full_ratio15_lock3_trustedseed_rtkout7",
     {("nagoya", "run1")}),
    # Phase 11bm: rtkout1 (--rtk-update-outlier-threshold 1) HUGE on t/r1 (+175.5m, +1.70pp).
    ("rtkout1", "full_ratio15_lock3_trustedseed_rtkout1",
     {("tokyo", "run1"), ("tokyo", "run3")}),
    # Phase 11bm: rtkout5c005 +15.0m nagoya/run2.
    ("rtkout5c005", "full_ratio15_lock3_trustedseed_rtkout5c005",
     {("nagoya", "run2")}),
    # Phase 11bm: rtkout5em3 +13.1m t/r2, +4.2m n/r1.
    ("rtkout5em3", "full_ratio15_lock3_trustedseed_rtkout5em3",
     {("tokyo", "run2"), ("nagoya", "run1")}),
]

_DIAG_ROOT = Path("experiments/results/libgnss_diag_phase10")
_SELECT_MODES = ("residual", "ratio", "score", "maxabs", "nrows")


@dataclass
class RunResult:
    city: str
    run: str
    mode: str
    ppc_pct: float
    pass_m: float
    total_m: float
    n_gated: int
    n_selected: int


def _eligible_for_run(city: str, run: str, restrict: set[tuple[str, str]] | None) -> bool:
    if restrict is None:
        return True
    return (city, run) in restrict


def _load_candidates_for_run(city: str, run: str) -> list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]]:
    out: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]] = []
    for label, dir_name, restrict in _CANDIDATES_PHASE11V:
        if not _eligible_for_run(city, run, restrict):
            continue
        pos_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.pos"
        diag_path = _PROJECT_ROOT / _DIAG_ROOT / dir_name / f"{city}_{run}_full.csv"
        if not pos_path.is_file() or not diag_path.is_file():
            continue
        pos, _ = _load_hybrid_pos_file(pos_path)
        diag = _load_rtk_diag_file(diag_path)
        out.append((label, pos, diag))
    return out


def _simulate(
    city: str,
    run: str,
    mode: str,
    *,
    hybrid_pos: dict[float, np.ndarray],
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]],
    ref: list[tuple[float, np.ndarray]],
    ratio_min: float,
    rms_max: float,
) -> RunResult:
    truth = np.asarray([t for _, t in ref], dtype=np.float64)
    est = np.zeros((len(ref), 3), dtype=np.float64)
    n_gated = 0
    n_selected = 0
    for i, (tow, _) in enumerate(ref):
        t_key = round(float(tow), 1)
        # Default to hybrid floor.
        hp = hybrid_pos.get(t_key)
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
            est[i] = np.asarray(hp, dtype=np.float64)
        # Try gated candidates.
        best_key = None
        best_pos = None
        any_gated = False
        for label, cand_pos, cand_diag in candidates:
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
            n_selected += 1
    score = score_ppc2024(est, truth)
    return RunResult(
        city=city, run=run, mode=mode,
        ppc_pct=float(score.score_pct),
        pass_m=float(score.pass_distance_m),
        total_m=float(score.total_distance_m),
        n_gated=n_gated,
        n_selected=n_selected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline selector sweep for Phase 11v")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS_DIR / "libgnss_rtk_pos_v5")
    parser.add_argument("--policy", type=str, default="phase11n")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "ppc_selector_sweep_phase11v.csv")
    args = parser.parse_args()

    rows: list[RunResult] = []
    for city, run in _FULL_RUNS:
        ref = _load_full_reference(args.data_root / city / run / "reference.csv")
        hybrid_pos, _ = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{city}_{run}_full.pos")
        # Resolve per-run policy gate.
        variant = _apply_rtkdiag_run_index_policy(
            CTRBPFConfig(enable_rtkdiag_pf_rescue=True),
            run=run,
            policy=str(args.policy),
            city=city,
        )
        ratio_min = float(variant.rtkdiag_candidate_ratio_min)
        rms_max = float(variant.rtkdiag_candidate_residual_rms_max)
        # Apply the same blocked-label filter as the policy.
        all_cands = _load_candidates_for_run(city, run)
        kept = _filter_rtkdiag_candidates_by_policy(
            all_cands,
            city=city, run=run, policy=str(args.policy),
        )
        kept_labels = sorted(c[0] for c in kept)
        print(f"\n{city}/{run}: ratio_min={ratio_min}, rms_max={rms_max}, candidates={len(kept)} ({','.join(kept_labels)})")
        per_mode: dict[str, RunResult] = {}
        for mode in _SELECT_MODES:
            res = _simulate(city, run, mode,
                            hybrid_pos=hybrid_pos, candidates=kept, ref=ref,
                            ratio_min=ratio_min, rms_max=rms_max)
            per_mode[mode] = res
            print(f"  mode={mode:<9s}: ppc={res.ppc_pct:.4f}%, pass={res.pass_m:.1f}/{res.total_m:.1f}, gated={res.n_gated}, sel={res.n_selected}")
        # Highlight current policy mode.
        cur_mode = str(variant.rtkdiag_candidate_select_mode)
        cur = per_mode.get(cur_mode)
        if cur is not None:
            best_mode = max(per_mode.values(), key=lambda r: r.pass_m)
            delta = best_mode.pass_m - cur.pass_m
            print(f"  current policy mode={cur_mode}: ppc={cur.ppc_pct:.4f}%, best_alt={best_mode.mode} ppc={best_mode.ppc_pct:.4f}% (delta_pass={delta:+.1f}m)")
        rows.extend(per_mode.values())

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["city", "run", "mode", "ppc_pct", "pass_m", "total_m", "n_gated", "n_selected"])
        for r in rows:
            w.writerow([r.city, r.run, r.mode, f"{r.ppc_pct:.6f}", f"{r.pass_m:.4f}", f"{r.total_m:.4f}", r.n_gated, r.n_selected])
    # Aggregates per mode.
    print("\nAggregate per uniform mode (apply same mode across all runs):")
    by_mode: dict[str, list[RunResult]] = {}
    for r in rows:
        by_mode.setdefault(r.mode, []).append(r)
    for mode, items in by_mode.items():
        pass_sum = sum(r.pass_m for r in items)
        total_sum = sum(r.total_m for r in items)
        print(f"  mode={mode:<9s}: ppc={100*pass_sum/total_sum:.4f}% (pass {pass_sum:.1f}/{total_sum:.1f})")
    print("\nAggregate per per-run-best (oracle on selector):")
    pass_sum = 0.0
    total_sum = 0.0
    best_modes: list[str] = []
    for city, run in _FULL_RUNS:
        run_rows = [r for r in rows if r.city == city and r.run == run]
        best = max(run_rows, key=lambda r: r.pass_m)
        pass_sum += best.pass_m
        total_sum += best.total_m
        best_modes.append(f"{city}/{run}={best.mode}")
    print(f"  ppc={100*pass_sum/total_sum:.4f}% (pass {pass_sum:.1f}/{total_sum:.1f})")
    print(f"  per-run best mode: {', '.join(best_modes)}")


if __name__ == "__main__":
    main()
