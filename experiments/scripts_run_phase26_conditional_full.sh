#!/usr/bin/env bash
# Phase 26: Viterbi α=0.5 smoothed ranker predictions on all 6 runs.
# n/r2 smoke gave 63.13% (+0.48pp on n/r2 vs stickiness +0.23pp).
# Expected OFFICIAL: 85.47% + 0.08pp ≈ 85.55%.
set -uo pipefail
cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
source .venv/bin/activate

FGO_V2_DIR=experiments/results/libgnss_diag_phase10/fgo_v2_gap
FGO_V14_DIR=experiments/results/libgnss_diag_phase10/fgo_v14_snr38
FGO_V17_DIR=experiments/results/libgnss_diag_phase10/fgo_v17_el25
G_DEFAULT=experiments/results/libgnss_diag_phase19/gici_tc_esdfix
G_Z=experiments/results/libgnss_diag_phase19/gici_full_zeroarm
G_R=experiments/results/libgnss_diag_phase19/gici_full_ratio25
G_LP=experiments/results/libgnss_diag_phase19/gici_full_loosepr
G_LH=experiments/results/libgnss_diag_phase19/gici_full_loosephase
G_R4=experiments/results/libgnss_diag_phase19/gici_full_ratio40
G_COMBO=experiments/results/libgnss_diag_phase19/gici_full_combo
G_C4=experiments/results/libgnss_diag_phase19/gici_full_combo4
G_LL=experiments/results/libgnss_diag_phase19/gici_full_lprlph
G_ZR=experiments/results/libgnss_diag_phase19/gici_full_zr
G_OA=experiments/results/libgnss_diag_phase19/gici_full_onarm
G_LA=experiments/results/libgnss_diag_phase19/gici_full_lowacc
G_HS=experiments/results/libgnss_diag_phase19/gici_full_hisnr
G_HS45=experiments/results/libgnss_diag_phase19/gici_full_hisnr45
G_HS30=experiments/results/libgnss_diag_phase19/gici_full_hisnr30
G_HE=experiments/results/libgnss_diag_phase19/gici_full_hielev
G_IR=experiments/results/libgnss_diag_phase19/gici_full_imurot
G_MB=experiments/results/libgnss_diag_phase19/gici_full_himuba
G_W5=experiments/results/libgnss_diag_phase19/gici_full_window5

VITERBI_CSV=experiments/results/selector_ranker_predictions_conditional_nr2vit.csv

# All 6 runs with conditional
for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  OLD_LABELS=$(cat /tmp/${CITY}_${RUN}_phase11fa_labels.txt)
  OLD_DIRS=$(cat /tmp/${CITY}_${RUN}_phase11fa_dirs.txt)
  NEW_DIRS="${OLD_DIRS},$FGO_V2_DIR,$FGO_V14_DIR,$FGO_V17_DIR,$G_DEFAULT,$G_Z,$G_R,$G_LP,$G_LH,$G_R4,$G_COMBO,$G_C4,$G_LL,$G_ZR,$G_OA,$G_LA,$G_HS,$G_HS45,$G_HS30,$G_HE,$G_IR,$G_MB,$G_W5"
  NEW_LABELS="${OLD_LABELS},xd_fgo_v2_gap,xd_fgo_v14_snr38,xd_fgo_v17_el25,xd_gici_def,xd_gici_z,xd_gici_r,xd_gici_lp,xd_gici_lh,xd_gici_r4,xd_gici_combo,xd_gici_c4,xd_gici_lprlph,xd_gici_zr,xd_gici_oa,xd_gici_la,xd_gici_hs,xd_gici_hs45,xd_gici_hs30,xd_gici_he,xd_gici_ir,xd_gici_mb,xd_gici_w5"

  echo "=== Conditional n/r2-viterbi ${CITY}/${RUN} ==="
  PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py --runs "${CITY}/${RUN}" \
    --methods "rbpf+dd+gate+hybrid+rtkdiag_pf" \
    --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 --hybrid-sigma-m 1.0 \
    --rtkdiag-candidate-pos-dirs "$NEW_DIRS" --rtkdiag-candidate-diag-dirs "$NEW_DIRS" \
    --rtkdiag-candidate-labels "$NEW_LABELS" \
    --rtkdiag-candidate-block-labels-by-run "" \
    --rtkdiag-candidate-run-index-policy phase11ep \
    --rtkdiag-candidate-select-mode ranker \
    --rtkdiag-candidate-ranker-score-path "$VITERBI_CSV" \
    --rtkdiag-candidate-emit-mode candidate \
    --rtkdiag-candidate-residual-rms-max 50.0 --rtkdiag-candidate-ratio-min 1.0 \
    --rtkdiag-candidate-recenter-max-shift-m 10000.0 --rtkdiag-candidate-emit-max-diff-m 0.4 \
    --rtkdiag-candidate-max-to-hybrid-m 0 \
    --rtkdiag-candidate-fallback-mode hybrid \
    --rtkdiag-candidate-bridge-enable --rtkdiag-candidate-bridge-max-s 6.0 --rtkdiag-candidate-bridge-residual-rms-m 0.2 \
    --rtkdiag-candidate-rms-prefilter-k 3 \
    --n-particles 2000 --pos-dir "/tmp/ppc_phase26_conditional_${CITY}_${RUN}" \
    --results-prefix "ppc_ctrbpf_fgo_phase26_conditional_${CITY}_${RUN}_full" \
    >"/tmp/phase26_conditional_${CITY}_${RUN}.log" 2>&1
  PPC=$(awk -F, 'NR==2 {printf "%.4f", $11}' "experiments/results/ppc_ctrbpf_fgo_phase26_conditional_${CITY}_${RUN}_full_runs.csv")
  echo "FIN ${CITY}/${RUN}: ${PPC}%"
done

echo ""
echo "=== Phase 26 Conditional n/r2-viterbi OFFICIAL aggregate ==="
TOTAL=0.0
N=0
for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  P=$(awk -F, 'NR==2 {printf "%.4f", $11}' "experiments/results/ppc_ctrbpf_fgo_phase26_conditional_${CITY}_${RUN}_full_runs.csv" 2>/dev/null)
  if [ -n "$P" ]; then
    TOTAL=$(awk "BEGIN {printf \"%.4f\", $TOTAL + $P}")
    N=$((N + 1))
    echo "  ${CITY}/${RUN}: ${P}%"
  fi
done
AVG=$(awk "BEGIN {printf \"%.4f\", $TOTAL / $N}")
echo "  OFFICIAL: ${AVG}% (n=${N}) [Phase 20 baseline 85.47%]"
