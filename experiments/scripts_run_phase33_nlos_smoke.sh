#!/usr/bin/env bash
# Phase 33 Stage B smoke: PPC selector with Phase 29 pool + NLOS-soft k3 candidate.
# Each run gets Phase 29 best per-run config; we ADD the new NLOS-soft candidate
# (label `xd_nlos_k3`) to the pool and check if PPC ranker picks it on enough epochs
# to move OFFICIAL beyond Phase 29 85.7623%.
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

NLOS_DIR=experiments/results/libgnss_diag_phase33_nlos_soft_k3

V1_CSV=experiments/results/selector_ranker_predictions.csv
V3_CSV=experiments/results/selector_ranker_predictions_v3.csv

declare -A CONFIG
CONFIG[tokyo_run1_csv]="$V1_CSV"; CONFIG[tokyo_run1_k]="3"
CONFIG[tokyo_run2_csv]="$V3_CSV"; CONFIG[tokyo_run2_k]="3"
CONFIG[tokyo_run3_csv]="$V1_CSV"; CONFIG[tokyo_run3_k]="3"
CONFIG[nagoya_run1_csv]="$V3_CSV"; CONFIG[nagoya_run1_k]="99"
CONFIG[nagoya_run2_csv]="$V3_CSV"; CONFIG[nagoya_run2_k]="99"
CONFIG[nagoya_run3_csv]="$V1_CSV"; CONFIG[nagoya_run3_k]="3"

for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  KEY="${CITY}_${RUN}"
  CSV="${CONFIG[${KEY}_csv]}"
  K="${CONFIG[${KEY}_k]}"
  RANKER_LABEL=$(if [ "$CSV" = "$V3_CSV" ]; then echo "v3"; else echo "v1"; fi)

  OLD_LABELS=$(cat /tmp/${CITY}_${RUN}_phase11fa_labels.txt)
  OLD_DIRS=$(cat /tmp/${CITY}_${RUN}_phase11fa_dirs.txt)
  NEW_DIRS="${OLD_DIRS},$FGO_V2_DIR,$FGO_V14_DIR,$FGO_V17_DIR,$G_DEFAULT,$G_Z,$G_R,$G_LP,$G_LH,$G_R4,$G_COMBO,$G_C4,$G_LL,$G_ZR,$G_OA,$G_LA,$G_HS,$G_HS45,$G_HS30,$G_HE,$G_IR,$G_MB,$G_W5,$NLOS_DIR"
  NEW_LABELS="${OLD_LABELS},xd_fgo_v2_gap,xd_fgo_v14_snr38,xd_fgo_v17_el25,xd_gici_def,xd_gici_z,xd_gici_r,xd_gici_lp,xd_gici_lh,xd_gici_r4,xd_gici_combo,xd_gici_c4,xd_gici_lprlph,xd_gici_zr,xd_gici_oa,xd_gici_la,xd_gici_hs,xd_gici_hs45,xd_gici_hs30,xd_gici_he,xd_gici_ir,xd_gici_mb,xd_gici_w5,xd_nlos_k3"

  echo "=== Phase 33 smoke ${CITY}/${RUN} (ranker=${RANKER_LABEL}, k=${K}, +xd_nlos_k3) ==="
  PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py --runs "${CITY}/${RUN}" \
    --methods "rbpf+dd+gate+hybrid+rtkdiag_pf" \
    --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 --hybrid-sigma-m 1.0 \
    --rtkdiag-candidate-pos-dirs "$NEW_DIRS" --rtkdiag-candidate-diag-dirs "$NEW_DIRS" \
    --rtkdiag-candidate-labels "$NEW_LABELS" \
    --rtkdiag-candidate-block-labels-by-run "" \
    --rtkdiag-candidate-run-index-policy phase11ep \
    --rtkdiag-candidate-select-mode ranker \
    --rtkdiag-candidate-ranker-score-path "$CSV" \
    --rtkdiag-candidate-emit-mode candidate \
    --rtkdiag-candidate-residual-rms-max 50.0 --rtkdiag-candidate-ratio-min 1.0 \
    --rtkdiag-candidate-recenter-max-shift-m 10000.0 --rtkdiag-candidate-emit-max-diff-m 0.4 \
    --rtkdiag-candidate-max-to-hybrid-m 0 \
    --rtkdiag-candidate-fallback-mode hybrid \
    --rtkdiag-candidate-bridge-enable --rtkdiag-candidate-bridge-max-s 6.0 --rtkdiag-candidate-bridge-residual-rms-m 0.2 \
    --rtkdiag-candidate-rms-prefilter-k "$K" \
    --n-particles 2000 --pos-dir "/tmp/ppc_phase33_${KEY}" \
    --results-prefix "ppc_ctrbpf_fgo_phase33_smoke_${KEY}_full" \
    >"/tmp/phase33_smoke_${KEY}.log" 2>&1
  PPC=$(awk -F, 'NR==2 {printf "%.4f", $11}' "experiments/results/ppc_ctrbpf_fgo_phase33_smoke_${KEY}_full_runs.csv")
  echo "FIN ${CITY}/${RUN}: ${PPC}%"
done

echo ""
echo "=== Phase 33 Stage B smoke OFFICIAL aggregate ==="
TOTAL=0.0
N=0
for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  P=$(awk -F, 'NR==2 {printf "%.4f", $11}' "experiments/results/ppc_ctrbpf_fgo_phase33_smoke_${CITY}_${RUN}_full_runs.csv" 2>/dev/null)
  if [ -n "$P" ]; then
    TOTAL=$(awk "BEGIN {printf \"%.4f\", $TOTAL + $P}")
    N=$((N + 1))
    echo "  ${CITY}/${RUN}: ${P}%"
  fi
done
AVG=$(awk "BEGIN {printf \"%.4f\", $TOTAL / $N}")
echo "  OFFICIAL: ${AVG}% (n=${N}) [Phase 29 = 85.7623%, gain target +0.5pp+ from NLOS rejection]"
