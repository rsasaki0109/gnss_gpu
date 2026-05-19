#!/usr/bin/env bash
# Phase 33 Stage C-prime: PF 6-run sweep with v5_nlos ranker.
# Per-run conditional based on Phase 29 layout, swapping v3 spots → v5_nlos
# and keeping v1 spots as v1 (Phase 29 production-best preserved on those).
# Then a 2nd pass: replace v1 spots with v5_nlos too, see if v5 dominates v1.
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

V1_CSV=experiments/results/selector_ranker_predictions.csv
V3_CSV=experiments/results/selector_ranker_predictions_v3.csv
V5_CSV=experiments/results/selector_ranker_predictions_v5_nlos.csv

# RANKER_VARIANT can be: phase29_replace_v3 (v1 stays, v3 spots → v5) | all_v5 (replace v1 too)
RANKER_VARIANT=${RANKER_VARIANT:-phase29_replace_v3}

# Per-run config based on Phase 29 winning layout
declare -A CONFIG_PHASE29
CONFIG_PHASE29[tokyo_run1]="v1:3"
CONFIG_PHASE29[tokyo_run2]="v3:3"
CONFIG_PHASE29[tokyo_run3]="v1:3"
CONFIG_PHASE29[nagoya_run1]="v3:99"
CONFIG_PHASE29[nagoya_run2]="v3:99"
CONFIG_PHASE29[nagoya_run3]="v1:3"

# Resolve ranker for this variant
resolve_csv() {
  local ranker="$1"
  case "$ranker" in
    v1) echo "$V1_CSV" ;;
    v3) echo "$V3_CSV" ;;
    v5) echo "$V5_CSV" ;;
    *) echo "$V5_CSV" ;;
  esac
}

map_ranker() {
  local orig="$1"
  case "${RANKER_VARIANT}" in
    phase29_replace_v3)
      if [ "$orig" = "v3" ]; then echo "v5"; else echo "$orig"; fi
      ;;
    all_v5)
      echo "v5"
      ;;
    *)
      echo "$orig"
      ;;
  esac
}

for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  KEY="${CITY}_${RUN}"
  RAW="${CONFIG_PHASE29[${KEY}]}"
  RANKER_ORIG=${RAW%:*}
  K=${RAW#*:}
  RANKER_NEW=$(map_ranker "$RANKER_ORIG")
  CSV=$(resolve_csv "$RANKER_NEW")

  OLD_LABELS=$(cat /tmp/${CITY}_${RUN}_phase11fa_labels.txt)
  OLD_DIRS=$(cat /tmp/${CITY}_${RUN}_phase11fa_dirs.txt)
  NEW_DIRS="${OLD_DIRS},$FGO_V2_DIR,$FGO_V14_DIR,$FGO_V17_DIR,$G_DEFAULT,$G_Z,$G_R,$G_LP,$G_LH,$G_R4,$G_COMBO,$G_C4,$G_LL,$G_ZR,$G_OA,$G_LA,$G_HS,$G_HS45,$G_HS30,$G_HE,$G_IR,$G_MB,$G_W5"
  NEW_LABELS="${OLD_LABELS},xd_fgo_v2_gap,xd_fgo_v14_snr38,xd_fgo_v17_el25,xd_gici_def,xd_gici_z,xd_gici_r,xd_gici_lp,xd_gici_lh,xd_gici_r4,xd_gici_combo,xd_gici_c4,xd_gici_lprlph,xd_gici_zr,xd_gici_oa,xd_gici_la,xd_gici_hs,xd_gici_hs45,xd_gici_hs30,xd_gici_he,xd_gici_ir,xd_gici_mb,xd_gici_w5"

  echo "=== Phase 33 v5sweep ${CITY}/${RUN} (variant=${RANKER_VARIANT}, ranker=${RANKER_NEW}, k=${K}) ==="
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
    --n-particles 2000 --pos-dir "/tmp/ppc_phase33_v5_${RANKER_VARIANT}_${KEY}" \
    --results-prefix "ppc_ctrbpf_fgo_phase33_v5_${RANKER_VARIANT}_${KEY}_full" \
    >"/tmp/phase33_v5_${RANKER_VARIANT}_${KEY}.log" 2>&1
  PPC=$(awk -F, 'NR==2 {printf "%.4f", $11}' "experiments/results/ppc_ctrbpf_fgo_phase33_v5_${RANKER_VARIANT}_${KEY}_full_runs.csv")
  echo "FIN ${CITY}/${RUN}: ${PPC}%"
done

echo ""
echo "=== Phase 33 v5 sweep (${RANKER_VARIANT}) OFFICIAL aggregate ==="
TOTAL=0.0
N=0
for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  P=$(awk -F, 'NR==2 {printf "%.4f", $11}' "experiments/results/ppc_ctrbpf_fgo_phase33_v5_${RANKER_VARIANT}_${CITY}_${RUN}_full_runs.csv" 2>/dev/null)
  if [ -n "$P" ]; then
    TOTAL=$(awk "BEGIN {printf \"%.4f\", $TOTAL + $P}")
    N=$((N + 1))
    echo "  ${CITY}/${RUN}: ${P}%"
  fi
done
AVG=$(awk "BEGIN {printf \"%.4f\", $TOTAL / $N}")
echo "  OFFICIAL: ${AVG}% (n=${N}) [Phase 29 baseline 85.7623%, v5 LORO sim 86.18%]"
