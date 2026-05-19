#!/usr/bin/env bash
# Phase 33 Stage B: generate libgnss++ RTK candidate with PLATEAU NLOS code-variance scaling.
# Outputs `experiments/results/libgnss_diag_phase33_nlos_soft_k3/{city}_{run}_full.pos`.
# CLI: --rtk-nlos-mask-path <csv> --rtk-nlos-k-weak 3.0 inflates DD code variance for NLOS PRNs.
set -uo pipefail
cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
source .venv/bin/activate

BIN=/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/third_party/gnssplusplus/build/apps/gnss_solve
DATA=/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data
MASK_DIR=experiments/results/plateau_nlos_phase33
OUT_DIR=${OUT_DIR:-experiments/results/libgnss_diag_phase33_nlos_soft_k3}
LOG_DIR=${LOG_DIR:-/tmp/phase33_nlos_soft_k3}
K_WEAK=${K_WEAK:-3.0}

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_one() {
  local CR="$1"
  local CITY=${CR%/*}
  local RUN=${CR#*/}
  local PROFILE
  if [ "$CITY" = "tokyo" ]; then PROFILE="tokyo"; else PROFILE="nagoya"; fi
  local MASK="${MASK_DIR}/${CITY}_${RUN}_per_epoch_nlos.csv"
  local OUT_POS="${OUT_DIR}/${CITY}_${RUN}_full.pos"
  local OUT_DIAG="${OUT_DIR}/${CITY}_${RUN}_full.csv"
  local LOG="${LOG_DIR}/${CITY}_${RUN}.log"

  if [ ! -f "$MASK" ]; then
    echo "[skip] mask missing: $MASK" >&2
    return 1
  fi

  echo "[start] ${CITY}/${RUN}: k_weak=${K_WEAK}" >&2

  case "$PROFILE" in
    tokyo)
      PROFILE_ARGS=(--preset low-cost --arfilter --arfilter-margin 0.35 --min-hold-count 8 --hold-ratio-threshold 2.6)
      ;;
    nagoya)
      PROFILE_ARGS=(--preset low-cost --min-hold-count 7 --hold-ratio-threshold 2.4)
      ;;
  esac

  "$BIN" \
    --rover "${DATA}/${CITY}/${RUN}/rover.obs" \
    --base "${DATA}/${CITY}/${RUN}/base.obs" \
    --nav "${DATA}/${CITY}/${RUN}/base.nav" \
    --out "$OUT_POS" \
    --no-kml \
    "${PROFILE_ARGS[@]}" \
    --diagnostics-csv "$OUT_DIAG" \
    --rtk-nlos-mask-path "$MASK" \
    --rtk-nlos-k-weak "$K_WEAK" \
    >"$LOG" 2>&1
  local RC=$?
  if [ $RC -ne 0 ]; then
    echo "[fail] ${CITY}/${RUN} rc=${RC}" >&2
    return $RC
  fi
  local LINE=$(grep -E "fix rate" "$LOG" | head -1)
  echo "[done] ${CITY}/${RUN}: ${LINE}" >&2
}

export -f run_one
export BIN DATA MASK_DIR OUT_DIR LOG_DIR K_WEAK

RUNS=(tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3)
# Sequential (gnss_solve is single-threaded so up to 6 in parallel works on this box):
printf '%s\n' "${RUNS[@]}" | xargs -n 1 -P 6 -I {} bash -c 'run_one "$@"' _ {}

echo ""
echo "=== Phase 33 Stage B k=${K_WEAK} candidate generation complete ==="
ls -1 "${OUT_DIR}" | head -10
