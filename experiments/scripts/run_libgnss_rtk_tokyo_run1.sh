#!/usr/bin/env bash
set -euo pipefail
GNSS=/mnt/c/Users/rsasa/Workspace/old/gnss_gpu/third_party/gnssplusplus/build/apps/gnss_solve
DATA=/mnt/e/datasets/PPC-Dataset-data/tokyo/run1
OUT=/mnt/c/Users/rsasa/Workspace/old/gnss_gpu/experiments/results/libgnss_rtk_pos_v5
mkdir -p "$OUT"
echo "[rtk] tokyo/run1 start"
"$GNSS" \
  --rover "$DATA/rover.obs" \
  --base "$DATA/base.obs" \
  --nav "$DATA/base.nav" \
  --skip-epochs 0 \
  --out "$OUT/tokyo_run1_full.pos" \
  --no-kml \
  --preset low-cost \
  --arfilter --arfilter-margin 0.35 \
  --min-hold-count 8 --hold-ratio-threshold 2.6
echo "[rtk] tokyo/run1 done"
