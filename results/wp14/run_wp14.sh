#!/bin/sh
# WP14: run the gnssplusplus GTSAM TC-FGO backend (README recommended preset)
# on one full PPC Tokyo run and dump per-epoch solutions for the campaign scorer.
# Usage: run_wp14.sh <runN>   (e.g. run_wp14.sh run1)
set -eu
RUN="$1"
ROOT="C:/Users/rsasa/Workspace/old/gnss_gpu"
D="$ROOT/datasets/PPC-Dataset-data/tokyo/$RUN"
OUT="$ROOT/results/wp14"
EXE="$ROOT/results/wp14/harness/build_fgo/Release/wp14_fgo_dump.exe"
# gtsam.dll from the E:/gtsam install (system-Eigen GTSAM build)
export PATH="/e/gtsam/install/bin:$PATH"

"$EXE" \
  --rover "$D/rover.obs" --base "$D/base.obs" --nav "$D/base.nav" \
  --imu "$D/imu.csv" --ref "$D/reference.csv" \
  --fixed-lag 5 --multi-freq --partial-ar --hold \
  --elev-mask 25 --snr-mask 30 --imu-preset-tactical \
  --cmc --cmc-level 0.75 --cp-hold --cp-hold-res 2.0 \
  --exc-recovery --ddpr-anchor --fde --varerr \
  --pos-out "$OUT/tokyo_${RUN}_fgo_gtsam.csv" \
  > "$OUT/tokyo_${RUN}_parity.log" 2>&1
echo "DONE $RUN exit=$?"
tail -6 "$OUT/tokyo_${RUN}_parity.log"
