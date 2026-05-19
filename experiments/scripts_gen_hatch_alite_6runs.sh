#!/usr/bin/env bash
# Generate hatch_alite candidate for all 6 PPC runs (parallel).
set -uo pipefail
cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
source .venv/bin/activate

OUT=experiments/results/libgnss_hatch_alite_v1
mkdir -p "$OUT"

run_one() {
  local CITY=$1
  local RUN=$2
  local LOG=/tmp/hatch_alite_${CITY}_${RUN}.log
  echo "Starting hatch_alite ${CITY}/${RUN} -> $LOG"
  PYTHONPATH=python python experiments/build_hatch_smoothed_spp_candidate.py \
    --city "$CITY" --run "$RUN" --output-dir "$OUT" \
    >"$LOG" 2>&1 &
}

run_one tokyo run1
run_one tokyo run2
run_one tokyo run3
run_one nagoya run1
run_one nagoya run2
run_one nagoya run3
wait

echo "=== ALL 6 generations done ==="
for f in $OUT/*_full.pos; do
  base=$(basename "$f")
  lines=$(grep -c -v "^%" "$f" 2>/dev/null || echo 0)
  echo "  $base: $lines positions"
done
