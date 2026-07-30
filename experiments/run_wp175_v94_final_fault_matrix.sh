#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_case() {
  local city="$1"
  local fault="$2"
  local seed baseline stem
  if [[ "${city}" == "tokyo" ]]; then
    seed="dist/tokyo-supply/wp160_seed.pos"
    baseline="results/wp174/tokyo_integrity_cauchy_srcpar_hard_replace_dual_promoted_full_wp175_v92.pos"
  else
    seed="dist/tokyo-supply/nagoya_wp100_seed.pos"
    baseline="results/wp174/nagoya_integrity_cauchy_srcpar_hard_replace_dual_promoted_full_wp175_v93.pos"
  fi
  stem="${city}_integrity_final_${fault}_epochs1000_wp175_v94"

  python3 experiments/inject_wp174_rinex_faults.py \
    --input "/mnt/e/datasets/PPC-Dataset-data/${city}/run1/rover.obs" \
    --output "results/wp174/${stem}.obs" \
    --manifest "results/wp174/${stem}_manifest.json" \
    --fault "${fault}" \
    --events 8 \
    --anchor-pos "${baseline}" \
    --anchor-streak-epochs 10 \
    --recovery-horizon-s 10 \
    --maximum-epochs 1000 \
    >"results/wp174/${stem}_inject.log"

  third_party/gnssplusplus/build/apps/gnss_fuse \
    --data-dir "/mnt/e/datasets/PPC-Dataset-data/${city}/run1" \
    --rover "results/wp174/${stem}.obs" \
    --out "results/wp174/${stem}.pos" \
    --preset low-cost \
    --ratio 1.5 \
    --ar-policy demo5-continuous \
    --glonass-ar autocal \
    --max-pos-jump 0 \
    --rtk-update-outlier-threshold 3 \
    --prefer-trusted-seed \
    --rover-seed-pos "${seed}" \
    --library-fix-integrity-gate \
    --library-fix-integrity-csv "results/wp174/${stem}_integrity.csv" \
    --integrity-disjoint-partition gj-erc \
    --integrity-disjoint-ensemble \
    --integrity-max-statistical-separation-m 2.0 \
    --integrity-student-t-all-measurements \
    --integrity-student-t-degrees-of-freedom 1 \
    --integrity-causal-arc-promotion \
    --integrity-satellite-par-consensus-promotion \
    --integrity-src-par-consensus-promotion \
    --integrity-disjoint-satellite-par-shadow \
    --integrity-l1-l2-wlnl-cascade-shadow \
    --max-epochs 1000 \
    --quiet \
    >"results/wp174/${stem}.log" 2>&1

  python3 experiments/analyze_wp174_raw_fault_replay.py \
    --debug "results/wp174/${stem}_integrity.csv" \
    --positions "results/wp174/${stem}.pos" \
    --reference "/mnt/e/datasets/PPC-Dataset-data/${city}/run1/reference.csv" \
    --manifest "results/wp174/${stem}_manifest.json" \
    --output "results/wp174/${stem}_audit.json" \
    >"results/wp174/${stem}_audit.log"
  echo "DONE ${city} ${fault}"
}

if [[ "$#" -eq 2 ]]; then
  run_case "$1" "$2"
  exit 0
elif [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [CITY FAULT]" >&2
  exit 2
fi

for city in tokyo nagoya; do
  for fault in cycle_slip nlos satellite_loss outage; do
    run_case "${city}" "${fault}"
  done
done
