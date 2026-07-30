#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

run_case() {
  local city="$1"
  local fault="$2"
  local version="$3"
  local seed
  if [[ "${city}" == "tokyo" ]]; then
    seed="dist/tokyo-supply/wp160_seed.pos"
  else
    seed="dist/tokyo-supply/nagoya_wp100_seed.pos"
  fi

  local input_stem="${city}_raw_${fault}_libraryfixgate_anchored_epochs1000_wp175_library_${version}"
  local output_stem="${city}_integrity_cauchy_causal_arc_no_reanchor_${fault}_epochs1000_wp175_v48"
  third_party/gnssplusplus/build/apps/gnss_fuse \
    --data-dir "/mnt/e/datasets/PPC-Dataset-data/${city}/run1" \
    --rover "results/wp174/${input_stem}.obs" \
    --out "results/wp174/${output_stem}.pos" \
    --preset low-cost \
    --ratio 1.5 \
    --ar-policy demo5-continuous \
    --glonass-ar autocal \
    --max-pos-jump 0 \
    --rtk-update-outlier-threshold 3 \
    --prefer-trusted-seed \
    --rover-seed-pos "${seed}" \
    --library-fix-integrity-gate \
    --library-fix-integrity-csv \
      "results/wp174/${output_stem}_integrity.csv" \
    --integrity-disjoint-partition gj-erc \
    --integrity-disjoint-ensemble \
    --integrity-max-statistical-separation-m 2.0 \
    --integrity-student-t-all-measurements \
    --integrity-student-t-degrees-of-freedom 1 \
    --integrity-causal-arc-promotion \
    --max-epochs 1000 \
    --quiet \
    >"results/wp174/${output_stem}.log" 2>&1
  echo "DONE ${city} ${fault}"
}

if [[ "$#" -eq 2 ]]; then
  city="$1"
  fault="$2"
  if [[ "${city}" != "tokyo" && "${city}" != "nagoya" ]]; then
    echo "city must be tokyo or nagoya" >&2
    exit 2
  fi
  case "${fault}" in
    cycle_slip|satellite_loss|outage)
      version="v2"
      ;;
    nlos)
      version="v1"
      ;;
    *)
      echo "unsupported fault: ${fault}" >&2
      exit 2
      ;;
  esac
  run_case "${city}" "${fault}" "${version}"
  exit 0
elif [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [CITY FAULT]" >&2
  exit 2
fi

run_case tokyo cycle_slip v2 &
run_case nagoya cycle_slip v2 &
wait
run_case tokyo nlos v1 &
run_case nagoya nlos v1 &
wait
run_case tokyo satellite_loss v2 &
run_case nagoya satellite_loss v2 &
wait
run_case tokyo outage v2 &
run_case nagoya outage v2 &
wait
