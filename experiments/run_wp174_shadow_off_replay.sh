#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ("$1" != "tokyo" && "$1" != "nagoya") ]]; then
  echo "usage: $0 {tokyo|nagoya}" >&2
  exit 2
fi

city="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
solver="${repo_root}/third_party/gnssplusplus/build/apps/gnss_solve"
dataset_parent="${WP174_DATASET_ROOT:-/mnt/e/datasets/PPC-Dataset-data}"
dataset_root="${dataset_parent}/${city}/run1"
output_root="${repo_root}/results/wp174"

if [[ "${city}" == "tokyo" ]]; then
  seed="${repo_root}/dist/tokyo-supply/wp160_seed.pos"
else
  seed="${repo_root}/dist/tokyo-supply/nagoya_wp100_seed.pos"
fi

mkdir -p "${output_root}"
{
  echo "submodule_commit=$(git -C "${repo_root}/third_party/gnssplusplus" rev-parse HEAD)"
  echo "solver_sha256=$(sha256sum "${solver}" | awk '{print $1}')"
  echo "working_diff_sha256=$(git -C "${repo_root}/third_party/gnssplusplus" diff --binary | sha256sum | awk '{print $1}')"
  echo "candidate_count=0"
} >"${output_root}/${city}_shadow_off_provenance.txt"
/usr/bin/time -f "%e" -o "${output_root}/${city}_shadow_off_elapsed_s.txt" \
  "${solver}" \
    --rover "${dataset_root}/rover.obs" \
    --base "${dataset_root}/base.obs" \
    --nav "${dataset_root}/base.nav" \
    --out "${output_root}/${city}_shadow_off.pos" \
    --no-kml \
    --preset low-cost \
    --ratio 1.5 \
    --ar-policy demo5-continuous \
    --glonass-ar autocal \
    --max-pos-jump 0 \
    --rtk-update-outlier-threshold 3 \
    --prefer-trusted-seed \
    --rover-seed-pos "${seed}" \
    --lambda-shadow-candidates 0 \
    --debug-epoch-log "${output_root}/${city}_shadow_off_debug.csv" \
    >"${output_root}/${city}_shadow_off.log" 2>&1

sha256sum "${output_root}/${city}_shadow_off.pos"
