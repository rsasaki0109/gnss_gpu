#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
solver="${repo_root}/third_party/gnssplusplus/build/apps/gnss_solve"
dataset_root="${WP174_TOKYO_RUN1_ROOT:-/mnt/e/datasets/PPC-Dataset-data/tokyo/run1}"
output_root="${WP174_OUTPUT_ROOT:-${repo_root}/results/wp174/smoke}"
max_epochs="${WP174_MAX_EPOCHS:-300}"
skip_epochs="${WP174_SKIP_EPOCHS:-0}"
src_par_success_rate="${WP174_SRC_PAR_SUCCESS_RATE:-0}"
src_par_covariance_scale="${WP174_SRC_PAR_COVARIANCE_SCALE:-1}"
reuse_default="${WP174_REUSE_DEFAULT:-0}"
enable_l5="${WP174_ENABLE_L5:-0}"
enable_wide_lane="${WP174_ENABLE_WIDE_LANE:-0}"
enable_wlnl_fallback="${WP174_ENABLE_WLNL_FALLBACK:-0}"
mkdir -p "${output_root}"

common=(
  --no-kml
  --preset low-cost
  --ratio 1.5
  --ar-policy demo5-continuous
  --glonass-ar autocal
  --max-pos-jump 0
  --rtk-update-outlier-threshold 3
  --prefer-trusted-seed
  --rover-seed-pos "${repo_root}/dist/tokyo-supply/wp160_seed.pos"
  --max-epochs "${max_epochs}"
  --skip-epochs "${skip_epochs}"
)
if [[ "${enable_l5}" == "1" ]]; then
  common+=(--enable-l5)
fi
if [[ "${enable_wide_lane}" == "1" ]]; then
  common+=(--enable-wide-lane-ar)
fi
if [[ "${enable_wlnl_fallback}" == "1" ]]; then
  common+=(--enable-wlnl-fallback)
fi

shadow_extra=()
if [[ "${src_par_success_rate}" != "0" ]]; then
  shadow_extra+=(
    --lambda-src-par-shadow-success-rate "${src_par_success_rate}"
    --lambda-src-par-shadow-covariance-scale "${src_par_covariance_scale}"
  )
fi

if [[ "${reuse_default}" == "1" &&
      -s "${output_root}/default.pos" &&
      -s "${output_root}/default_debug.csv" ]]; then
  echo "Reusing ${output_root}/default.pos"
else
  /usr/bin/time -f "%e" -o "${output_root}/default_elapsed_s.txt" \
    "${solver}" \
    --rover "${dataset_root}/rover.obs" \
    --base "${dataset_root}/base.obs" \
    --nav "${dataset_root}/base.nav" \
    --out "${output_root}/default.pos" \
    --debug-epoch-log "${output_root}/default_debug.csv" \
    "${common[@]}" \
    >"${output_root}/default.log" 2>&1
fi

/usr/bin/time -f "%e" -o "${output_root}/shadow_elapsed_s.txt" \
  "${solver}" \
  --rover "${dataset_root}/rover.obs" \
  --base "${dataset_root}/base.obs" \
  --nav "${dataset_root}/base.nav" \
  --out "${output_root}/shadow.pos" \
  --debug-epoch-log "${output_root}/shadow_debug.csv" \
  --lambda-shadow-candidates 8 \
  "${shadow_extra[@]}" \
  "${common[@]}" \
  >"${output_root}/shadow.log" 2>&1

cmp "${output_root}/default.pos" "${output_root}/shadow.pos"

python3 - "${output_root}/shadow_debug.csv" "${max_epochs}" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == expected_rows, (len(rows), expected_rows)
attempted = sum(int(row["lambda_shadow_attempted"]) for row in rows)
solved = sum(int(row["lambda_shadow_solved"]) for row in rows)
assert attempted > 0 and solved > 0, (attempted, solved)
assert all(
    row["lambda_shadow_candidate_count"] == "8"
    for row in rows
    if row["lambda_shadow_solved"] == "1"
)
print(
    f"WP174 shadow smoke passed: rows={len(rows)}, "
    f"attempted={attempted}, solved={solved}"
)
PY
