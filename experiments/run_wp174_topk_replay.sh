#!/usr/bin/env bash
set -euo pipefail

if [[ ($# -lt 1 || $# -gt 2) ||
      ("$1" != "tokyo" && "$1" != "nagoya") ]]; then
  echo "usage: $0 {tokyo|nagoya} [candidate-count]" >&2
  exit 2
fi

city="$1"
candidate_count="${2:-8}"
if ! [[ "${candidate_count}" =~ ^[0-9]+$ ]] ||
   ((candidate_count < 2 || candidate_count > 32)); then
  echo "candidate-count must be an integer in [2, 32]" >&2
  exit 2
fi
src_par_success_rate="${WP174_SRC_PAR_SUCCESS_RATE:-0}"
src_par_covariance_scale="${WP174_SRC_PAR_COVARIANCE_SCALE:-1}"
satellite_par_max_drops="${WP174_SATELLITE_PAR_MAX_DROPS:-0}"
satellite_par_covariance_scale="${WP174_SATELLITE_PAR_COVARIANCE_SCALE:-16}"
safe_fix_shadow="${WP174_SAFE_FIX_SHADOW:-0}"
robust_consensus="${WP174_SAFE_FIX_ROBUST_CONSENSUS:-0}"
if [[ "${robust_consensus}" != "0" &&
      "${robust_consensus}" != "1" ]]; then
  echo "WP174_SAFE_FIX_ROBUST_CONSENSUS must be 0 or 1" >&2
  exit 2
fi
library_fixed_quality_gate="${WP175_LIBRARY_FIXED_QUALITY_GATE:-0}"
if [[ "${library_fixed_quality_gate}" != "0" &&
      "${library_fixed_quality_gate}" != "1" ]]; then
  echo "WP175_LIBRARY_FIXED_QUALITY_GATE must be 0 or 1" >&2
  exit 2
fi
l1_l5_causal_arcs="${WP175_L1_L5_CAUSAL_ARCS:-0}"
if [[ "${l1_l5_causal_arcs}" != "0" &&
      "${l1_l5_causal_arcs}" != "1" ]]; then
  echo "WP175_L1_L5_CAUSAL_ARCS must be 0 or 1" >&2
  exit 2
fi
student_t_front_end="${WP175_STUDENT_T_FRONT_END:-0}"
if [[ "${student_t_front_end}" != "0" &&
      "${student_t_front_end}" != "1" ]]; then
  echo "WP175_STUDENT_T_FRONT_END must be 0 or 1" >&2
  exit 2
fi
independent_failure_budget="${WP175_INDEPENDENT_FAILURE_BUDGET:-0}"
if [[ "${independent_failure_budget}" != "0" &&
      "${independent_failure_budget}" != "1" ]]; then
  echo "WP175_INDEPENDENT_FAILURE_BUDGET must be 0 or 1" >&2
  exit 2
fi
quality_diverse_par="${WP174_QUALITY_DIVERSE_PAR:-0}"
if [[ "${quality_diverse_par}" != "0" &&
      "${quality_diverse_par}" != "1" ]]; then
  echo "WP174_QUALITY_DIVERSE_PAR must be 0 or 1" >&2
  exit 2
fi
safe_float_continuity="${WP174_SAFE_FLOAT_CONTINUITY:-0}"
if [[ "${safe_float_continuity}" != "0" &&
      "${safe_float_continuity}" != "1" ]]; then
  echo "WP174_SAFE_FLOAT_CONTINUITY must be 0 or 1" >&2
  exit 2
fi
safe_availability_fallback="${WP174_SAFE_AVAILABILITY_FALLBACK:-0}"
if [[ "${safe_availability_fallback}" != "0" &&
      "${safe_availability_fallback}" != "1" ]]; then
  echo "WP174_SAFE_AVAILABILITY_FALLBACK must be 0 or 1" >&2
  exit 2
fi
maximum_epochs="${WP174_MAX_EPOCHS:-}"
if [[ -n "${maximum_epochs}" ]] &&
   (! [[ "${maximum_epochs}" =~ ^[0-9]+$ ]] ||
    ((maximum_epochs < 24))); then
  echo "WP174_MAX_EPOCHS must be empty or an integer >= 24" >&2
  exit 2
fi
skip_epochs="${WP174_SKIP_EPOCHS:-0}"
if ! [[ "${skip_epochs}" =~ ^[0-9]+$ ]]; then
  echo "WP174_SKIP_EPOCHS must be a non-negative integer" >&2
  exit 2
fi
tag="topk${candidate_count}"
if [[ "${src_par_success_rate}" != "0" ]]; then
  src_tag="${src_par_success_rate//./}"
  scale_tag="${src_par_covariance_scale//./}"
  tag="${tag}_srcp${src_tag}_s${scale_tag}"
fi
if ! [[ "${satellite_par_max_drops}" =~ ^[0-9]+$ ]] ||
   ((satellite_par_max_drops > 32)); then
  echo "WP174_SATELLITE_PAR_MAX_DROPS must be an integer in [0, 32]" >&2
  exit 2
fi
if [[ "${satellite_par_max_drops}" != "0" ]]; then
  tag="${tag}_satpar${satellite_par_max_drops}"
fi
if [[ "${safe_fix_shadow}" != "0" && "${safe_fix_shadow}" != "1" ]]; then
  echo "WP174_SAFE_FIX_SHADOW must be 0 or 1" >&2
  exit 2
fi
if [[ "${safe_fix_shadow}" == "1" ]]; then
  tag="${tag}_safefix"
fi
if [[ "${robust_consensus}" == "1" ]]; then
  tag="${tag}_robust"
fi
if [[ "${library_fixed_quality_gate}" == "1" ]]; then
  tag="${tag}_libraryfixgate"
fi
if [[ "${l1_l5_causal_arcs}" == "1" ]]; then
  tag="${tag}_l1l5causalarc"
fi
if [[ "${student_t_front_end}" == "1" ]]; then
  tag="${tag}_studentt"
fi
if [[ "${independent_failure_budget}" == "1" ]]; then
  tag="${tag}_failurebudget"
fi
if [[ "${safe_float_continuity}" == "1" ]]; then
  tag="${tag}_continuity"
fi
if [[ "${safe_availability_fallback}" == "1" ]]; then
  tag="${tag}_availability"
fi
if [[ -n "${maximum_epochs}" ]]; then
  tag="${tag}_epochs${maximum_epochs}"
fi
if [[ "${skip_epochs}" != "0" ]]; then
  tag="${tag}_skip${skip_epochs}"
fi
tag_suffix="${WP174_TAG_SUFFIX:-}"
if [[ -n "${tag_suffix}" ]]; then
  if ! [[ "${tag_suffix}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "WP174_TAG_SUFFIX must contain only letters, digits, _ or -" >&2
    exit 2
  fi
  tag="${tag}_${tag_suffix}"
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
solver="${repo_root}/third_party/gnssplusplus/build/apps/gnss_solve"
dataset_parent="${WP174_DATASET_ROOT:-/mnt/e/datasets/PPC-Dataset-data}"
dataset_root="${dataset_parent}/${city}/run1"
output_root="${repo_root}/results/wp174"

if [[ "${city}" == "tokyo" ]]; then
  seed="${repo_root}/dist/tokyo-supply/wp160_seed.pos"
  locked="${repo_root}/dist/tokyo-supply/wp160_seeded_demo5.pos"
else
  seed="${repo_root}/dist/tokyo-supply/nagoya_wp100_seed.pos"
  locked="${repo_root}/dist/tokyo-supply/nagoya_wp100_seeded_demo5.pos"
fi

mkdir -p "${output_root}"
shadow_extra=()
if [[ "${src_par_success_rate}" != "0" ]]; then
  shadow_extra+=(
    --lambda-src-par-shadow-success-rate "${src_par_success_rate}"
    --lambda-src-par-shadow-covariance-scale "${src_par_covariance_scale}"
  )
fi
if [[ "${satellite_par_max_drops}" != "0" ]]; then
  shadow_extra+=(
    --lambda-satellite-par-shadow-max-drops "${satellite_par_max_drops}"
    --lambda-satellite-par-shadow-covariance-scale "${satellite_par_covariance_scale}"
  )
fi
if [[ "${safe_fix_shadow}" == "1" ]]; then
  shadow_extra+=(--safe-fix-shadow-state-machine)
fi
if [[ "${robust_consensus}" == "1" ]]; then
  shadow_extra+=(--safe-fix-robust-consensus-shadow)
fi
if [[ "${library_fixed_quality_gate}" == "1" ]]; then
  shadow_extra+=(--library-fixed-quality-gate)
fi
if [[ "${l1_l5_causal_arcs}" == "1" ]]; then
  shadow_extra+=(
    --enable-l5
    --lambda-l1-l5-wlnl-shadow
    --lambda-l1-l5-wlnl-causal-arcs
  )
fi
if [[ "${student_t_front_end}" == "1" ]]; then
  shadow_extra+=(--student-t-rtk-front-end)
fi
if [[ "${independent_failure_budget}" == "1" ]]; then
  shadow_extra+=(--safe-fix-independent-failure-budget)
fi
if [[ "${safe_float_continuity}" == "1" ]]; then
  shadow_extra+=(--safe-float-continuity)
fi
if [[ "${safe_availability_fallback}" == "1" ]]; then
  shadow_extra+=(--safe-availability-fallback)
fi
if [[ "${quality_diverse_par}" == "1" ]]; then
  shadow_extra+=(--lambda-satellite-par-shadow-quality-diverse)
fi
limit_args=()
if [[ "${skip_epochs}" != "0" ]]; then
  limit_args+=(--skip-epochs "${skip_epochs}")
fi
if [[ -n "${maximum_epochs}" ]]; then
  limit_args+=(--max-epochs "${maximum_epochs}")
fi
{
  echo "submodule_commit=$(git -C "${repo_root}/third_party/gnssplusplus" rev-parse HEAD)"
  echo "solver_sha256=$(sha256sum "${solver}" | awk '{print $1}')"
  echo "working_diff_sha256=$(git -C "${repo_root}/third_party/gnssplusplus" diff --binary | sha256sum | awk '{print $1}')"
  echo "candidate_count=${candidate_count}"
  echo "src_par_success_rate=${src_par_success_rate}"
  echo "src_par_covariance_scale=${src_par_covariance_scale}"
  echo "satellite_par_max_drops=${satellite_par_max_drops}"
  echo "satellite_par_covariance_scale=${satellite_par_covariance_scale}"
  echo "safe_fix_shadow=${safe_fix_shadow}"
  echo "robust_consensus=${robust_consensus}"
  echo "library_fixed_quality_gate=${library_fixed_quality_gate}"
  echo "l1_l5_causal_arcs=${l1_l5_causal_arcs}"
  echo "student_t_front_end=${student_t_front_end}"
  echo "independent_failure_budget=${independent_failure_budget}"
  echo "satellite_par_quality_diverse=${quality_diverse_par}"
  echo "safe_float_continuity=${safe_float_continuity}"
  echo "safe_availability_fallback=${safe_availability_fallback}"
  echo "maximum_epochs=${maximum_epochs:-full_route}"
  echo "skip_epochs=${skip_epochs}"
} >"${output_root}/${city}_${tag}_provenance.txt"
/usr/bin/time -f "%e" -o "${output_root}/${city}_${tag}_elapsed_s.txt" \
  "${solver}" \
    --rover "${dataset_root}/rover.obs" \
    --base "${dataset_root}/base.obs" \
    --nav "${dataset_root}/base.nav" \
    --out "${output_root}/${city}_${tag}.pos" \
    --no-kml \
    --preset low-cost \
    --ratio 1.5 \
    --ar-policy demo5-continuous \
    --glonass-ar autocal \
    --max-pos-jump 0 \
    --rtk-update-outlier-threshold 3 \
    --prefer-trusted-seed \
    --rover-seed-pos "${seed}" \
    --lambda-shadow-candidates "${candidate_count}" \
    "${shadow_extra[@]}" \
    "${limit_args[@]}" \
    --debug-epoch-log "${output_root}/${city}_${tag}_debug.csv" \
    >"${output_root}/${city}_${tag}.log" 2>&1

if cmp -s "${locked}" "${output_root}/${city}_${tag}.pos"; then
  echo "locked_match=true"
else
  echo "locked_match=false"
fi
sha256sum "${locked}" "${output_root}/${city}_${tag}.pos"
