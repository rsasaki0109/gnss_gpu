#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]] ||
   [[ "$1" != "tokyo" && "$1" != "nagoya" ]] ||
   [[ "$2" != "outage" && "$2" != "cycle_slip" &&
      "$2" != "satellite_loss" && "$2" != "nlos" ]]; then
  echo "usage: $0 {tokyo|nagoya} {outage|cycle_slip|satellite_loss|nlos} [maximum-epochs]" >&2
  exit 2
fi

city="$1"
fault="$2"
maximum_epochs="${3:-}"
if [[ -n "${maximum_epochs}" ]] &&
   (! [[ "${maximum_epochs}" =~ ^[0-9]+$ ]] || ((maximum_epochs < 24))); then
  echo "maximum-epochs must be an integer >= 24" >&2
  exit 2
fi

event_count="${WP174_RAW_FAULT_EVENTS:-8}"
if ! [[ "${event_count}" =~ ^[0-9]+$ ]] || ((event_count < 1)); then
  echo "WP174_RAW_FAULT_EVENTS must be a positive integer" >&2
  exit 2
fi
fault_duration_s="${WP174_RAW_FAULT_DURATION_S:-5}"
if ! [[ "${fault_duration_s}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "WP174_RAW_FAULT_DURATION_S must be a positive number" >&2
  exit 2
fi
fix_source="${WP174_RAW_FIX_SOURCE:-solver}"
if [[ "${fix_source}" != "solver" &&
      "${fix_source}" != "safe_fix_shadow" ]]; then
  echo "WP174_RAW_FIX_SOURCE must be solver or safe_fix_shadow" >&2
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
library_fix_anchored="${WP175_LIBRARY_FIX_ANCHORED_FAULTS:-0}"
if [[ "${library_fix_anchored}" != "0" &&
      "${library_fix_anchored}" != "1" ]]; then
  echo "WP175_LIBRARY_FIX_ANCHORED_FAULTS must be 0 or 1" >&2
  exit 2
fi
if [[ "${library_fixed_quality_gate}" == "1" &&
      "${robust_consensus}" != "1" ]]; then
  echo "library FIX quality gate requires robust consensus" >&2
  exit 2
fi
quality_diverse_par="${WP174_QUALITY_DIVERSE_PAR:-0}"
if [[ "${quality_diverse_par}" != "0" &&
      "${quality_diverse_par}" != "1" ]]; then
  echo "WP174_QUALITY_DIVERSE_PAR must be 0 or 1" >&2
  exit 2
fi
tag_suffix="${WP174_TAG_SUFFIX:-}"
if [[ -n "${tag_suffix}" &&
      ! "${tag_suffix}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "WP174_TAG_SUFFIX contains unsupported characters" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_parent_wsl="${WP174_DATASET_ROOT:-/mnt/e/datasets/PPC-Dataset-data}"
dataset_parent_windows="${WP174_DATASET_ROOT_WINDOWS:-E:/datasets/PPC-Dataset-data}"
dataset_root_wsl="${dataset_parent_wsl}/${city}/run1"
dataset_root_windows="${dataset_parent_windows}/${city}/run1"
solver="${repo_root}/third_party/gnssplusplus/build/apps/gnss_solve"
output_root="${repo_root}/results/wp174"
document_root="${repo_root}/internal_docs"
tag="${city}_raw_${fault}"
if [[ "${fix_source}" == "safe_fix_shadow" ]]; then
  tag="${tag}_safe"
fi
if [[ "${library_fixed_quality_gate}" == "1" ]]; then
  tag="${tag}_libraryfixgate"
fi
if [[ "${library_fix_anchored}" == "1" ]]; then
  tag="${tag}_anchored"
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
if [[ -n "${tag_suffix}" ]]; then
  tag="${tag}_${tag_suffix}"
fi

if [[ "${city}" == "tokyo" ]]; then
  seed="${repo_root}/dist/tokyo-supply/wp160_seed.pos"
  anchor_pos="${output_root}/tokyo_topk2_robust_libraryfixgate_continuity_availability_wp175_full_v1.pos"
else
  seed="${repo_root}/dist/tokyo-supply/nagoya_wp100_seed.pos"
  anchor_pos="${output_root}/nagoya_topk2_robust_libraryfixgate_continuity_availability_wp175_full_v1.pos"
fi

mkdir -p "${output_root}" "${document_root}"
mutator_args=(
  experiments/inject_wp174_rinex_faults.py
  --input "${dataset_root_windows}/rover.obs"
  --output "results/wp174/${tag}.obs"
  --manifest "internal_docs/wp174_${tag}_manifest_2026_07_29.json"
  --fault "${fault}"
  --events "${event_count}"
  --duration-s "${fault_duration_s}"
)
if [[ -n "${maximum_epochs}" ]]; then
  mutator_args+=(--maximum-epochs "${maximum_epochs}")
fi
if [[ "${library_fix_anchored}" == "1" ]]; then
  mutator_args+=(
    --anchor-pos "${anchor_pos}"
    --anchor-streak-epochs 10
    --recovery-horizon-s 10
  )
fi

cd "${repo_root}"
python.exe "${mutator_args[@]}"

{
  echo "submodule_commit=$(git -C third_party/gnssplusplus rev-parse HEAD)"
  echo "working_diff_sha256=$(git -C third_party/gnssplusplus diff --binary | sha256sum | awk '{print $1}')"
  echo "solver_sha256=$(sha256sum "${solver}" | awk '{print $1}')"
  echo "city=${city}"
  echo "fault=${fault}"
  echo "event_count=${event_count}"
  echo "maximum_epochs=${maximum_epochs:-full_route}"
  echo "fix_source=${fix_source}"
  echo "safe_float_continuity=${safe_float_continuity}"
  echo "safe_availability_fallback=${safe_availability_fallback}"
  echo "fault_duration_s=${fault_duration_s}"
  echo "robust_consensus=${robust_consensus}"
  echo "library_fixed_quality_gate=${library_fixed_quality_gate}"
  echo "library_fix_anchored_faults=${library_fix_anchored}"
  if [[ "${library_fix_anchored}" == "1" ]]; then
    echo "anchor_pos_sha256=$(sha256sum "${anchor_pos}" | awk '{print $1}')"
  fi
  echo "satellite_par_quality_diverse=${quality_diverse_par}"
  echo "runtime_fgo=false"
} >"${output_root}/${tag}_provenance.txt"

safe_fix_args=()
if [[ "${fix_source}" == "safe_fix_shadow" ]]; then
  safe_fix_args+=(
    --lambda-shadow-candidates 2
    --safe-fix-shadow-state-machine
  )
fi
if [[ "${robust_consensus}" == "1" ]]; then
  safe_fix_args=(
    --safe-fix-robust-consensus-shadow
  )
fi
if [[ "${library_fixed_quality_gate}" == "1" ]]; then
  safe_fix_args+=(--library-fixed-quality-gate)
fi
if [[ "${safe_float_continuity}" == "1" ]]; then
  safe_fix_args+=(--safe-float-continuity)
fi
if [[ "${safe_availability_fallback}" == "1" ]]; then
  safe_fix_args+=(--safe-availability-fallback)
fi
if [[ "${quality_diverse_par}" == "1" ]]; then
  safe_fix_args+=(--lambda-satellite-par-shadow-quality-diverse)
fi

/usr/bin/time -f "%e" -o "${output_root}/${tag}_elapsed_s.txt" \
  "${solver}" \
    --rover "${output_root}/${tag}.obs" \
    --base "${dataset_root_wsl}/base.obs" \
    --nav "${dataset_root_wsl}/base.nav" \
    --out "${output_root}/${tag}.pos" \
    --no-kml \
    --preset low-cost \
    --ratio 1.5 \
    --ar-policy demo5-continuous \
    --glonass-ar autocal \
    --max-pos-jump 0 \
    --rtk-update-outlier-threshold 3 \
    --prefer-trusted-seed \
    --rover-seed-pos "${seed}" \
    "${safe_fix_args[@]}" \
    --debug-epoch-log "${output_root}/${tag}_debug.csv" \
    >"${output_root}/${tag}.log" 2>&1

python.exe experiments/analyze_wp174_raw_fault_replay.py \
  --debug "results/wp174/${tag}_debug.csv" \
  --positions "results/wp174/${tag}.pos" \
  --reference "${dataset_root_windows}/reference.csv" \
  --manifest "internal_docs/wp174_${tag}_manifest_2026_07_29.json" \
  --fix-source "${fix_source}" \
  --output "internal_docs/wp174_${tag}_audit_2026_07_29.json"
