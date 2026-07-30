#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [maximum-epochs]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

for city in tokyo nagoya; do
  for fault in outage cycle_slip satellite_loss nlos; do
    WP174_RAW_FIX_SOURCE=safe_fix_shadow \
      bash experiments/run_wp174_raw_fault_replay.sh \
        "${city}" "${fault}" "$@"
  done
done
