#!/usr/bin/env bash
set -euo pipefail

# Frozen Odaiba reference run for PF smoother refactor checks.
# Extra CLI args are appended after the frozen preset so later duplicates override
# earlier values under argparse.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${URBANNAV_DATA_ROOT:-/tmp/UrbanNav-Tokyo}"

cd "${SCRIPT_DIR}"
export PYTHONPATH="../third_party/gnssplusplus/build/python:../third_party/gnssplusplus/python:../python:."

cmd=(
  python3
  exp_pf_smoother_eval.py
  --data-root "${DATA_ROOT}"
  --preset odaiba_reference
)

cmd+=("$@")
printf '+'
for arg in "${cmd[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

"${cmd[@]}"
