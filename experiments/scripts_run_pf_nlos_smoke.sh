#!/usr/bin/env bash
# Smoke: PPC PF-domain NLOS soft mask on one urban-heavy run.
# Requires mask CSV from build_per_epoch_nlos_csv.py (not committed).
set -uo pipefail
cd "$(dirname "$0")/.."

RUN="${1:-tokyo/run1}"
CITY="${RUN%/*}"
RUN_NAME="${RUN#*/}"
MASK="experiments/results/plateau_nlos_phase33/${CITY}_${RUN_NAME}_per_epoch_nlos.csv"

if [[ ! -f "$MASK" ]]; then
  echo "missing mask CSV: $MASK" >&2
  echo "generate with: python experiments/build_per_epoch_nlos_csv.py ..." >&2
  exit 1
fi

PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py \
  --runs "$RUN" \
  --methods "rbpf+dd" \
  --pf-nlos-mask-path "$MASK" \
  --pf-nlos-k-weak 3 \
  --pf-nlos-k-strong 3 \
  --n-particles 2000 \
  --max-epochs 120 \
  --pos-dir "/tmp/pf_nlos_smoke_${CITY}_${RUN_NAME}" \
  --results-prefix "ppc_pf_nlos_smoke_${CITY}_${RUN_NAME}"
