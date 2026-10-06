#!/usr/bin/env bash
# Run the shipped RB-FGO-PF configuration (WP18 + output guards) on one
# Tokyo PPC run. Equivalent to repro_tc_fgo/results/wp18/run_rbpf18.sh.
#
#   PPC_DATA_ROOT=.../PPC-Dataset-data/tokyo \
#   bash run_ship.sh <name> <run 1|2|3> <max_epochs> [extra wp16_run_rbpf.py args]
#
# Writes results/<name>/run<N>.{pos,npz} and results/<name>.log. The published
# tables additionally relabel fixes below the report floor as float:
#   python relabel_nb9_pos.py results/<name>/runN.pos results/<name>/runN.npz out.pos 12
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$1"; RUN="$2"; MAXEP="$3"; shift 3
PY="${RBPF_PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x "$HERE/.venv/Scripts/python.exe" ]; then PY="$HERE/.venv/Scripts/python.exe"; else PY="$HERE/.venv/bin/python"; fi
fi

GUARDS="--env RBPF_FIX_VOTE_DD=6.0 --env RBPF_FIX_VOTE_DPR=3.5 --env RBPF_FB_COMMIT_MAX_DPR=1.5"
R1FLAGS=""
if [ "$RUN" = "1" ]; then
  R1FLAGS="--sanity-enable 1 --env SANITY_PR_ONLY=1 --env RECOV_CP_HOLD_EPOCHS=5 --env RECOV_CP_RELEASE_THRESH=2.0 --env RECOV_CP_RELEASE_COUNT=5 --env SANITY_MAX_MEDIAN_RATIO=5.0"
fi
mkdir -p "$HERE/results"
# shellcheck disable=SC2086
"$PY" "$HERE/wp16_run_rbpf.py" "$RUN" --max-ep "$MAXEP" \
  --out-dir "$HERE/results/$NAME" \
  --env SEED_N_CYCLES=1 --env COND_HOLD=1 --env RBPF_FB=1 --env RBPF_SPAWN=1 \
  $R1FLAGS $GUARDS "$@" > "$HERE/results/$NAME.log" 2>&1
tail -n 6 "$HERE/results/$NAME.log"
