#!/usr/bin/env bash
# Phase79 production candidate: Phase71 plus n/r2-only top1 label-pair override.
#
# Expected projection:
#   Phase71 official: 86.205492%
#   Phase79 official: 86.457091% (+0.251599pp)
set -euo pipefail

cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
source .venv/bin/activate

OSM_PY="${OSM_PY:-/usr/bin/python3}"
V5_CSV=experiments/results/selector_ranker_predictions_v5_nlos.csv
NAGOYA_RUN2_INTERNAL=experiments/results/ppc_phase57_gap_nagoya_run2_internal_epochs.csv

PHASE79_OSM_DIR="${PHASE79_OSM_DIR:-/tmp/phase79_osmroad_hs_alpha05_triggered}"
PHASE79_PHASE71_OVERLAY="${PHASE79_PHASE71_OVERLAY:-/tmp/selector_ranker_predictions_phase79_phase71_osmroad_overlay_v5.csv}"
PHASE79_TOP1_OVERLAY="${PHASE79_TOP1_OVERLAY:-/tmp/selector_ranker_predictions_phase79_nr2_top1_label_pair_overlay.csv}"
PHASE79_OVERLAY_PREFIX="${PHASE79_OVERLAY_PREFIX:-experiments/results/phase79_top1_production_label_pair_overlay}"
PHASE79_RESULTS_PREFIX_BASE="${PHASE79_RESULTS_PREFIX_BASE:-ppc_phase79_top1_prod}"
PHASE79_POS_DIR_BASE="${PHASE79_POS_DIR_BASE:-/tmp/ppc_phase79_top1_prod}"
PHASE79_SUMMARY="${PHASE79_SUMMARY:-experiments/results/phase79_top1_production_summary.csv}"

echo "=== Phase79 prep: materialize n/r2 Phase71 OSM road candidate ==="
if [[ ! -f "$NAGOYA_RUN2_INTERNAL" ]]; then
  echo "missing n/r2 Phase43 internal diagnostics: $NAGOYA_RUN2_INTERNAL" >&2
  exit 1
fi

PYTHONPATH=.:python:experiments "$OSM_PY" experiments/materialize_phase70_osm_road_centerline_candidate.py \
  "$NAGOYA_RUN2_INTERNAL" \
  --source-pos experiments/results/libgnss_diag_phase19/gici_full_hisnr/nagoya_run2_full.pos \
  --source-diag-csv experiments/results/libgnss_diag_phase19/gici_full_hisnr/nagoya_run2_full.csv \
  --out-dir "$PHASE79_OSM_DIR" \
  --city nagoya \
  --run run2 \
  --selected-label xd_gici_hs \
  --start-epoch 0 \
  --end-epoch 9434 \
  --bbox-margin-deg 0.002 \
  --epsg 32653 \
  --alpha 0.5 \
  --road-dist-min-m 2.5 \
  --min-contiguous-epochs 40

echo "=== Phase79 prep: build n/r2 Phase71 OSM ranker overlay ==="
PYTHONPATH=python python3 experiments/build_phase70_osm_road_ranker_overlay.py \
  --base-predictions "$V5_CSV" \
  --trigger-epochs "$PHASE79_OSM_DIR/nagoya_run2_phase70_osm_road_materialize_epochs.csv" \
  --out-csv "$PHASE79_PHASE71_OVERLAY" \
  --run-id nagoya_run2 \
  --label xd_gici_osmroad_hs \
  --p-pass 999.0

echo "=== Phase79 prep: build n/r2 top1 label-pair ranker overlay ==="
PYTHONPATH=python python3 experiments/build_phase79_nr2_top1_label_pair_overlay.py \
  --base-predictions "$PHASE79_PHASE71_OVERLAY" \
  --out-csv "$PHASE79_TOP1_OVERLAY" \
  --out-prefix "$PHASE79_OVERLAY_PREFIX"

if [[ "${PHASE79_PREP_ONLY:-0}" == "1" ]]; then
  echo "PHASE79_PREP_ONLY=1; stopping after artifact preparation."
  exit 0
fi

runner_args=(
  --phase79-score-path "$PHASE79_TOP1_OVERLAY"
  --osm-dir "$PHASE79_OSM_DIR"
  --results-prefix-base "$PHASE79_RESULTS_PREFIX_BASE"
  --pos-dir-base "$PHASE79_POS_DIR_BASE"
  --summary-path "$PHASE79_SUMMARY"
)

if [[ "${PHASE79_SKIP_EXISTING:-0}" == "1" ]]; then
  runner_args+=(--skip-existing)
fi
if [[ "${PHASE79_DRY_RUN:-0}" == "1" ]]; then
  runner_args+=(--dry-run)
fi
if [[ "${PHASE79_WRITE_INTERNAL:-0}" == "1" ]]; then
  runner_args+=(--write-internal-diagnostics)
fi
if [[ -n "${PHASE79_RUNS:-}" ]]; then
  # Space-separated run keys, e.g. PHASE79_RUNS="nagoya_run2" for a quick smoke.
  read -r -a phase79_runs <<< "$PHASE79_RUNS"
  runner_args+=(--runs "${phase79_runs[@]}")
fi

echo "=== Phase79 PROD six-run replay ==="
PYTHONPATH=python python3 experiments/run_phase79_sixrun_neutral_check.py "${runner_args[@]}"
