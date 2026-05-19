#!/usr/bin/env bash
# Phase 71 production candidate: Phase43 plus n/r2-only OSM road-centerline candidate.
#
# Expected projection:
#   Phase43 official: 85.998294%
#   Phase71 official: 86.205492% (+0.207198pp)
set -euo pipefail

cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
source .venv/bin/activate
OSM_PY=/usr/bin/python3

FGO_V2_DIR=experiments/results/libgnss_diag_phase10/fgo_v2_gap
FGO_V14_DIR=experiments/results/libgnss_diag_phase10/fgo_v14_snr38
FGO_V17_DIR=experiments/results/libgnss_diag_phase10/fgo_v17_el25
G_DEFAULT=experiments/results/libgnss_diag_phase19/gici_tc_esdfix
G_Z=experiments/results/libgnss_diag_phase19/gici_full_zeroarm
G_R=experiments/results/libgnss_diag_phase19/gici_full_ratio25
G_LP=experiments/results/libgnss_diag_phase19/gici_full_loosepr
G_LH=experiments/results/libgnss_diag_phase19/gici_full_loosephase
G_R4=experiments/results/libgnss_diag_phase19/gici_full_ratio40
G_COMBO=experiments/results/libgnss_diag_phase19/gici_full_combo
G_C4=experiments/results/libgnss_diag_phase19/gici_full_combo4
G_LL=experiments/results/libgnss_diag_phase19/gici_full_lprlph
G_ZR=experiments/results/libgnss_diag_phase19/gici_full_zr
G_OA=experiments/results/libgnss_diag_phase19/gici_full_onarm
G_LA=experiments/results/libgnss_diag_phase19/gici_full_lowacc
G_HS=experiments/results/libgnss_diag_phase19/gici_full_hisnr
G_HS45=experiments/results/libgnss_diag_phase19/gici_full_hisnr45
G_HS30=experiments/results/libgnss_diag_phase19/gici_full_hisnr30
G_HE=experiments/results/libgnss_diag_phase19/gici_full_hielev
G_IR=experiments/results/libgnss_diag_phase19/gici_full_imurot
G_MB=experiments/results/libgnss_diag_phase19/gici_full_himuba
G_W5=experiments/results/libgnss_diag_phase19/gici_full_window5
G_OSM="${PHASE71_OSM_DIR:-/tmp/phase71_osmroad_hs_alpha05_triggered}"

V1_CSV=experiments/results/selector_ranker_predictions.csv
V3_CSV=experiments/results/selector_ranker_predictions_v3.csv
V5_CSV=experiments/results/selector_ranker_predictions_v5_nlos.csv
V5_OSM="${PHASE71_V5_OSM:-/tmp/selector_ranker_predictions_phase71_osmroad_overlay_v5.csv}"
NAGOYA_RUN2_INTERNAL=experiments/results/ppc_phase57_gap_nagoya_run2_internal_epochs.csv

declare -A CONFIG
CONFIG[tokyo_run1_csv]="$V1_CSV"; CONFIG[tokyo_run1_k]="3";  CONFIG[tokyo_run1_mode]="ranker"
CONFIG[tokyo_run2_csv]="$V3_CSV"; CONFIG[tokyo_run2_k]="3";  CONFIG[tokyo_run2_mode]="ranker"
CONFIG[tokyo_run3_csv]="$V1_CSV"; CONFIG[tokyo_run3_k]="3";  CONFIG[tokyo_run3_mode]="ranker"
CONFIG[nagoya_run1_csv]="$V3_CSV"; CONFIG[nagoya_run1_k]="99"; CONFIG[nagoya_run1_mode]="ranker"
CONFIG[nagoya_run2_csv]="$V5_OSM"; CONFIG[nagoya_run2_k]="99"; CONFIG[nagoya_run2_mode]="ranker_gici_cluster_override"
CONFIG[nagoya_run3_csv]="$V1_CSV"; CONFIG[nagoya_run3_k]="3";  CONFIG[nagoya_run3_mode]="ranker"

base_extra_labels() {
  printf '%s' "xd_fgo_v2_gap,xd_fgo_v14_snr38,xd_fgo_v17_el25,xd_gici_def,xd_gici_z,xd_gici_r,xd_gici_lp,xd_gici_lh,xd_gici_r4,xd_gici_combo,xd_gici_c4,xd_gici_lprlph,xd_gici_zr,xd_gici_oa,xd_gici_la,xd_gici_hs,xd_gici_hs45,xd_gici_hs30,xd_gici_he,xd_gici_ir,xd_gici_mb,xd_gici_w5"
}

base_extra_dirs() {
  printf '%s' "$FGO_V2_DIR,$FGO_V14_DIR,$FGO_V17_DIR,$G_DEFAULT,$G_Z,$G_R,$G_LP,$G_LH,$G_R4,$G_COMBO,$G_C4,$G_LL,$G_ZR,$G_OA,$G_LA,$G_HS,$G_HS45,$G_HS30,$G_HE,$G_IR,$G_MB,$G_W5"
}

candidate_labels() {
  local city=$1
  local run=$2
  local old_labels
  old_labels=$(cat "/tmp/${city}_${run}_phase11fa_labels.txt")
  if [[ "$city" == "nagoya" && "$run" == "run2" ]]; then
    printf '%s,%s,xd_gici_osmroad_hs' "$old_labels" "$(base_extra_labels)"
  else
    printf '%s,%s' "$old_labels" "$(base_extra_labels)"
  fi
}

candidate_dirs() {
  local city=$1
  local run=$2
  local old_dirs
  old_dirs=$(cat "/tmp/${city}_${run}_phase11fa_dirs.txt")
  if [[ "$city" == "nagoya" && "$run" == "run2" ]]; then
    printf '%s,%s,%s' "$old_dirs" "$(base_extra_dirs)" "$G_OSM"
  else
    printf '%s,%s' "$old_dirs" "$(base_extra_dirs)"
  fi
}

echo "=== Phase 71 prep: materialize n/r2 OSM road candidate ==="
if [[ ! -f "$NAGOYA_RUN2_INTERNAL" ]]; then
  echo "missing n/r2 Phase43 internal diagnostics: $NAGOYA_RUN2_INTERNAL" >&2
  exit 1
fi
PYTHONPATH=.:python:experiments "$OSM_PY" experiments/materialize_phase70_osm_road_centerline_candidate.py \
  "$NAGOYA_RUN2_INTERNAL" \
  --source-pos experiments/results/libgnss_diag_phase19/gici_full_hisnr/nagoya_run2_full.pos \
  --source-diag-csv experiments/results/libgnss_diag_phase19/gici_full_hisnr/nagoya_run2_full.csv \
  --out-dir "$G_OSM" \
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

echo "=== Phase 71 prep: build n/r2 ranker overlay ==="
PYTHONPATH=python python3 experiments/build_phase70_osm_road_ranker_overlay.py \
  --base-predictions "$V5_CSV" \
  --trigger-epochs "$G_OSM/nagoya_run2_phase70_osm_road_materialize_epochs.csv" \
  --out-csv "$V5_OSM" \
  --run-id nagoya_run2 \
  --label xd_gici_osmroad_hs \
  --p-pass 999.0

if [[ "${PHASE71_PREP_ONLY:-0}" == "1" ]]; then
  echo "PHASE71_PREP_ONLY=1; stopping after artifact preparation."
  exit 0
fi

for CR in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  CITY=${CR%/*}
  RUN=${CR#*/}
  KEY="${CITY}_${RUN}"
  CSV="${CONFIG[${KEY}_csv]}"
  K="${CONFIG[${KEY}_k]}"
  MODE="${CONFIG[${KEY}_mode]}"

  echo "=== Phase 71 OSM road PROD ${CITY}/${RUN} (mode=${MODE}, k=${K}) ==="
  PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py --runs "${CITY}/${RUN}" \
    --methods "rbpf+dd+gate+hybrid+rtkdiag_pf" \
    --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 --hybrid-sigma-m 1.0 \
    --rtkdiag-candidate-pos-dirs "$(candidate_dirs "$CITY" "$RUN")" \
    --rtkdiag-candidate-diag-dirs "$(candidate_dirs "$CITY" "$RUN")" \
    --rtkdiag-candidate-labels "$(candidate_labels "$CITY" "$RUN")" \
    --rtkdiag-candidate-block-labels-by-run "" \
    --rtkdiag-candidate-run-index-policy phase11ep \
    --rtkdiag-candidate-select-mode "$MODE" \
    --rtkdiag-candidate-ranker-score-path "$CSV" \
    --rtkdiag-candidate-emit-mode candidate \
    --rtkdiag-candidate-residual-rms-max 50.0 --rtkdiag-candidate-ratio-min 1.0 \
    --rtkdiag-candidate-recenter-max-shift-m 10000.0 --rtkdiag-candidate-emit-max-diff-m 0.4 \
    --rtkdiag-candidate-max-to-hybrid-m 0 \
    --rtkdiag-candidate-fallback-mode hybrid \
    --rtkdiag-candidate-bridge-enable --rtkdiag-candidate-bridge-max-s 6.0 --rtkdiag-candidate-bridge-residual-rms-m 0.2 \
    --rtkdiag-candidate-rms-prefilter-k "$K" \
    --n-particles 2000 --pos-dir "/tmp/ppc_phase71_osmroad_prod_${KEY}" \
    --results-prefix "ppc_phase71_osmroad_prod_${KEY}_full"
  PPC=$(awk -F, 'NR==2 {printf "%.6f", $11}' "experiments/results/ppc_phase71_osmroad_prod_${KEY}_full_runs.csv")
  echo "FIN ${CITY}/${RUN}: ${PPC}%"
done

PYTHONPATH=python python3 - <<'PY'
import csv
from pathlib import Path

runs = ["tokyo_run1", "tokyo_run2", "tokyo_run3", "nagoya_run1", "nagoya_run2", "nagoya_run3"]
rows = []
for key in runs:
    b = next(csv.DictReader(open(f"experiments/results/ppc_ctrbpf_fgo_phase43_prod_{key}_full_runs.csv")))
    p = next(csv.DictReader(open(f"experiments/results/ppc_phase71_osmroad_prod_{key}_full_runs.csv")))
    rows.append(
        {
            "run": key,
            "phase43_ppc_pct": b["honest_ppc_pct"],
            "phase71_ppc_pct": p["honest_ppc_pct"],
            "delta_pp": float(p["honest_ppc_pct"]) - float(b["honest_ppc_pct"]),
            "phase43_pass_m": b["honest_pass_m"],
            "phase71_pass_m": p["honest_pass_m"],
            "delta_pass_m": float(p["honest_pass_m"]) - float(b["honest_pass_m"]),
        }
    )

path = Path("experiments/results/phase71_osmroad_production_summary.csv")
with path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
avg43 = sum(float(r["phase43_ppc_pct"]) for r in rows) / len(rows)
avg71 = sum(float(r["phase71_ppc_pct"]) for r in rows) / len(rows)
print(f"Saved: {path}")
print(f"Phase43 official: {avg43:.6f}%")
print(f"Phase71 official: {avg71:.6f}%")
print(f"Delta: {avg71 - avg43:+.6f}pp")
PY
