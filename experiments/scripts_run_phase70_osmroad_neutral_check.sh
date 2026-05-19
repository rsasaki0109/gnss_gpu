#!/usr/bin/env bash
# Phase70 OSM-road candidate neutral check across all PPC2024 runs.
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
G_OSM="${PHASE70_OSM_DIR:-/tmp/phase70_osm_road_hs_alpha05_triggered_allruns}"

V1_CSV=experiments/results/selector_ranker_predictions.csv
V3_CSV=experiments/results/selector_ranker_predictions_v3.csv
V5_CSV=experiments/results/selector_ranker_predictions_v5_nlos.csv
V1_OSM="${PHASE70_V1_OSM:-/tmp/selector_ranker_predictions_phase70_osmroad_overlay_v1.csv}"
V3_OSM="${PHASE70_V3_OSM:-/tmp/selector_ranker_predictions_phase70_osmroad_overlay_v3.csv}"
V5_OSM="${PHASE70_V5_OSM:-/tmp/selector_ranker_predictions_phase70_osmroad_overlay_v5.csv}"

declare -A CONFIG
CONFIG[tokyo_run1_csv]="$V1_CSV"; CONFIG[tokyo_run1_osm_csv]="$V1_OSM"; CONFIG[tokyo_run1_k]="3";  CONFIG[tokyo_run1_mode]="ranker"
CONFIG[tokyo_run2_csv]="$V3_CSV"; CONFIG[tokyo_run2_osm_csv]="$V3_OSM"; CONFIG[tokyo_run2_k]="3";  CONFIG[tokyo_run2_mode]="ranker"
CONFIG[tokyo_run3_csv]="$V1_CSV"; CONFIG[tokyo_run3_osm_csv]="$V1_OSM"; CONFIG[tokyo_run3_k]="3";  CONFIG[tokyo_run3_mode]="ranker"
CONFIG[nagoya_run1_csv]="$V3_CSV"; CONFIG[nagoya_run1_osm_csv]="$V3_OSM"; CONFIG[nagoya_run1_k]="99"; CONFIG[nagoya_run1_mode]="ranker"
CONFIG[nagoya_run2_csv]="$V5_CSV"; CONFIG[nagoya_run2_osm_csv]="$V5_OSM"; CONFIG[nagoya_run2_k]="99"; CONFIG[nagoya_run2_mode]="ranker_gici_cluster_override"
CONFIG[nagoya_run3_csv]="$V1_CSV"; CONFIG[nagoya_run3_osm_csv]="$V1_OSM"; CONFIG[nagoya_run3_k]="3";  CONFIG[nagoya_run3_mode]="ranker"

extra_labels() {
  printf '%s' "xd_fgo_v2_gap,xd_fgo_v14_snr38,xd_fgo_v17_el25,xd_gici_def,xd_gici_z,xd_gici_r,xd_gici_lp,xd_gici_lh,xd_gici_r4,xd_gici_combo,xd_gici_c4,xd_gici_lprlph,xd_gici_zr,xd_gici_oa,xd_gici_la,xd_gici_hs,xd_gici_hs45,xd_gici_hs30,xd_gici_he,xd_gici_ir,xd_gici_mb,xd_gici_w5"
}

extra_dirs() {
  printf '%s' "$FGO_V2_DIR,$FGO_V14_DIR,$FGO_V17_DIR,$G_DEFAULT,$G_Z,$G_R,$G_LP,$G_LH,$G_R4,$G_COMBO,$G_C4,$G_LL,$G_ZR,$G_OA,$G_LA,$G_HS,$G_HS45,$G_HS30,$G_HE,$G_IR,$G_MB,$G_W5"
}

phase43_labels() {
  local city=$1
  local run=$2
  local old_labels
  old_labels=$(cat "/tmp/${city}_${run}_phase11fa_labels.txt")
  printf '%s,%s' "$old_labels" "$(extra_labels)"
}

phase43_dirs() {
  local city=$1
  local run=$2
  local old_dirs
  old_dirs=$(cat "/tmp/${city}_${run}_phase11fa_dirs.txt")
  printf '%s,%s' "$old_dirs" "$(extra_dirs)"
}

phase70_labels() {
  printf '%s,xd_gici_osmroad_hs' "$(phase43_labels "$1" "$2")"
}

phase70_dirs() {
  printf '%s,%s' "$(phase43_dirs "$1" "$2")" "$G_OSM"
}

run_phase43_internal() {
  local city=$1
  local run=$2
  local key="${city}_${run}"
  local out="experiments/results/ppc_phase70_baseline_internal_${key}_full_internal_epochs.csv"
  if [[ -f "$out" ]]; then
    echo "SKIP baseline internal ${city}/${run}: $out"
    return
  fi
  echo "RUN baseline internal ${city}/${run}"
  PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py --runs "${city}/${run}" \
    --methods "rbpf+dd+gate+hybrid+rtkdiag_pf" \
    --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 --hybrid-sigma-m 1.0 \
    --rtkdiag-candidate-pos-dirs "$(phase43_dirs "$city" "$run")" \
    --rtkdiag-candidate-diag-dirs "$(phase43_dirs "$city" "$run")" \
    --rtkdiag-candidate-labels "$(phase43_labels "$city" "$run")" \
    --rtkdiag-candidate-block-labels-by-run "" \
    --rtkdiag-candidate-run-index-policy phase11ep \
    --rtkdiag-candidate-select-mode "${CONFIG[${key}_mode]}" \
    --rtkdiag-candidate-ranker-score-path "${CONFIG[${key}_csv]}" \
    --rtkdiag-candidate-emit-mode candidate \
    --rtkdiag-candidate-residual-rms-max 50.0 --rtkdiag-candidate-ratio-min 1.0 \
    --rtkdiag-candidate-recenter-max-shift-m 10000.0 --rtkdiag-candidate-emit-max-diff-m 0.4 \
    --rtkdiag-candidate-max-to-hybrid-m 0 \
    --rtkdiag-candidate-fallback-mode hybrid \
    --rtkdiag-candidate-bridge-enable --rtkdiag-candidate-bridge-max-s 6.0 --rtkdiag-candidate-bridge-residual-rms-m 0.2 \
    --rtkdiag-candidate-rms-prefilter-k "${CONFIG[${key}_k]}" \
    --n-particles 2000 --pos-dir "/tmp/ppc_phase70_baseline_internal_${key}" \
    --results-prefix "ppc_phase70_baseline_internal_${key}_full" \
    --write-internal-diagnostics
}

materialize_osm_candidate() {
  local city=$1
  local run=$2
  local key="${city}_${run}"
  local epsg=32653
  if [[ "$city" == "tokyo" ]]; then
    epsg=32654
  fi
  echo "RUN materialize ${city}/${run} epsg=${epsg}"
  PYTHONPATH=.:python:experiments "$OSM_PY" experiments/materialize_phase70_osm_road_centerline_candidate.py \
    "experiments/results/ppc_phase70_baseline_internal_${key}_full_internal_epochs.csv" \
    --source-pos "experiments/results/libgnss_diag_phase19/gici_full_hisnr/${key}_full.pos" \
    --source-diag-csv "experiments/results/libgnss_diag_phase19/gici_full_hisnr/${key}_full.csv" \
    --out-dir "$G_OSM" \
    --city "$city" \
    --run "$run" \
    --selected-label xd_gici_hs \
    --bbox-margin-deg 0.002 \
    --epsg "$epsg" \
    --alpha 0.5 \
    --road-dist-min-m 2.5 \
    --min-contiguous-epochs 40
}

build_overlays() {
  cp "$V1_CSV" "$V1_OSM"
  cp "$V3_CSV" "$V3_OSM"
  cp "$V5_CSV" "$V5_OSM"
  for cr in tokyo/run1 tokyo/run3 nagoya/run3; do
    local city=${cr%/*}
    local run=${cr#*/}
    local key="${city}_${run}"
    PYTHONPATH=python python3 experiments/build_phase70_osm_road_ranker_overlay.py \
      --base-predictions "$V1_OSM" \
      --trigger-epochs "$G_OSM/${key}_phase70_osm_road_materialize_epochs.csv" \
      --out-csv "$V1_OSM" \
      --run-id "$key" \
      --label xd_gici_osmroad_hs \
      --p-pass 999.0
  done
  for cr in tokyo/run2 nagoya/run1; do
    local city=${cr%/*}
    local run=${cr#*/}
    local key="${city}_${run}"
    PYTHONPATH=python python3 experiments/build_phase70_osm_road_ranker_overlay.py \
      --base-predictions "$V3_OSM" \
      --trigger-epochs "$G_OSM/${key}_phase70_osm_road_materialize_epochs.csv" \
      --out-csv "$V3_OSM" \
      --run-id "$key" \
      --label xd_gici_osmroad_hs \
      --p-pass 999.0
  done
  PYTHONPATH=python python3 experiments/build_phase70_osm_road_ranker_overlay.py \
    --base-predictions "$V5_OSM" \
    --trigger-epochs "$G_OSM/nagoya_run2_phase70_osm_road_materialize_epochs.csv" \
    --out-csv "$V5_OSM" \
    --run-id nagoya_run2 \
    --label xd_gici_osmroad_hs \
    --p-pass 999.0
}

run_phase70_overlay() {
  local city=$1
  local run=$2
  local key="${city}_${run}"
  echo "RUN phase70 overlay ${city}/${run}"
  PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py --runs "${city}/${run}" \
    --methods "rbpf+dd+gate+hybrid+rtkdiag_pf" \
    --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 --hybrid-sigma-m 1.0 \
    --rtkdiag-candidate-pos-dirs "$(phase70_dirs "$city" "$run")" \
    --rtkdiag-candidate-diag-dirs "$(phase70_dirs "$city" "$run")" \
    --rtkdiag-candidate-labels "$(phase70_labels "$city" "$run")" \
    --rtkdiag-candidate-block-labels-by-run "" \
    --rtkdiag-candidate-run-index-policy phase11ep \
    --rtkdiag-candidate-select-mode "${CONFIG[${key}_mode]}" \
    --rtkdiag-candidate-ranker-score-path "${CONFIG[${key}_osm_csv]}" \
    --rtkdiag-candidate-emit-mode candidate \
    --rtkdiag-candidate-residual-rms-max 50.0 --rtkdiag-candidate-ratio-min 1.0 \
    --rtkdiag-candidate-recenter-max-shift-m 10000.0 --rtkdiag-candidate-emit-max-diff-m 0.4 \
    --rtkdiag-candidate-max-to-hybrid-m 0 \
    --rtkdiag-candidate-fallback-mode hybrid \
    --rtkdiag-candidate-bridge-enable --rtkdiag-candidate-bridge-max-s 6.0 --rtkdiag-candidate-bridge-residual-rms-m 0.2 \
    --rtkdiag-candidate-rms-prefilter-k "${CONFIG[${key}_k]}" \
    --n-particles 2000 --pos-dir "/tmp/ppc_phase70_osmroad_neutral_${key}" \
    --results-prefix "ppc_phase70_osmroad_neutral_${key}_full"
}

for cr in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  city=${cr%/*}
  run=${cr#*/}
  run_phase43_internal "$city" "$run"
  materialize_osm_candidate "$city" "$run"
done

build_overlays

for cr in tokyo/run1 tokyo/run2 tokyo/run3 nagoya/run1 nagoya/run2 nagoya/run3; do
  city=${cr%/*}
  run=${cr#*/}
  run_phase70_overlay "$city" "$run"
done

PYTHONPATH=python python3 - <<'PY'
import csv
from pathlib import Path

runs = ["tokyo_run1", "tokyo_run2", "tokyo_run3", "nagoya_run1", "nagoya_run2", "nagoya_run3"]
baseline = {}
phase70 = {}
for key in runs:
    p = Path(f"experiments/results/ppc_ctrbpf_fgo_phase43_prod_{key}_full_runs.csv")
    if p.is_file():
        with p.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        baseline[key] = row
    p = Path(f"experiments/results/ppc_phase70_osmroad_neutral_{key}_full_runs.csv")
    if p.is_file():
        with p.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        phase70[key] = row

out = []
for key in runs:
    b = baseline.get(key)
    n = phase70.get(key)
    if not b or not n:
        continue
    out.append({
        "run": key,
        "phase43_ppc_pct": b["honest_ppc_pct"],
        "phase70_ppc_pct": n["honest_ppc_pct"],
        "delta_pp": float(n["honest_ppc_pct"]) - float(b["honest_ppc_pct"]),
        "phase43_pass_m": b["honest_pass_m"],
        "phase70_pass_m": n["honest_pass_m"],
        "delta_pass_m": float(n["honest_pass_m"]) - float(b["honest_pass_m"]),
    })
path = Path("experiments/results/phase70_osmroad_neutral_check_summary.csv")
with path.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(out[0])) if out else None
    if writer:
        writer.writeheader()
        writer.writerows(out)
print(f"saved: {path}")
if out:
    avg43 = sum(float(r["phase43_ppc_pct"]) for r in out) / len(out)
    avg70 = sum(float(r["phase70_ppc_pct"]) for r in out) / len(out)
    print(f"phase43_avg={avg43:.6f} phase70_avg={avg70:.6f} delta={avg70-avg43:.6f} n={len(out)}")
PY
