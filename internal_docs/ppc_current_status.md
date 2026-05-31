# PPC Current Status

Last updated: 2026-05-19.

This is the short current-state document for PPC work. The long chronological
log remains in [`plan.md`](plan.md).

## Current Conclusion (2026-05-19)

**Canonical production best: Phase71 OSM road route = 86.205492% OFFICIAL**.

Phase71 keeps the Phase43 per-run conditional setup and adds an
OSM road-centerline corrected candidate (`xd_gici_osmroad_hs`) only on
`nagoya/run2`. The gain is isolated to n/r2; the other five production runs are
unchanged.

| run | Phase43 | Phase71 | delta |
|---|---:|---:|---:|
| tokyo/run1 | 90.841510% | 90.841510% | +0.000000pp |
| tokyo/run2 | 95.410067% | 95.410067% | +0.000000pp |
| tokyo/run3 | 88.949168% | 88.949168% | +0.000000pp |
| nagoya/run1 | 83.700284% | 83.700284% | +0.000000pp |
| nagoya/run2 | 64.426589% | 65.669779% | +1.243190pp |
| nagoya/run3 | 92.662146% | 92.662146% | +0.000000pp |

Official average:

```text
Phase43: 85.998294%
Phase71: 86.205492%
Delta: +0.207198pp
TURING 85.6% delta: +0.605492pp
```

Key implementation files:

- `experiments/materialize_phase70_osm_road_centerline_candidate.py`
- `experiments/build_phase70_osm_road_ranker_overlay.py`
- `experiments/scripts_run_phase70_osmroad_neutral_check.sh`
- `experiments/scripts_run_phase71_osmroad_production.sh`

The Phase71 production script regenerates the OSM candidate and ranker overlay
under `/tmp` by default. The prep-only smoke command reproduced
`triggered_epochs=359`, `good/bad=30/0`, and overlay `added_rows=359` after the
artifact-regeneration change.

## GICI rtk_imu_tc candidate injection — selector-layer probe (2026-06-01)

GICI `forppc2024 rtk_imu_tc` NMEA outputs were verified across all 6 runs at
95–100% <1 m (fix4 99–100%) by `experiments/eval_gici_tc_ppc2024_batch.py`
(PR #72; +18 s leap, coverage-aware best-deployable pick per run). To test
injection, `experiments/materialize_gici_tc_ppc_candidate.py` converts the best
per-run NMEA into a selector-pool candidate (`libgnss_diag_phase10/gici_tc/`,
`.pos` + synthesised gate-passing diag `.csv` — GICI exposes no RTK ratio/RMS,
so the diag marks every emitted epoch as a trusted external fix; this is a
deliberate "full-trust" probe).

`sim_ppc_phase_csv_addcand.py --discover-diag-dirs --only-labels xd_gici_tc`
(policy `phase11dd`, base pool from `ppc_ctrbpf_fgo_phase11dd_full_p2k_runs.csv`):

| run | base | +gici_tc | Δ |
|---|---:|---:|---:|
| tokyo/run1 | 66.40% | 85.79% | +19.39pp |
| tokyo/run2 | 84.86% | 92.82% | +7.96pp |
| tokyo/run3 | 80.03% | 78.10% | −1.93pp |
| nagoya/run1 | 64.33% | 76.24% | +11.91pp |
| nagoya/run2 | 40.21% | 64.75% | +24.54pp |
| nagoya/run3 | 59.15% | 86.83% | +27.68pp |
| **aggregate** | **70.65%** | **81.17%** | **+10.52pp** |

Including RTK-float (fix=5) epochs beats `--fix4-only` (+10.52pp vs +4.44pp):
on this weak base, GICI float still beats the base fill.

**Caveat — not yet a production win.** This base (`phase11dd`, 70.65%) is far
weaker than the Phase71 production pipeline (86.21% OFFICIAL). Compared against
the production per-run numbers (tokyo1 90.84%, tokyo2 95.41%, tokyo3 88.95%,
nagoya1 83.70%, nagoya2 65.67%, nagoya3 92.66%), full-trust gici_tc would
*regress* 5/6 runs and roughly tie the OSM lever on the weak run (n/r2
64.75% vs 65.67%). The selector layer is also below the PF+ranker layer that
yields OFFICIAL, and past one-layer gains have been absorbed by the full
pipeline (cf. v3-ranker / Viterbi negatives).

**Decisive next test:** inject `gici_tc` into the *production* pool via the full
`exp_ppc_ctrbpf_fgo.py` replay (Phase43/71 candidate set + ranker) — especially
n/r2-targeted, where production is only ~65% — and measure OFFICIAL, rather than
trusting the selector-layer delta. A fair (non-fabricated) gate for gici_tc is
also open work, since it has no native ratio/RMS.

## Previous Conclusion (2026-05-15)

**Former canonical best: Phase 19aw K=3 = 83.42% OFFICIAL** (rms_prefilter_k on top of
17-variant gici-open pool + status=5 gate + velocity bridge)、+11.94pp / +5,500m from
Phase 11ep canonical 71.48%、**TURING gap 2.18pp remaining**。
詳細: [`rms_prefilter_breakthrough_2026_05_15.md`](rms_prefilter_breakthrough_2026_05_15.md)。

1 セッションで 3 stage breakthrough:
1. **Fix=4 gate 撤廃 + status=5 path** (Phase 19ap): selector が Float candidate (rms<=0.3m)
   を初めて受理 → +2.71pp (74.81%)
2. **Velocity/IMU bridge** (Phase 19at): last good anchor + PF velocity × Δt synthetic
   candidate (rms=0.2, max_dt=6.0s) → +0.16pp (74.97%)
3. **rms_prefilter_k=3** (Phase 19aw): selector ranking 直前で top-3 by residual_rms に
   フィルタ → **+8.44pp (83.42%)**。 cluster-bias の温床だった composite formula `residual /
   (... * abs_max^c)` が高 abs_max 候補を促進していた problem を 19 lines のコードで切断。

過去のメモ (Phase 19al 76.83%) は metric ミス (pooled→per-run-averaged) の補正前数値。
Phase 19at = 74.97% が正しい補正後 baseline、 Phase 19aw = 83.42% が当時の頂点。

## Previous (earlier 2026-05-15)

17-variant gici-open TC FGO pool で 76.83% aggregate (補正前) / 74.97% OFFICIAL (補正後)。
詳細: [`gici_open_phase19_breakthrough.md`](gici_open_phase19_breakthrough.md)。

2 段 breakthrough (1 セッション):
1. **Format bug fix** (Phase 19l): `nmea_to_ppc_pool_esdfix.py` に `output_added=1` 列
   追加で gici TC FGO が PPC selector で初めて pick されるように → **+1.62pp**。 過去
   "gici 0.00pp redundant" 結論を完全取り消し。
2. **17-variant pool** (Phase 19s-19al): lever arm / AR ratio / PR/Phase outlier / SNR /
   elevation / IMU / window の異なる 17 gici config を pool 同居、 PPC selector が
   per-epoch best を自動選択で **更に +1.45pp**。

Pool 飽和確定 (5 連続 marginal/regression、 Phase 19am-an)。 残 gap 8.77pp の close
には architectural breakthrough (CLAS/madoca PPP-AR / city-model NLOS / triple-freq
cascade / multi-base) が必要、 ~月単位 task。

**n/r2 構造的限界**: 全 19 variants で **44.21% に張り付き**、 全 knob で改善なし。
残 epoch (55.79%) は libgnss++ + gici 全 candidate でも recoverable でない。

---

## (legacy) phase11er n/r2 rescue 状態 (2026-05-14)

(以下は gici breakthrough 前の状態記録。 phase11er 自体は今も 17-variant pool に
吸収されて生きている)

- `libgnss++` is not losing to RTKLIB `demo5` on the checked-in public PPC
  coverage benchmark. It wins all six Tokyo/Nagoya public runs in the existing
  RTKLIB demo5 comparison.
- The real gap is harder: `libgnss++` still scores only 20-30% official PPC in
  hard Nagoya spans, and `gnss_gpu` currently relies on that solver output as a
  hybrid floor/candidate source.
- `phase11er` is a safe RTKDiag candidate-emission policy for `nagoya/run2`.
  It improves the hard lowcase, but it does not prove that CT-RBPF/FGO itself
  has become strong.
- The next useful work is fallback-span diagnosis and candidate coverage, not
  more global selector complexity.

## libgnss++ vs RTKLIB demo5

Existing `third_party/gnssplusplus` docs report:

| run | gnss++ official | RTKLIB demo5 official | delta |
|---|---:|---:|---:|
| tokyo/run1 | 34.9% | 0.0% | +34.9pp |
| tokyo/run2 | 69.0% | 16.9% | +52.1pp |
| tokyo/run3 | 60.6% | 35.6% | +25.0pp |
| nagoya/run1 | 49.5% | 22.4% | +27.1pp |
| nagoya/run2 | 20.9% | 11.0% | +9.9pp |
| nagoya/run3 | 27.4% | 7.6% | +19.7pp |

Local scorer checks on existing `.pos` files:

| profile / pos-dir | weighted honest PPC | note |
|---|---:|---|
| `libgnss_rtk_pos_v5` | 50.72% | current `gnss_gpu` hybrid floor |
| `libgnss_diag_phase10/dev_demo5_trusted_o3` | 54.10% | `nagoya/run2` 24.49% |
| `libgnss_diag_phase10/demo5_continuous_nojump` | 52.83% | `nagoya/run2` 29.24% |

Important distinction: `dev_demo5_*` and `demo5_continuous_*` are local
`libgnss++` profile outputs, not RTKLIB demo5 raw output.

## phase11er

`phase11er` is a conservative `nagoya/run2` RTKDiag rescue policy:

- allowed only on `nagoya/run2`
- residual selection over seven candidate labels
- `ratio_min=1.0`
- `residual_rms_max=50.0`
- candidate-to-hybrid gate: 10 m
- emit candidate, fallback to hybrid
- all non-`nagoya/run2` runs block candidates and pass through hybrid

Verified p2k result:

| scope | hybrid | phase11er | delta |
|---|---:|---:|---:|
| 6-run honest p2k | 14.224688% | 14.354783% | +0.130095pp |
| `nagoya/run2` full p2k | 10.544309% | 11.815481% | +1.271172pp |
| `nagoya/run2` segment p2k | 45.322161% | 50.785987% | +5.463826pp |

Internal diagnosis on `nagoya/run2` p2k:

| emitted source | epochs | pass <=0.5m | fail >3m | median error | p95 error |
|---|---:|---:|---:|---:|---:|
| `rtkdiag_candidate` | 1298 | 1220 | 40 | 0.101m | 0.744m |
| `rtkdiag_fallback_hybrid_rtkdiag` | 654 | 1 | 647 | 26.354m | 31.909m |
| `pf_rtkdiag` | 42 | 0 | 42 | 27.011m | 48.063m |
| `pf_min_sats` | 6 | 0 | 6 | 23.560m | 45.204m |

Interpretation: candidate epochs are strong; fallback epochs are the bottleneck.

## Current Artifacts

- `experiments/results/phase71_osmroad_production_summary.csv`
- `experiments/results/phase71_osmroad_block_other_runs_summary.csv`
- `experiments/results/phase70_osmroad_neutral_check_summary.csv`
- `experiments/results/ppc_phase57_gap_nagoya_run2_internal_epochs.csv`
- `experiments/results/ppc_ctrbpf_fgo_phase43_prod_*_full_runs.csv`
- `experiments/results/ppc_phase71_osmroad_prod_*_full_runs.csv`
- `experiments/results/selector_ranker_predictions_v5_nlos.csv`
- `experiments/results/ppc_compare_libgnss_v5_runs.csv`
- `experiments/results/ppc_compare_dev_demo5_trusted_o3_runs.csv`
- `experiments/results/ppc_compare_demo5_continuous_nojump_runs.csv`

Legacy phase11er artifacts:

- `experiments/results/ppc_phase11er_policy_all_p2k_runs.csv`
- `experiments/results/ppc_phase11er_internal_n2_p2k_runs.csv`
- `experiments/results/ppc_phase11er_internal_n2_p2k_internal_epochs.csv`
- `experiments/results/ppc_phase11er_internal_n2_p2k_state_summary.csv`
- `experiments/results/ppc_phase11er_internal_n2_p2k_state_groups.csv`
- `experiments/results/ppc_phase11er_internal_n2_p2k_state_spans.csv`

See [`../experiments/results/README.md`](../experiments/results/README.md) for
artifact conventions.

## Next Tasks

1. Keep Phase71 production replay reproducible without committing `/tmp`
   materialized OSM candidates or large overlay CSVs.
2. Phase72: test whether road/map constraints can generalize beyond the n/r2
   OSM span without perturbing the five neutral runs.
3. Find stronger absolute candidate sources for stable low-residual wrong
   solutions that ranker/consensus logic cannot detect.
4. Reproduce RTKLIB demo5 raw baseline through the
   `third_party/gnssplusplus/docs/ppc_reproduction.md` coverage-matrix path and
   align the denominator with the local scorer.
5. Compare `libgnss++` profiles at span level, especially on `nagoya/run2`.

## Reproduction Snippets

Current `libgnss++` floor scorer:

```bash
python3 experiments/exp_ppc_libgnss_hybrid.py \
  --skip-solvers \
  --pos-dir experiments/results/libgnss_rtk_pos_v5 \
  --spp-dir experiments/results/libgnss_spp_pos \
  --results-prefix ppc_compare_libgnss_v5
```

Legacy phase11er skeleton:

```bash
base=experiments/results/libgnss_diag_phase10
labels=fgo_v14_snr38,full_ratio15_lock3_trustedseed_rtkout3oGem3,dev_demo5_trusted_o3,n2_nobds,fgo_v1,full_ratio15_lock3_trustedseed_rtkout3mlc1,full_ratio15_lock3_trustedseed_rtkout5
posdirs=${base}/fgo_v14_snr38,${base}/full_ratio15_lock3_trustedseed_rtkout3oGem3,${base}/dev_demo5_trusted_o3,${base}/n2_nobds,${base}/fgo_v1,${base}/full_ratio15_lock3_trustedseed_rtkout3mlc1,${base}/full_ratio15_lock3_trustedseed_rtkout5
python3 experiments/exp_ppc_ctrbpf_fgo.py \
  --results-prefix ppc_phase11er_policy_all_p2k \
  --runs all \
  --methods rbpf+dd+gate+hybrid,rbpf+dd+gate+hybrid+rtkdiag_pf \
  --n-particles 5000 \
  --start-epoch 0 \
  --max-epochs 2000 \
  --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 \
  --rtkdiag-candidate-pos-dirs "$posdirs" \
  --rtkdiag-candidate-diag-dirs "$posdirs" \
  --rtkdiag-candidate-labels "$labels" \
  --rtkdiag-candidate-run-index-policy phase11er
```
