# Fable5 advice request — NLOS Wave 1 closed, Wave 2 ranker pivot

Date: 2026-07-04  
Status: **answered — Wave 2 6-pool CLOSED (gate failed)**  
Related: PR #117 merged; PR #118 draft

## Goal

Decide whether to invest in **ranker/rtkdiag + PLATEAU NLOS features** after Wave 1 PF/DD soft-weight closed as a PPC improvement path (6/6 runs Δ=0).

## Wave 1 (merged, PR #117)

**Strategy (Fable-approved):** PLATEAU BVH → per-epoch mask CSV → soft down-weight at PF/DD (not SPP, not hard gate).

**Production A/B:** `rbpf+dd+gate+hybrid`, soft-k3 vs baseline, `start_epoch=1000`, `max_epochs=1200`, all 6 runs.

| Run | baseline honest | segment | hybrid_applied | Δ honest |
|-----|----------------:|--------:|---------------:|---------:|
| tokyo/run1 | 5.77% | 36.73% | 1131/1200 | **0** |
| tokyo/run2 | 0.60% | 4.67% | 1018/1200 | **0** |
| tokyo/run3 | 2.13% | 33.63% | 829/1200 | **0** |
| nagoya/run1 | 10.79% | 37.46% | 689/1200 | **0** |
| nagoya/run2 | 4.35% | 25.40% | 654/1191 | **0** |
| nagoya/run3 | 0.05% | 0.39% | 438/1200 | **0** |

**Root cause:** hybrid PU dominates emitted trajectory (55–94% of epochs); PF weight changes do not reach output. Hybrid-missing windows have ~80–100 m PF error — wrong scale for 1.5 m PPC threshold.

**Verdict:** CLOSE Wave 1 as PPC path. Keep mask pipeline on SSD for downstream features.

## Wave 2 exploratory (local, not merged)

**Hypothesis:** NLOS helps at **ranker/rtkdiag** layer (Phase 33 historically +1.07 pp on nagoya/run2 with v5_nlos + full 50+ candidate pool).

**Bootstrap constraint:** Phase 10/19 50+ candidate dirs not on SSD. Built minimal pool instead:

- 6 libgnss variants/run (`w2_def`, `w2_hold5/7/10`, `w2_ratio20/30`)
- `bootstrap_rtkdiag_candidate_pool.py` → manifests in `experiments/results/rtkdiag_manifest/`
- v3 features (230k rows) → v5_nlos augment → LightGBM LORO wrong-fix **41.68%**
- Smoke: nagoya/run2, same window as Wave 1

| Config | honest PPC | segment | rtkdiag PU |
|--------|-----------|---------|------------|
| Wave 1 hybrid-only | **4.35%** | **25.40%** | 0 |
| Wave 2 ranker+rtkdiag (6-pool) | **2.31%** | **13.47%** | **1191/1191** |

**Mechanism (smoke log):** rtkdiag fully engaged; selection `pf_bridge+rnk:1050`, `w2_def+rnk:141`. Ranker path works mechanically but **regresses vs hybrid-only** with 6-candidate pool.

NLOS feature importances (gain): `nlos_min_elev_deg`, `nlos_n_sats`, `nlos_frac`, `nlos_mean_elev_deg` — signal present in wrong-fix model, not yet translated to PPC gain.

## Questions for Fable5

1. **Wave 1 closure:** Confirm PR #117 merge was correct (tooling + negative result, no PPC gain expected)?

2. **PF-layer retention:** Keep PF/DD NLOS wiring in mainline behind `--pf-nlos-preset`, or demote to experiments-only?

3. **Wave 2 pivot approval:** Given 6-pool regression, which path?

   | Option | Description | Cost |
   |--------|-------------|------|
   | **A** | Build full 50+ candidate pool on SSD, re-run Phase 33 recipe with refreshed PLATEAU masks | ~days RTK + storage |
   | **B** | Train/eval ranker features only; do not change emit until pool is complete | Low |
   | **C** | Abandon ranker+NLOS combo; try hybrid-light / mask-gated hybrid fallback on NLOS-heavy epochs | Medium |
   | **D** | Other (specify) | — |

4. **Validation target:** Is historical Phase 33 nagoya/run2 **+1.07 pp** still the right success criterion for refreshed masks?

5. **Minimum smoke before pool investment:** Single run (n/r2)? Full 6-run? Different epoch window?

6. **Do-not confirmation:** Still avoid SPP NLOS, hard gate default, PF k-sweep on hybrid-dominated windows without new hypothesis?

## Constraints (unchanged)

- Do not commit `experiments/results/`
- Block on EGM96 grids optional (city constant geoid OK for Tokyo/Nagoya)
- No ML NLOS classifier at PF layer

## Local artifacts (SSD, gitignored)

- Masks: `experiments/results/plateau_nlos_phase33/{city}_{run}_per_epoch_nlos.csv`
- Wave 2 pool: `experiments/results/libgnss_rtk_wave2/{label}/`
- Manifests: `experiments/results/rtkdiag_manifest/{city}_{run}.json`
- Ranker: `selector_training_features_v5_nlos.csv`, `selector_ranker_predictions_v5_nlos.csv`
- Smoke: `ppc_pf_nlos_wave2_nagoya_run2_runs.csv`

## Commands (Wave 2 pipeline)

```bash
python experiments/prepare_pf_nlos_production.py wave2-bootstrap --runs all
python experiments/prepare_pf_nlos_production.py wave2-features --runs all
python experiments/prepare_pf_nlos_production.py wave2-train
python experiments/prepare_pf_nlos_production.py wave2-smoke --run nagoya/run2 --start-epoch 1000 --max-epochs 1200
```
