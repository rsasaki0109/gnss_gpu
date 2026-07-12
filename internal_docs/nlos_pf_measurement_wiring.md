# NLOS geometry mask at PF measurement layer (Waves 1-4)

Status: **Wave 1 complete — negative on PPC** (2026-07-04). CPU wiring and 6-run production A/B measured; **PF/DD soft-k3 does not move official PPC on any run**.

## Strategy (Fable)

- PLATEAU BVH → per-epoch mask CSV → **soft down-weight at PF/DD update** (not SPP, not hard gate).
- SPP-domain and ranker-layer NLOS hooks were **negative or absorbed** in prior PPC phases.
- Do-not: hard gate default, ML classifier, commit `experiments/results/`.

## Waves

| Wave | Deliverable | Status |
|---|---|---|
| 1 | `gnss_gpu.nlos_mask`, soft-weight sweep, GMM eval | Done |
| 2 | PF smoother + PPC PF undiff weights | Done |
| 3 | PPC CPU eval, DD pair weights, demo doc | Done |
| 4 | Presets + this doc | Done |
| 5 | 6-run SSD prep + production smoke | Done (Δ=0 all runs) |

## Production A/B (6-run, 2026-07-04)

Method: `rbpf+dd+gate+hybrid`, soft-k3 vs baseline, `start_epoch=1000`, `max_epochs=1200`.

| Run | baseline honest | segment | hybrid_applied | Δ honest |
|-----|----------------:|--------:|---------------:|---------:|
| tokyo/run1 | 5.77% | 36.73% | 1131/1200 | **0** |
| tokyo/run2 | 0.60% | 4.67% | 1018/1200 | **0** |
| tokyo/run3 | 2.13% | 33.63% | 829/1200 | **0** |
| nagoya/run1 | 10.79% | 37.46% | 689/1200 | **0** |
| nagoya/run2 | 4.35% | 25.40% | 654/1191 | **0** |
| nagoya/run3 | 0.05% | 0.39% | 438/1200 | **0** |

Additional tokyo/run1 checks: k=3/5/10/20 sweep → Δ=0; hybrid-gap window (epoch 7370, 82% hybrid-missing) → Δ=0 (per-epoch mean +0.6 m worse).

**Interpretation:** mask applies on all epochs in mask-soft runs (wiring OK). Hybrid PU dominates emitted trajectory on most epochs; on hybrid-missing windows PF error is ~80–100 m so soft down-weight at k=3 is the wrong scale for the 1.5 m PPC threshold.

Aggregated JSON (local): `experiments/results/ppc_pf_nlos_batch_smoke_summary.json`

## Presets

### PPC (`exp_ppc_ctrbpf_fgo.py`)

```bash
PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py \
  --runs tokyo/run1 \
  --methods rbpf+dd \
  --pf-nlos-preset soft-k3
```

Sets:

- `--pf-nlos-mask-path experiments/results/plateau_nlos_phase33/{city}_{run}_per_epoch_nlos.csv`
- `--pf-nlos-k-weak 3 --pf-nlos-k-strong 3`

Mask CSVs are produced by `experiments/build_per_epoch_nlos_csv.py` (not committed).

**Note:** When undiff rover weights are already NLOS-scaled (`epoch_observation_inputs`),
smoother DD-PR skips a second `scale_dd_result_weights_by_nlos_mask` pass to avoid
double-counting (`sqrt(w_k*w_ref)` already embeds the mask).

### PF smoother (`exp_pf_smoother_eval.py`)

```bash
PYTHONPATH=python python experiments/exp_pf_smoother_eval.py \
  --preset odaiba_pf_nlos_soft \
  --nlos-mask-csv path/to/mask.csv
```

`odaiba_pf_nlos_soft` extends `odaiba_reference` with `k_weak=k_strong=3`. Without `--nlos-mask-csv`, weights stay unchanged.

## CPU replay results (PLATEAU demo, deterministic)

| Evaluator | Naive RMS (m) | Mask-soft RMS (m) | Notes |
|---|---:|---:|---|
| `replay_plateau_nlos_demo_pf.py` | 11.18 | 1.40 | Suite summary (BVH demo) |
| `exp_ppc_pf_nlos_eval.py` | 10.35 | 3.00 | `gnss_gpu.nlos_mask` production path |
| `exp_gmm_nlos_eval.py` | (naive) | mask-soft < naive | GMM optional on top |
| `exp_nlos_soft_weight_sweep.py` | — | best grid row | 2B/3B residual+accel |

These are **geometry replay** numbers, not PPC2024 official scores.

## Module map

| Layer | Module / flag |
|---|---|
| Mask load | `gnss_gpu.nlos_mask` |
| Presets | `gnss_gpu.nlos_presets` |
| UrbanNav PF | `--nlos-mask-csv` via `epoch_observation_inputs.py` |
| PPC PF undiff | `--pf-nlos-mask-path`, `--pf-nlos-preset soft-k3` |
| DD carrier/PR | `scale_dd_result_weights_by_nlos_mask` (PPC + smoother DD carrier; smoother DD-PR uses rover-weight path) |

## Next (post–Wave 1)

Wave 1 PF/DD soft-weight is **closed as a PPC improvement path**. Do not extend k-sweep or 6-run PF A/B without a new hypothesis.

Candidate Wave 2 (needs explicit approval — strategy pivot):

1. **Ranker/rtkdiag layer:** PLATEAU NLOS features already exist in `train_selector_ranker_v5_nlos.py` (`nlos_frac`, `nlos_count`, …). Production scripts use `selector_ranker_predictions_v5_nlos.csv` with `--rtkdiag-candidate-select-mode ranker` (see `scripts_run_phase33_perrun_production.sh`). Re-run nagoya/run2 with v5_nlos ranker + current BVH masks.
2. **Do-not revive:** SPP NLOS, hard gate default, more PF k-sweep on hybrid-dominated windows.

## Fable review (2026-07-03, initial)

**Verdict: PASS with warnings** (strategy aligned; do-nots clean).

Post-review hardening on this branch:

- Mask lookup prefers ``tow`` over ``epoch_idx`` (skipped-epoch drift guard).
- PPC DD-PR anchor now receives the same NLOS scale as DD carrier.
- Strong-only NLOS PRNs are down-weighted even when absent from the weak set.

## Fable-style post-mortem (2026-07-04, after 6-run A/B)

**Verdict: CLOSE Wave 1; do not merge expecting PPC gain from PF soft-weight alone.**

| Finding | Action |
|---|---|
| 6/6 runs Δ honest = 0 | Accept negative result; PR #117 documents tooling + evidence |
| Hybrid applies 55–94% of epochs | PF weight changes do not reach emitted trajectory |
| Hybrid-missing windows: ~80–100 m PF error | Soft-k3 is wrong scale; down-weight can harm mean error |
| Mask + BVH pipeline works | Keep assets on SSD; reuse for ranker features |
| rtkdiag_pf_pu=0 in smoke (single libgnss pool) | Not a Wave 1 blocker; production uses 50+ candidate pool + ranker CSV |

Recommended next experiment (if approved): **nagoya/run2** with `selector_ranker_predictions_v5_nlos.csv` + full candidate pool — historically +1.07 pp in Phase 33; validate whether refreshed PLATEAU masks change ranker inputs materially.

## Wave 2 exploratory (2026-07-04, local — pending Fable5)

Bootstrap with **6-candidate pool** (Phase 10/19 50+ dirs not on SSD). nagoya/run2 smoke, same window as Wave 1:

| Config | honest | segment | rtkdiag PU |
|--------|-------:|--------:|-----------:|
| hybrid-only (Wave 1) | 4.35% | 25.40% | 0 |
| ranker+rtkdiag (6-pool) | 2.31% | 13.47% | 1191/1191 |

rtkdiag engages (`pf_bridge+rnk:1050`, `w2_def+rnk:141`) but **regresses** vs hybrid-only. Do not treat 6-pool smoke as Phase 33 reproduction.

**Gate experiments (2026-07-04, Fable5 action items):**

| Check | Result |
|-------|--------|
| Config-parity re-smoke (+ emit guards) | honest still **2.31%** (no recovery to 4.35%) |
| Oracle ceiling (4853 tows) | **headroom 0.0 pp** — hybrid beats all w2 variants |

**Verdict: CLOSE Wave 2 6-pool bootstrap.** No oracle headroom; 50+ pool rebuild not justified for PPC gain from this avenue alone (Fable5 gate failed).

Advice: `internal_docs/fable5_nlos_wave2_advice_request_2026_07_04.md`, response `fable5_nlos_wave2_advice_response_2026_07_04.md`. Follow-up: PR #118.

~~Remaining before PPC score validation:~~

~~1. Install PPC data~~ ✓  
~~2. Smoke A/B~~ ✓ (6-run batch)  
~~3. Real BVH masks~~ ✓  
~~4. Record deltas~~ ✓ (all zero)

## Local smoke prep (no PPC dataset required)

```bash
PYTHONPATH=python:. python experiments/seed_pf_nlos_smoke_mask.py
```

Writes `experiments/results/plateau_nlos_phase33/tokyo_run1_per_epoch_nlos.csv` from the
PLATEAU demo mask (gitignored). Pair with `scripts_run_pf_nlos_smoke.sh` once PPC data exists.

## Production prep (PPC on mobile SSD + real BVH mask)

```bash
PYTHONPATH=python python experiments/prepare_pf_nlos_production.py check --run tokyo/run1
PYTHONPATH=python python experiments/prepare_pf_nlos_production.py batch-prep --runs all --with-diagnostics
PYTHONPATH=python python experiments/prepare_pf_nlos_production.py batch-smoke --runs all --profile signal --start-epoch 1000 --max-epochs 1200
PYTHONPATH=python python experiments/prepare_pf_nlos_production.py gaps --run tokyo/run1
```

Single-run equivalents: `fetch`, `mask`, `hybrid`, `smoke`, `gaps`.

SSD layout (when `E:` is present):

- `E:/datasets/PPC-Dataset-data`
- `E:/datasets/plateau/{city}_{run}`
- `E:/datasets/plateau_cache/{city}_{run}_triangles.npz`

Geoid note: if `pyproj` EGM96 grids are missing on Windows, use the city
constant from `prepare_pf_nlos_production.py` (Tokyo `36.7` m). Triangle
cache must be rebuilt when geoid correction changes.
