# NLOS geometry mask at PF measurement layer (Waves 1-4)

Status: **CPU wiring complete** (2026-07-03). PPC full-run score impact is **not yet measured** on the 6-run dataset.

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

## Next (outside Wave 4)

1. Generate `plateau_nlos_phase33` CSVs for target PPC runs (GPU/BVH, manual).
2. Smoke `scripts_run_pf_nlos_smoke.sh` on `tokyo/run1` with `rbpf+dd+gate+hybrid+rtkdiag_pf` if ranker pool is available.
3. Record PPC2024 delta vs baseline; expect heavy-NLOS runs (`n/r2`) to show the most PF-domain gain.

## Local smoke prep (no PPC dataset required)

```bash
PYTHONPATH=python:. python experiments/seed_pf_nlos_smoke_mask.py
```

Writes `experiments/results/plateau_nlos_phase33/tokyo_run1_per_epoch_nlos.csv` from the
PLATEAU demo mask (gitignored). Pair with `scripts_run_pf_nlos_smoke.sh` once PPC data exists.
