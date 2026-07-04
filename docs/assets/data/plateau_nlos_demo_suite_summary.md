# PLATEAU NLOS Demo Suite

Mask: 70 epochs x 14 satellites, NLOS 641/980 (65.4%), ray source: native BVH.

| Estimator | Baseline RMS (m) | Mask-soft RMS (m) | RMS gain | Wins |
|---|---:|---:|---:|---:|
| SPP | 11.85 | 4.07 | 65.6% | 48/68 |
| PF | 11.18 | 1.40 | 87.4% | 70/70 |
| FGO | 8.10 | 0.38 | 95.4% | 70/70 |

Best mask-soft RMS: FGO at 0.38 m.

## Measurement-level wiring (NLOS Wave 2-3)

Geometry mask CSVs (`tow,epoch_idx,prn,is_los`) now feed soft weights at the
PF update layer instead of SPP-only hooks:

| Path | Entry | Flags / module |
|---|---|---|
| UrbanNav PF smoother | `epoch_observation_inputs.py` | `--nlos-mask-csv`, `--nlos-k-weak` |
| PPC PF | `exp_ppc_ctrbpf_fgo.py` | `--pf-nlos-mask-path`, `--pf-nlos-k-weak` |
| DD carrier / DD-PR | `nlos_mask.scale_dd_result_weights_by_nlos_mask` | same mask tables, min(k, ref) per pair |

CPU replay evaluators: `experiments/exp_ppc_pf_nlos_eval.py`,
`experiments/exp_nlos_soft_weight_sweep.py`, `experiments/exp_gmm_nlos_eval.py`.
