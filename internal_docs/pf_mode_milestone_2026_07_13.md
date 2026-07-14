# PF weighted-mode milestone (2026-07-13)

## Implementation

- Added `python/gnss_gpu/particle_modes.py`.
- The extractor uses deterministic weighted systematic reduction, weighted
  voxel density, dense-cell connected components, bounded tail assignment,
  per-mode mass/mean/covariance, and explicit noise mass.
- The selector combines posterior mass with a constant-velocity reachability
  prior and abstains on weak mass, weak score ratio, excessive prediction
  distance, single-mode posteriors, or an immaterial global-mean displacement.
- Added PPC policies `off`, `diagnostic`, and `emit`. Diagnostic mode writes
  per-epoch mode fields automatically. Emit mode may replace only an output
  whose source starts with `pf`; hybrid/candidate fallbacks are protected.
- Safe emission defaults require epoch >= 10, at least two modes, and at least
  0.5 m between the selected mode and global weighted mean. The overall policy
  remains `off` by default.

## Verification

- 21 focused and adjacent tests passed.
- Synthetic tests cover separated unequal modes, equal-mode abstention,
  temporal selection of a reachable secondary mode, sparse bridges,
  deterministic 50k-particle reduction, single-mode abstention, diagnostic
  invariance, PF-source-only replacement, and diagnostic fields.
- A 20-epoch Tokyo run1 PPC smoke showed diagnostic and off `.pos` data rows
  were byte-identical (`max_abs_delta = 0`).
- The installed PF CUDA extension was stale. The already-built 2026-07-12
  extension was verified to export `pf_device_weight_dd_joint` and installed
  over the stale generated binary before real-data replay.

## Six-run fixed-setting diagnostic

Command shape:

```text
--runs all --max-epochs 200 --methods pf --n-particles 5000
--pf-mode-policy diagnostic --pf-mode-voxel-size-m 2
--pf-mode-assignment-radius-m 6
```

This is a candidate-emission ablation, not an official score: only the first
200 epochs of each run are emitted, so the honest full-run denominator makes
the printed official score zero. Reference positions were used only after the
run to compare the counterfactual selected mode with the emitted weighted
mean; they are not selector inputs.

With the initial permissive selector, 1182/1200 epochs were accepted. The
pooled mode-minus-mean 3D error was -0.015 m and only 47.9% of epochs improved.
Nagoya run1 worsened by +0.149 m on average. This rejected unconditional mode
emission and motivated the stricter non-reference gate above.

## Six-run emit smoke

The strict fixed gate was replayed on the first 100 epochs of every run. It
changed 13/600 epochs, with maximum consecutive changes of two epochs.

| Run | Applied | Mean mode-minus-mean error | Improved |
|---|---:|---:|---:|
| Tokyo run1 | 1 | -0.289 m | 100% |
| Tokyo run2 | 2 | -0.263 m | 100% |
| Tokyo run3 | 4 | -0.258 m | 75% |
| Nagoya run1 | 4 | +0.191 m | 25% |
| Nagoya run2 | 2 | -0.479 m | 100% |
| Nagoya run3 | 0 | n/a | n/a |
| Pooled | 13 | -0.157 m | 69.2% |

The emission did not cascade, and pooled error improved, but the fixed gate
still harmed Nagoya run1 in three of four applications. There is no honest
non-reference discriminator in the current fields that safely separates those
events. Therefore:

- keep `pf_mode_policy=off` in production;
- retain `diagnostic` for posterior analysis;
- do not claim a Phase71 improvement from this short PF-only ablation;
- use mode identities as constraints for the next fixed-lag FFBSi milestone,
  where temporal trajectory evidence is stronger than one-epoch emission.

Artifacts:

- `experiments/results/pf_mode_all_200_v2_internal_epochs.csv`
- `experiments/results/pf_mode_emit_all_100_internal_epochs.csv`
- `results/pf_mode_emit_all_100_pos/`

## Full six-run 2,000-particle diagnostic

The fixed strict gate was subsequently replayed on every usable epoch of all
six official runs. Decisions below use only epoch 200 onward, leaving the
earlier prefixes as development data. The holdout contains 57,398 epochs with
100% reference coverage. Mode extraction evaluated 57,371 epochs, found two
or more modes on 97.41%, accepted 30,341 counterfactual emissions, and
abstained on 47.11%.

The counterfactual selected mode improved only 48.34% of accepted epochs. Its
`mode_error - emitted_weighted_mean_error` averaged **+0.044 m**, and the p95
was **+1.330 m**. Thus the small mean degradation is not compensated by a
majority win rate, while individual downside remains metre-scale. The full
holdout contradicts the favorable 13-emission prefix smoke. Production
emission is rejected; `off` remains the default and `diagnostic` remains useful
for posterior structure analysis.

Full-run artifacts:

- `experiments/results/pf_mode_full6_p2k_diagnostic_runs.csv`
- `experiments/results/pf_mode_full6_p2k_diagnostic_internal_epochs.csv`
- `experiments/results/pf_mode_full6_p2k_diagnostic_ablation_summary.csv`
- `results/pf_mode_full6_p2k_pos/`
