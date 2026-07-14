# Fixed-lag particle smoothing milestone (2026-07-13)

## Scope

This milestone implements a bounded-memory particle smoother for the PPC
CT-RBPF path.  It supports both ancestry tracing and a marginal FFBSi backward
kernel, optional conditioning on the selected terminal particle mode, multiple
backward paths, and delayed output replacement. Forward Doppler-updated
particle velocities parameterize the fixed-lag backward transition. The older
full backward-pass smoother now applies the complete time reversal explicitly:
receiver prediction velocity, satellite velocity, and observed Doppler all
change sign together, preserving `range_rate = -wavelength * Doppler` and the
clock-drift nuisance convention. A regression test checks both reversed arrays.

Production defaults remain unchanged: `--enable-pf-ffbsi-smoother` is opt-in.
Reservoir-Stein resampling is rejected when smoothing is enabled because it does
not supply a discrete ancestry map.

## Safety gates

Delayed replacement is limited to PF-sourced epochs.  It abstains for a rejected
terminal mode, fewer than two distinct oldest particles, excessive path spread,
non-finite output, or a correction larger than the configured limit.  Hybrid and
candidate outputs are never overwritten.

## Tests

The unit and integration suite covers terminal masks, global and per-particle
velocity transitions, marginal and genealogical paths, lag/flush behavior,
lineage degeneracy, and protected output sources.  The focused suite passed 35
tests.

## Six-run ablation

Each PPC run was evaluated independently for its first 50 usable epochs with
1,000 particles, lag 10, and 8 backward paths.  These short prefixes are an
implementation ablation, not an official PPC score.

| Run | accepted/evaluated | mean error delta (m) | fraction improved | mean distinct oldest particles |
|---|---:|---:|---:|---:|
| tokyo/run1 | 23 | +0.0503 | 43.5% | 1.78 |
| tokyo/run2 | 25 | +0.1005 | 44.0% | 1.72 |
| tokyo/run3 | 31 | +0.0397 | 35.5% | 2.00 |
| nagoya/run1 | 35 | -0.1874 | 57.1% | 2.02 |
| nagoya/run2 | 35 | +0.0570 | 54.3% | 2.26 |
| nagoya/run3 | 36 | -0.0295 | 50.0% | 2.32 |

Across 185 safely evaluable delayed estimates, the mean error delta was
-0.00325 m and 48.1% improved.  Mean correction magnitude was 0.586 m; mean and
maximum reported positional standard deviation were 1.168 m and 2.816 m.

The pure genealogy experiment collapsed to one oldest particle at every tested
epoch and produced essentially zero covariance.  Marginal FFBSi restored some
path diversity and meaningful covariance, but its accuracy effect was not
consistent across runs.  Therefore the implementation is retained for research
and diagnostics, while production emission stays disabled.

## Official full-six and blocked replay

The 2,000-particle, lag-10, eight-path marginal replay is complete for all six
official runs and every declared blocked scope. On 57,398 holdout epochs,
57,371 delayed outputs were evaluable and 34,943 passed the safety gates
(39.09% abstention). Of those applied counterfactuals, 50.66% improved and the
mean error delta was -0.0124 m, but the worsening-side delta p95 was +0.803 m.
The five blocked holdouts split three improving versus two worsening; Nagoya
run2/run3 worsened by +0.133/+0.215 m on average. Runtime rose from 47.79 to
58.48 ms/epoch and the correction-magnitude p95 was 1.756 m. The standalone PF
official score remained 0%, so this does not support production adoption.

During aggregation, the original diagnostic compared the post-replacement
`emit_to_ref_m` with the same FFBSi error and therefore reported a false zero
delta. The runner now preserves `pf_ffbsi_baseline_to_ref_m`; the summarizer
uses that field and has a compatibility fallback to the already-recorded
`pf_before_emit_to_ref_m`, allowing the completed artifact to be corrected
without rerunning or using truth for selection. A regression test covers this
ordering bug.

## Artifacts

- `experiments/results/pf_ffbsi_marginal_smoke_internal_epochs.csv`
- `experiments/results/pf_ffbsi_t2_50_internal_epochs.csv`
- `experiments/results/pf_ffbsi_t3_50_internal_epochs.csv`
- `experiments/results/pf_ffbsi_n1_50_internal_epochs.csv`
- `experiments/results/pf_ffbsi_n2_50_internal_epochs.csv`
- `experiments/results/pf_ffbsi_n3_50_internal_epochs.csv`
- `experiments/results/pf_ffbsi_full6_p2k_lag10_paths8_ablation_summary.csv`
- `experiments/results/pf_vs_ffbsi_full6_comparison.csv`
