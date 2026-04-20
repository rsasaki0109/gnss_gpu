# Region-Aware Proper RBPF Velocity Gate Results

Date: 2026-04-21 JST
Branch: `feature/carrier-phase-imu`

## Setup

Goal: extend codex14's Odaiba 3k subset submeter result to full Odaiba by
applying the proper RBPF Doppler velocity KF only in selected regions.

Common Odaiba config:

```bash
--preset odaiba_best_accuracy
--rbpf-velocity-kf
--rbpf-velocity-process-noise 0.1
--rbpf-doppler-sigma 2.0
--rbpf-velocity-kf-gate
```

Gate semantics:
- `min_dd_pairs=0`, `min_ess_ratio=0.0`, `max_doppler_residual=inf` is the
  no-gate codex14 tuned candidate.
- `min_dd_pairs` uses kept DD carrier pairs after DD carrier gating/support
  guards.
- Doppler residual is the centered median absolute range-rate residual after
  removing receiver clock drift in the same style as the CUDA KF update.

CSV artifacts:
- `experiments/results/rbpf_velocity_gated_phase1_dd_pairs.csv`
- `experiments/results/rbpf_velocity_gated_phase2_ess.csv`
- `experiments/results/rbpf_velocity_gated_phase3_residual.csv`
- `experiments/results/rbpf_velocity_gated_phase4_subset.csv`
- `experiments/results/rbpf_velocity_gated_phase5_shinjuku.csv`

## Phase 1: DD Pair Gate

Full Odaiba, ESS gate disabled, Doppler residual gate disabled.

| min DD pairs | Doppler KF used | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | best, no gate |
| 5 | 10524 | 1.214 | 5.428 | 1.326 | 4.657 | worse |
| 10 | 6968 | 1.402 | 5.792 | 1.362 | 5.105 | worse |
| 15 | 1293 | 1.323 | 5.311 | 1.328 | 5.166 | worse |
| 17 | 257 | 1.419 | 5.609 | 1.377 | 5.462 | worse |
| 20 | 0 | 1.413 | 5.964 | 1.357 | 5.812 | worse |

DD pair gating did not transfer the subset win to full Odaiba. Stricter DD
gates remove most Doppler KF updates and degrade the smoother median.

## Phase 2: ESS Gate

Best Phase 1 setting fixed: `min_dd_pairs=0`, residual gate disabled.

| min ESS ratio | Doppler KF used | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.000 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | reused from Phase 1 |
| 0.005 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | no effect |
| 0.010 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | no effect |
| 0.020 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | no effect |
| 0.050 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | no effect |

At the Doppler KF update point, ESS never fell below these thresholds, so the
ESS gate did not change behavior.

## Phase 3: Doppler Residual Gate

Best Phase 1-2 setting fixed: `min_dd_pairs=0`, `min_ess_ratio=0.0`.

| max Doppler residual | Doppler KF used | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| inf | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | reused from Phase 2 |
| 5.0 | 12252 | 1.268 | 4.980 | 1.209 | 4.292 | no effect |
| 3.0 | 12250 | 1.271 | 5.002 | 1.270 | 4.358 | worse |
| 1.5 | 12103 | 1.280 | 5.088 | 1.416 | 4.435 | worse |

The residual gate only began to skip updates at 3.0 m/s and lower, and those
skips harmed the smoother median.

## Phase 4: Odaiba 3k Subset Reproduction

Best full-run setting remained no gate. With `max_epochs=3000`:

| Config | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Doppler KF used |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Q_v=0.1`, Doppler sigma 2.0, no gate | 0.878 | 1.968 | 0.890 | 2.098 | 3000 |

This reproduces codex14's subset submeter result.

## Phase 5: Shinjuku Regression

Same no-gate setting on full Shinjuku:

| FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Doppler KF used | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2.265 | 7.540 | 2.270 | 7.738 | 20127 | RMS < 9.5 pass |

## Decision

Region-aware gates did not beat the current Odaiba best (`odaiba_best_accuracy`
SMTH P50 1.14 m), and did not beat the no-gate proper RBPF tuned full result
(`SMTH P50 1.209 m`). No `odaiba_rbpf_velocity_gated` preset should be promoted.

Because the result is negative, the gate implementation should be reverted and
kept as a documented failed path. The next submeter attempts should move to the
plan's BBB/CCC directions rather than more DD/ESS/Doppler residual gating.
