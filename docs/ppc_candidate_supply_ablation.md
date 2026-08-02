# PPC candidate-supply ablation

This note records the frozen, truth-free candidate-supply decision used by the
PPC GNSS+IMU PF/FGO pipeline. Reference trajectories are opened only by the
post-estimator audit subprocess. They are never present in the solver or
tracker commands.

## Decision

Use native quality-ranked partial ambiguity resolution (`--quality-ranked-par`)
with top-K 8. Combine it with the previously validated native fixed-lag IMU
aperture (0.30 m, 0.05 m winner margin), two-epoch IMU-consistent acquisition,
and a two-epoch validation-gap tolerance. Independent GNSS holdout validation
and the 0.99 PF posterior gate remain mandatory.

The policy is uniform across all six routes. We do not choose a policy per
route after looking at reference errors.

## Frozen 300-epoch six-route comparison

Both arms used the same binary, data, route order, top-K 8, native IMU FGO,
FIX streak 2, gap tolerance 1, and CUDA mode off. The only changed solver input
was quality-ranked PAR.

| Policy | Denominator | Correct FIX | False FIX | >1 m false | Correct FIX rate | Candidate oracle | Unique pass / correct | Weighted solver mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline PAR | 1,798 | 558 | 0 | 0 | 31.03% | 1,115 / 1,349 | 671 / 671 | 13.228 ms |
| Quality-ranked PAR | 1,798 | 662 | 0 | 0 | 36.82% | 1,136 / 1,329 | 766 / 765 | 13.030 ms |

Quality-ranked PAR added 104 correct FIX epochs (+18.6% relative) without a
tracker-output false FIX. One individually unique holdout pass was wrong, but
the temporal tracker did not promote it to FIX. This is why the holdout and
streak gates remain part of the production contract.

Per-route correct FIX counts were:

| Route | Baseline | Quality-ranked | Delta |
|---|---:|---:|---:|
| Tokyo run1 | 68 | 122 | +54 |
| Tokyo run2 | 164 | 176 | +12 |
| Tokyo run3 | 201 | 225 | +24 |
| Nagoya run1 | 26 | 20 | -6 |
| Nagoya run2 | 70 | 90 | +20 |
| Nagoya run3 | 29 | 29 | 0 |

Artifacts:

- `Testing/basin_fgo_supply_six_baseline_k8_e300/summary.json`
- `Testing/basin_fgo_supply_six_quality_k8_e300/summary.json`

## Rejected alternatives

On the frozen first 300 epochs of Nagoya run3:

- top-K 16 increased candidate oracle coverage from 122/171 to 130/171, but
  produced three false FIX epochs (all below 1 m). It violates the zero-false
  contract and is rejected.
- constellation/interleaved PAR with four candidate groups and two-group
  fallback consensus left correct FIX at 29, reduced oracle coverage to
  125/200, and increased solver mean time from 10.772 ms to 14.755 ms. It is
  rejected.
- quality-ranked PAR kept 29/29 correct FIX, increased oracle coverage to
  131/167, and remained close to baseline runtime. The six-route ablation above
  demonstrates that its benefit is not limited to the Nagoya run3 slice.

These are engineering results on the available PPC routes, not a blind-test or
state-of-the-art claim.

## Full-length six-route confirmation

The frozen quality-ranked policy was subsequently replayed over every available
epoch of all six routes. The official denominator contained 48,778 epochs. It
produced 8,626 correct FIX epochs (17.684%), zero false FIX, and zero false FIX
above 1 m. The preceding uniform baseline produced 7,475 correct FIX epochs
(15.325%), so the frozen change adds 1,151 correct FIX epochs (+15.4% relative)
without relaxing the integrity contract.

| Route | Baseline correct FIX | Quality-ranked correct FIX | Delta |
|---|---:|---:|---:|
| Tokyo run1 | 743 | 897 | +154 |
| Tokyo run2 | 1,337 | 1,535 | +198 |
| Tokyo run3 | 3,366 | 4,088 | +722 |
| Nagoya run1 | 910 | 926 | +16 |
| Nagoya run2 | 926 | 988 | +62 |
| Nagoya run3 | 193 | 192 | -1 |
| **All routes** | **7,475** | **8,626** | **+1,151** |

The candidate audit found a correct candidate in 15,069 of 25,515 evaluated
epochs. There were 10,147 unique holdout passes, of which seven were wrong;
the independent temporal tracker promoted none of those seven to FIX. Both the
route-level audits and aggregate integrity gate passed. This confirms the
candidate-ranking improvement on the available public routes, but it remains
public-data engineering evidence rather than a blind SOTA claim.

Artifacts:

- `Testing/basin_fgo_quality_full_parallel_v5_aggregate.json`
- aggregate SHA-256:
  `64f0f4e58faad0d1fb0f1bff130abd9592ecbe08521a6f4f6e2278815ae1e5ad`

## Integrated IMU continuity promotion

The quality-ranked basin stream was then consumed by the already implemented
native IMU aperture and accelerated-acquisition gates. A 300-epoch six-route
check increased correct FIX from 662 to 698 with zero false FIX. Increasing
the validation-gap tolerance from one to two epochs was selected on run1/run2
(426 to 456 correct FIX) and then confirmed on the untouched run3 blocks (272
to 279), again with zero false FIX.

The frozen combined policy produced the following full-length result:

| Route | Quality-ranked only | Combined safe policy | Delta | False / >1 m |
|---|---:|---:|---:|---:|
| Tokyo run1 | 897 | 1,081 | +184 | 0 / 0 |
| Tokyo run2 | 1,535 | 1,691 | +156 | 0 / 0 |
| Tokyo run3 | 4,088 | 4,744 | +656 | 0 / 0 |
| Nagoya run1 | 926 | 1,014 | +88 | 0 / 0 |
| Nagoya run2 | 988 | 1,228 | +240 | 0 / 0 |
| Nagoya run3 | 192 | 206 | +14 | 0 / 0 |
| **All routes** | **8,626** | **9,964** | **+1,338** | **0 / 0** |

The final rate is 20.427% of the 48,778-epoch official denominator. Relative
to the original uniform 7,475-FIX policy, this is +2,489 correct FIX epochs
(+33.3% relative), with every route improving and both integrity counters
remaining zero.

Four GNSS fault injections (outage, ambiguous holdout, cycle slip, and NLOS)
and four IMU fault injections (bias jump, dropout, time offset, and vibration)
were rerun with the combined policy. All eight audits passed with zero false
FIX and zero false FIX above 1 m. The GNSS fault interval itself emitted no
FIX. Machine-readable counts, settings, and artifact hashes are recorded in
`internal_docs/ppc_quality_imu_gap2_promotion_evidence_2026_08_02.json`.

This is the promoted safe-FIX availability policy for the available PPC data.
It is not a blind score or a SOTA claim: the workspace contains no sealed route,
and safe-FIX availability is not the official trajectory-distance score.

## CUDA plus native IMU FGO check

The combined GTSAM+CUDA binary was also tested on the same Nagoya run3 slice.
CPU and CUDA runs produced identical audited decisions (29 correct FIX, zero
false FIX), candidate oracle coverage (131/167), and unique passes (33/33).
CUDA reduced the native IMU stage mean from 78.446 ms to 67.718 ms, but the
reported solver mean increased from 11.420 ms to 41.366 ms and p95 from
16.449 ms to 93.349 ms. At top-K 8 this workload is too small to amortize GPU
launch and transfer overhead, so CUDA is available but not selected by the
frozen production run.

Artifact:

- `Testing/basin_fgo_supply_nagoya3_quality_k8_e300_gtsam_cuda/summary.json`
