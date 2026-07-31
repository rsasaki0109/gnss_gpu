# GNSS-only MultiSD FGO / integer-aperture research audit (2026-08-01)

## Scope and hard constraints

- Inputs are PPC `rover.obs`, `base.obs`, and `base.nav` only.
- IMU, LiDAR, camera, map, and external-route training data are excluded.
- The production-library target is correct FIX >=70% for Tokyo and >=80% for
  Nagoya, false/FIX <=0.1%, zero >1 m false fixes, and end-to-end p95 <=100 ms.
- All new acquisition paths remain default-off until nested blocked CV, injected
  faults, CPU/CUDA parity, and latency gates pass.

## Primary literature and OSS audit

1. Teunissen and Verhagen's ratio-test analysis shows that a fixed ratio is not
   itself a correctness test. Hou, Verhagen, and Wu's FFRT implementation makes
   the acceptance threshold conditional on a required failure rate. Therefore
   the regular LAMBDA ratio remains a candidate-quality feature; the authority
   to publish FIX belongs to the independent aperture/holdout validator.
   Sources: [ratio-test revisited](https://doi.org/10.1179/003962609X390058),
   [efficient FFRT](https://doi.org/10.3390/s16070945).
2. Success-rate PAR fixes a high-confidence subset rather than requiring full
   ambiguity resolution. The two-step data-driven PAR work uses the integer
   bootstrapping success-rate lower bound. The 2026 entropy-weighted MPAR work
   ranks with ambiguity variance, carrier residual, and signal strength. We can
   implement the variance and residual terms from the existing PPC graph; any
   SNR term must come from the same RINEX observations and needs a missing-value
   fail-safe.
   Sources: [two-step success-rate PAR](https://doi.org/10.1016/j.asr.2016.07.029),
   [entropy-weighted MPAR](https://doi.org/10.3390/s26144388).
3. GraphGNSSLib demonstrates GNSS-only FGO with DD pseudorange, DD carrier, and
   Doppler followed by LAMBDA. `gtsam_gnss` exposes pseudorange, Doppler, TDCP,
   robust error, and ambiguity examples. GICI-LIB is broader and includes
   inertial/visual factors, but its useful comparison here is only the GNSS
   temporal/spatial factor organization and outlier rejection. No IMU/camera
   code or measurements are adopted.
   Sources: [GraphGNSSLib](https://github.com/weisongwen/GraphGNSSLib),
   [gtsam_gnss](https://github.com/taroz/gtsam_gnss),
   [GICI-LIB](https://github.com/chichengcn/gici-open),
   [RTKLIB](https://github.com/tomojitakasu/RTKLIB).
4. Window carrier phase FGO uses temporal carrier correlation together with
   pseudorange and Doppler. This supports the current fixed-lag MultiSD graph
   and TDCP/arc continuity direction without requiring inertial aiding.
   Source: [window carrier-phase FGO](https://arxiv.org/abs/2109.00683).

This is an algorithm/design audit, not source-code transplantation. OSS code is
used for behavioral comparison; implementation remains native to gnssplusplus.

## Measured PPC blocker before validator-aware PAR fallback

The production-library baselines are 5,984/11,928 correct FIX (50.1677%) for
Tokyo and 5,047/7,602 (66.3904%) for Nagoya, both with zero false fixes.

With causal MultiSD FGO, constellation PAR, candidate ratio 1.0, and the strict
disjoint validator, the baseline-priority union reached:

| City | Correct FIX | Total | Rate | False FIX |
|---|---:|---:|---:|---:|
| Tokyo | 6,151 | 11,928 | 51.5677% | 0 |
| Nagoya | 5,304 | 7,602 | 69.7711% | 0 |

The full-route FLOAT ledger showed candidate supply, not only the validator
threshold, as the limiting stage. More importantly, native PAR stopped at the
first subset/pool whose LAMBDA search succeeded. If every top-K hypothesis from
that group failed the independent validator, it never tried the next valid PAR
subset. Candidate generation and validation were therefore sequential but not
integrated.

## Implemented experiment: validator-aware candidate groups

- Generate up to `multisd_max_candidate_groups` successful LAMBDA top-K groups.
- Evaluate one group at a time using only disjoint satellite/time observations.
- If a group has zero passing hypotheses, evaluate the next PAR group.
- Accept only one passing hypothesis within a group.
- If multiple hypotheses pass in any group, fail closed and do not search for a
  more convenient later group.
- Keep the default at one group, preserving production behavior.
- Expose `--multisd-fgo-shadow-candidate-groups` (1..32) and include it in the
  PPC CV policy/sidecar command identity.

The first comparison is locked to candidate ratio 1.0, eight groups, top-K 4,
constellation PAR, three-epoch causal history, 0.75 carrier pass fraction, four
holdout satellites, and minimum six fixed ambiguities. It must first beat the
one-group six-route 300-epoch result with zero false fixes before full-route and
fault testing.

### Result: groups=8 is diagnostic-only and rejected for promotion

The six-route 300-epoch probe improved Tokyo from 565 to 569 correct shadow
fixes and Nagoya from 444 to 476. It had zero false fixes, zero >1 m fixes, and
a worst-route p95 of 69.12 ms. Nagoya run1 reproduced 208 correct fixes with
`groups=1`, while `groups=8` produced 219; all 11 additions had selected ranks
at least four and therefore exercised the intended fallback.

The complete routes exposed the safety failure that the short probe missed:

| City | Shadow correct | Shadow false | >1 m false | p95 (ms) |
|---|---:|---:|---:|---:|
| Tokyo | 1,236 | 6 | 2 | 51.19 |
| Nagoya | 1,692 | 2 | 0 | 49.78 |

After production-library priority, Tokyo still had 6 false rescues (2 above
1 m), while Nagoya had 269 correct rescues and zero false rescues. Therefore
groups=8 is not an admissible policy. All six Tokyo false candidates came from
later groups (selected ranks 5--23); a two-group cap is also unsafe because one
false candidate had rank 5. The feature remains defaulted to one group.

An oracle-only diagnostic found that a later-group condition such as seed
separation <=0.18 m OR maximum integer distance <=0.18 cycles removed these
full-route false candidates, but it is not adopted: those thresholds were
observed on the final route and must be treated only as candidates for blocked
nested CV. The next implementation should prefer cross-subset consensus or an
FFRT-calibrated fallback aperture over a route-fitted scalar cutoff.

## Next ranked experiments

1. Validator-aware groups=8 versus groups=1 on all six 300-epoch routes.
2. If safe, run both complete production routes and measure union FIX, false
   FIX, >1 m errors, candidate-group depth, and p95/max latency.
3. Interleave constellation pools at equal subset depth so the group budget
   covers satellite/source diversity rather than only successively smaller
   prefixes of the first pool.
4. Add a success-rate/ADOP score and same-observation carrier-residual quality
   score to subset ordering; tune only inside nested blocked CV.
5. Calibrate an FFRT/IA lookup or Monte-Carlo boundary per ambiguity dimension
   and covariance-quality band, while retaining disjoint validation as final
   publication authority.
6. Run ambiguity-arc blocked nested CV plus NLOS, outage, cycle-slip, and
   satellite-loss injection, then CPU/CUDA parity and p95 <=100 ms gates.

Failure to reach 70%/80% is reported as a measured candidate/oracle boundary;
it does not authorize relaxing the false-fix integrity limits.
