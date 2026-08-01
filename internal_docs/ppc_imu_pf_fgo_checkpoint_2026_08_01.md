# PPC IMU PF/FGO integration checkpoint (2026-08-01)

## Runtime and evaluation contract

- Inputs: the existing PPC rover/base/navigation/IMU files only.
- No LiDAR, camera, map, or additional route data.
- `reference.csv` is forbidden to the estimator. It is opened only by offline
  audit/scoring after estimator artifacts exist.
- The legacy safe PF/FGO output remains the disabled-mode baseline:
  Tokyo 6488/11928 correct FIX and Nagoya 5277/7602 correct FIX, with zero
  false FIX and zero false FIX above 1 m.
- IMU may improve continuous-state prediction and ambiguity-basin proposal,
  but it may not bypass the independent GNSS carrier/code holdout.

## Literature and OSS decisions

1. Use on-manifold preintegration with analytic bias correction and propagated
   covariance, following Forster et al., rather than the former position-only
   constant-velocity correction:
   <https://arxiv.org/abs/1512.02363>.
2. Reuse GTSAM `CombinedImuFactor` and the existing project implementation of
   Pose3, velocity, accelerometer bias, gyroscope bias, and bias random walk:
   <https://borglab.github.io/gtsam/imufactor/>.
3. Use an incremental fixed-lag smoother so old states are marginalized and
   online cost remains bounded:
   <https://borglab.github.io/gtsam/fixedlagsmoother/>.
4. Keep the discrete ambiguity/NLOS/slip modes outside the continuous IMU
   state initially, then exchange proposals and likelihoods with the PF. GTSAM
   also provides hybrid incremental inference, but it is an ablation candidate
   rather than a prerequisite:
   <https://borglab.github.io/gtsam/hybrid/>.
5. Treat time offset and IMU intrinsic scale/misalignment as calibration
   quantities, not silent constants. OpenVINS exposes these quantities as
   explicit estimator states; PPC has hardware synchronization, so this work
   first audits them and keeps runtime offset at zero unless blocked-CV evidence
   supports a change:
   <https://docs.openvins.com/classov__msckf_1_1State.html>.

The repository already contains the required GTSAM IMU backend in
`src/algorithms/fgo_gtsam_backend.cpp`. The missing production link is in
`apps/native/gnss_solve.cpp`: it neither loads an IMU CSV nor populates
`FGOProblem::ImuInput`, and the current MultiSD build did not discover GTSAM.

## Root cause of the previous no-gain bridge

`python/gnss_gpu/basin_imu_bridge.py` only modifies the ambiguity PF's
constant-velocity position/velocity proposal. It does not add attitude,
velocity, accelerometer-bias, or gyroscope-bias nodes to FGO, does not preserve
attitude across rolling windows, and cannot relinearize IMU residuals jointly
with TDCP/Doppler/DD carrier factors. On the existing Nagoya 300-epoch artifact
it produced the same 18 correct FIX as the no-IMU tracker, so it is retained as
an ablation rather than the integration target.

The upstream PPC README calls the axes FRD, while the actual logs and existing
fusion stack use FLU-like signed measurements (stationary Z is approximately
+9.81 m/s^2). This ambiguity is now resolved by a six-route offline dynamic
contract audit instead of comments or a single stationary sample.

## Six-route input and contract audit

Artifacts:

- `Testing/ppc_imu_fgo/imu_coverage.json`
- `Testing/ppc_imu_fgo/imu_contract_audit.json`

All six routes pass:

- exactly 100 Hz median rate;
- no non-monotonic/duplicate timestamps;
- no gaps above 1.5 nominal sample periods;
- identical start/end times and GPS weeks to the 5 Hz route interval;
- forward-X versus speed derivative correlation: 0.85--0.96;
- left-Y versus centripetal acceleration correlation: 0.74--0.93;
- up-Z gyro versus counter-clockwise yaw correlation: effectively 1.00;
- best gyro/reference time shift: 0.00 s on five routes and +0.02 s on Tokyo
  run2, within the pre-registered +/-0.05 s synchronized-input gate.

The runtime body contract is therefore identity raw-to-FLU for these PPC CSVs.
The dataset lever arms are converted from README FRD coordinates to FGO FLU:

- Tokyo: `[0.31, 0.0, 0.55]` m;
- Nagoya: `[0.593, 0.670, 1.216]` m.

These values are fixed metadata, never inferred from reference at runtime.

## Native integration status

`gnss_solve` now accepts the opt-in options
`--multisd-fgo-imu`, `--multisd-fgo-imu-lever-arm-flu`, and
`--multisd-fgo-imu-fixed-lag`.  The existing Eigen MultiSD solve remains the
discrete top-K and independent-GNSS-validation authority.  A GTSAM companion
now estimates Pose3, ENU velocity, accelerometer bias, and gyro bias using
`CombinedImuFactor` in a fixed-lag graph.  Its states are proposal telemetry;
they cannot directly set RTK status or bypass the carrier/code holdout.

The initial 30-epoch smoke exposed an output-contract bug rather than an
optimizer failure: all 21 GTSAM windows converged with FLOAT states, but the
fixed-lag backend did not copy the already-computed satellite count into
`PositionSolution`, so the common `isValid()` rejected every result.  The
backend now publishes `epoch_diagnostics[i].num_satellites`; Tokyo run1 is
21/21 valid with zero smoother recovery.

Rolling windows now latch one causal route ENU origin and warm-start the next
overlapping window from the matching previously estimated rotation matrix,
velocity, accelerometer bias, and gyro bias.  On the Tokyo 30-epoch smoke,
20/21 windows warm-start, all 21 remain valid, and maximum estimated bias
norms are 0.0206 m/s^2 and 0.00389 rad/s.  Biases and warm-start state are
exported in both the shadow CSV and basin JSONL.

The no-IMU legacy and GTSAM builds produce byte-identical `.pos` and basin
JSONL on the 30-epoch parity case.  Every non-timing shadow CSV field is also
identical; only the four measured runtime columns differ across compilers.
When IMU is disabled, no IMU CSV columns or JSON object are emitted.

## PF/FGO feedback experiments

The tracker can now consume embedded native IMU-FGO states as proposals.  It
uses relative FGO displacement and ENU-to-ECEF velocity, never the absolute
IMU-FGO position as an additional measurement likelihood.  FIX still requires
a GNSS-holdout-passing candidate, posterior concentration, and a consecutive
validation streak.

Six-route, 300-epoch, top-K4 artifact:

- IMU FGO available: 1746/1746 windows;
- warm-start: 1740/1746 windows;
- proposal-only motion feedback: same 569 correct tracker FIX as GNSS-only,
  false FIX 0;
- a relative-motion aperture of 0.30 m with 0.05 m runner-up margin resolves
  26 otherwise-multiple GNSS-passing epochs and raises tracker correct FIX
  from 569 to 617 (+48), false FIX 0 and >1 m false FIX 0;
- these first-300 gains overlap primary RTK FIX, so baseline-priority union
  gain is zero on that slice.

The aperture compares each candidate with the prior validated position plus
the native IMU-FGO displacement.  It never selects a candidate which failed
the independent GNSS holdout.  An additional opt-in reacquisition gate allows
FIX after two, rather than three, consecutive GNSS-passing epochs only when
the relative IMU motion residual is at most 0.30 m.

Blocked-CV evidence with the rule fixed:

| Slice | Role | GNSS-only tracker | IMU tracker | Union delta | New IMU false |
|---|---:|---:|---:|---:|---:|
| Nagoya/run1 K8, epochs 0--599 | development | 192 | 257 | +11 correct | 0 |
| Tokyo/run1 K8, epochs 1500--1799 | temporal holdout | 3 | 6 | +2 correct | 0 |
| Nagoya/run1 K8, epochs 5700--5999 | temporal holdout | 41 | 50 | 0 | 0 |

The Tokyo holdout's primary solver already contains one 0.5--1.0 m false FIX;
the IMU tracker adds none, and >1 m false remains zero.  Therefore this slice
supports incremental safety of the IMU rescue but is not yet sufficient for
the final global `false FIX = 0` promotion contract.

Artifacts are under `Testing/ppc_imu_fgo/native_e300`,
`Testing/ppc_imu_fgo/blocked_cv_nagoya1_k8_e600`,
`Testing/ppc_imu_fgo/blocked_cv_tokyo1_k8_s1500`, and
`Testing/ppc_imu_fgo/blocked_cv_nagoya1_k8_s5700`.

## Causal PF-to-FGO ambiguity feedback

The bidirectional bridge is now implemented as an explicit two-pass replay of
the online causal contract.  The PF writes only selected integer modes whose
native candidate passed the independent GNSS holdout.  Each row carries the
`gnss_gpu_pf_fgo_feedback_v1` schema and an explicit holdout-pass assertion.
`gnss_solve` rejects the entire file on malformed schema, time ordering,
duplicate identity, or invalid values.

For each IMU-FGO window, only the latest feedback group strictly older than
the current epoch and no more than one second old is eligible.  Every row must
uniquely match satellite, reference satellite, signal, target segment, and
wavelength in the newly built problem, with at least six matches; otherwise
the window receives no priors.  GTSAM adds each matched integer as a soft
ambiguity prior at most once per ambiguity symbol.  It never pins the symbol,
changes the RTK/MultiSD result, bypasses holdout validation, or labels a state
FIXED.  CSV and basin JSON telemetry expose source time, age, rows, matches,
reason, and backend requested/applied counts.

The first smoke exposed and fixed a repeated-evidence bug: 15 PF modes had
initially produced 150 factors over a ten-epoch window.  The corrected backend
reports requested=applied and at most one prior-addition epoch per solve.
Across the 600-epoch development replay, all 4,056 GNSS basin payload rows are
identical after removing the opt-in `imu_fgo` telemetry object.

The prior-sigma sweep was confined to Nagoya/run1 epochs 0--599.  Strong
0.05/0.2-cycle priors regressed correct tracker FIX by one.  Conservative
0.5 and 1.0-cycle priors both changed 257 to 258 correct tracker FIX and the
baseline-priority union from 315 to 316, with false FIX 0 and >1 m false FIX
0.  The more conservative 1.0-cycle setting was frozen for temporal holdout:

| Slice | PF-to-FGO applied windows | Correct before/after | Union before/after | False / >1 m false |
|---|---:|---:|---:|---:|
| Nagoya/run1 0--599 (development) | 342 | 257 / 258 | 315 / 316 | 0 / 0 |
| Tokyo/run1 1500--1799 (holdout) | 21 | 6 / 6 | 179 / 179 | 0 / 0 |
| Nagoya/run1 5700--5999 (holdout) | 87 | 50 / 50 | 202 / 202 | 0 / 0 |

All three runs have zero same/future-epoch consumption and zero requested vs
applied factor-count mismatch.  This proves causal/safe operation and one
development gain, but the neutral holdouts do not justify default promotion.
The feedback therefore remains opt-in.

Artifacts are under `Testing/ppc_imu_fgo/pf_feedback_smoke`,
`Testing/ppc_imu_fgo/pf_feedback_nagoya1_k8_e600`,
`Testing/ppc_imu_fgo/pf_feedback_tokyo1_k8_s1500`, and
`Testing/ppc_imu_fgo/pf_feedback_nagoya1_k8_s5700`.

## IMU fault injection and residual diagnostics

`experiments/inject_ppc_imu_fault.py` now generates deterministic PPC IMU
fault streams with no reference/truth input.  Supported faults are bounded
dropout, timestamp offset, accelerometer bias jump, gyro bias jump, combined
bias jump, and sinusoidal vibration.  Every run writes a
`gnss_gpu_ppc_imu_fault_v1` manifest with exact parameters, row counts, and
input/output SHA-256 hashes.  Timestamp faults are sorted and rejected if they
create duplicate or non-increasing sample times.  The existing basin fault
injector remains the GNSS-outage path.

The native companion now reports sample count, maximum coverage gap (including
both window boundaries), a categorical fault reason, IMU-prediction-to-
optimized pose correction, adjacent accel/gyro bias steps, and the actual
CombinedImuFactor NIS (`2 * factor.error`) both absolute and per 15 DoF.  These
are telemetry-only and do not affect the solver.

An opt-in IMU run now fails closed when its maximum sample/coverage gap exceeds
`--multisd-fgo-imu-max-gap` (default 0.05 s).  A five-second Nagoya dropout
previously allowed partially covered edge windows and produced a 1.283 m pose
correction.  The coverage gate explicitly abstains on 16 gap windows and 17
insufficient-sample windows, reduces the maximum observed correction to
0.228 m, and resumes once a complete rolling window exists: 1.8 s after the
dropout ends.  Correct tracker FIX is 27 clean versus 26 under dropout; the
lost epoch is an IMU-accelerated FIX inside the outage.  False FIX and >1 m
false FIX remain zero.

The first deterministic 300-epoch matrix (faults at TOW 550400--550405) is:

| IMU input | Correct tracker FIX | False / >1 m false | Valid companion windows |
|---|---:|---:|---:|
| clean | 27 | 0 / 0 | 291 |
| dropout, 5 s (coverage-gated) | 26 | 0 / 0 | 258 |
| timestamp offset, 4 ms | 27 | 0 / 0 | 291 |
| accel 0.5 m/s^2 + gyro 2 deg/s bias jump | 27 | 0 / 0 | 291 |
| 20 Hz, accel 2 m/s^2 + gyro 5 deg/s vibration | 27 | 0 / 0 | 291 |

Single-window NIS maxima are not suitable gates: clean reaches 22.6 per DoF.
Over the five-second fault interval, however, median NIS/DoF changes from
0.75 clean to 1.63 for the bias jump and 1.64 for vibration.  No NIS gate is
enabled yet; a sustained rule must be pre-registered from all-six-route clean
distributions rather than tuned on this one development slice.

Artifacts are under `Testing/ppc_imu_fgo/imu_fault_nagoya1_e300`.

### Six-route clean health calibration

All six routes were replayed for the first 300 epochs with the new health
telemetry.  Every one of 1,746 attempted IMU-FGO windows is available and has
fault reason `ok`.  `experiments/audit_ppc_imu_fgo_health.py` performs the
truth-free, time-order-checked, hash-recorded aggregation.  Combined NIS/DoF
has median 1.183, p95 16.404, and maximum 67.957.  The 25-window rolling
median maximum is 6.880.  A provisional 20%-margin monitor is therefore 8.5,
but remains telemetry-only and explicitly `promotion_ready=false`: first-300
coverage cannot establish a production fault gate, and the injected moderate
bias/vibration faults do not cross it reliably.

Artifacts and the audit are under
`Testing/ppc_imu_fgo/clean_nis_six_route_e300`.

## GPU dispatch and parity

A combined GTSAM+CUDA 12.8, sm_75 build now exists at
`third_party/gnssplusplus/build-codex-gtsam-cuda-clang-v2`.  Forced CUDA on
Nagoya/run1 first 300 epochs executed 3,715 dense solves and 196 batched
hypothesis solves successfully with zero fallback.  CPU/GPU parity passes:
integer acceptance is identical, maximum ECEF difference is 0.014 mm, and
maximum incremental-log-likelihood difference is 3.20e-5.

Forced GPU is substantially slower on the GTX 1660 Ti and these 96--97 state
problems: GNSS p95 is 61.67 ms versus 12.82 ms CPU (4.81x).  This is PCIe and
cuSOLVER launch overhead, not a failed kernel.  The existing `auto` dispatch
threshold of 2,048 states is therefore correct: an auto-mode smoke selected
CUDA for 0/21 windows and made zero CUDA attempts.  The safe accelerated
configuration is a CUDA-capable build with `auto`, not forced GPU.  The
GTSAM fixed-lag companion remains CPU incremental; its small sparse graph is
not a suitable dense GPU workload on this device.

Parity evidence is under
`Testing/ppc_imu_fgo/cuda_nis_parity_nagoya1_e300/combined/parity.json`.

## Full six-route frozen evaluation

The frozen top-K8 configuration (`native_imu_aperture_m=0.30`, margin 0.05 m,
and IMU-consistent GNSS streak 2) was replayed over all six complete routes.
Reference truth was opened only by the post-estimator audit subprocess after
the basin and tracker artifacts were fixed and hashed.

| Route | GNSS-only correct | IMU correct | Delta | IMU false / >1 m |
|---|---:|---:|---:|---:|
| Nagoya run1 | 739 | 898 | +159 | 0 / 0 |
| Nagoya run2 | 656 | 942 | +286 | 0 / 0 |
| Nagoya run3 | 145 | 178 | +33 | 0 / 0 |
| Tokyo run1 | 548 | 764 | +216 | 0 / 0 |
| Tokyo run2 | 980 | 1,289 | +309 | 0 / 0 |
| Tokyo run3 | 2,446 | 3,689 | +1,243 | 0 / 0 |
| **total** | **5,514** | **7,760** | **+2,246** | **0 / 0** |

The common denominator is 48,778 reference-covered baseline epochs.  Correct
FIX rate rises from 11.304% to 15.909%, a 40.73% relative gain.  Every one of
the 5,514 GNSS-only FIX decisions remains fixed; the candidate adds 2,246 and
loses zero.  All six routes improve, giving a conservative route-level
one-sided sign-test p-value of 0.015625.  The baseline-priority union adds 240
correct epochs, but it still inherits 1,528 false FIX (773 above 1 m) from the
legacy primary baseline.  Those inherited statuses are explicitly excluded
from the standalone safe IMU component claim; only the tracker/IMU rescue has
false FIX zero.

The full replay exposed 447 rows where GTSAM recovered after an indeterminate
incremental update.  The recovered trajectory can be numerically finite but
is re-anchored, so the PF parser now fails closed whenever
`recovery_epochs > 0` or `fault_reason != ok`.  Before that gate, exactly one
correct Tokyo run1 accelerated FIX occurred on a recovery row.  After the
gate, no recovery row is marked IMU-available and no recovery row fixes; the
full result loses only that one epoch and retains the +2,246 gain.  Recovery,
coverage, and fault gaps also split the rolling-NIS calculation, preventing a
health window from bridging invalid telemetry.

Across the six routes the native fixed-lag p95 runtime is 110.5--135.5 ms,
within the 200 ms 5 Hz epoch budget.  The PF-side IMU processing adds no
material measured tracking cost.  Full clean, non-recovery health telemetry
contains 57,648 NIS samples: median 0.911, p95 10.236, 25-window rolling
median p95 3.087 and maximum 16.663.  Because this clean distribution overlaps
the moderate injected bias/vibration cases, NIS remains telemetry-only; no
unsafe data-tuned rejection threshold is enabled.

The combined GNSS-outage and simultaneous GNSS+IMU-outage checks both retain
26/26 correct tracker FIX with zero false and zero >1 m false.  Both first
post-outage fixes occur 17.8 seconds after the fault interval; GNSS candidate
supply, not IMU propagation, is the reacquisition bottleneck on this slice.

Machine-readable evidence is at
`internal_docs/ppc_imu_pf_fgo_promotion_evidence_2026_08_02.json`.  It combines
the six full-route pairs, two frozen temporal holdouts, six deterministic
fault audits, recovery fail-closed checks, runtime budgets, full-route health,
and CPU/GPU parity.  Every component gate passes and the IMU configuration is
a default candidate while remaining disabled by default, preserving legacy
behaviour until an explicit release change.

`experiments/compose_ppc_imu_safe_output.py` now closes the output-composition
contract: the IMU PF/FGO tracker is the sole status=FIX authority, while the
legacy position is retained only as FLOAT fallback.  It never inherits a
legacy status=FIX.  Materialized full-route outputs match all 7,760 tracker
FIX decisions exactly, with false FIX 0 and >1 m false FIX 0 on every route.
The machine-readable evidence therefore reports both
`default_candidate_ready=true` and `default_promotion_ready=true`, while
`default_enabled=false`.  The unsafe baseline-priority union remains an
explicit diagnostic only and must not be used as the promoted output.

The CPU CI smoke now includes the 17 PF/FGO, IMU-contract, fault-injection,
parity, and promotion-evidence suites; the corresponding experiment scripts
are Ruff-checked.  Final local evidence is 97 tests passed, Ruff clean, and
Actionlint clean.  Both the MSVC legacy native build and the Clang
GTSAM+CUDA 12.8 build complete incrementally; native `fgo_tests` and
`fgo_multisd_smoke` both pass.  (`fgo_tests` requires the vcpkg runtime DLL
directory on `PATH` when invoked outside a Visual Studio developer shell.)

Artifacts are under
`Testing/ppc_imu_fgo/full_six_route_k8_imu`, with parity under
`Testing/ppc_imu_fgo/cuda_nis_parity_nagoya1_e300/combined` and faults under
`Testing/ppc_imu_fgo/imu_fault_nagoya1_e300` and
`Testing/ppc_imu_fgo/combined_fault_nagoya1_e300`.

## Remaining implementation slice

1. Keep PF-to-FGO ambiguity priors opt-in: their temporal holdouts are neutral
   and do not justify default promotion.
2. Treat smoother recovery and NIS as operational telemetry.  Do not tune a
   production NIS gate without a new pre-registered fault-separation study.
3. Before an explicit release/default-enable change, package the native
   GTSAM+CUDA build and run the same evidence builder in the release job; keep
   CUDA dispatch on `auto` for these small states.
