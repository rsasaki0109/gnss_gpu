# Code and literature improvement audit (2026-07-13)

## Bottom line

The next accuracy work should not be another sweep over pseudorange weights,
NLOS thresholds, RTK reset constants, or ranker thresholds. Those axes have
already been exercised heavily. The strongest remaining improvements are
structural:

1. implement the designed-but-missing tightly coupled DD-RTK/IMU bridge;
2. stop collapsing a multimodal particle cloud to one weighted mean, and wire
   the existing FFBSi machinery into the production PPC path;
3. add a genuinely multi-epoch carrier factor (WCP/ambiguity-eliminated window
   carrier phase) and switchable pseudorange constraints;
4. make Doppler observation-code and constellation aware;
5. harden evaluation so no same-run in-sample selector output can be mistaken
   for transferable gain.

The current production best remains Phase71 at 86.205492% official. The gain is
localized to Nagoya run2 (64.426589% to 65.669779%); the other five official
runs are neutral. That is useful evidence, but not evidence of general map-aid
transferability.

## Direct code findings

### P0: tightly coupled DD-RTK/IMU is explicitly missing

`third_party/gnssplusplus/include/libgnss++/fusion/dd_imu_bridge.hpp` says it is
"Stage 2 (design-only)", declares no functions, and has no `.cpp`. It already
specifies the intended architecture: a 15-state ESKF, variable-length DD
ambiguities, reuse of existing DD measurement construction, cycle-slip logic,
and ambiguity lifecycle.

Meanwhile dynamic RTK calls `resetPositionToSPP()` during reset/reacquisition.
That function overwrites the three position states and clears their covariance
cross-terms before assigning a fresh diagonal variance. This explains the
canyon benchmark's loss of cross-epoch memory: once trust is lost, the filter
keeps returning to a weak absolute SPP seed instead of being carried through the
gap by inertial dynamics.

Recommended implementation:

- turn `dd_imu_bridge.hpp` into an operational `dd_imu_bridge.cpp`;
- propagate position, velocity, attitude, gyro/accelerometer biases with IMU;
- evaluate DD code and carrier residuals against the fused state;
- augment the covariance with live ambiguity states and retain the existing
  slip/lock/hold lifecycle;
- use INS-aided partial ambiguity resolution (posterior residual, elevation,
  and body-frame azimuth ordering);
- replace unconditional hard SPP position overwrite with an innovation-gated
  soft reset or covariance inflation when a valid propagated fused state exists.

This is the largest change, but it attacks the observed root cause rather than
another symptom. The inuex35 comparison is also consistent with this priority:
the external pipeline fixes about 49.5% of epochs at 0.815 m, while this stack's
fixes are much rarer (about 6.5%) but very accurate (about 0.048 m) and
front-loaded.

### P0/P1: preserve and emit particle modes

`ParticleFilterDevice.estimate()` returns only a weighted mean. The PPC path
calls it and explicitly emits that weighted mean. However, the same runtime can
apply a Gaussian-mixture position likelihood, so the posterior is allowed to be
multimodal and is then collapsed to a point that may lie between streets or
between two sides of a canyon.

Recommended smallest milestone:

- copy particle position and log weight only at emission epochs;
- cluster in local ENU using weighted connected components/DBSCAN or
  region-growing over an adaptive grid;
- compute mass, mean, covariance, peak density, and road/topology consistency
  for every mode;
- choose a mode using temporal reachability, Doppler/IMU heading, previous-mode
  persistence, and an abstention threshold;
- emit the selected mode mean or MAP particle, never the global mean when two
  material modes exist;
- log mode count, dominant mass, inter-mode separation, covariance, selected
  mode, and fallback reason.

This directly matches recent 3DMA-GNSS evidence that simple averaging of modes
causes accuracy loss and that temporal cluster selection reduces solution
shifting.

### P1: replace the PPC smoother's equal average with existing FFBSi

`pf_device_smoother.py` disables Doppler during the backward pass because the
direction-aware model is missing, then combines estimates as
`(forward_pos + backward_pos) / 2`. This is not a Bayesian particle smoother.

The repository already contains `particle_ffbsi.py`, GPU-exported resampling
ancestors, and `exp_ffbsi_eval.py`. Therefore this is an integration task:

- store filtering particles/log weights and systematic-resampling ancestors for
  a bounded lag;
- use genealogy smoothing or FFBSi inside the PPC production path;
- smooth within a selected mode so trajectories do not jump across posterior
  modes;
- add a correct reverse-time Doppler transition or omit Doppler with an explicit
  uncertainty penalty rather than averaging two differently informed paths.

### P1: add WCP and switchable constraints to the factor graph

The current stack has carrier and TC-FGO infrastructure, but much of the
remaining pseudorange work is still single-epoch weighting/exclusion. Two
literature-backed factors are materially different from the exhausted tuning:

- Window Carrier Phase (WCP): stack carrier observations over a window and use
  a left-nullspace projection to eliminate shared ambiguities. Fuse WCP,
  pseudorange, and Doppler in FGO.
- Switchable pseudorange constraints: attach a latent switch to each factor and
  regularize it with an integrity/chi-square constraint. This preserves geometry
  while suppressing bad urban measurements instead of hard exclusion.

First milestone: implement WCP on continuous, slip-free arcs; compare against
TDCP using blocked spans. Second milestone: add switch variables only after the
base WCP factor passes residual and Jacobian tests.

### P1/P2: make Doppler signal- and constellation-aware

The PPC default accepts G/E/J Doppler but excludes BeiDou until clock handling
is validated. The Doppler API accepts one wavelength and one clock-drift state.
That is insufficient for mixed observation codes, BeiDou B1, or GLONASS FDMA.

Recommended change:

- derive wavelength per observation row from its signal code/frequency channel;
- solve a shared receiver velocity plus constellation-group clock-drift terms;
- add per-system observability/rank guards and robust prefit gates;
- validate G/E/J equivalence first, then enable BeiDou, then GLONASS only when
  channel metadata is present;
- use the same model for backward smoothing.

This can add usable velocity geometry without relying on biased absolute
pseudorange.

### P2: improve road aiding from centerline attraction to a topology factor

Phase71's dual-gated OSM candidate is currently the only demonstrated map gain,
and it is localized. A safer generalization is not a stronger centerline pull.
Use a road factor that:

- penalizes lateral distance, while leaving along-road motion weakly constrained;
- compares velocity/IMU heading with road tangent;
- keeps multiple reachable road hypotheses through intersections;
- applies switchable/abstaining factors when map coverage, road class, or mode
  dominance is weak;
- uses lane/drivable polygons when available instead of assuming centerline is
  truth.

Evaluate on untouched runs and on predeclared contiguous spans. Do not tune a
trigger on Nagoya run2 and report the same run as transfer evidence.

### P2 research experiment: faithful Recurrence Vector and subset clustering

The candidate-3DMA work first tested multipivot, robust subsets, temporal
satellite-bias removal, PLATEAU clustering, and OSM dual gating. Pseudorange-only
variants did not beat production.  The later `recurrence_vector` branch now
implements every step exposed by the 2025 paper's public ION abstract: actual
four-satellite subset solutions, candidate recurrence-vector differencing, LOS
projection, probabilistic signal-type/visibility comparison, and cumulative
argmax selection.  Because the proceedings PDF is credit-gated, exact
paper-internal numeric parameter fidelity is not claimed.  Safe-gated and raw
counterfactual evaluations remain separated, and neither may feed production
unless it beats Phase71 on untouched runs with a production-like persistence
gate.

## Evaluation and code-quality findings

`train_nr2_pb40_nonref_ranker.py` computes a five-fold time prediction but writes
the same-run `p_pass_lgb_insample` values into the output. The word
"non-reference" describes its feature contract, not an out-of-sample result;
the labels still come from reference truth. This script may be experimental,
but its exported predictions are optimistic by construction.

Required policy:

- export only blocked-time, leave-one-run-out, or leave-one-route-out predictions;
- tune thresholds inside the training fold, not on the reported fold;
- group adjacent epochs and all candidates from an epoch in the same fold;
- separate "competition score" from "algorithm transfer" reports;
- mark any truth-derived artifact in filenames and schemas;
- require per-run, per-span, tail-error, coverage, and abstention metrics.

The ~10k-line `exp_ppc_ctrbpf_fgo.py` also makes inert knobs and accidental
phase coupling likely. Split it gradually into observation construction,
posterior update, mode extraction, emission policy, and evaluation modules.
This is primarily a reliability improvement, but it will make honest ablations
much easier.

## Methods not worth repeating unchanged

The repository already has negative or exhausted evidence for:

- generic NLOS downweighting/exclusion;
- Hatch smoothing and standalone pseudorange-domain corrections;
- more robust-loss/IRLS sweeps on the same absolute pseudorange model;
- naive LAMBDA/ambiguity threshold sweeps;
- TC-FGO anchor-parameter sweeps;
- first-order reflection features and simplified UTD/bridge features;
- absolute pseudorange candidate likelihood without a strong clock/common-bias
  treatment;
- more same-run selector/ranker threshold search.

Full ray-traced delay likelihood is possible but expensive and has a low prior
after the reflection-feature null results. Direct position estimation is not a
PPC option with RINEX-only observations because it needs correlator/raw signal
data.

## Paper-to-code map

| Paper/method | Mechanism taken from the paper | Current repository implementation | Current evidence / disposition |
|---|---|---|---|
| Ng et al., grid 3DMA clustering + Doppler FGO (2025) | Avoid averaging separated modes; select temporally consistent clusters | `particle_modes.py` plus abstaining diagnostic/emission wiring in `exp_ppc_ctrbpf_fgo.py` | Full-six/blocked diagnostics complete; accepted modes improve only 48.3% and worsen mean error by 0.044 m, so emission stays off |
| Godsill, Doucet & West, FFBSi (2004) | Forward particle filtering followed by backward trajectory simulation | `particle_ffbsi.py`, `particle_fixed_lag.py`, ancestry export, mode masks, and direction-consistent reverse Doppler replay | Full-six/blocked lag-10/path-8 replay complete; mean improves 0.012 m but worsening-tail delta p95 is +0.803 m, so research opt-in only |
| Chai et al., tightly coupled RTK/INS PAR (2025) | INS-aided sequential partial ambiguity resolution in blocked urban conditions | Operational `dd_imu_bridge.cpp`: joint DD code/carrier update, ambiguity covariance, slip lifecycle, partial AR, and gated bounded soft reset | 12/12 directly linked C++ tests pass; Tokyo1 development improves p50/p95/p99, while the remaining independent replay matrix is active |
| Bai et al., WCP-FGO (2021/2022) | Stack a slip-free carrier window and eliminate common ambiguities with a left-nullspace projection | `wcp_factor.py` integrated into TC-FGO continuous arcs | Full-six pooled score improves 0.644545→3.448119%, but remains 82.757373 points below Phase71 with mixed Nagoya tails; research-only |
| Xia et al., integrity-constrained FGO (2024) | Latent switches suppress inconsistent pseudorange factors without hard satellite deletion | `switchable_factor.py`; ordinary flag is a geometry-preserving shadow, committed switching is experimental-only | All full/blocked shadow pairs preserve exact positions and metrics over 2,275,766 full-run rows; committed switching was catastrophic and remains rejected |
| Signal/constellation-aware Doppler model | Resolve frequency per observation code and estimate observable constellation clock groups | `doppler_signals.py`, mixed G/E/J/C/R normalization, GLONASS FDMA metadata abstention, grouped clock WLS | GEJCR improves position p95/p99 but worsens p50 and Doppler fit/drift tails; retain GEJ default and reject unconditional C/R |
| Lee et al., Recurrence Vector (2025) | Four-satellite subset solutions, recurrence differences, LOS projection, visibility-probability comparison | `candidate_3dma.py` recurrence-vector strategy with safe gate plus separately named raw counterfactual | Safe full-six abstains on all 38,749 evaluated epochs and preserves 29.533785%; raw accepts all, worsens 35,972 versus 2,726 improvements, and collapses pooled score to 0.059943%; reject emission |
| Gutierrez et al., measurement-subset clustering (2024) | Cluster consistent subset solutions after availability filtering | Candidate confidence/subset diagnostics used in the bounded 3DMA research path | Retained as diagnostic support; no production adoption without untouched-run gain |

Primary sources:

- https://doi.org/10.1017/S0373463325000220
- https://doi.org/10.1198/016214504000000151
- https://doi.org/10.3390/electronics14132712
- https://arxiv.org/abs/2109.00683
- https://doi.org/10.33012/navi.660
- https://doi.org/10.33012/2025.20423
- https://doi.org/10.33012/2024.19605

## Recommended next experiment

Start with **mode-aware PF emission**, because it is the fastest high-value test
and reuses the current particle cloud. It should be a small, auditable milestone:

1. implement a pure CPU weighted-mode extractor with synthetic bimodal tests;
2. add diagnostic-only mode logging to PPC with no trajectory changes;
3. replay all six official runs and measure how often the global mean lies in a
   low-density valley or outside the dominant mode;
4. enable selected-mode emission only behind an abstaining dominance/separation
   gate;
5. evaluate all six runs plus blocked spans, then decide whether to integrate
   FFBSi mode smoothing.

In parallel planning, scope the DD-RTK/IMU bridge as the next structural
milestone. It is more likely to close the inuex35 canyon gap, but it should not
be mixed into the smaller PF emission experiment.
