# Improvement 1--7 completion audit (2026-07-13)

This is an evidence checklist, not a completion claim. A row is complete only
when the referenced artifact covers the scope in that column.

| Item | Implementation evidence | Unit / synthetic evidence | Real-data evidence | Remaining proof |
|---|---|---|---|---|
| 1. Weighted PF modes | `particle_modes.py`; PPC diagnostic/emit/abstain wiring | mode and emission suites | six-run 2k-particle diagnostic complete; 57,398 holdout epochs, 47.1% abstention, accepted emission improves 48.3% and worsens mean by 0.044 m | emission rejected; keep default off and diagnostic-only support |
| 2. Fixed-lag FFBSi | `particle_ffbsi.py`, `particle_fixed_lag.py`; marginal/genealogy and terminal masks; full backward replay reverses receiver velocity, satellite velocity, and Doppler together | FFBSi/fixed-lag/smoother suites, including exact reverse-time Doppler arrays; counterfactual-diagnostic regression test | full-six 2k-particle lag-10/path-8 replay and all blocked scopes complete; 57,398 holdout epochs, 34,943/57,371 applied (39.09% abstention), 50.66% improved, mean delta -0.0124 m but worsening-tail delta p95 +0.803 m; five blocked holdouts split 3 improve/2 worsen | default emission rejected; retain opt-in research/diagnostic support because benefit is inconsistent and standalone PF remains 0% |
| 3. DD-RTK/IMU ESKF | operational `fusion/dd_imu_bridge`: augmented INS/ambiguity covariance, joint DD update, partial AR, slip/arc retirement, positive-definite innovation checks, patient bounded soft reset, and carrier-to-code fallback; multi-GNSS/reference/signal-aware DD ambiguity keys; `RTKProcessor::formTightlyCoupledObservations` assembles real rover/base RINEX DD rows at the propagated INS ECEF position; `LooseCouplingProcessor::processTightlyCoupledDD` owns propagation/cross-covariance synchronization; `--tight-dd-imu` commits code DD while shadow-testing carrier candidates, and `--tight-dd-carrier-experimental` explicitly enables carrier/ambiguity/PAR commits | 12 directly linked bridge tests plus live fusion-state and runner tests pass; the complete app builds under MSVC 19.44 with Eigen 5.0.1 and C++20; all full/blocked artifacts match executable SHA-256 `f6c32e…fd6021` and pass count/coverage/tail/runtime consistency audit | all-six pooled score improves 32.246385→33.726454% (+1.480069). Four of five untouched full holdouts improve score, while Nagoya/run2 worsens 5.741607→5.567837%; blocked holdouts show three positive and one zero score delta, with mixed central/tail behavior. Across full runs it processes 57,004 DD epochs, accepts/rejects 33,729/23,275, emits 1,197,961 rows, makes 21,766 carrier fallbacks and 621 soft resets, but fixes zero ambiguities | research-only/opt-in: pooled performance remains 52.479038 points below Phase71, one full holdout regresses, and PAR never activates; omit both tight flags from Phase71 production and keep experimental carrier commits rejected |
| 4. WCP | whitened carrier design and left-nullspace projection in `wcp_factor.py`; slip-free arc integration in TC-FGO | WCP and TC-FGO tests, including automatic geometry-corrected slip split; complete 24-row full-six and blocked matrices | Full-six pooled score improves 0.644545→3.448119% (+2.803574) with 1,341,642 WCP residual rows; all five full holdouts improve score. Tokyo2/3 improve by 4.4996/4.3186 points. Nagoya1/2 improve by 3.7174/2.0523 points but have mixed central/tail behavior. Nagoya3 improves 0→0.000127% and p50/p95 100.66/312.41→89.69/303.34 m, while p99 worsens 368.62→368.76 m; its four causal static FIX epochs never exceed the 1 m/s heading threshold. Blocked holdouts improve pooled 0→6.0093% | research-only: score remains 82.757373 points below Phase71, Nagoya tails are inconsistent, and Nagoya3 heading is unobservable; omit `--wcp` from production |
| 5. Switchable pseudorange | analytically eliminated switch residual/Jacobian, switch-value/downweighted-row diagnostics, and a geometry-preserving integrity gate; committed switching is isolated behind `--switchable-pseudorange-experimental`, while the ordinary flag shadow-evaluates then emits the baseline factor exactly; this is not a claim to reproduce Xia et al.'s full batch chi-square factor or protection-level calculation | focused switch/TC-FGO/runner suite passes; synthetic shadow residual and Jacobian exactly match baseline; artifact audit requires exact position SHA-256 and metric identity for both shadow pairs | the original Tokyo/run1 committed switch collapsed geometry (p50 112 km), and a gated 1,000-epoch retry also regressed, so threshold tuning stopped. Across every full and blocked scope, baseline/switch and WCP/WCP+switch positions and metrics are exactly identical. Full shadows evaluate 2,275,766 rows over 291,254 shadow epochs; the integrity gate abstains from every commit while preserving diagnostics | retain only opt-in shadow telemetry; reject committed switching and omit both switch flags from Phase71 production |
| 6. Signal-aware Doppler | per-code G/E/J/C/R wavelength resolution, GLONASS FDMA abstention, constellation clock-drift WLS/normalization | all-system Doppler suites | matched GEJ and GEJCR six-run/blocked summaries cover 57,398 holdout epochs at 100% diagnostic-update coverage; GEJCR raises mean fitted clock groups 2.9970→3.9970 and improves position p95/p99 by 1.001/1.649 m, but worsens p50 by 0.0649 m, Doppler fit-RMS p95 by 0.0658 m/s, and drift-span p95 by 0.2308 m/s; both standalone RBPF variants score 0% | retain code/signal-aware model and diagnostics; keep staged GEJ as production default and reject unconditional C/R enablement until a competitive estimator supplies adoption evidence |
| 7. Recurrence Vector | four-satellite solves, recurrence projections, visibility classification probability and subset accumulation | candidate-3DMA, resumable/parallel scope runner, terminal-source coverage handling, policy-provenance, finite-population and arithmetic-consistency tests pass; safe/raw outputs are separately named and audited | safe full-six evaluates 38,749 epochs at 48.01–83.87% per-run coverage, abstains on all 38,749, and preserves pooled source score 29.533785%. Raw full-six accepts every evaluated epoch, improves 2,726, worsens 35,972, and collapses pooled score to 0.059943%; every run scores at most 0.118453%. Safe blocked preserves 56.656641%, while raw blocked worsens 593/595 and collapses to 0.276064% | safe gating is non-regressing but completely inactive and remains diagnostic-only; raw emission is rejected by both full-six and blocked evidence; exact paper-internal parameter fidelity is not claimed because the full paper is access-controlled |

## Evaluation separation

- PF full-run decisions use epoch 200 onward (`run_holdout` / `pooled_holdout`),
  not the development prefixes.
- Recurrence's initial model/gates used Nagoya/run1 as development.  The later
  0.05 confidence gate used Tokyo/run1 full development evidence, so both
  Nagoya/run1 and Tokyo/run1 are now conservatively labelled development;
  Nagoya/run2-3 and Tokyo/run2-3 are the four untouched-run holdouts.
- TC-FGO's Nagoya/run2 segment was inspected for five-epoch start-index
  validation and is therefore labelled `development_smoke`; Nagoya/run1 is
  development and the other four blocked spans remain unseen holdout.
- Nagoya/run3's declared TC-FGO span has only four static RTK FIX epochs in
  the causal history available through the end of that span.  The runner now
  records all four variants as explicit
  `insufficient_causal_static_fix_history` abstentions (zero coverage and zero
  honest score over 232.94 m), rather than borrowing a later FIX or aborting
  the remaining evaluation queue.
- The full Nagoya/run3 replay likewise has only four static RTK FIX epochs in
  the complete causal trajectory. Its runner therefore uses all four for
  phase-1 initialization, records `phase_init_static_fixes=4` in every variant,
  and leaves the other five official runs fixed at five. This is a declared
  data-feasibility exception, not a performance-selected parameter; the
  artifact audit enforces the exact per-scope protocol. All four displacements
  are below the existing 1 m/s heading threshold, so phase 2 cannot initialize
  yaw from RTK FIX velocity and retains the initializer's fallback attitude;
  this observability limitation must count against adoption.
- Runtime measured under deliberate concurrent load is retained as a factual
  measurement but is not used for cross-method speed ranking.

## Phase71 reproducibility

The tracked Phase11fa manifests are recoverable. The three ranker prediction
CSVs, Phase57 Nagoya/run2 diagnostics, three Phase10 candidate directories and
19 Phase19 GICI candidate directories are absent from both the worktree and
repository history. Therefore 86.205492% is a documented canonical reference,
not a freshly replayed baseline, until those artifacts are regenerated from a
compatible candidate pool.

All structural runners now compute the same honest distance-weighted 0.5 m
PPC score used for Phase71 comparisons. Missing output epochs contribute no
pass distance while the full requested reference distance remains in the
denominator. Completed Tokyo/run1 TC-FGO development telemetry gives 0.1344%
for baseline and 0.2194% for WCP, far below the canonical 86.205492% despite
WCP's relative improvement.

## Build and artifact audit

- The root CUDA/C++ project now requires C++20 for both host C++ and CUDA C++
  (`CMAKE_CXX_STANDARD=20`, `CMAKE_CUDA_STANDARD=20`, required, extensions
  disabled).  A clean MSVC 19.44 / CUDA 12.8 Release rebuild of
  `gnss_gpu_core` completed with NVCC visibly using `-std=c++20` for all four
  core translation units.
- The current improvement-focused Python regression is 136/136 passing. The
  standalone executable linked directly from `test_dd_imu_bridge.cpp` runs
  12/12 DD/IMU tests successfully; the app build directory itself has no
  CTest registrations, so a zero-test `ctest` invocation is explicitly not
  counted as evidence.
- Repository-wide pytest discovery now collects 3,606 tests with zero
  collection errors.  The prior ROS package-path, missing UrbanNav harness,
  optional pybind11 extension, duplicate module-name, and broken external
  `xonsh` plugin issues are handled explicitly.  The full run completed in
  453.25 s with 3,326 passed, 180 skipped, 26 passing subtests and 90 failures.
  Those failures are outside items 1--7 and are classified rather than hidden:
  50 require gnssplusplus CLI/package build products at the test's expected
  path, 15 require a working Windows PROJ/geoid-grid environment, 11 are
  pre-existing zero-spread CUDA-wrapper tests against an unchanged positive
  input validator, 12 are GSDC external-artifact/Windows-path tests, one needs
  the absent product prediction CSV already identified in the Phase71
  reproducibility gap, and one was a PF-forward mock-signature mismatch at the
  new `tow` propagation boundary.  That mock now accepts and asserts the
  propagated TOW; its 11 adjacent tests pass.  The prior full run has not been
  relabelled, while the item-focused 138 tests and direct 12 C++ bridge tests
  remain green.
- `audit_improvement_1_7_artifacts.py` is fail-closed over the required
  scope/variant matrices and non-empty metric fields.  In addition to the
  12-file source/C++20/production-policy contract, it requires exact position SHA-256 and metric
  identity between each safe shadow pair (`baseline`/`switch` and
  `wcp`/`wcp_switch`), finite decision scores, and existing decision-evidence
  artifacts.  It also validates the six official full-run score rows
  for PF modes, FFBSi, GEJ Doppler and GEJCR Doppler, including coverage,
  honest pass/total distance, score and runtime. Their score must equal
  `100 * pass_distance / total_distance`, coverage must lie in 0--100%, and
  distance/runtime values must be physically ordered and nonnegative. Duplicate/extra scopes and
  non-finite numeric values are rejected.  TC-FGO, tight DD/IMU, and
  Recurrence summaries also require their method-specific development/holdout
  roles to match the manifest; Recurrence additionally requires coverage,
  p50/p95/p99 tails, explicit abstention counts/rates, runtime, and the
  declared safe/raw source, confidence, and boundary policies. Coverage and
  acceptance must also agree arithmetically with requested, evaluated, and
  abstained epoch counts. Tight DD/IMU likewise requires p50/p95/p99 and checks
  coverage against emitted/requested epochs, accepted+rejected against DD
  epochs, bounded diagnostic counts, and zero tight diagnostics on baselines.
  TC-FGO similarly checks requested/output/evaluated ordering, coverage and
  runtime normalization, pass-rate bounds, variant-specific zero diagnostics,
  and switch-row count bounds; an output-zero blocked span may leave only the
  mathematically undefined per-output runtime non-finite.
  Matrix checks now reject unexpected and duplicate `(scope, variant)` rows;
  diagnostic-only base scopes are enumerated explicitly rather than accepted
  as silent extras.
  A separate seven-row decision ledger
  is also fail-closed: every item must name its evidence, production decision,
  integrated configuration, and exact canonical Phase71 reference; any
  `pending` row keeps the audit red.  Those four official
  full-run checks and their aggregate/blocked checks pass; the audit remains
  intentionally non-zero until the active TC-FGO, tight DD/IMU and recurrence
  queues produce their complete matrices.

## Provisional integrated configuration

The evidence-supported production configuration remains Phase71 with staged
G/E/J Doppler (`--doppler-systems G,E,J`).  None of the new structural methods
is allowed to change production emission while its independent full-six matrix
is incomplete or inconsistent.

The Phase71 production script now spells out `--pf-mode-policy off` and
`--doppler-systems G,E,J` instead of relying on mutable parser defaults.  The
source-contract audit rejects accidental production use of FFBSi, tight DD,
experimental carrier, WCP, either switch mode, or Recurrence Vector.

- PF modes: keep `--pf-mode-policy off`; `diagnostic` is safe for telemetry,
  while `emit` is rejected by the full-six counterfactual.
- FFBSi: keep `--enable-pf-ffbsi-smoother` absent in production; retain the
  lag/path controls as research-only because mean gain and worsening tail
  disagree.
- Tight DD/IMU: omit both tight flags from Phase71 production. The safe code-DD
  path remains an opt-in research ablation; committed carrier/ambiguity updates
  remain rejected.
- TC-FGO: omit `--wcp` and both switch flags from Phase71 production. WCP is a
  research-only ablation; `--switchable-pseudorange` remains safe opt-in shadow
  telemetry, while `--switchable-pseudorange-experimental` stays rejected.
- Doppler: retain signal-aware code/frequency resolution and grouped clocks,
  but do not promote unconditional C/R beyond the validated G/E/J default.
- Recurrence Vector: safe diagnostics use a 20 m source-projection gate,
  boundary abstention, and a recorded 0.05 minimum selected probability.  The
  separately named raw counterfactual sets the source/probability gates to
  zero and permits boundary maxima; it can never feed production emission.
  Resume checks reject missing/mismatched policy metadata and epoch-CSV
  population mismatches, while accepting declared terminal source-data gaps.
  Twenty-five early full-run chunks with pre-policy metadata were excluded and
  replayed rather than silently mixed into the safe aggregate.

All seven method decisions are final. None displaces the canonical Phase71
production configuration; safe diagnostics and research ablations remain
available only behind explicit non-production flags.
