# Paper Draft (2026-04-01)

This file is a paper-writing starter aligned with the current literature audit and the
current state of the experiments. It is intentionally conservative: it does not claim
results that have not yet been demonstrated by the repository outputs.

## Working Title

GPU-Resident Particle Filtering and BVH-Accelerated 3D Likelihoods for Robust Urban GNSS Positioning

## Abstract Draft

Urban GNSS positioning remains brittle in dense city streets because non-line-of-sight
reception, building shadowing, and temporary low satellite availability produce
multi-modal measurement likelihoods and occasional catastrophic failures. Prior work has
already studied GNSS particle filters, 3D-mapping-aided GNSS ranging, shadow matching,
and GPU-accelerated large-scale particle filtering in adjacent domains. The contribution
we target in this work is therefore not a single "first" component, but a practical
combination: a GPU-resident particle-filtering framework for urban GNSS, a robust
observation model for heavy-tailed pseudorange behavior, an optional 3D ray-tracing
likelihood path accelerated by BVH, and a tail-aware real-data evaluation protocol.
Across six PPC holdout segments, the best exploratory gate improves mean RMS horizontal
error from 66.92 m to 65.54 m while keeping mean p95 nearly unchanged at 81.69 m to
81.22 m, indicating that the PPC gate contribution is measurable but modest. The
stronger empirical result comes from external validation on UrbanNav Tokyo with trimble
observations and `G,E,J` multi-GNSS measurements, where `PF+RobustClear-10K` achieves
66.60 m mean RMS horizontal error and 98.53 m mean p95, outperforming `EKF` at
93.25 m and 178.18 m while reducing the >100 m failure rate from 16.29% to 4.80% and
the >500 m catastrophic rate from 0.161% to 0.000%. Separately, on a real PLATEAU
subset, BVH acceleration preserves PF3D accuracy while reducing runtime from
1028.29 ms/epoch to 17.78 ms/epoch, a 57.8x speedup. These results support a paper
positioned around robust urban GNSS inference, honest cross-dataset evaluation, and
practical 3D-likelihood systems acceleration rather than a claim of a single
unprecedented algorithmic primitive.

## Introduction Draft

### 1. Motivation

Reliable urban GNSS positioning remains difficult because the measurement process is not
well described by a single narrow Gaussian error model. In dense city streets,
pseudorange measurements are corrupted by non-line-of-sight reception, partial
occlusion, building-induced multipath, and intermittent reductions in visible
satellites. These effects create multi-modal or heavy-tailed likelihoods that are hard
to handle with point-estimate solvers and fragile under low-satellite conditions.

This failure mode is visible in the current PPC-Dataset baseline results in this
repository. Across six full real-data runs, the best multi-constellation WLS
configurations (`G,E` or `G,E,J`) achieve mean p95 horizontal errors of about 102.7 m,
but their mean RMS errors still rise to 1080-1221 m because a small number of
catastrophic failures dominate the average. The worst epochs reach roughly 20 km error,
and the largest concentration of severe failures occurs when only 4-6 satellites are
available. These observations make two points clear: urban GNSS needs inference methods
that can represent ambiguity more faithfully than local least-squares updates, and the
evaluation must report tail behavior rather than RMS alone.

The repository now also contains an external result that matters more than the original
PPC-only picture. On UrbanNav Tokyo with trimble observations and repaired `G,E,J`
measurement loading, `PF+RobustClear-10K` achieves 66.60 m mean RMS horizontal error
and 98.53 m mean p95 across Odaiba and Shinjuku, while `EKF` remains at 93.25 m and
178.18 m. The outlier rate above 100 m falls from 16.29% for `EKF` to 4.80%, and the
catastrophic rate above 500 m falls from 0.161% to 0.000%. This means the paper no
longer needs to rely only on a diagnostic PPC story. It now has an external validation
result in which the PF family is genuinely better.

### 2. Position Relative to Prior Work

The goal of this project is not to claim the first GNSS particle filter, the first urban
GNSS particle filter, or the first use of 3D building models in GNSS positioning. Those
claims are contradicted by prior work. GNSS particle-filter positioning has been studied
by Suzuki and by Gupta and Gao. 3D-mapping-aided GNSS and shadow matching have been
studied by Groves and by Adjrad and Groves, and earlier PF-based 3D map integration also
exists in Suzuki and Kubo. Meanwhile, Koide et al. demonstrated that GPU-accelerated
Stein particle filtering at the one-million-particle scale is feasible in LiDAR
localization, providing strong adjacent prior art for the systems angle.

What remains open is the combination relevant to this repository: a GNSS-oriented system
that keeps particle state resident on the GPU, evaluates heavy-tailed and optionally
3D-aware likelihoods for urban measurements, and analyzes robustness on real urban GNSS
runs using tail-aware metrics. That is the line this paper should defend.

### 3. Technical Thesis

Our technical thesis is that urban GNSS can benefit from combining three ingredients.
First, GPU-resident particle inference can represent ambiguous or multi-modal posterior
structure that is poorly handled by local solvers. Second, robust observation modeling is
necessary because urban pseudorange errors are heavy-tailed even when explicit blocked
satellite reasoning is not active. Third, 3D map reasoning should be made practical by a
systems path that keeps the 3D likelihood affordable enough to test on real city-model
subsets.

The repository now contains evidence for each axis. PPC holdout experiments show that the
strategy-gate family survives holdout only with small gains, so it should be treated as a
secondary result rather than the paper's main novelty. UrbanNav external evaluation shows
that the PF family with a robust clear-mixture observation model is the strongest current
accuracy result. Real-PLATEAU PF3D experiments show that BVH acceleration makes the
3D-likelihood path practical even when accuracy stays unchanged on that short subset.

### 4. Claimed Contributions

The paper should frame its contributions as follows.

1. A GPU-resident particle-filtering implementation for urban GNSS with practical runtime
   characteristics on real datasets.
2. A robust urban-GNSS observation path, including the `PF+RobustClear` variant and a
   reusable multi-GNSS quality-veto utility for stabilizing raw multi-GNSS WLS.
3. A BVH-accelerated 3D ray-tracing likelihood path that preserves PF3D accuracy while
   reducing runtime by 57.8x on a real PLATEAU subset.
4. A robustness-oriented empirical protocol that separates PPC design/holdout from
   UrbanNav external validation and reports percentile, outlier-rate, catastrophic-rate,
   and failure-segment metrics rather than RMS alone.

### 5. Evaluation Framing

The empirical section should be organized around three questions.

First, on PPC holdout, how much gain is left after enforcing holdout discipline on the
strategy-gate family? Second, on UrbanNav external validation, does the PF family still
beat classical baselines when run without UrbanNav-specific retuning? Third, does BVH
make the 3D-likelihood path practical without changing PF3D accuracy on the tested real
subset?

Because PPC and UrbanNav both expose tail behavior, the paper should report at least the
following metrics for every method: p50, p95, RMS, outlier rate above 100 m, catastrophic
rate above 500 m, and longest continuous failure-segment duration. Runtime per epoch
should be reported alongside these metrics so that robustness gains are not presented
without cost.

## Related Work Draft

### GNSS Particle Filtering

Particle filtering for GNSS positioning is not new. Suzuki's multiple-update particle
filter shows that direct GNSS particle-based positioning remains an active topic,
including urban vehicle experiments and the use of pseudorange and carrier-phase
observations. Gupta and Gao further show that particle-filter-based GNSS localization can
be framed around reliability against multiple faults, which is especially relevant for
urban canyon behavior and integrity-style analysis. These papers mean that our novelty
cannot be framed as "particle filtering for GNSS" in the abstract. Instead, the relevant
question is whether large-scale GPU execution, explicit 3D likelihood modeling, and
robustness-oriented evaluation materially change what is practical.

### 3D-Mapping-Aided GNSS and Shadow Matching

The 3DMA GNSS literature also predates this project by a wide margin. Groves introduced
shadow matching as a technique for urban canyons using 3D city models, establishing that
predicted satellite visibility from map geometry is prior art rather than a new
contribution. Adjrad and Groves later integrated shadow matching with 3D-mapping-aided
GNSS ranging, demonstrating that map reasoning and ranging information can already be
combined in a single positioning framework. Earlier PF-based map integration also appears
in Suzuki and Kubo, while later UCL work, including Zhong's thesis, discusses multi-epoch
filtering and particle-filter perspectives. Accordingly, this paper should not claim the
first use of 3D models, the first real-time 3DMA GNSS system, or the first integration of
shadow matching and GNSS ranging.

### Large-Scale GPU Particle Filtering

On the systems side, MegaParticles is the strongest adjacent prior art. Koide et al.
show that GPU-accelerated Stein particle filtering can scale to the one-million-particle
regime in range-based LiDAR localization. That result matters because it removes any
credible claim that GPU + SVGD + million-particle scale is broadly unprecedented.
However, the state space, likelihood model, and failure modes in GNSS differ from those
in LiDAR localization. A defensible systems contribution remains available if we show that
the same scale can be realized for GNSS while integrating satellite geometry, clock-bias
structure, and 3D map-aware likelihoods.

### Our Intended Positioning

The cleanest way to position this paper is therefore as a combined contribution across
three axes: robust urban inference, systems acceleration, and empirical discipline.
Robust urban inference covers the PF family and the robust observation path that now wins
on UrbanNav external validation. Systems acceleration covers GPU-resident execution and
BVH-accelerated 3D likelihood evaluation. Empirical discipline covers PPC holdout,
UrbanNav external validation, catastrophic-tail analysis, and honest separation between
main results and auxiliary utilities. Framing the paper this way keeps it aligned with
the literature while preserving a strong and defensible contribution statement.

## Results Draft

### 1. Experimental Organization

The empirical section should be deliberately split by role, not by implementation
module. PPC is the design-and-holdout dataset. It is where we analyze catastrophic
tails, compare gate families under the same dump contract, and decide which exploratory
ideas survive holdout. UrbanNav is the external validation dataset. It is where we test
whether the resulting PF family still improves over classical baselines without
UrbanNav-specific retuning. Finally, the PF3D versus PF3D-BVH comparison is a systems
experiment that isolates the cost of explicit 3D likelihood evaluation on a real PLATEAU
subset.

The paper-ready summary of these three roles is already fixed in
[paper_main_table.md](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_main_table.md).

### 2. PPC Holdout Result

The PPC holdout result should be presented as a discipline result rather than a headline
accuracy result. The safe baseline `always_robust` achieves 66.92 m mean RMS horizontal
error and 81.69 m mean p95, while the best exploratory gate
`entry_veto_negative_exit_rescue...` reaches 65.54 m and 81.22 m. This is a real but
small improvement. The right interpretation is not that the gate family is the core
scientific contribution. The right interpretation is that the repository now enforces an
experiment process in which only small, holdout-surviving gains are retained, and larger
but brittle tuned gains are rejected. The paired segment comparison is already rendered in
[paper_ppc_holdout.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_ppc_holdout.png).

### 3. UrbanNav External Validation Result

The main empirical claim should come from UrbanNav external validation with trimble
observations and `G,E,J` measurements. Under this fixed setting, `PF+RobustClear-10K`
achieves 66.60 m mean RMS horizontal error and 98.53 m mean p95, while `PF-10K` reaches
67.61 m and 101.46 m. Both PF variants clearly outperform `EKF`, which remains at
93.25 m mean RMS and 178.18 m mean p95. The failure-rate differences are even more
important: `EKF` has a 16.29% >100 m rate and 0.161% >500 m rate, whereas
`PF+RobustClear-10K` reaches 4.80% and 0.000%, respectively. The per-run numbers show
that this is not a single-sequence artifact: on Odaiba, `PF+RobustClear-10K` reaches
61.86 m RMS versus 89.42 m for `EKF`; on Shinjuku, it reaches 71.33 m versus 97.07 m.

This is the cleanest current evidence that the PF family matters beyond PPC. The figure
that should visualize this result is
[paper_urbannav_external.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_urbannav_external.png),
which combines a CDF view with tail-rate bars. That figure makes it easy to defend the
claim that the PF family is not merely better in average error, but materially better in
tail behavior on the external dataset.

One remaining concern is whether this gain is concentrated in only one favorable part of
each run. The current repository now has a fixed-window analysis over the same external
epoch dump. With 500-epoch windows and 250-epoch stride, `PF+RobustClear-10K` beats
`EKF` in 90 of 127 windows by RMS and 102 of 127 windows by p95, while matching or
improving the >500 m catastrophic rate in all 127 windows. This does not create new
geographic diversity, but it does make the external result harder to dismiss as a lucky
single-interval artifact.

### 4. WLS+QualityVeto as a Utility, Not the Main Method

One tempting but misleading narrative would be to present `WLS+QualityVeto` as the main
answer to UrbanNav multi-GNSS instability. The data do not support that. The promoted
quality-veto hook improves raw multi-GNSS WLS from 3179.37 m mean RMS to 2933.77 m and
reduces the catastrophic rate from 2.909% to 2.552%, but it is still far from the PF
results. The correct role of this component is therefore architectural rather than
headline empirical: it is a minimal reusable core utility that stabilizes the measurement
path and gives us a defensible shared interface between experiments and production-style
code. It should appear in the method and implementation sections, and possibly in a
supporting ablation table, but not as the main claimed result.

### 5. BVH Systems Result

The 3D map path is currently strongest as a systems result. On the real PLATEAU subset,
`PF3D-10K` and `PF3D-BVH-10K` match exactly in accuracy at 55.50 m RMS and 58.39 m p95,
but the runtime drops from 1028.29 ms/epoch to 17.78 ms/epoch with BVH, a 57.8x speedup.
This is the cleanest way to defend the explicit 3D-likelihood implementation in the
current paper. We should not overstate it as a real-data accuracy gain, because the
current external accuracy leader is `PF+RobustClear`, not the full map-aware path. But as
a systems contribution, the result is strong and easy to communicate. The intended figure
for this section is [paper_bvh_runtime.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_bvh_runtime.png).

## Venue-Shaped Section Plan

If this draft is converted into a submission manuscript now, the safest structure is:

1. Introduction.
   Lead with brittle urban GNSS tails, not with a claim of a novel primitive.
2. Related Work.
   Keep the scope tight: GNSS PF, 3DMA GNSS, and GPU PF systems.
3. Method.
   Put `PF+RobustClear` first, then the optional 3D BVH path, then the shared
   multi-GNSS quality-veto utility.
4. Experimental Protocol.
   Explicitly separate PPC design/holdout from UrbanNav external validation and
   state that all main figures come from fixed CSV snapshots.
5. Results.
   Use Table 1 first, then the UrbanNav figure as the main empirical result, then
   PPC holdout as a discipline/ablation result, and finally BVH as a systems result.
6. Discussion and Limitations.
   State directly that the external winner is `PF+RobustClear`, not the full
   PLATEAU path, and that the PPC gate gain is modest.
7. Conclusion.
   Close on robustness, systems practicality, and honest evaluation.

This ordering keeps the strongest evidence near the front and pushes the weaker but
still valuable PPC gate story into a supporting role.

## Table and Figure Captions Draft

The current generated caption pack is
[paper_captions.md](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_captions.md).
The shortest safe versions are below so they can be pasted into a manuscript without
recomputing the prose.

### Table 1

Main quantitative summary of the paper. PPC holdout is reported as a design-discipline
result rather than a headline accuracy claim. UrbanNav external uses fixed
`trimble + G,E,J` settings without UrbanNav-specific retuning. `PF+RobustClear-10K`
is the strongest external method, improving mean RMS from 93.25 m (`EKF`) to 66.60 m
and mean p95 from 178.18 m to 98.53 m, while reducing the >100 m rate from 16.29% to
4.80% and the >500 m rate from 0.161% to 0.000%. `WLS+QualityVeto` is included as a
promoted core utility, not as the main external method. BVH rows isolate runtime on a
real PLATEAU subset and show unchanged PF3D accuracy with large acceleration.

### Figure 1

Segment-wise PPC holdout comparison between `always_robust` and the best exploratory
gate `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`.
The gain survives holdout but remains modest: mean RMS decreases from 66.92 m to
65.54 m and mean p95 decreases from 81.69 m to 81.22 m. This figure supports the
paper's experiment-discipline claim rather than the main accuracy claim.

### Figure 2

UrbanNav external validation on Odaiba and Shinjuku using trimble observations and
`G,E,J` measurements. Left: empirical CDF of 2D error. Right: rates of failures above
100 m and catastrophic failures above 500 m. `PF+RobustClear-10K` achieves 66.60 m
mean RMS and 98.53 m mean p95, outperforming `EKF` at 93.25 m and 178.18 m.
Relative to `EKF`, the >100 m rate falls from 16.29% to 4.80% and the >500 m rate
falls from 0.161% to 0.000%.

### Figure 3

Runtime and accuracy on the real PLATEAU PF3D subset. `PF3D-BVH-10K` preserves the
same accuracy as `PF3D-10K` at 55.50 m RMS and 58.39 m p95 while reducing runtime
from 1028.29 ms/epoch to 17.78 ms/epoch, a 57.8x speedup. This figure should be framed
as a systems contribution rather than a real-data accuracy gain from explicit 3D
reasoning.

## Discussion / Limitations Draft

The strongest limitation is also the cleanest honesty point. The paper now has a real
external win for the PF family, but the winning external method is `PF+RobustClear`, not
the explicit 3D ray-tracing path. Accordingly, the paper should not imply that the
PLATEAU-based LOS/NLOS likelihood is the direct cause of the UrbanNav accuracy gain. The
causal story is narrower: robust observation modeling and GPU-resident PF inference are
already enough to beat classical baselines on the repaired multi-GNSS UrbanNav setting,
while BVH makes the map-aware extension computationally practical.

The second limitation is that the PPC gate family, although well-controlled and
holdout-validated, contributes only a small numerical gain. This means the gate result is
best framed as a process and ablation success, not as the paper's headline method.

The third limitation is external breadth. The UrbanNav result is now more robust than a
two-run average alone because fixed-window analysis across 127 external windows still
favors the PF family over `EKF` in most windows. However, the geography is still
concentrated on the Tokyo runs currently extracted in this repository. The paper can
defensibly claim cross-dataset evidence across PPC and UrbanNav, and it can say that the
UrbanNav gain is not restricted to a single short interval, but it should not imply broad
deployment-level generalization without additional datasets.

## Method Draft

### 1. State and Baseline Likelihood

The core particle-filter state in this repository is a 4D ECEF-plus-clock vector,
`x = [x, y, z, b]`, where `[x, y, z]` is receiver position and `b` is receiver clock bias
expressed in meters. Given satellite positions `s_j` and pseudorange observations
`rho_j`, the standard predicted pseudorange under particle `i` is

`rho_hat_ij = ||x_i - s_j|| + b_i`.

The simplest observation model is a Gaussian pseudorange likelihood over the residual
`r_ij = rho_j - rho_hat_ij`. This produces the standard PF baseline used by `PF-10K`.
In implementation terms, that path is the GPU particle filter in
`gnss_gpu.particle_filter` and the experiment wrapper `run_pf_standard(...)`.

### 2. Robust Clear-Mixture Observation Model

The most important practical method component in the current repository is not the full
3D blocked-satellite path, but a robust observation model that softens the effect of
heavy-tailed pseudorange errors even when explicit blocking is not active. In the code,
this is represented by the `clear_nlos_prob` branch in the 3D-capable likelihood path and
is surfaced experimentally as `PF+RobustClear-10K`.

Conceptually, each measurement is scored by a mixture between a narrow LOS-like component
and a broader NLOS-like component, even when the map path indicates clear visibility. The
result is a heavy-tailed likelihood that is less eager to collapse particle weight around
single-epoch pseudorange outliers. This is the current best explanation for the UrbanNav
external win: the external gain is driven primarily by robust observation modeling plus
GPU-resident PF inference, not by blocked-map reasoning alone.

### 3. Optional 3D Ray-Tracing Likelihood

The repository also supports an explicit 3D-aware likelihood in which candidate receiver
states are tested against city-model geometry. For each particle-satellite pair, the code
evaluates whether the path is clear or blocked using triangle intersections or an
equivalent BVH-accelerated query. That binary visibility hypothesis then selects or mixes
between LOS-like and NLOS-like residual models.

This 3D likelihood is implemented in the `ParticleFilter3D` and `ParticleFilter3DBVH`
paths. The repository results support two careful claims here. First, the 3D path is
implemented end-to-end on real PLATEAU subsets. Second, BVH makes that path practical.
The current results do not yet support the stronger claim that explicit 3D likelihoods are
the main cause of the best external accuracy numbers, so the method section should keep
that distinction explicit.

### 4. Multi-GNSS Measurement Quality Veto

The repository now exposes a small reusable multi-GNSS quality-veto utility. This utility
compares a raw multi-GNSS WLS epoch against a reference-system fallback, computes
residual- and bias-based diagnostics, and decides whether to keep the multi-GNSS solution
or fall back to the reference-system one. The key diagnostics are the multi-solution
residual p95, maximum absolute residual, inter-system bias spread relative to GPS, and
the number of extra satellites contributed by the additional constellations.

This component should be described as a supporting architectural utility, not as the main
algorithmic contribution. Its purpose is to stabilize the measurement path and to provide
a clear core interface between experiments and production-style code. The current results
show that it improves raw multi-GNSS WLS, but it is still not competitive with the PF
family on the main UrbanNav external table.

### 5. GPU Execution and BVH Acceleration

The systems side of the method rests on GPU-resident particle arrays and GPU-side
likelihood evaluation. PF and PF3D avoid repeated host-device reinitialization of particle
state and instead update weights, resample, and estimate on the device. For the 3D path,
the main computational bottleneck is repeated particle-satellite-versus-geometry
intersection. The BVH variant reduces the number of triangle tests by replacing
flat-per-triangle traversal with hierarchical spatial culling. That implementation detail
is what supports the 57.8x runtime reduction in the real-PLATEAU result.

### 6. Evaluation Protocol

The paper should explicitly define two data roles. PPC is the design-and-holdout dataset.
It is where exploratory strategy families are compared, rejected, or retained under a
controlled process. UrbanNav is the external dataset. It is where the final PF family is
evaluated without retuning the method on UrbanNav itself. This separation matters because
it prevents the paper from presenting a tuned external result as if it were genuine
generalization.

For every method, the paper should report at least RMS, p50, p95, >100 m rate, >500 m
rate, and longest continuous failure-segment duration. This is not cosmetic. The PPC
analysis already shows that RMS alone can hide or exaggerate the wrong behavior depending
on the distribution tail.

## Conclusion Draft

This repository now supports a defensible paper centered on robust urban GNSS inference,
systems acceleration, and empirical discipline. The strongest current empirical result is
that on UrbanNav Tokyo with trimble observations and repaired `G,E,J` measurement
loading, the PF family clearly outperforms `EKF`: `PF+RobustClear-10K` reaches
66.60 m mean RMS and 98.53 m mean p95, while `EKF` remains at 93.25 m and 178.18 m, and
the PF family removes catastrophic failures above 500 m on the evaluated runs. The
strongest systems result is that BVH preserves PF3D accuracy while reducing runtime from
1028.29 ms/epoch to 17.78 ms/epoch on a real PLATEAU subset.

The right claim is therefore not that this repository introduces a single world-first
algorithmic primitive. The right claim is that it assembles a practical stack for urban
GNSS: GPU-resident particle inference, robust heavy-tailed observation modeling, an
optional 3D map-aware likelihood path, and a disciplined evaluation protocol that
separates design from external validation and reports tail behavior honestly.

At the same time, the paper should remain explicit about what is still missing. The PPC
holdout gate improvement is small, so it should remain a secondary result. The 3D map path
is presently strongest as a systems contribution rather than a demonstrated source of the
best external accuracy. And the external validation, while now much stronger than before,
still rests on the currently extracted UrbanNav Tokyo runs rather than a broad dataset
suite. A Hong Kong control run remains useful here: raw PF collapses there, but guide-policy
experiments show that single-constellation sparse regimes prefer an always-on `EKF`-derived
velocity guide, while repaired multi-GNSS Tokyo sequences prefer a safer robust-clear
init-guide path. A simple adaptive guide narrows this weakness, but it is still a
supplemental cross-geometry mitigation rather than the headline method. Likewise, the older
`EKF`-anchored rescue path should be described as a safety utility, not as the headline
method. In particular, a full-run Tokyo confirmation still leaves `PF+AdaptiveGuide-10K`
slightly behind `PF+RobustClear-10K`, so the adaptive path should stay out of the main
table.

If written with those boundaries intact, the paper can make a strong and defensible case:
the repository does not merely contain an interesting GNSS PF implementation, but a
coherent urban localization system whose best external result, systems efficiency, and
evaluation discipline all materially improved during development.

## Current Quantitative Facts Safe to Cite

- PPC full-run best-configuration summary:
  `G,E` mean RMS 2D = 1221.12 m, mean p95 = 102.72 m across 6 runs.
- PPC full-run best-configuration summary:
  `G,E,J` mean RMS 2D = 1080.25 m, mean p95 = 102.67 m across 6 runs.
- PPC full-run outlier analysis:
  catastrophic errors reach about 20 km.
- PPC full-run outlier analysis:
  `>=500 m` catastrophic epochs = 329.
- PPC full-run outlier analysis:
  severe failures are concentrated at `n_sat = 4-6`.
- PPC holdout strategy summary:
  `always_robust` = 66.92 m mean RMS 2D, 81.69 m mean p95.
- PPC holdout strategy summary:
  `entry_veto_negative_exit_rescue...` = 65.54 m mean RMS 2D, 81.22 m mean p95.
- UrbanNav external `trimble + G,E,J` summary:
  `PF+RobustClear-10K` = 66.60 m mean RMS 2D, 98.53 m mean p95, 4.80% >100 m, 0.000% >500 m.
- UrbanNav external `trimble + G,E,J` summary:
  `PF-10K` = 67.61 m mean RMS 2D, 101.46 m mean p95, 5.44% >100 m, 0.000% >500 m.
- UrbanNav external `trimble + G,E,J` summary:
  `EKF` = 93.25 m mean RMS 2D, 178.18 m mean p95, 16.29% >100 m, 0.161% >500 m.
- UrbanNav external 500-epoch window summary:
  `PF+RobustClear-10K` beats `EKF` in `90 / 127` windows by RMS and `102 / 127` by p95.
- UrbanNav external 500-epoch window summary:
  `PF-10K` beats `EKF` in `88 / 127` windows by RMS and `101 / 127` by p95.
- UrbanNav external 500-epoch window summary:
  both PF variants satisfy `>500m rate <= EKF` in `127 / 127` windows.
- UrbanNav cross-geometry adaptive-guide summary:
  `PF+AdaptiveGuide-10K` = 62.90 m mean RMS 2D, 90.35 m mean p95, 2.77% >100 m on Tokyo `trimble + G,E,J` 3k slices.
- UrbanNav Hong Kong adaptive-guide summary:
  `PF+AdaptiveGuide-10K` = 66.85 m RMS 2D, 97.45 m p95, 3.85% >100 m on `HK_20190428`.
- Real-PLATEAU PF3D runtime summary:
  `PF3D-10K` = 1028.29 ms/epoch and `PF3D-BVH-10K` = 17.78 ms/epoch with matching 55.50 m RMS 2D.

## Text That Must Wait for More Experiments

Do not yet claim any of the following in the abstract or conclusion.

- That PLATEAU-based 3D likelihoods improve robustness on real data.
- That `WLS+QualityVeto` is the final best multi-GNSS method.
- That the exploratory PPC gate is a major source of the paper's gain.
- That the same external gain already holds on more than the current UrbanNav Tokyo runs.
- That `PF+AdaptiveGuide-10K` has already replaced `PF+RobustClear-10K` as the main UrbanNav external method.

## Immediate Next Writing Tasks

1. Convert this draft into the actual manuscript source and keep the current section
   ordering: UrbanNav main result first, PPC holdout second, BVH systems third.
2. Move the detailed PPC strategy family discussion to an appendix or supplemental note.
3. Convert the named papers in this file into a `.bib` file or a reference appendix once
   the target venue is fixed.
4. If submission timing permits, add one more external sequence group or additional
   UrbanNav geography so the conclusion can make a broader generalization claim.
