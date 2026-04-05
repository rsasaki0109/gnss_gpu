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
1028.29 ms/epoch to 17.78 ms/epoch, a 57.8x speedup. A particle count scaling
experiment across both Tokyo sequences reveals a phase transition: PF performance
crosses the EKF baseline at approximately N=1,000 particles, mean RMS saturates near
N=5,000, but the >100 m failure rate continues to improve up to N=1,000,000 — from
3.31% to 1.97% on Odaiba and from 7.46% to 4.49% on Shinjuku. This shows that
GPU-scale particle inference enables a tail-robustness regime unreachable at
conventional particle counts. These results support a paper positioned around robust
urban GNSS inference, scale-dependent robustness, and practical 3D-likelihood systems
acceleration rather than a claim of a single unprecedented algorithmic primitive.

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
by Suzuki \cite{suzuki2024} and by Gupta and Gao \cite{gupta2021}. 3D-mapping-aided GNSS and shadow matching have been
studied by Groves \cite{groves2011} and by Adjrad and Groves \cite{adjrad2018}, and earlier PF-based 3D map integration also
exists in Suzuki and Kubo \cite{suzuki2013}. Meanwhile, Koide et al. \cite{koide2024} demonstrated that GPU-accelerated
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
5. An empirical scaling analysis showing that particle count governs a phase transition
   in urban GNSS robustness: mean RMS saturates at moderate counts (~5K), but tail
   failure rates continue to improve up to one million particles, demonstrating that
   GPU-scale inference is not merely faster but enables qualitatively better robustness.

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

### GNSS Positioning Fundamentals

Standard GNSS single-point positioning is formulated as a nonlinear least-squares problem
over pseudorange measurements, typically solved iteratively via weighted least-squares
(WLS) or extended Kalman filtering (EKF) \cite{groves2013, kaplan2017}. These methods
assume approximately Gaussian measurement noise, which breaks down in urban canyons
where NLOS reception and multipath produce heavy-tailed error distributions
\cite{wen2020, hsu2015}. The need for integrity-aware evaluation beyond mean or RMS
metrics is well established in the GNSS community, with protection-level concepts
\cite{blanch2017} motivating our use of tail metrics such as P95 and >100 m failure
rate.

### GNSS Particle Filtering

Particle filtering for positioning was established by Gustafsson et al.
\cite{gustafsson2002} as a natural extension of the bootstrap particle filter
\cite{gordon1993} to navigation problems. In the GNSS domain, Suzuki's multiple-update
particle filter \cite{suzuki2024} shows that direct GNSS particle-based positioning
remains an active topic, including urban vehicle experiments and the use of pseudorange
and carrier-phase observations. Gupta and Gao \cite{gupta2021} further show that
particle-filter-based GNSS localization can be framed around reliability against
multiple faults, which is especially relevant for urban canyon behavior and
integrity-style analysis. More recently, Niimi et al. \cite{niimi2025} demonstrate a
tightly coupled Rao-Blackwellized particle filter for GNSS-only urban positioning
achieving sub-meter accuracy, and Zocca et al. \cite{zocca2022} study improved
weighting schemes for GNSS particle filters at up to 60,000 particles. The convergence
properties of particle filters are characterized by Crisan and Doucet
\cite{crisan2002}, who show that posterior approximation error decreases as
$O(1/\sqrt{N})$. However, none of these works systematically study how positioning
accuracy metrics — particularly tail metrics like failure rates — scale as a function of
particle count from hundreds to millions.

### 3D-Mapping-Aided GNSS and Shadow Matching

The 3DMA GNSS literature also predates this project by a wide margin. Groves \cite{groves2011} introduced
shadow matching as a technique for urban canyons using 3D city models, establishing that
predicted satellite visibility from map geometry is prior art rather than a new
contribution. Adjrad and Groves \cite{adjrad2018} later integrated shadow matching with 3D-mapping-aided
GNSS ranging, demonstrating that map reasoning and ranging information can already be
combined in a single positioning framework. Earlier PF-based map integration also appears
in Suzuki and Kubo \cite{suzuki2013}. Zhong and Groves \cite{zhong2022} demonstrate
multi-epoch 3D-mapping-aided positioning using both grid filters and particle filters,
reducing horizontal RMS by approximately 68\% compared to single-epoch 3DMA. Zhong's
thesis \cite{zhong2023} further discusses multi-epoch filtering and particle-filter
perspectives. More recently, Neamati et al. \cite{neamati2022} propose set-valued shadow matching
using zonotopes, offering guaranteed uncertainty bounds rather than point estimates.
Hsu et al. \cite{hsu2015} combine RAIM with 3D city models for NLOS
correction and exclusion. Accordingly, this paper should not claim the first use of 3D
models, the first real-time 3DMA GNSS system, or the first integration of shadow
matching and GNSS ranging.

### Large-Scale GPU Particle Filtering

On the systems side, MegaParticles is the strongest adjacent prior art. Koide et al. \cite{koide2024}
show that GPU-accelerated Stein particle filtering can scale to the one-million-particle
regime in range-based LiDAR localization. That result matters because it removes any
credible claim that GPU + SVGD + million-particle scale is broadly unprecedented.
However, the state space, likelihood model, and failure modes in GNSS differ from those
in LiDAR localization. A defensible systems contribution remains available if we show that
the same scale can be realized for GNSS while integrating satellite geometry, clock-bias
structure, and 3D map-aware likelihoods.

### Factor Graph Optimization Baselines

A separate and increasingly competitive line of work formulates urban GNSS as factor
graph optimization (FGO). Wen et al. \cite{wen2021icra} propose robust FGO for GNSS
at ICRA, and \cite{wen2021} provide GraphGNSSLib as an open-source FGO package. More
recently, Wen et al. \cite{wen2024} introduce integrity-constrained FGO with switch
variables that reweight measurements to satisfy chi-square integrity bounds, improving
the proportion of sub-10 m fixes from 55\% to 78\% on UrbanNav. The robust
optimization literature provides theoretical grounding: Olson and Agarwal
\cite{olson2013} introduce max-mixture models for outlier-robust estimation, and
Wen et al. \cite{wen2020} demonstrate that adaptive Gaussian mixture models
effectively handle non-Gaussian GNSS noise in urban scenarios. These FGO methods are
optimization-based and produce point estimates, whereas our particle filter approach
maintains an explicit posterior sample that enables scaling analysis and tail-robustness
characterization. The UrbanNav dataset itself is documented in Hsu et al.
\cite{hsu2023}.

### Our Intended Positioning

The cleanest way to position this paper is therefore as a combined contribution across
four axes: robust urban inference, scale-dependent tail robustness, systems
acceleration, and empirical discipline.
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

> **Table 1.** Main quantitative summary of the paper. PPC holdout is reported as a
> design-discipline result rather than a headline accuracy claim. UrbanNav external uses
> fixed `trimble + G,E,J` settings without UrbanNav-specific retuning.
> `PF+RobustClear-10K` is the strongest external method, improving mean RMS from
> 93.25 m (`EKF`) to 66.60 m and mean p95 from 178.18 m to 98.53 m, while reducing the
> >100 m rate from 16.29% to 4.80% and the >500 m rate from 0.161% to 0.000%.
> `WLS+QualityVeto` is included as a promoted core utility, not as the main external
> method. BVH rows isolate runtime on a real PLATEAU subset and show unchanged PF3D
> accuracy with large acceleration.

### 2. PPC Holdout Result

The PPC holdout result should be presented as a discipline result rather than a headline
accuracy result. The safe baseline `always_robust` achieves 66.92 m mean RMS horizontal
error and 81.69 m mean p95, while the best exploratory gate
`entry_veto_negative_exit_rescue...` reaches 65.54 m and 81.22 m. This is a real but
small improvement. The right interpretation is not that the gate family is the core
scientific contribution. The right interpretation is that the repository now enforces an
experiment process in which only small, holdout-surviving gains are retained, and larger
but brittle tuned gains are rejected. The full list of strategies evaluated and the holdout methodology are described in
Appendix A. The paired segment comparison is already rendered in
[paper_ppc_holdout.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_ppc_holdout.png).

> **Figure 1.** Segment-wise PPC holdout comparison between `always_robust` and the best
> exploratory gate. The gain survives holdout but remains modest: mean RMS decreases from
> 66.92 m to 65.54 m and mean p95 decreases from 81.69 m to 81.22 m. This figure
> supports the paper's experiment-discipline claim rather than the main accuracy claim.

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

> **Figure 2.** UrbanNav external validation on Odaiba and Shinjuku using trimble
> observations and `G,E,J` measurements. Left: empirical CDF of 2D error. Right: rates
> of failures above 100 m and catastrophic failures above 500 m. `PF+RobustClear-10K`
> achieves 66.60 m mean RMS and 98.53 m mean p95, outperforming `EKF` at 93.25 m and
> 178.18 m. Relative to `EKF`, the >100 m rate falls from 16.29% to 4.80% and the
> >500 m rate falls from 0.161% to 0.000%.

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

### 5. Hong Kong Supplemental Result

The strongest current limitation of the Tokyo mainline is geographic breadth: does the
PF family still win in a different urban geometry? Hong Kong provides an answer across
three sequences with increasing canyon severity.

On all three sequences, the frozen mainline `PF+RobustClear-10K` collapses under raw
GPS+BeiDou multi-GNSS input, confirming that the Tokyo configuration does not transfer
directly to Hong Kong's satellite geometry. However, when the PF framework is configured
with an adaptive EKF-derived velocity guide, `PF+AdaptiveGuide-10K` outperforms `EKF`
on all three sequences:

- **HK-20190428** (medium urban, 2019 pilot): `PF+AdaptiveGuide-10K` achieves 66.85 m
  RMS vs `EKF` at 69.49 m. The cleanest HK result.
- **HK TST** (medium urban, F9P splitter, 2021): `PF+AdaptiveGuide-10K` achieves
  152.37 m RMS vs `EKF` at 301.04 m. A 49% improvement despite poor absolute accuracy.
- **HK Whampoa** (deep urban, F9P splitter, 2021): `PF+AdaptiveGuide-10K` achieves
  413.68 m RMS vs `EKF` at 463.09 m. The hardest canyon, but PF still wins.

The absolute accuracy degrades with canyon severity, but the relative advantage of the
PF family over `EKF` is consistent across all three sequences. This does not make
`PF+AdaptiveGuide-10K` the frozen mainline — that remains `PF+RobustClear-10K` on
Tokyo. But it demonstrates that the PF framework generalizes to Hong Kong when the
appropriate guide policy is enabled, supporting a cross-geography breadth claim.

### 6. Particle Count Scaling Result

A key advantage of GPU-resident particle inference is the ability to run at particle
counts that are impractical on CPU. To quantify when scale matters, we evaluated
the PF on both UrbanNav Tokyo sequences (Odaiba and Shinjuku, trimble + G,E,J) across
particle counts from 100 to 1,000,000 using `ParticleFilterDevice` with persistent GPU
memory.

Both sequences reveal a consistent phase transition. At N=100, the PF is worse than
`EKF` on both sequences (Odaiba: 135.88 m vs 89.42 m; Shinjuku: 120.17 m vs 97.07 m).
At N≈1,000, the PF crosses over `EKF` in RMS on both sequences. Mean RMS saturates
around N=5,000 (Odaiba: ~62 m; Shinjuku: ~71 m). However, the tail continues to
improve with scale on both sequences: the >100 m failure rate on Odaiba drops from
3.31% at 10K to 1.97% at 1M, and on Shinjuku from 7.46% to 4.49%.

This means GPU-scale particle inference is not merely faster — it enables a
tail-robustness regime that is unreachable at conventional particle counts. The RMS
headline stabilizes early, but the failure-rate headline requires large-scale inference
to reach its floor. This finding is reproducible across both Tokyo sequences. The
intended figure for this result is
[paper_particle_scaling.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_particle_scaling.png).

> **Figure 4.** Particle count scaling on UrbanNav Tokyo (Odaiba and Shinjuku). Left:
> mean RMS 2D. Center: mean P95. Right: >100 m failure rate. Both sequences show a
> phase transition at N≈1,000 where PF crosses the EKF baseline. RMS saturates near
> N=5,000, but the >100 m failure rate continues to improve up to 1M particles. The
> shaded region marks the crossover zone.

### 7. BVH Systems Result

The 3D map path is currently strongest as a systems result. On the real PLATEAU subset,
`PF3D-10K` and `PF3D-BVH-10K` match exactly in accuracy at 55.50 m RMS and 58.39 m p95,
but the runtime drops from 1028.29 ms/epoch to 17.78 ms/epoch with BVH, a 57.8x speedup.
This is the cleanest way to defend the explicit 3D-likelihood implementation in the
current paper. We should not overstate it as a real-data accuracy gain, because the
current external accuracy leader is `PF+RobustClear`, not the full map-aware path. But as
a systems contribution, the result is strong and easy to communicate. The intended figure
for this section is [paper_bvh_runtime.png](/workspace/ai_coding_ws/gnss_gpu/experiments/results/paper_assets/paper_bvh_runtime.png).

> **Figure 3.** Runtime and accuracy on the real PLATEAU PF3D subset. `PF3D-BVH-10K`
> preserves the same accuracy as `PF3D-10K` at 55.50 m RMS and 58.39 m p95 while
> reducing runtime from 1028.29 ms/epoch to 17.78 ms/epoch, a 57.8x speedup. This figure
> should be framed as a systems contribution rather than a real-data accuracy gain from
> explicit 3D reasoning.

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

The third limitation is external breadth. The UrbanNav result now spans two geographies:
Tokyo (where `PF+RobustClear-10K` wins as mainline) and Hong Kong (where
`PF+AdaptiveGuide-10K` wins as a supplemental variant with GPS+BeiDou). Across five
sequences total — Odaiba, Shinjuku, HK-20190428, HK TST, and HK Whampoa — the PF family
consistently outperforms `EKF`. The winning HK method is not the frozen mainline (it
requires adaptive guide), and absolute HK accuracy degrades with canyon severity.
Nevertheless, the consistent relative advantage across five sequences in two cities
strengthens the cross-geography claim beyond what a Tokyo-only paper could support.

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
separates design from external validation and reports tail behavior honestly. A
distinctive empirical finding is that particle count scaling reveals a phase transition
in urban GNSS robustness: mean RMS saturates at moderate counts, but tail failure rates
continue to improve up to one million particles. This result, reproduced on both Tokyo
sequences, demonstrates that GPU-scale particle inference is not merely a faster version
of conventional PF — it enables a tail-robustness regime that is qualitatively
unreachable at the particle counts typical of CPU-based implementations.

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
- HK TST `G,C` summary:
  `PF+AdaptiveGuide-10K` = 152.37 m RMS 2D, 254.41 m p95, 60.94% >100 m.
  `EKF` = 301.04 m RMS 2D, 545.51 m p95, 84.48% >100 m.
- HK Whampoa `G,C` summary:
  `PF+AdaptiveGuide-10K` = 413.68 m RMS 2D, 643.11 m p95, 90.29% >100 m.
  `EKF` = 463.09 m RMS 2D, 769.90 m p95, 91.73% >100 m.
- Particle scaling on Odaiba trimble + G,E,J:
  N=100: 135.88 m RMS, 46.01% >100 m (worse than EKF).
  N=1000: 70.59 m RMS, 11.64% >100 m (crossover with EKF at 89.42 m).
  N=10000: 61.86 m RMS, 3.31% >100 m.
  N=1000000: 60.40 m RMS, 1.97% >100 m (tail continues to improve).
- Particle scaling on Shinjuku trimble + G,E,J:
  N=100: 120.17 m RMS, 36.00% >100 m (worse than EKF).
  N=1000: 78.46 m RMS, 10.49% >100 m (crossover with EKF at 97.07 m).
  N=10000: 71.72 m RMS, 7.46% >100 m.
  N=1000000: 73.26 m RMS, 4.49% >100 m (tail continues to improve).
- Real-PLATEAU PF3D runtime summary:
  `PF3D-10K` = 1028.29 ms/epoch and `PF3D-BVH-10K` = 17.78 ms/epoch with matching 55.50 m RMS 2D.

## Text That Must Wait for More Experiments

Do not yet claim any of the following in the abstract or conclusion.

- That PLATEAU-based 3D likelihoods improve robustness on real data.
- That `WLS+QualityVeto` is the final best multi-GNSS method.
- That the exploratory PPC gate is a major source of the paper's gain.
- That the same external gain already holds on more than the current UrbanNav Tokyo runs.
- That `PF+AdaptiveGuide-10K` has already replaced `PF+RobustClear-10K` as the main UrbanNav external method.

## Appendix A: PPC Strategy Family Details

The PPC holdout experiment evaluated a series of strategy gates of increasing complexity.
All gates share the same particle-filter core and differ only in how the observation-model
branch is selected each epoch. The family was grown incrementally:

1. `always_robust` — always use the robust clear-mixture likelihood. Safe baseline.
2. `always_blocked` — always use the blocked-satellite likelihood. Overfits to NLOS-heavy epochs.
3. `disagreement_gate` — switch on WLS/PF position disagreement.
4. `rule_chain_gate` — handcrafted multi-rule chain.
5. `weighted_score_gate` — continuous score combining residual and geometry features.
6. `clock_veto_gate` — add clock-bias anomaly detection.
7. `dual_mode_regime_gate` — separate open-sky and urban regimes.
8. `quality_veto_regime_gate` — integrate the multi-GNSS quality veto into regime selection.
9. `hysteresis_quality_veto_regime_gate` — add hysteresis to prevent rapid switching.
10. `branch_aware_hysteresis_quality_veto_regime_gate` — condition switching on current branch state.
11. `rescue_branch_aware_hysteresis_quality_veto_regime_gate` — add EKF-anchored rescue fallback.
12. `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate` — add entry veto and negative-exit conditions. Final surviving gate.

The evaluation protocol uses a positive/holdout split of 6 PPC segments each (positive6 /
holdout6). Strategies are first tuned on the positive split, then frozen and evaluated on
holdout. On the tuned split, several complex gates show meaningful gains over
`always_robust`. On holdout, most gains disappear. The only non-trivial survivor is gate
12, which improves mean RMS from 66.92 m to 65.54 m and mean p95 from 81.69 m to 81.22 m.

This outcome supports two conclusions. First, holdout discipline is necessary: tuned-only
numbers overstate the real improvement. Second, further growing the strategy family is
unlikely to yield large holdout-surviving gains. The 12-strategy progression already
exhausts the obvious feature space, and the surviving gain is small.

Sources:

- `experiments/pf_strategy_lab/strategies.py`
- `experiments/results/pf_strategy_lab_positive6_summary.csv`
- `experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv`
- `experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv`

## Immediate Next Writing Tasks

1. Convert this draft into the actual manuscript source and keep the current section
   ordering: UrbanNav main result first, PPC holdout second, BVH systems third.
2. ~~Move the detailed PPC strategy family discussion to an appendix or supplemental note.~~ Done: Appendix A added with 12-strategy progression; main text refers to appendix.
3. ~~Convert the named papers in this file into a `.bib` file or a reference appendix once
   the target venue is fixed.~~ Done: `docs/references.bib` created with 7 entries; `\cite{}` keys inserted inline.
4. If submission timing permits, add one more external sequence group or additional
   UrbanNav geography so the conclusion can make a broader generalization claim.
