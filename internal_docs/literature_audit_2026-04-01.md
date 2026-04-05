# Literature Audit (2026-04-01)

## Scope

This note records a quick primary-source audit of papers most relevant to:

- particle filtering for GNSS positioning,
- 3D-mapping-aided GNSS / shadow matching,
- GPU-accelerated large-scale particle filtering.

The goal is not to be exhaustive. The goal is to stop us from making novelty claims
that are obviously too strong.

## Papers checked

### 1. MegaParticles

- Kenji Koide et al., "MegaParticles: Range-based 6-DoF Monte Carlo Localization with GPU-Accelerated Stein Particle Filter," ICRA 2024.
- Source: https://arxiv.org/abs/2404.16370

What it establishes:

- GPU-accelerated SVGD particle filtering at the one-million-particle scale is real and recent.
- The problem is LiDAR localization, not GNSS.

Implication for us:

- This is strong adjacent prior art for the systems angle.
- We cannot present GPU+SVGD+1M particles as generally unprecedented.
- We can still argue that applying this scale and architecture to GNSS urban positioning is a distinct contribution.

### 2. Multiple Update Particle Filter

- Taro Suzuki, "Multiple Update Particle Filter: Position Estimation by Combining GNSS Pseudorange and Carrier Phase Observations," 2024.
- Source: https://arxiv.org/abs/2403.03394

What it establishes:

- Direct GNSS particle-filter positioning in urban environments is active and recent.
- The paper explicitly reports vehicle position estimation experiments in urban environments.
- The methodological novelty is multiple-update resampling for sharp likelihoods, not GPU acceleration.

Implication for us:

- Any claim like "first GNSS particle filter" is false.
- Any claim like "first urban GNSS PF" is also unsafe.

### 3. Reliable GNSS Localization Against Multiple Faults Using a Particle Filter Framework

- Shubh Gupta and Grace X. Gao, "Reliable GNSS Localization Against Multiple Faults Using a Particle Filter Framework," 2021.
- Source: https://arxiv.org/abs/2101.06380

What it establishes:

- PF-based GNSS localization under urban faults already exists.
- Integrity / availability are already part of the PF framing in GNSS.

Implication for us:

- If we want a robust paper, we should compare or at least discuss integrity-style failure cases.
- Our full-epoch PPC catastrophic tails are directly relevant to this literature.

### 4. Shadow Matching

- Paul D. Groves, "Shadow Matching: A New GNSS Positioning Technique for Urban Canyons," Journal of Navigation, published online June 7, 2011.
- Source: https://www.cambridge.org/core/journals/journal-of-navigation/article/shadow-matching-a-new-gnss-positioning-technique-for-urban-canyons/5ED5573D9D3EAAFDEED212BF2AAAC9B5

What it establishes:

- 3D building models have been used with GNSS for more than a decade.
- Predicting satellite visibility from 3D maps is foundational prior art, not novelty.

Implication for us:

- "Uses 3D building models for urban GNSS" is not a contribution statement.

### 5. Intelligent Urban Positioning: Integration of Shadow Matching with 3D-Mapping-Aided GNSS Ranging

- Mounir Adjrad and Paul D. Groves, Journal of Navigation, published online August 3, 2017.
- Source: https://www.cambridge.org/core/journals/journal-of-navigation/article/intelligent-urban-positioning-integration-of-shadow-matching-with-3dmappingaided-gnss-ranging/7A8E0DFB8DC82EC5AADD74C5D94166E5

What it establishes:

- Shadow matching and 3DMA GNSS ranging were already integrated in 2017.
- The paper explicitly discusses multi-constellation use and real-time smartphone demonstrations in the surrounding UCL line of work.

Implication for us:

- "First integration of 3D map reasoning with GNSS ranging" is false.
- "Real-time 3D-map-aided GNSS" is also false.

### 6. Investigation of 3D-Mapping-Aided GNSS Navigation in Urban Canyons

- Qiming Zhong, UCL PhD thesis, 2024.
- Source: https://discovery-pp.ucl.ac.uk/id/eprint/10196136/

What it establishes:

- UCL's 3DMA GNSS line already includes multi-epoch filtering and explicitly mentions particle filtering and grid filtering.

Implication for us:

- "Multi-epoch 3DMA GNSS with particle filtering" is not new as a research direction.
- The thesis is not the same evidentiary weight as a flagship journal paper, but it is strong enough that we should not ignore it.

## What is still defensible

These are plausible claims after this audit:

1. A GNSS-specific system that combines:
   - GPU-resident large-scale particle inference,
   - explicit 3D-map / ray-tracing-style urban likelihoods,
   - and practical real-data evaluation.

2. A systems contribution around scale:
   - e.g. one-million-particle-class GNSS filtering at practical runtime.

3. A GNSS-specific adaptation of SVGD / particle mechanics:
   - but only if we explain exactly what is adapted for GNSS and why it is not a straight port of MegaParticles.

4. A robustness / integrity contribution:
   - if we explicitly analyze catastrophic tails and show mitigation.

## Claims that should be weakened or removed

Avoid these:

- "world first GNSS particle filter"
- "world first urban GNSS particle filter"
- "first use of 3D building models for GNSS particle likelihood"
- "first real-time 3D-map-aided GNSS"
- "first integration of shadow matching and GNSS ranging"

If we need bold language, use:

- "To the best of our knowledge, we are not aware of prior work that combines ..."

But even that should only be used for the exact combined stack, not for any individual component.

## Immediate paper-writing consequences

1. Related work must cite at least:
   - Groves 2011
   - Adjrad & Groves 2017
   - Gupta & Gao 2021
   - Suzuki 2024
   - Koide et al. 2024

2. The paper should separate:
   - algorithmic novelty,
   - systems novelty,
   - empirical novelty.

3. Because PPC full-epoch runs show catastrophic outliers with `n_sat=4-6`, the paper should not rely on RMS alone.
   Recommended metrics:
   - p50 / p95
   - outlier rate above 100 m
   - catastrophic rate above 500 m
   - longest failure segment duration

4. A useful baseline framing is:
   - WLS / EKF / RTK-like
   - PF prior art discussion
   - our GPU PF / 3D-aware PF / BVH variants

## Open gaps in this audit

- I have not yet completed an exhaustive search for older GNSS-specific GPU particle-filter papers.
- I have not yet checked every PolyU / UCL / ION paper in the 3DMA GNSS line.
- I have not yet mapped all claims in `docs/design.md` against this audit.

So this note is strong enough to prune bad claims, but not strong enough yet to finalize a camera-ready related-work section.
