# Candidate-centred 3DMA GNSS multi-method investigation

Date: 2026-07-13

## Conclusion

The existing Phase71 triggered OSM road candidate remains the production-best
route. None of the new pseudorange-only candidate scorers improved the PPC
Nagoya run2 controlled-offset pilot. The only locally useful new mode was a
strong OSM centreline prior, and it was safe only after adding two truth-free
abstention conditions:

1. the source must remain at least 2.5 m from the road for 10 consecutive
   epochs; and
2. the candidate grid must actually reach within 0.5 m of a mapped road.

With those gates, the method improved two of three Nagoya run2 windows, was
neutral in the third, preserved a clean source in all three, and abstained on
Nagoya run1 and the untouched Nagoya run3 holdout. This is a useful safety
mechanism and diagnostic mode, but is not yet a replacement for Phase71.

## Literature basis

The implementation was guided by the following primary sources:

- Lee et al., *Enhancing GNSS Performance in Urban Canyon using GNSS
  Visibility Map and Recurrence Vector* (ION GNSS+ 2025): forms four-satellite
  subsets, projects candidate recurrence vectors onto satellite LOS directions,
  and compares the resulting ranging-error probabilities with a 3D visibility
  map. DOI: <https://doi.org/10.33012/2025.20423>
- Groves and Adjrad, *Likelihood-based GNSS positioning using LOS/NLOS
  predictions from 3D mapping and pseudoranges*: statistically scores LOS and
  NLOS pseudorange innovations without explicitly ray tracing every reflection.
  DOI: <https://doi.org/10.1007/s10291-017-0654-1>
- Wang, Groves and Ziebart, *GNSS Shadow Matching ... Optimized Visibility
  Scoring Scheme*: candidate visibility scoring that accounts for reflection
  and diffraction. DOI: <https://doi.org/10.1002/navi.38>
- Zhong and Groves, *Multi-Epoch 3D-Mapping-Aided Positioning using Bayesian
  Filtering Techniques*: shows the benefit of temporal integration, especially
  for moving receivers in dense environments. DOI:
  <https://doi.org/10.33012/navi.515>
- Ng et al., *Grid-based 3DMA GNSS with clustering and Doppler velocity using
  factor graph optimisation*: uses region-growing to isolate likelihood modes
  and multi-epoch constraints to mitigate multimodality and solution shifting.
  DOI: <https://doi.org/10.1017/S0373463325000220>
- Ng, Zhang and Hsu, *Range-based 3D Mapping Aided GNSS with NLOS Correction
  based on Skyplot with Building Boundaries*: scores simulated candidate
  pseudoranges including atmospheric and clock terms. DOI:
  <https://doi.org/10.33012/2019.16774>

The later `recurrence_vector` branch implements the complete sequence stated in
the paper's public ION abstract: it solves the initial position for each actual
four-satellite subset, differences every candidate against each subset
solution, projects those recurrence vectors onto the corresponding satellite
LOS directions, compares the resulting signal-type probabilities with the 3D
visibility classification, and selects the cumulative-probability argmax.  The
separate `multipivot` and `robust_subset` branches remain distinct ablations.
The proceedings PDF is credit-gated, so this is not a claim that unpublished
numeric parameter choices were reproduced verbatim.

## Implemented methods

Core implementation: `python/gnss_gpu/candidate_3dma.py`

Evaluator: `experiments/eval_candidate_3dma_ppc.py`

- horizontal ENU candidate generation around an external source;
- transmit-time satellite geometry, Sagnac correction, broadcast atmosphere
  correction, and constellation-specific receiver clocks;
- asymmetric LOS/NLOS likelihood with C/N0-derived visibility probability;
- multiple-pivot candidate-relative pseudorange consensus;
- robust four-satellite subset consensus using Cauchy costs and quantiles;
- per-satellite multi-epoch bias removal and temporal consistency scoring;
- PLATEAU visibility-mask region-growing and mode selection;
- optional source displacement prior;
- OSM road likelihood;
- truth-free OSM source-mismatch, contiguous-duration, and grid-reachability
  gates;
- epoch and fixed-window selection, CSV diagnostics, and JSON summaries.

The multipivot implementation also now treats a candidate with no usable LOS
pivot pair as a penalized candidate instead of aborting the whole grid.

## Synthetic verification

`tests/test_candidate_3dma.py` contains 15 passing tests covering:

- ENU candidate geometry;
- C/N0 visibility probabilities;
- exact candidate recovery and per-constellation clock removal;
- asymmetric NLOS and road terms;
- outlier-tolerant multipivot and robust-subset consensus;
- candidates without LOS pivot pairs;
- per-satellite temporal intercept removal;
- visibility-mode clustering;
- contiguous road mismatch and candidate-grid reachability gates.

Verification command:

```text
python -m pytest tests/test_candidate_3dma.py -q
python -m ruff check python/gnss_gpu/candidate_3dma.py \
  python/gnss_gpu/__init__.py experiments/eval_candidate_3dma_ppc.py \
  tests/test_candidate_3dma.py
```

Result: 15 passed; Ruff clean.

## Real-data protocol

The rover observations, navigation data, C/N0, PLATEAU triangles, and OSM roads
are real PPC data. Candidate scoring never reads the reference trajectory. For
this controlled sensitivity diagnostic only, the source was constructed as the
reference shifted 1.7 m north; a second run used the unshifted reference to test
whether a method damages a clean source. The reference is used after selection
to calculate errors.

Thus these experiments test whether a truth-free scorer can recover a known
small lateral source error on real measurements. They are not a deployable
source-trajectory evaluation. The actual Phase71 `xd_gici_hs` source artifacts
were not locally available to this evaluator.

Fixed pilot:

- PPC Nagoya run2, TOW 556226.6--556230.4, 20 epochs;
- systems G,R,E,C,J;
- 13 x 13 ENU grid, radius 3 m, spacing 0.5 m;
- baseline p50 1.700 m and grid oracle p50 0.200 m.

## Pilot comparison

| Method | Selected p50 (m) | Decision |
|---|---:|---|
| Absolute asymmetric 3DMA likelihood, window | 3.330 | reject |
| Candidate-relative multipivot | 2.973 | reject alone |
| Robust subset, quantile 0.2 | 3.680 | reject |
| Robust subset, best-subset quantile 0 | 2.973 | reject alone |
| Per-satellite temporal bias removal, 20 epochs | 2.973 | reject alone |
| Per-satellite temporal bias removal, 362 epochs | 3.774 | reject |
| Absolute + PLATEAU visibility clustering | 3.330 | neutral to absolute |
| Multipivot + PLATEAU visibility clustering | 2.973 | neutral to multipivot |
| Broad OSM corridor | 3.330 | neutral |
| Strong ungated OSM road mode | 1.393 | improves shifted source, unsafe |

Visibility clustering was neutral because all competitive candidates shared the
same predicted visibility mode. Persistent satellite-specific code/multipath
biases still dominated the 1.7 m candidate-relative geometry signal.

The ungated strong road mode pulled clean run2 sources by 1.414--3.041 m and is
therefore not acceptable. On Nagoya run1 it also pulled a clean source to the
grid corner, producing 4.243 m error. The latter case had a source-road distance
of 8.43 m but even the best grid candidate remained 5.28 m from the road. This
motivated the grid-reachability abstention gate.

## Final dual-gated road-mode results

Parameters were fixed at source-road distance 2.5 m, closest-candidate road
distance 0.5 m, and 10 contiguous epochs.

| Dataset/window | Source | Gate | Baseline p50 | Selected p50 | Outcome |
|---|---|---:|---:|---:|---|
| run2 pilot 556226.6--556230.4 | +1.7 m north | on | 1.700 | 1.393 | improve |
| run2 pilot | clean | off | 0.000 | 0.000 | preserve |
| run2 holdout A 556250.0--556253.8 | +1.7 m north | on | 1.700 | 1.700 | neutral |
| run2 holdout A | clean | off | 0.000 | 0.000 | preserve |
| run2 holdout B 556280.0--556283.8 | +1.7 m north | on | 1.700 | 1.393 | improve |
| run2 holdout B | clean | off | 0.000 | 0.000 | preserve |
| run1 regression 550400.0--550403.8 | +1.7 m north | off | 1.700 | 1.700 | abstain |
| run1 regression | clean | off | 0.000 | 0.000 | preserve |
| run3 untouched holdout 553820.0--553823.8 | +1.7 m north | off | 1.700 | 1.700 | abstain |
| run3 untouched holdout | clean | off | 0.000 | 0.000 | preserve |

The reachability condition was designed after observing the run1 failure, so
run1 is correctly labelled a regression test, not an untouched holdout. Nagoya
run3 was selected only after the rule was frozen.

## Production decision

1. Keep Phase71 as canonical production. Its real-source six-run replay remains
   86.205492% official: Nagoya run2 improves 64.426589% to 65.669779%
   (+1.243190 pp), while the other five runs are unchanged.
2. Do not connect absolute, multipivot, robust-subset, temporal-only, or
   visibility-cluster candidate scores directly to production.
3. Retain the dual-gated road mode as an experimental abstaining correction and
   as a safety improvement for future OSM candidate generation.
4. Before production use, evaluate it with the actual Phase71 source trajectory,
   use a longer production-like contiguous threshold (Phase71 uses 40 epochs),
   and replay all six runs.
5. The next pseudorange experiment should implement the recurrence-vector
   paper's four-satellite subset-position construction exactly and add Doppler
   or FGO temporal constraints. The current residual-only approximations do not
   expose the small lateral offset reliably.
