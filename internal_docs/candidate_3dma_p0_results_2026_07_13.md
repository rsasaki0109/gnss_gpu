# Candidate-centered 3DMA P0 result (2026-07-13)

Status: **minimal implementation complete; naive absolute pseudorange
likelihood is negative on the first real-data gate**.

## What was implemented

- `python/gnss_gpu/candidate_3dma.py`
  - local ENU candidate-grid generation;
  - per-constellation clock-free pseudorange innovations;
  - Sagnac-corrected geometric ranges;
  - narrow LOS and positive-tail asymmetric NLOS likelihoods;
  - C/N0-to-LOS probability and visibility consistency score;
  - optional road-corridor distance factor;
  - normalized candidate posterior and component diagnostics.
- `experiments/eval_candidate_3dma_ppc.py`
  - PPC reception-time or transmit-time window loading;
  - batched PLATEAU BVH LOS evaluation;
  - broadcast ionosphere/troposphere correction;
  - epoch or constant-offset window selection;
  - optional OSM road-corridor integration;
  - truth is used only to report final errors.
- `tests/test_candidate_3dma.py`: seven CPU tests covering candidate geometry,
  clock removal, multi-constellation clocks, NLOS asymmetry, visibility, road
  scoring, input validation and recovery on clean synthetic measurements.

## Real-data gate

Dataset and window:

- PPC `nagoya/run2`, TOW `556226.6..556230.4`, 20 epochs;
- satellite positions recomputed at transmission time with two iterations;
- PLATEAU cache `nagoya_run2_triangles.npz`;
- all available `G,R,E,C,J` observations with per-system clocks;
- controlled source trajectory = reference shifted `+1.7 m` north, matching the
  scale and direction of the Phase62 stable-wrong-solution diagnosis;
- search grid `+-3 m`, spacing `0.5 m`, 169 candidates per epoch.

This reference-shift input is a diagnostic construction, not a deployable
source. Reference coordinates do not enter candidate scoring.

| Selector | P50 (m) | RMS (m) | improved epochs |
|---|---:|---:|---:|
| shifted source | 1.700 | 1.700 | -- |
| grid oracle (evaluation only) | 0.200 | 0.200 | 20/20 |
| single-epoch 3DMA | 3.720 | 3.693 | 0/20 |
| window-summed 3DMA | 3.330 | 3.330 | 0/20 |
| window + OSM corridor | 3.330 | 3.330 | 0/20 |

Representative command:

```bash
PYTHONPATH=build/python:python python experiments/eval_candidate_3dma_ppc.py \
  --data-dir E:/datasets/PPC-Dataset-data/nagoya/run2 \
  --source-from-reference-offset --reference-offset-north-m 1.7 \
  --triangle-cache-npz E:/datasets/plateau_cache/nagoya_run2_triangles.npz \
  --out-prefix experiments/results/candidate_3dma_nagoya_run2_tx_atmos_window \
  --start-tow 556226.6 --end-tow 556230.4 --systems G,R,E,C,J \
  --radius-m 3 --spacing-m 0.5 --selection-mode window
```

## Interpretation

The implementation and CUDA LOS wiring work, but the initial statistical model
does not expose the needed absolute offset. Disabling the visibility term did
not change the selected direction. Adding transmission-time geometry, Sagnac,
broadcast atmosphere correction and elevation weighting reduced some tail error
but did not reverse the result. The failure therefore sits in the absolute
single-epoch pseudorange model: persistent satellite-specific multipath and
remaining code biases create a stronger false likelihood gradient than a
`1.7 m` position displacement.

The OSM corridor was neutral in this short window because the competing
candidates remained within the same road corridor. This is consistent with the
earlier Phase68 result: a broad road constraint is not sufficient to determine
the correct lateral mode.

## Decision and next hypothesis

Do not connect this absolute likelihood directly to Phase71 production, and do
not tune its sigma constants on truth. Keep the small scorer as experimental
infrastructure.

The next justified experiment is the 2025/2026 **consensus recovery-vector**
model: score candidate-relative range changes from multiple high-elevation
pivots/subsets, rather than forcing absolute per-satellite innovations toward a
zero/constant-bias distribution. Accumulate that relative score over a span and
combine it with distinct PLATEAU visibility modes. This directly removes the
persistent satellite-specific intercepts that defeated P0 while retaining the
candidate-centered GPU/BVH machinery delivered here.
