# Experiment Log

This file records the current, README-facing experiment results. Long-running
work-in-progress notes stay under `internal_docs/`.

## 2026-06-07 - PLATEAU NLOS Replay Suite

Goal: verify that one exported PLATEAU LOS/NLOS ray mask can be consumed by
multiple downstream estimators, rather than only improving one bespoke demo.

Command:

```bash
PYTHONPATH=python:. python3 experiments/run_plateau_nlos_demo_suite.py
```

Inputs:

- Mesh: `data/sample_plateau.gml`
- Trajectory: 70 synthetic receiver epochs through the sample PLATEAU mesh
- Satellites: 14 fixed azimuth/elevation directions
- Mask contract: `tow,epoch_idx,prn,is_los`
- Current generated mask: 980 rows, 641 NLOS rows, 65.4% NLOS

Outputs:

- `experiments/results/plateau_nlos_demo_mask.csv`
- `experiments/results/plateau_nlos_demo_{spp,pf,fgo}_replay_summary.json`
- `experiments/results/plateau_nlos_demo_suite_summary.{json,md,csv}`
- `docs/assets/data/plateau_nlos_demo_suite_summary.{csv,md}`
- `docs/assets/media/plateau_nlos_visualization.html`

Result:

| Estimator | Baseline RMS | Mask-soft RMS | RMS gain | Wins |
|---|---:|---:|---:|---:|
| SPP | 11.85 m | 4.07 m | 65.6% | 48/68 |
| PF | 11.18 m | 1.40 m | 87.4% | 70/70 |
| FGO | 8.10 m | 0.38 m | 95.4% | 70/70 |

Interpretation:

- The exported mask is estimator-agnostic enough to drive SPP, particle-filter,
  and local-FGO consumers.
- The suite intentionally separates geometry/ray classification from solver
  behavior: replay scripts consume the CSV path, not the PLATEAU mesh.
- Robust residual weighting alone is not the main result here; the geometry mask
  provides the decisive downstream information.

Limitations:

- The replay uses a deterministic synthetic measurement model over a small
  shipped CityGML sample, not a real drive log.
- The local-FGO replay fixes receiver clock bias to isolate position and NLOS
  mask effects.
- This result supports the mask contract and demo pipeline; it is not a
  replacement for UrbanNav/PPC real-data evaluation.
