# Examples

Runnable demos for `gnss_gpu`. Run them from the repo root with `PYTHONPATH=python`.

## Start here — no GPU, no build, no data

```bash
PYTHONPATH=python python3 examples/demo_urban_canyon_sim.py
```

[`demo_urban_canyon_sim.py`](demo_urban_canyon_sim.py) simulates a car driving
through an urban canyon (buildings block some satellites → NLOS multipath) and
compares plain least squares against the package's robust SPP solver. It uses
only NumPy and pure-Python package code, runs in ~1 second, and shows the core
idea behind the project: robust down-weighting of NLOS-biased measurements.

```text
naive WLS (L2)        P50 10.30 m / RMS 10.21 m
robust SPP (Cauchy)   P50  2.00 m / RMS  2.39 m   → 81% better P50
```

## Demos that need the native CUDA build

These import the compiled kernels (signal sim, acquisition, particle filters,
ray tracing, multi-GNSS). Build them first — see
[Building the CUDA/C++ kernels](../README.md#building-the-cudac-kernels) — then
copy the generated `.so` files into `python/gnss_gpu/`.

| Demo | What it shows |
|---|---|
| [`demo_signal_sim.py`](demo_signal_sim.py) | GPU GNSS signal simulation + acquisition round-trip |
| [`demo_acquisition.py`](demo_acquisition.py) | GPU-accelerated GPS signal acquisition |
| [`demo_rinex.py`](demo_rinex.py) | RINEX observation parsing, WLS positioning, and particle filter |
| [`demo_interference.py`](demo_interference.py) | GPU-accelerated GNSS interference detection and excision |
| [`demo_full_pipeline.py`](demo_full_pipeline.py) | Full pipeline on an urban multipath scenario |
| [`demo_plateau_urban.py`](demo_plateau_urban.py) | PLATEAU 3D city model + particle-filter urban positioning |
| [`demo_real_data.py`](demo_real_data.py) | Real-data end-to-end positioning pipeline |
| [`demo_visualization.py`](demo_visualization.py) | Generates the visualization outputs (skyplots, maps, etc.) |

Some of these also expect downloaded datasets; check the demo's docstring and
[`internal_docs/plan.md`](../internal_docs/plan.md) for the inputs they need.

## See also

- [Benchmarks](../benchmarks/RESULTS.md) — GPU throughput numbers.
- [Live results snapshot](https://rsasaki0109.github.io/gnss_gpu/) — current
  figures and method freeze.
- [Project README](../README.md) — overview, results, and setup.
