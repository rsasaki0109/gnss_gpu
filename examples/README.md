# Examples

Runnable demos for `gnss_gpu`. Run them from the repo root with `PYTHONPATH=python`.

## Which demo should I run?

| Goal | Run this | Needs GPU build? | Needs downloaded data? |
|---|---|---:|---:|
| Try it in the browser, zero install | [`colab_urban_canyon_quickstart.ipynb`](colab_urban_canyon_quickstart.ipynb) ([open in Colab](https://colab.research.google.com/github/rsasaki0109/gnss_gpu/blob/main/examples/colab_urban_canyon_quickstart.ipynb)) | No | No |
| See the core idea quickly | [`demo_urban_canyon_sim.py`](demo_urban_canyon_sim.py) | No | No |
| Show the particle-filter localization improvement | [`demo_pf_localization_improvement.py`](demo_pf_localization_improvement.py) | No | No |
| Inspect NLOS measurement effects | [`demo_nlos_simulation.py`](demo_nlos_simulation.py) | No | No |
| Try the shipped PLATEAU sample | [`demo_plateau_nlos_simulation.py`](demo_plateau_nlos_simulation.py) | No | No |
| Generate a standalone PLATEAU HTML report | [`demo_plateau_nlos_visualization.py`](demo_plateau_nlos_visualization.py) | No | No |
| Exercise signal simulation/acquisition kernels | [`demo_signal_sim.py`](demo_signal_sim.py) | Yes | No |
| Run real-data or full particle-filter flows | [`demo_real_data.py`](demo_real_data.py), [`demo_full_pipeline.py`](demo_full_pipeline.py) | Yes | Usually |

Start with the first row after a fresh clone. Build CUDA/C++ only when you want
to run the kernel-backed demos listed near the bottom of this page.

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

The same scenario, with a sky plot and trajectory/error figures, is available as
[`colab_urban_canyon_quickstart.ipynb`](colab_urban_canyon_quickstart.ipynb) —
[open it in Colab](https://colab.research.google.com/github/rsasaki0109/gnss_gpu/blob/main/examples/colab_urban_canyon_quickstart.ipynb)
to run everything in the browser with zero install.

## Particle-filter localization improvement demo -- no GPU, no build, no data

```bash
PYTHONPATH=python:. python3 examples/demo_pf_localization_improvement.py
```

[`demo_pf_localization_improvement.py`](demo_pf_localization_improvement.py)
reads checked-in result artifacts instead of rerunning UrbanNav. It prints the
Odaiba OpenStreetMap particle-filter comparison against RTKLIB demo5 and the
PLATEAU LOS/NLOS mask replay gain for the particle-filter consumer.

```text
RTKLIB demo5                              P50 2.67 m / RMS 13.08 m
PF 100K (DD + smoother + stop-detect)     P50 1.36 m / RMS  4.11 m
Improvement vs RTKLIB demo5: P50 49%, RMS 69%.
PLATEAU PF mask-soft replay: RMS 11.18 m -> 1.40 m, gain 87.4%.
```

## NLOS simulation research demo — no GPU, no build, no data

```bash
PYTHONPATH=python python3 examples/demo_nlos_simulation.py
```

[`demo_nlos_simulation.py`](demo_nlos_simulation.py) builds a deterministic
box-building street canyon, ray-casts each satellite path, then simulates
NLOS excess delay, C/N0 attenuation, and larger code tracking noise. It compares
naive SPP, robust SPP, and a geometry-aware SPP proxy that uses the ray mask.

```bash
PYTHONPATH=python python3 examples/demo_plateau_nlos_simulation.py
```

[`demo_plateau_nlos_simulation.py`](demo_plateau_nlos_simulation.py) runs the
same measurement model on the shipped PLATEAU CityGML sample. It uses the native
BVH ray tracer when available and falls back to CPU triangle ray-casting
otherwise.

```bash
PYTHONPATH=python python3 examples/demo_plateau_nlos_visualization.py
```

[`demo_plateau_nlos_visualization.py`](demo_plateau_nlos_visualization.py)
writes a standalone HTML report to `/tmp/gnss_gpu_plateau_nlos_viz.html` with
the PLATEAU mesh, receiver trajectory, worst-epoch LOS/NLOS sky plot, and error
timeline. The checked-in Pages copy lives at
[`../docs/assets/media/demos/plateau_nlos_visualization.html`](../docs/assets/media/demos/plateau_nlos_visualization.html).

To run the full mask export plus SPP/PF/FGO replay comparison:

```bash
PYTHONPATH=python:. python3 experiments/run_plateau_nlos_demo_suite.py
```

To turn the same ray mask into the experiment CSV contract used by SPP/FGO/PF:

```bash
PYTHONPATH=python:. python3 experiments/export_plateau_nlos_demo_mask.py \
  --out-csv experiments/results/plateau_nlos_demo_mask.csv \
  --summary-json experiments/results/plateau_nlos_demo_mask_summary.json
```

Then replay the mask through the downstream SPP consumer path:

```bash
PYTHONPATH=python:. python3 experiments/replay_plateau_nlos_demo_spp.py \
  --mask-csv experiments/results/plateau_nlos_demo_mask.csv \
  --summary-json experiments/results/plateau_nlos_demo_spp_replay_summary.json
```

Replay the same mask through the downstream particle-filter consumer path:

```bash
PYTHONPATH=python:. python3 experiments/replay_plateau_nlos_demo_pf.py \
  --mask-csv experiments/results/plateau_nlos_demo_mask.csv \
  --summary-json experiments/results/plateau_nlos_demo_pf_replay_summary.json
```

Replay it through the local-FGO consumer path:

```bash
PYTHONPATH=python:. python3 experiments/replay_plateau_nlos_demo_fgo.py \
  --mask-csv experiments/results/plateau_nlos_demo_mask.csv \
  --summary-json experiments/results/plateau_nlos_demo_fgo_replay_summary.json
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
