# gnss_gpu — GPU-Accelerated GNSS Positioning Library

GPU-accelerated GNSS signal processing and positioning library using CUDA, featuring mega-particle filtering for urban multimodal positioning.

## Features

### Core Positioning
- **WLS Batch Positioning** — GPU-parallel Weighted Least Squares (9.2M epochs/s)
- **EKF Positioning** — Extended Kalman Filter with constant-velocity model
- **Multi-GNSS** — GPS/GLONASS/Galileo/BeiDou with ISB estimation
- **RTK** — Carrier phase positioning with LAMBDA ambiguity resolution

### Particle Filter
- **Mega Particle Filter** — 1M+ particles on GPU (13.3M particles/s)
- **Megopolis Resampling** — Numerically stable, no prefix-sum
- **SVGD** — Stein Variational Gradient Descent (no sample impoverishment)
- **3D-Aware PF** — Ray tracing integrated NLOS-aware likelihood

### Signal Processing
- **Signal Acquisition** — cuFFT-based parallel GPS L1 C/A acquisition
- **Vector Tracking** — Coupled DLL/PLL + EKF navigation filter
- **Interference Detection** — STFT-based jamming detection & excision

### Urban Environment
- **3D Ray Tracing** — Moller-Trumbore NLOS detection
- **Multipath Simulation** — Signal-level DLL error model
- **Vulnerability Mapping** — DOP/visibility grid evaluation (11M pts/s)
- **PLATEAU Integration** — Japanese 3D city model (CityGML) loader

### Atmospheric Correction
- **Troposphere** — Saastamoinen model
- **Ionosphere** — Klobuchar broadcast model

### I/O
- RINEX 3.x observation & navigation file parser
- NMEA reader/writer (GGA, RMC, GSA, GSV, VTG)
- PLATEAU CityGML loader

### Integration
- ROS2 node with PointCloud2 particle visualization
- Matplotlib & Plotly visualization tools

## Quick Start

### Build
```bash
pip install .
# or
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc)
```

### Usage
```python
import gnss_gpu

# WLS positioning
positions, iters = gnss_gpu.wls_batch(sat_ecef, pseudoranges, weights)

# Mega particle filter
pf = gnss_gpu.ParticleFilter(n_particles=1_000_000)
pf.initialize(initial_position)
pf.predict(dt=1.0)
pf.update(sat_ecef, pseudoranges)
estimate = pf.estimate()

# 3D-aware particle filter
from gnss_gpu import BuildingModel, ParticleFilter3D
buildings = BuildingModel.from_obj("city.obj")
pf3d = ParticleFilter3D(buildings, n_particles=100_000)

# Load PLATEAU 3D city model
from gnss_gpu.io import load_plateau
buildings = load_plateau("path/to/plateau/", zone=9)
```

## Architecture
```
Python API (gnss_gpu)
    | pybind11
CUDA Kernels
    |-- Positioning (WLS, EKF, RTK, Multi-GNSS)
    |-- Particle Filter (predict, weight, resampling, SVGD)
    |-- Signal Processing (acquisition, tracking, interference)
    |-- Ray Tracing (LOS/NLOS, multipath)
    +-- Utilities (coordinates, atmosphere, ephemeris)
```

## Performance Benchmarks
| Module | Input | Time | Throughput |
|--------|-------|------|------------|
| WLS Batch | 10K epochs | 1.1 ms | 9.2M epoch/s |
| Particle Filter | 1M particles | 75 ms | 13.3M part/s |
| Acquisition | 32 PRN | 263 ms | 122 PRN/s |
| Vulnerability Map | 10K grid | 0.9 ms | 11M pts/s |
| Ray Tracing | 1K tri x 8 sat | 0.9 ms | 9.5M checks/s |

## References
- Koide et al., "MegaParticles", ICRA 2024. arXiv:2404.16370
- Chesser et al., "Megopolis Resampling", 2021. arXiv:2109.13504
- Liu & Wang, "Stein Variational Gradient Descent", 2016

## License
Apache-2.0
