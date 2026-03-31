# PLATEAU 3D City Model + Particle Filter Urban Positioning -- Expected Output

## Run Command

```bash
PYTHONPATH=python python3 examples/demo_plateau_urban.py
```

## Expected Output Format

```
========================================================================
  PLATEAU 3D City Model + Particle Filter Urban Positioning
========================================================================

[1] Loading PLATEAU CityGML data
    File: /path/to/gnss_gpu/data/sample_plateau.gml
    Source    : gnss_gpu (PlateauLoader)
    Triangles: 36
    Buildings: 3 (bldg_001: 20m, bldg_002: 45m, bldg_003: 12m)

[2] Defining pedestrian trajectory
    Epochs: 60 at 1s intervals
    Total distance : ~100 m
    Start (ENU)    : (-10.0, -20.0) m
    Turn  (ENU)    : (32.0, -20.0) m  [epoch 35]
    End   (ENU)    : (32.0, 10.0) m

[3] Generating observations (8 satellites x 60 epochs)
    LOS check  : CPU ray tracing (or GPU ray tracing if CUDA available)
    LOS signals: ~350-400 / 480
    NLOS signals: ~80-130 / 480
    Multipath bias : mean=~35 m (NLOS only)
    Noise (LOS)    : sigma=3.0 m
    Noise (NLOS)   : sigma=15.0 m

[4] WLS positioning (no building model)
    Source: CPU (or GPU if CUDA available)
    Mean error: ~15-30 m

[5] ParticleFilter (standard, no building model)
    Source    : CPU (or GPU if CUDA available)
    Particles: 20000
    Mean error: ~10-25 m

[6] ParticleFilter3D (NLOS-aware with building model)
    Source    : CPU (or GPU if CUDA available)
    Particles: 20000
    Mean error: ~5-15 m

[7] Epoch-by-epoch errors (3D position error in metres)
    Epoch  NLOS       WLS        PF      PF3D    Best
    --------------------------------------------------
        0     2     18.43     12.56      8.21    PF3D
        1     2     19.12     11.34      7.88    PF3D
      ...   ...       ...       ...       ...     ...
       59     1     14.56      9.23      6.45    PF3D

[8] Summary statistics
    Method                              Mean      Std     95th      Max
    ------------------------------------------------------------------------
    WLS (no building model)            18.50     6.30    28.90    35.20
    PF (no building model)             12.40     4.80    20.50    26.30
    PF3D (NLOS-aware, PLATEAU model)    7.20     3.10    12.80    16.50

    PF3D improvement over WLS: +11.30 m mean error
    PF3D improvement over PF : +5.20 m mean error
    PF3D best in 48/60 epochs (80%)

[9] Generating VulnerabilityMap GeoJSON
    Source : CPU (or GPU if CUDA available)
    Output : /path/to/gnss_gpu/output/plateau_vulnerability_map.geojson
    Features: 441

========================================================================
  Experiment complete.
  Key finding: PF3D with PLATEAU 3D building model achieves
  +11.3 m improvement over WLS and +5.2 m over standard PF
  in this urban canyon scenario with ~100 NLOS observations.
========================================================================
```

## Notes

- Exact numerical values vary with the random seed (default: 42).
- The pattern -- PF3D outperforming PF outperforming WLS -- is consistent
  across seeds when NLOS satellites are present.
- With GPU (CUDA) available, the GPU kernels handle ray tracing and particle
  filter updates. Without CUDA, pure-Python CPU fallbacks are used, which
  produce the same algorithmic results but run slower.
- The VulnerabilityMap GeoJSON can be viewed in any GeoJSON viewer
  (e.g., geojson.io) to visualise HDOP across the Tokyo Station area.

## Key Observations

1. **WLS** treats all pseudoranges equally. NLOS multipath bias (20-50 m)
   directly contaminates the position solution.

2. **PF (standard)** improves over WLS through temporal filtering and
   motion model constraints, but still weights NLOS observations the same
   as LOS ones.

3. **PF3D (NLOS-aware)** uses the PLATEAU 3D building model to classify
   each satellite-particle pair as LOS or NLOS. NLOS observations are
   down-weighted (wider sigma) and bias-corrected, preventing multipath
   from dragging the position estimate.
