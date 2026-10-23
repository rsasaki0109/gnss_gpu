# gnss_gpu v0.4.0

This release brings the PLATEAU city simulation to the browser, adds five-system
GNSS playback and recorded signal-strength comparison, and consolidates exact-output
performance work in the native and browser pipelines since v0.3.0.

## Interactive city simulation

- Explore the real PLATEAU Odaiba building mesh and recorded UrbanNav receiver route
  over an OpenStreetMap basemap. Inspect direct, reflected and blocked paths in 3D.
- GPS, Galileo, BeiDou, QZSS and GLONASS are enabled together by default, with
  constellation filters, sky directions and receiver-height/elevation controls.
- Replay per-satellite relative power, elevation, path excess and geometric range
  bias. Compare receiver conditions at the same recorded pose.
- Evaluate complete routes or selected intervals, pause/cancel without inventing
  uncomputed results, and export CSV, JSON and reference-receiver GeoJSON.
  Route summaries retain missing fits and report paired comparisons only where
  both conditions have a fit.
- Compare 454,213 original first-band RINEX Sxx samples from two UrbanNav receivers
  on a separate signal-strength axis. Data load on demand, retain original epochs,
  and match only within 2 ms; gaps remain gaps. Export the selected recorded window.

## Performance and precision

- Native PF, ray-tracing, signal-simulation and RB-FGO-PF changes reuse epoch data,
  vectorize double-difference operations and reduce repeated likelihood work.
  Real-input CUDA likelihood replay is available for parity/performance evaluation;
  this does not imply the complete RB-FGO-PF/FGO pipeline runs on the GPU.
- Browser reflection tracing shares geometry and uses strict Float64 WebAssembly
  SIMD with a JavaScript fallback. It retains the original mesh and orbit precision.
- A recorded CPU experiment for shared A/B route evaluation reduced compute time
  from 407.3 ms to 325.8 ms (20.0%, 1.25x) over six poses and seven ABBA rounds,
  with 216 exact-result comparisons. This excludes rendering, startup, transfer
  and GPU work; it is not a whole-application speedup claim.
- A trace-phase profiling tool checks complete snapshots against the baseline
  while reporting direct-ray and reflection work separately.

## Interpretation and limits

The browser models geometric paths and a controlled geometric position fit.
It does not reproduce a complete receiver solution or establish measured positioning
accuracy. Uniform-material Fresnel reflection, polarization and ideal antenna
options are explicit assumptions; they do not modify the positioning solver.
Relative modeled power is not calibrated C/N0. The source RINEX files omit the
signal-strength unit header, so dB-Hz is inferred from the RTKCONV producer and is
shown as inferred. Receiver/band differences, unknown antenna lever arms and
differences between observation and building-model dates limit interpretation.
GLONASS uses broadcast state vectors with Float64 RK4 propagation; numerical
convergence is not an external orbit-accuracy measurement. More constellations
can introduce additional reflected paths and do not guarantee a lower bias.

The research workspace also retains negative results and experimental presets.
Their presence does not promote them into production. Native GPU availability
and performance depend on the CUDA runtime and workload.

## Distribution and reproducibility

The tag-triggered release workflow builds a CUDA wheel (compute capabilities 7.5,
8.0, 8.6 and 8.9), a source distribution, a deterministic reproducibility archive,
and CUDA/ROS 2 container images tagged `v0.4.0`. Assets become available after
their release jobs succeed. The archive preserves the July 2026 v0.3 locked audit
artifacts and labels them as historical; those figures are not new v0.4 benchmarks.

```bash
python tools/build_release_bundle.py --output dist/reproducibility --archive dist/gnss_gpu-v0.4.0-reproducibility.zip
python tools/build_release_bundle.py --verify dist/reproducibility
```

[Browser demo](https://rsasaki0109.github.io/gnss_gpu/simulation.html) ·
[Changes since v0.3.0](https://github.com/rsasaki0109/gnss_gpu/compare/v0.3.0...v0.4.0)
