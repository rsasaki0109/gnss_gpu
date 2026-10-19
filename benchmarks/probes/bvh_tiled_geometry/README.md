# Explicit tiled reflection inputs

This probe stores the raw vertices and original GPU-computed normals of32
consecutive traversal entries in component arrays. Neighboring candidate
threads read adjacent doubles. The existing intersection, distances, winner
selection, tie order, LOS and fallback kernels remain unchanged.

The new scene methods are experimental and default off. The tiled path requires
an ordered mesh of at least4096triangles and at least512rays. It keeps the
original geometry and uses3200bytes per32-entry tile, capped at128MiB.
Small, serial and unsupported traversal cases retain the original path.
Toggling layouts can retain both caches; allocated_bytes includes both.

generate.py verifies the frozen baseline hashes and exactly reverses all
changes back to the original complete sources. It writes a separate project;
it does not edit or install the product. Build using the included CMake file
and explicit CODE_ROOT/Python3_EXECUTABLE/pybind11_DIR options.

Before adoption require exact original/off/on output bytes, all new kernels
under CUDA memory/race/synchronization checks, and whole trace_batch timings
against original and same-binary off controls with independent A/A controls.
Include first-use preparation and memory costs. No accuracy approximation or
new GPU-utilization/offload percentage follows from this layout experiment.
