# Exact grouped reflection directions (D296)

This explicit, default-off native experiment groups traversal entries only when
all bits of the original vertex v0 and the original GPU unit normal match.
Each ray then evaluates the original mirror and direction normalization once per
group. Every original triangle still runs the unchanged intersection, delay and
winner rules in its original traversal ordinal.

The constructor option `grouped_directions=True` captures v0 while the owner
takes its mesh snapshot. First eligible reflection copies the original GPU
normals once to the CPU, groups six uint64 components with full key equality,
and uploads the map and representative ordinals. No CPU normal approximation or
geometric tolerance participates in reuse. These costs belong in cold clocks.

Only ordered meshes with 4096 through 262144 entries, at most 70% unique groups,
at least 512 rays and eight satellites use the new cache. Its 56-byte per
group/ray entries share a 128 MiB bound and epochs are processed in bounded
chunks. A whole epoch that cannot fit falls back to the original path. The
original normal cache remains; all extra GPU buffers count in allocated_bytes.
Small, distinct or unsupported scenes retain the original kernels.

`set_direction_cache_enabled` toggles the experiment on a scene constructed with
the option. A scene without the captured inputs remains on the original path.
Previously allocated caches can remain after toggling; close releases them.

Generate into a new directory with `generate.py --output <new-directory>`.
Use the frozen CMake project in `../bvh_tiled_geometry` with CODE_ROOT pointing
to the projection, CUDA/C++20 Release and the original FMA policy. The projector
checks current D295 source hashes and completely reverses its edits as proof.

Before any speed adoption require exact outputs and resource/lifetime checks,
clean native exits, both whole-call controls and strict A/A calibration. Include
setup, cold and fallback cases; keep all raw observations. Native gains alone
do not establish urban pipeline, PF, FGO or positioning improvements.
