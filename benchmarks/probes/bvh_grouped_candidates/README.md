# Register reuse for exact reflection groups (D297)

This separate experiment removes D296's global group/ray direction cache.
Groups still require bitwise equality of original v0 and original GPU normals;
each thread computes the original mirror and normalization once, then evaluates
all group members using the original intersection and delay arithmetic. A group
has at most eight members. Every valid original ordinal remains in the CSR map,
and the existing (delay, ordinal) winner comparison preserves ties.

Only static integer group/member buffers are added. No group/ray floating-point
cache, preparation launch or large epoch buffer is needed. The original bounded
winner scratch and existing ray chunks are used. First-call normal readback,
CPU grouping and uploads still belong in cold clocks. Default is off; repeated
groups, bulk rays and the existing 70% group-count threshold are required.

The constructor option is `grouped_candidates=True`. Diagnostic properties are
`group_count`, `group_metadata_bytes`, `grouped_candidates_active`; toggling is
`set_grouped_candidates_enabled`. Unsupported and small calls use the old path.

The projector composes frozen D296 edits and reverses all composition edits,
then chains D296's full undo proof back to the frozen D295 source. Existing GPU
functions and CPU builder remain literal. Use the previous frozen CUDA/C++20
CMake project with an explicit CODE_ROOT to build the generated tree.

Require exact results, ownership checks and clean CUDA tools before speed.
Measure full warm and cold API calls with both controls and strict A/A, retain
all times and fallback cases, and freeze inputs and gates before clocks.
D296's rejection and its data remain unchanged.
