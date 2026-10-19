# D298: lazy exact reflection groups

This experimental projector builds on the frozen D297 register group candidate.
It removes the constructor's CPU vertex snapshot. LOS-only and ineligible calls
do no group preparation. The first eligible requested reflection packs original
vertices from the scene's owned GPU snapshot, reads them back, frees the temporary,
and applies the original full-bit normal/vertex key and bounded-eight CSR grouping.
All original reflection arithmetic, candidates, ordering and tie rules remain.
Default construction stays disabled; enabling later uses the same owned snapshot.

`group_preparation_count` reports successful vertex preparations, including a
grouping attempt rejected for insufficient sharing. `group_preparation_scratch_bytes`
reports the last temporary GPU request (24 bytes per ordered triangle), already
released before CPU grouping and API return. `allocated_bytes` reports live owned
allocations and does not include this released temporary. Peak memory must therefore
account for the temporary separately. No permanent CPU vertex snapshot is retained.

Generate with Python `generate.py --output <new-directory>`, then build using
`../bvh_tiled_geometry/CMakeLists.txt` with `CODE_ROOT` pointing to that directory.
This uses CUDA C++20 and the ordinary FMA policy without fast math. The projection
asserts its D297 parent hashes and proves full-source undo through D295. Existing
13 compiled GPU functions must match SASS exactly; the only additional kernel copies
vertices and performs no floating point arithmetic.

The first D298 screen is fixed before execution: four unchanged cases, original,
same-binary off/on and both A/A controls; five rounds, three groups, two warm
repetitions and eight interleaved cold repetitions. The additional cold repetitions
address the saved D297 identical-control variance; all raw times remain included.
Acceptance thresholds stay unchanged: bulk warm >=10% and four positive rounds,
small/cold >=-5% and four rounds >=-5%; each A/A absolute median <5% and four
absolute rounds <5%. Every condition must pass before a public/urban stage.
The completed D296/D297 measurements and their failed combined gates are retained.
