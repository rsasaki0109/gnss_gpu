# Frozen BVH source, 2026-10-19

These files retain the original CPU builder and GPU implementation used in D294.
Their raw SHA-256 values are checked before projection; the local `.gitattributes`
keeps their line endings intact. The original source contains a CPU recursion
overflow for long duplicate-triangle chains. Do not install this frozen builder.

The product uses the separately validated iterative scheduler from D295. Its
split arithmetic, sorting, ties, node numbering and entire GPU source are kept.
`bvh_iterative_build/generate.py` reproduces that CPU-only change from this tree.

The rejected D294 tile experiment can still be generated with
`python benchmarks/probes/bvh_tiled_geometry/generate_frozen.py --output <new-directory>`.
This driver points the unchanged original projector at this frozen tree. Build
instructions and the measured rejection remain in the D294 evaluation.
