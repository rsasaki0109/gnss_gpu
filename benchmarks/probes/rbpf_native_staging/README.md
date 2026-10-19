# Literal native candidate staging

Explicit CPU-only CPython 3.12/Win64 probe on the D236 RBPF caller. Registers
every particle's original candidates, in original order, through C++ Python C
API calls. Calls the original canonical callback for every move, retains every
value hash/lookup and first-seen key/hypothesis, and runs the original extend
and shrink methods. Each release creates a fresh empty dict.

Prior addition/division still use Python number operations in original order.
No native floating-point expressions, reassociation, integer conversion,
canonical/index alias cache, probability cache, common record sharing or CUDA
changes. Likelihood/normalization/weight/RNG choice and the entire remaining
step stay literal. No D239/D240 candidate or profiler is combined.

Private helper scopes refresh per step; unsupported caller/list/cache/prior
bindings use the original literal staging block once. New counters are outside
runtime state. GIL-held temporary lists/tuples/references are released normally;
no persistent staging workspace or cross-epoch hypothesis cache is added.

Use `NativeStagingStep(..., staging_extension=<explicit pyd>, extension=<canonical
pyd>, parent_extension=<parent pyd>, mass_extension=<mass pyd>)` with a caller
owned scorer for native numerical modes. Whole-step speed and exact state/
trajectory checks are required. Candidate-loop size or fewer Python frames
alone do not establish an effective improvement. Product/defaults stay unchanged.
