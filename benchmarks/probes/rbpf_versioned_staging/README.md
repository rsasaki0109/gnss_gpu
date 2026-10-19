# Per-request version-guarded candidate registration

Standalone CPython 3.12/Win64 CPU experiment, never installed. One extension
exports the complete unchanged D241 and D244 C++ control loops in a separate
namespace, and a new D244 variant. The source generator reverses every explicit
change to reconstruct the complete frozen original C++ text.

The new variant still checks mutable state on every original canonical/index
request. Retain dictionary objects and CPython dictionary slots after ordinary
attribute/type/binding guards. Pointer/version changes or class/layout changes
deoptimize before the next hit. Canonical code/globals/builtins, lookup, context,
memo dictionaries and counts remain guarded. Keep cached Python integer IDs and
validate retained H/key entries again when the identity dictionary version
changes. Unchanged identity versions require no repeated allocation/lookup.

All source canonical calls on misses, logical counters, Python prior arithmetic,
first-seen keys/H/pending, record order, particles/candidates, original weights,
normalization and RNG choice remain. Source adapter/AST restoration is the
unchanged D241 mechanism. No probability/common-record cache, numerical rewrite,
new CUDA code, transfer or cross-call H/problem/value cache. Only native loops
hold the GIL; existing original callbacks keep their own behavior.

The implementation uses CPython 3.12 private dictionary version/slot APIs.
Guard dictionaries, Python integer IDs and aliases own references only during
one staging call; release them on errors and before the original score/choice
tail. Logical map payload includes H/key/index/ID pointers, version and map key;
Python object/allocator/map bookkeeping and heap peak are separate, unmeasured.

Performance: full D236 CPU/GPU against candidate CPU/GPU, unchanged D244 off
CPU/GPU in the same extension, and matching original CPU/GPU A/A. Eight fixed
conditions, four initial RNG states, five rounds, three rotating groups, one
shared register-LU scorer. Require 10% paired reduction and four improving
rounds of five against original and off, with matching A/A below 5% absolute.
GPU joint additionally requires improvement against candidate CPU. Include all
guards, allocation, original callbacks/scoring/transfers/choice/conditional/FIX/
feedback/resampling. Preserve all negative/noisy results; no completed retiming.
Extract/search/time/satellite/graph-solve inputs are declared fixture boundaries.
