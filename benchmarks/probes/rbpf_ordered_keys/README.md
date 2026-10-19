# Exact ordered integer keys

Standalone CPU-only CPython 3.12 probe for the frozen experimental D236 RBPF
caller. Preserves outer/group/satellite insertion order, empty groups, absolute
integer gauges and original arbitrary-size integer objects. Exact builtin types
only; no sorting, integer conversion, arithmetic, floating point or CUDA code.

The move view still needs a full immutable key. The epoch memo needs only its
eligibility, so that path does not allocate an unused fingerprint. Every request
checks the original Python helper code and its six builtin/global bindings;
custom bindings execute the original helper. Existing memo identities, lifetime,
canonical calls and counters stay unchanged. No new persistent hypothesis cache,
score-index aliases, probability cache or profiling spans.

Use only through the explicit benchmark extension argument. Whole-step speed
and saved state/trajectory parity are required; reduced allocation or request
counts alone do not establish a speed improvement. No default model changes.
