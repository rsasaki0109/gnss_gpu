# Fused exact memo/index staging probe

Standalone CPU CPython3.12 experiment with off/on functions in one extension.
The exported module name is retained for the existing explicit staging loader;
the binary lives in a separate build directory and is never installed.
Original particle/move/prior/lookup/callback order is retained on misses.
Fuse only already completed builtin-key lookups with the literal memo hit;
increment the same logical memo counters, preserve all source calls on misses.
Guard code/globals/builtins/class versions, bound method, retained H/key,
counts and lookup mutations. Custom keys or changed bindings bypass fusion.
No floating math/score/RNG changes, probability cache or cross-epoch cache.
