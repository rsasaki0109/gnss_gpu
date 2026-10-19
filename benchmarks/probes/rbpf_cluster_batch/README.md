# Batched exact cluster-state selection

Explicit CPython 3.12 CPU extension and frozen corrected register CUDA scorer.
Only the score parameter block changes: state selection is batched, followed
by the same NumPy centre/prior packing. No GPU numerical code is changed.
All hypotheses, parents and observations remain. Arbitrary Python integer
subtraction and Python int true division preserve the original thresholds.
Streaming lexicographic maximum and tie rejection select the same state
object as stable ranking. Unsupported objects/bindings/hooks/policies use
the literal original parameter block once; no cross-call/epoch cache.

Full ordinary-input preflight precedes new integer arithmetic. Parent maps
are validated once per batch. Existing canonical/parent engine counters
remain logical request counts; batch_calls reports actual fused C API calls.
One owning returned list holds selected states only while parameters pack.
The extension retains no input references and allocates no CUDA storage.
No default installation or product enablement; complete-score controls and
state/diagnostic equivalence must pass before any speed claim.
