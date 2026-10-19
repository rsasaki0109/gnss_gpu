# Sparse complete PF wall boundaries

Diagnostic only. Reproduce the unchanged D235 complete-step AST and the unchanged D236 feedback wrapper, adding sparse direct integer wall markers at top-level phase boundaries. Three existing move adapters redirect only their callee through a closure; first/last math, record construction, canonical/DD/counter behavior, every candidate/prior/particle and every RNG choice are retained.

No per-particle trace, profile hook, context manager per method, runtime/global method replacement, native/CUDA rebuild or extra device transfer. The clone control retains the exact original inner code object and both owning wrapper code objects. All numeric globals stay live. Markers and their recorder are closure-owned; only an immutable tuple of strings and integers remains after each call, including errors and skipped steps.

Adjacent marks partition the recorded wall interval exactly. They include scheduling, allocation, waits and instrumentation. A phase share is neither GPU utilization nor a causal speedup estimate. Complete PF calibration includes all recording/setup/cleanup; state restore, callback rebinding and validation stay outside the external timer. Predeclare controls and calibration acceptance, retain every round and fail, and do not retime completed experiments for adoption.
