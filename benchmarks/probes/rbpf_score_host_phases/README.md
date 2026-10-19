# Host score statement diagnostics

Use benchmarks/rbpf_score_host_phases_v2.py and the checked test module.
The initial rbpf_score_host_phases.py is retained as a rejected import-time
bytecode assertion experiment;do not import it for execution. A separate
bytecode diagnostic records differing LOAD_GLOBAL/LOAD_ATTR call forms from
isolated compilation. v2 unmarked control reuses the original code object.
Marked code restores the original statement AST after diagnostic removal;
all original numeric globals remain live through the same module dict.
Helpers are closure cells,not inserted into original module globals.

Initial collection and later test-observation failures are retained with logs;
their source files moved byte-for-byte to tests/fixtures/non-test names.
The unsafe original scalar path raises math-domain ValueError;checked tests
compare the identical errors before examining trace cleanup. Implementation
v2 was not edited after numerical execution. No new CUDA source or binary,
observation/hypothesis/particle/prior/FP64/order/threshold/RNG change.
Host elapsed spans include callbacks,scheduling,marker overhead,ctypes and
native waits;native span is not kernel time,utilization or wholePF GPU share.
Whole-score paired calibration is required before interpreting percentages.
