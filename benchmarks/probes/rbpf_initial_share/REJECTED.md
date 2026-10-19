# Rejected first implementation

Do not enable this first implementation. Although its 110 numerical tests
passed, CUDA Racecheck reported two hazard groups (one error, one warning)
between a shared reuse-flag read and its overwrite. No timing used this DLL.

The separately saved `../rbpf_initial_share_checked` implementation separates
immutable eligibility from acceptance;it passed the same 110 tests and all
three 36-test CUDA checker subsets with zero errors/hazards/leaks. Preserve
this source and the original reports as a negative control.
