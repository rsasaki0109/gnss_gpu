# Exact extend support and integer assembly probe

Explicit CPU CPython 3.12/Win64 experiment. Build the separate CMake target;
never install it or change the frozen runtime/default dispatch. The instance
controller substitutes two blocks in the original `_extend`; undoing them
restores the complete original AST. D236 remains the comparison baseline.

Expand every pair in its original order, subtract arbitrary Python integers
before binary64 conversion, and preserve conversion errors even when there
will be no extension. Retain original held/missing index ordering, empty-move
decisions, `np.ix_`, `np.linalg.solve`, residual subtraction, matrix product,
and conditional posterior subtraction. Assembly uses the original `np.rint`
ufunc once on an exact finite contiguous binary64 vector, then the same
arbitrary Python integer conversion/subtraction in original pair order.
Preserve group/satellite insertion order, absolute gauge and independent copies.

Exact source/method/binding/type/buffer guards select the native domain. Other
inputs execute the original block once, including scalar warnings/conversion
errors; strict NumPy policies retain the original whole step. GIL held; no
cross-call H/DD/problem cache or new persistent/native/CUDA workspace. All
particles, candidates, prior arithmetic, score/choice/RNG/FIX/feedback/graph
work remains. No new CUDA numerical code or additional GPU transfer.

Performance protocol: one shared register-LU scorer, complete D236 CPU/GPU
against candidate CPU/GPU and corresponding original A/A controls. Eight fixed
conditions, five rounds, three rotating groups. All new guards, temporary
allocation, native calls and original solver/assembly work are timed. Preserve
every result; require at least 10% paired reduction, four improving rounds of
five and matching A/A median below 5% in magnitude. GPU joint screen additionally
compares candidate GPU with candidate CPU. No repeat of a completed timing to
force acceptance. Fixture extraction/search/graph inputs are declared boundaries.
