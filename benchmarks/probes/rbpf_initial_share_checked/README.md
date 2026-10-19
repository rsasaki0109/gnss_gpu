# Exact first-system GPU sharing

Explicit standalone experiment on the frozen corrected register-LU scorer.
First-stage normal/LU bits can be shared only for eight-group systems with
identical ordered finite-DD support and bit-identical prior covariance.
All hypotheses/observations,original row and FP64 arithmetic order,each RHS
solve,both IRLS updates,likelihood diagnostics and Python fallbacks remain.
Nonmatching/unsafe/failed templates take the original numerical path.

The old CPU body,original GPU kernels and register solver remain in this DLL.
Sharing is off at workspace creation;an explicit API enables it. Per-call
owners are packed into the existing input copy;templates/use flags are owned
by the workspace. Timing includes grouping and both GPU launches. The optional
accuracy query alone reads usage flags;normal scores add no output transfer.
No cross-call numerical cache or automatic runtime/default installation.

Checked version separates immutable reuse eligibility from the shared acceptance
flag. Acceptance is written only after normal/RHS preparation and read only
after the next block barrier;it is not read while being overwritten. The v1
read/write hazard and all failed-source/PE/checker evidence remain preserved.
