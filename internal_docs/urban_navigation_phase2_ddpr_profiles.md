# Phase 2 DDPR profile and satellite-screen contract

`gnss_gpu.ddpr_profiles` adds three truth-free offset models:

- constant;
- affine;
- continuous piecewise linear.

Fits use weighted least squares over ECEF offset evidence. They fail closed on
thin or ill-conditioned evidence. Model selection requires both a lower BIC
and at least 15% residual improvement before increasing complexity, so a
normal constant block remains constant.

## Satellite arc screen v2

The v2 screen operates per satellite arc rather than permanently excluding a
satellite across a whole segment. An epoch gap starts a new arc. Persistent,
well-supported outliers are hard-excluded; sparse or intermittent evidence is
soft-weighted. Equal-size triple-difference clusters are marked ambiguous
instead of selecting one by lexical/input order.

The final DDPR weight combines:

1. the estimator's base observation weight;
2. arc quality;
3. the geometric mean of independent Evidence API family supports;
4. family coverage;
5. temporal stability.

This avoids both single-channel domination and the WP158 behavior where a
satellite seen once could be excluded at outlier fraction 1.0.

## Structural cases

`experiments/audit_phase2_structural_cases.py` replays the stored truth-free
WP163/WP164 refit artifacts:

- WP163 is the offset-shape case. At least two carrier-reference ranks must
  recover one candidate under the 0.5 m affine block-residual and 4.0 m DDPR
  gates.
- WP164 is the thin/biased-evidence inverse. Its block offsets are already
  stable, but DDPR remains above 5.3 m. Affine fitting must not make any
  hypothesis pass the unchanged 4.0 m DDPR gate.
- M4 hashes must remain exact.

The audit reads no post-selection truth fields. `audit_*` fields present in
the historical development JSON are deliberately ignored.
