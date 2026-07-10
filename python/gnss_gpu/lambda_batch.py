"""Batched LAMBDA/MLAMBDA ambiguity resolution (CUDA).

Faithful batched port of cssrlib's ``mlambda.py`` (LDL -> LLL reduction
-> estimILS search; parmode=2 PAR via parsearch with exclmax=1). One
kernel launch evaluates MANY independent (ahat, Qahat) problems -- the
shape of the tc/-style subset/retry AR cascade, where a failing epoch
fires 10-30 sequential LAMBDA calls on slightly different candidate
subsets.

Each result mirrors what ``cssrlib.mlambda.mlambda`` returns for the
same input:

    afix : (n, ncands) float array (or the 1-D float vector on a
           parmode=2 rejection, exactly like cssrlib)
    s    : (ncands,) squared norms, or an empty array on a parmode=2
           rejection (cssrlib returns ``[]``)
    nfix : number of fixed ambiguities
    Ps   : bootstrapped success rate (NaN on a PAR rejection)

plus a ``status`` code: 0 = ok, 1 = non-positive-definite covariance
(cssrlib raises ``LambdaError``), 2 = unsupported dimensions.
"""

from dataclasses import dataclass

import numpy as np

try:
    from gnss_gpu._gnss_gpu_lambda_batch import (
        mlambda_batch as _mlambda_batch_native,
        lambda_batch_max_n as _lambda_batch_max_n,
    )
    HAS_LAMBDA_BATCH = True
except ImportError:
    HAS_LAMBDA_BATCH = False

STATUS_OK = 0
STATUS_NOT_POSITIVE_DEFINITE = 1
STATUS_UNSUPPORTED = 2


@dataclass(frozen=True)
class MlambdaResult:
    """One problem's mlambda-equivalent output."""

    afix: np.ndarray
    s: np.ndarray
    nfix: int
    Ps: float
    status: int
    zfix: np.ndarray  # decorrelated-domain solution (rows k: are integer)


def lambda_batch_max_n():
    """Maximum per-problem ambiguity count supported by the kernel."""
    if not HAS_LAMBDA_BATCH:
        raise RuntimeError("lambda_batch CUDA module not available")
    return int(_lambda_batch_max_n())


def mlambda_batch(ahats, Qs, ncands=2, parmode=2, P0=0.995):
    """Run cssrlib-equivalent mlambda on a batch of problems.

    Parameters
    ----------
    ahats : sequence of 1-D float arrays (float ambiguities)
    Qs    : sequence of matching (n, n) covariance matrices
    ncands, parmode, P0 : as in ``cssrlib.mlambda.mlambda``
        (parmode 1 = full ILS, 2 = PAR/parsearch)

    Returns
    -------
    list[MlambdaResult], one per problem, in input order.
    """
    if not HAS_LAMBDA_BATCH:
        raise RuntimeError("lambda_batch CUDA module not available")
    if len(ahats) != len(Qs):
        raise ValueError("ahats and Qs must have the same length")
    if len(ahats) == 0:
        return []

    n_arr = np.empty(len(ahats), dtype=np.int32)
    a_parts = []
    q_parts = []
    for i, (a, q) in enumerate(zip(ahats, Qs)):
        a = np.ascontiguousarray(a, dtype=np.float64).ravel()
        q = np.ascontiguousarray(q, dtype=np.float64)
        n = a.size
        if n < 1:
            raise ValueError(f"problem {i}: empty ahat")
        if q.shape != (n, n) and q.size != n * n:
            raise ValueError(f"problem {i}: Q shape does not match ahat")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(q))):
            raise ValueError(
                f"problem {i}: non-finite input (guard on the caller side, "
                "matching the pipeline's WP13m finite guard)")
        n_arr[i] = n
        a_parts.append(a)
        q_parts.append(q.ravel())

    ahat_flat = np.concatenate(a_parts)
    q_flat = np.concatenate(q_parts)

    afix_flat, zfix_flat, s_flat, s_len, nfix, ps, status = \
        _mlambda_batch_native(ahat_flat, q_flat, n_arr,
                              int(ncands), int(parmode), float(P0))

    results = []
    a_off = 0
    for i, n in enumerate(n_arr):
        n = int(n)
        afix = afix_flat[a_off * ncands:(a_off + n) * ncands].reshape(
            n, ncands)
        zfix = zfix_flat[a_off * ncands:(a_off + n) * ncands].reshape(
            n, ncands)
        if int(s_len[i]) == 0:
            s = np.zeros(0)
            afix_out = afix[:, 0].copy()  # cssrlib returns 1-D here
        else:
            s = s_flat[i * ncands:(i + 1) * ncands].copy()
            afix_out = afix.copy()
        results.append(MlambdaResult(
            afix=afix_out, s=s, nfix=int(nfix[i]), Ps=float(ps[i]),
            status=int(status[i]), zfix=zfix.copy()))
        a_off += n
    return results
