"""Time optimized CPU solves of frozen VD first-step matrices; component only."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    _pin()
    rows = {}
    with threadpool_limits(limits=1, user_api='blas'):
        for epochs in (5, 20, 64, 100):
            prefix = args.matrix_dir / f'vd_imu_{epochs}_off'
            paths = {label: Path(str(prefix) + suffix) for label, suffix in (
                ('matrix', '_H.csv'), ('rhs', '_g.csv'), ('native_step', '_accepted_delta.csv'))}
            matrix = np.loadtxt(paths['matrix'], delimiter=',')
            rhs = np.loadtxt(paths['rhs'])
            reference = np.loadtxt(paths['native_step'])
            assert matrix.shape == (epochs * 8, epochs * 8)
            asymmetry = float(np.max(np.abs(matrix - matrix.T)))
            # Native CPU Cholesky reads only the original lower triangle.
            # Do not average or modify the matrix; reflect that triangle only
            # in the diagnostic equation used to calculate its residual.
            lower = np.tril(matrix)
            expected_matrix = lower + np.tril(matrix, -1).T + np.eye(len(rhs)) * 1e-3

            def solve():
                work = matrix.copy(order='F')
                work.flat[::len(rhs) + 1] += 1e-3
                factor = cho_factor(work, lower=True, overwrite_a=True, check_finite=False)
                return cho_solve(factor, rhs, check_finite=False)

            solution = solve()
            absolute = float(np.max(np.abs(solution - reference)))
            residual = float(np.linalg.norm(expected_matrix @ solution - rhs, ord=np.inf))
            assert np.allclose(solution, reference, rtol=1e-10, atol=1e-10)
            samples = []
            for _ in range(5):
                for _ in range(3):
                    begin = time.perf_counter_ns()
                    result = solve()
                    samples.append((time.perf_counter_ns() - begin) / 1e6)
                    assert np.array_equal(result, solution)
            rows[str(len(rhs))] = dict(input_sha256={label: sha(path) for label, path in paths.items()},
                p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)), samples_ms=samples,
                maximum_native_first_step_abs_difference=absolute, equation_residual_inf_norm=residual,
                input_maximum_asymmetry=asymmetry, original_lower_triangle_used=True)
    args.output.write_text(json.dumps(dict(measurements=rows, helper_sha256=sha(Path(__file__)),
        scope='Single frozen first-step matrix only: matrix copy, original1e-3 diagonal jitter, CPU LAPACK Cholesky and solve; one pinned CPU and one BLAS thread. CSV loading/reference checks outside timing. Not full FGO or a directly paired GPU comparison.'), indent=2) + '\n', encoding='utf8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
