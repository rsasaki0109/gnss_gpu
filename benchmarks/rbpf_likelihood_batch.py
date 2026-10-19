"""Experimental C++/CUDA likelihood batch; no automatic runtime installation."""

import ctypes
import math
import os
from pathlib import Path
import time

import numpy as np

MAX_ROWS = 128
MAX_GROUPS = 8
MAX_D = 11
DIAG_SIZE = 3 + MAX_D + MAX_D * MAX_D


class BatchScorer:
    """Explicit DLL, owning workspace, per-call packing and scalar fallback.

    This is a component prototype. Model formulas/IRLS counts are retained;
    floating-point reductions and library results need not match NumPy bits.
    A completed score never changes the supplied graph, particles or RNG.
    """

    def __init__(self, dll, capacity=512):
        self.capacity = capacity
        self.cuda_directory = os.add_dll_directory(str(Path(os.environ['CUDA_PATH']) / 'bin'))
        self.dll = ctypes.CDLL(str(Path(dll).resolve()))
        self.dll.rbpf_create.argtypes = [ctypes.c_int]
        self.dll.rbpf_create.restype = ctypes.c_void_p
        self.dll.rbpf_destroy.argtypes = [ctypes.c_void_p]
        self.dll.rbpf_destroy.restype = None
        doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        integers = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.dll.rbpf_score.argtypes = [ctypes.c_void_p, doubles, integers, doubles, doubles, doubles, doubles,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, doubles, integers, ctypes.c_void_p]
        self.dll.rbpf_score.restype = ctypes.c_int
        start = time.perf_counter_ns()
        self.handle = self.dll.rbpf_create(capacity)
        self.setup_ms = (time.perf_counter_ns() - start) / 1e6
        if not self.handle:
            self.cuda_directory.close()
            raise RuntimeError('Standalone likelihood workspace allocation failed')

    def close(self):
        if self.handle:
            self.dll.rbpf_destroy(self.handle)
            self.handle = None
            self.cuda_directory.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    @staticmethod
    def scalar(hypotheses, problem, oracle):
        cache = {}
        return np.asarray([oracle._loglik(h, problem, cache) for h in hypotheses])

    def score(self, hypotheses, problem, oracle, mode='gpu', diagnostics=False):
        """One complete requested batch, including dedup/pack/transfers/readback."""
        assert self.handle
        assert mode in ('gpu', 'native_cpu')
        if not hypotheses:
            return np.empty(0), dict(unique=0, scalar_fallbacks=0)
        cfg = oracle.cfg
        # Same gauge-invariant key as the original cache; stable first request.
        keys, unique, scatter = {}, [], []
        for hypothesis in hypotheses:
            key = oracle.basin_id(hypothesis)
            if key not in keys:
                keys[key] = len(unique)
                unique.append(hypothesis)
            scatter.append(keys[key])
        m, nb = len(problem.rows_v0), len(problem.pairs)
        numeric = ('rows_J', 'rows_v0', 'rows_sig', 'rows_lam', 'rows_bmeas', 'y', 'Qb', 'pos', 'Ppos')
        if any(type(getattr(problem, key)) is not np.ndarray or getattr(problem, key).dtype != np.float64 for key in numeric):
            return self.scalar(hypotheses, problem, oracle), dict(unique=len(unique), scalar_fallbacks=len(unique), unsupported=True)
        groups = {}
        for r in range(m):
            if problem.rows_iscp[r]:
                group = problem.pairs[int(problem.rows_pair[r])][0]
                if group not in groups:
                    groups[group] = len(groups)
        if len(unique) > self.capacity or m > MAX_ROWS or not 0 < nb <= MAX_ROWS or len(groups) > MAX_GROUPS:
            return self.scalar(hypotheses, problem, oracle), dict(unique=len(unique), scalar_fallbacks=len(unique), unsupported=True)
        rows = np.ascontiguousarray(np.column_stack((problem.rows_J, problem.rows_v0, problem.rows_sig,
            problem.rows_lam, problem.rows_bmeas)), dtype=np.float64)
        meta = np.empty((m, 3), dtype=np.int32)
        meta[:, 0] = problem.rows_iscp
        meta[:, 1] = problem.rows_pair
        meta[:, 2] = [groups[problem.pairs[int(problem.rows_pair[r])][0]] if problem.rows_iscp[r] else -1 for r in range(m)]
        pairs = np.ascontiguousarray(np.column_stack((problem.y, np.diag(problem.Qb))))
        dd = np.ascontiguousarray([oracle._pair_dd(h, problem) for h in unique])
        parameters = np.empty((len(unique), 12))
        for i, hypothesis in enumerate(unique):
            centre, prior = np.zeros(3), problem.Ppos
            state = oracle._cluster_state(hypothesis)
            if state is not None:
                centre = np.asarray(state['pos'], dtype=float) - problem.pos
                prior = np.asarray(state['P'], dtype=float)
                if not np.all(np.isfinite(centre)):
                    centre, prior = np.zeros(3), problem.Ppos
            parameters[i, :3] = centre
            parameters[i, 3:] = np.asarray(prior).ravel()
        config = np.array([cfg.beta, cfg.marg_uniform, cfg.huber_c, cfg.sig_group,
            cfg.mix_eps, cfg.mix_sigma_cp, cfg.mix_sigma_pr], dtype=float)
        # Unsupported policies/values retain the literal scalar implementation.
        if (not np.all(np.isfinite(rows)) or not np.all(np.isfinite(pairs))
                or not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(config))
                or np.any(rows[:, 4] <= 0.) or cfg.beta <= 0. or cfg.sig_group <= 0.
                or not 0. < cfg.mix_eps < 1. or cfg.mix_sigma_cp <= 0. or cfg.mix_sigma_pr <= 0.
                or np.any(np.abs(meta[meta[:, 0] != 0, 1]) >= nb)
                or np.any(meta[meta[:, 0] != 0, 1] < 0)
                or any(value not in ('ignore', 'warn') for value in np.geterr().values())
                or np.geterr()['under'] != 'ignore'):
            return self.scalar(hypotheses, problem, oracle), dict(unique=len(unique), scalar_fallbacks=len(unique), unsafe=True)
        output = np.empty(len(unique))
        status = np.empty(len(unique), dtype=np.int32)
        diagnostic = np.empty((len(unique), DIAG_SIZE)) if diagnostics else None
        code = self.dll.rbpf_score(self.handle, rows, meta, pairs, dd, parameters, config,
            len(unique), m, nb, int(mode == 'gpu'), output, status,
            None if diagnostic is None else ctypes.c_void_p(diagnostic.ctypes.data))
        if code:
            return self.scalar(hypotheses, problem, oracle), dict(unique=len(unique), scalar_fallbacks=len(unique), native_error=int(code))
        fallback = np.flatnonzero(status)
        cache = {}
        for i in fallback:
            output[i] = oracle._loglik(unique[i], problem, cache)
        details = dict(unique=len(unique), scalar_fallbacks=len(fallback), statuses=status.copy())
        if diagnostics:
            details['diagnostic'] = diagnostic
            details['unique_hypotheses'] = unique
        return output[np.asarray(scatter)], details
