#!/usr/bin/env python3
"""Check that a run of this frozen copy reproduces a reference run.

Compares the common prefix (by tow) of two wp16_run_rbpf.py npz outputs:
the shipped mode, resolved-ambiguity count and position must be identical.

  python check_frozen_copy.py results/smoke_r3/run3.npz <reference run3.npz>
"""
import sys

import numpy as np


def main() -> int:
    new, ref = (np.load(p) for p in sys.argv[1:3])
    n = min(len(new["tow"]), len(ref["tow"]))
    if n == 0 or not np.array_equal(new["tow"][:n], ref["tow"][:n]):
        print("FAIL: epoch grids differ")
        return 1
    bad = []
    for key in ("smode", "rb_nb", "sol_xyz"):
        a, b = new[key][:n], ref[key][:n]
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            bad.append(key)
    if bad:
        print(f"FAIL: {', '.join(bad)} differ over the first {n} epochs")
        return 1
    print(f"OK: first {n} epochs identical (smode, rb_nb, sol_xyz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
