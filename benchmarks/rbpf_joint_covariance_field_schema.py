"""Admit the actual np.int32 satellite schema without changing index math."""

import numpy as np

from rbpf_joint_covariance_sized import make_type as make_sized_type


def make_type(original):
    chosen, engine = make_sized_type(original)
    ordinary = engine.ordinary
    engine.field_schema_calls = 0

    def field_ordinary(rtk, active, jm, key_x, dim, scope):
        if type(active) is list and any(
            type(row) is tuple and len(row) == 3 and type(row[0]) is np.int32
            for row in active
        ):
            if not (
                type(rtk.nav.na) is int
                and 0 <= rtk.nav.na < 2**30
                and all(
                    type(row) is tuple
                    and len(row) == 3
                    and type(row[0]) in (int, np.int32)
                    and type(row[1]) is int
                    and type(row[2]) is int
                    for row in active
                )
            ):
                return False
            # Validation-only metadata. copy_pairs still passes the ORIGINAL
            # np.int32 satellites to the literal IB, retaining its operations.
            normalized = [(int(s), f, k) for s, f, k in active]
            accepted = ordinary(rtk, normalized, jm, key_x, dim, scope)
            engine.field_schema_calls += int(accepted)
            return accepted
        return ordinary(rtk, active, jm, key_x, dim, scope)

    engine.ordinary = field_ordinary
    return chosen, engine
