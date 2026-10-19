"""Keep small covariance loops literal; group the measured larger cases."""

from rbpf_joint_covariance import make_type as make_base_type

MIN_BULK_AMBIGUITIES = 16


def make_type(original):
    chosen, engine = make_base_type(original)
    bulk = engine.copy_pairs
    engine.small_literal_calls = 0

    def sized(rtk, active, jm, key_x, dim, scope, literal):
        if type(active) is list and len(active) < MIN_BULK_AMBIGUITIES:
            engine.small_literal_calls += 1
            return literal(rtk, active, jm)
        return bulk(rtk, active, jm, key_x, dim, scope, literal)

    engine.copy_pairs = sized
    return chosen, engine
