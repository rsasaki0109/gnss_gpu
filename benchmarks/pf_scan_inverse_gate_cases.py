"""Exact all-field selection across the measured large-source dispatch gate."""

from pf_uniform_inverse_cases import check_inverse

COUNTS=(262143,262144,262145)
TARGETS=(129,257,131073)
VARIANTS=(0,2)

def validate_new(native,klass):
    return {f'scan_inverse_gate/{n}/target{t}/v{v}':check_inverse(native,klass,n,t,v)
            for n in COUNTS for t in TARGETS for v in VARIANTS}
