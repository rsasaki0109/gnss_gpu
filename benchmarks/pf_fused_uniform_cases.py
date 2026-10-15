"""Exact fused uniform gather: block tails, covariance lanes and scratch reuse."""

from pf_uniform_inverse_cases import check_inverse
from pf_uniform_cdf_cases import check_uniform_queue,check_uniform
from bench_pf_scan_inverse_compare import validate as validate_previous

COUNTS=(262144,500000,1000000)
TARGETS=(127,129,191,192,255,256,257,511,512,513,65535,65536,65537)
VARIANTS=(0,2)

def validate_new(native,klass):
    records={f'fused_uniform/{n}/target{t}/v{v}':check_inverse(native,klass,n,t,v)
             for n in COUNTS for t in TARGETS for v in VARIANTS}
    for n in COUNTS:
        records[f'fused_uniform/queue/{n}']=check_uniform_queue(native,klass,n)
    for n in COUNTS:
        for v in VARIANTS:
            records[f'fused_uniform/diagnostic/{n}/v{v}']=check_uniform(native,klass,n,257,v)
    return records

def validate(native,klass):
    records=validate_previous(native,klass)
    records.update(validate_new(native,klass))
    return records
