from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def main():
    root=Path(__file__).resolve().parent.parent
    rng=np.random.default_rng(20420261015)
    divisors={65535,65536,65537,100000,262144,500000,1000000,2**31-1}
    for exponent in range(16,31):
        divisors.update((2**exponent-1,2**exponent,2**exponent+1))
    divisors.update(map(int,rng.integers(65536,2**31,size=256)))
    cases=0
    records=[]
    for divisor in sorted(divisors):
        ceil_log=(divisor-1).bit_length()
        assert 1<=ceil_log<=31
        multiplier=((1<<(32+ceil_log))+divisor-1)//divisor
        assert 2**32<=multiplier<2**33
        magic=multiplier-2**32
        shift=ceil_log-1
        quotients=np.arange((2**32-1)//divisor+1,dtype=np.uint64)
        boundaries=quotients*np.uint64(divisor)
        values=np.unique(np.concatenate([rng.integers(0,2**32,size=32768,dtype=np.uint64),
            np.asarray([0,1,2**31-1,2**31,2**32-2,2**32-1],dtype=np.uint64),
            boundaries,boundaries[boundaries>0]-np.uint64(1),boundaries[boundaries<2**32-1]+np.uint64(1)]))
        high=(values*np.uint64(magic))>>np.uint64(32)
        fast_q=(high+((values-high)>>np.uint64(1)))>>np.uint64(shift)
        remainder=values-fast_q*np.uint64(divisor)
        np.testing.assert_array_equal(fast_q,values//np.uint64(divisor))
        np.testing.assert_array_equal(remainder,values%np.uint64(divisor))
        assert np.all(remainder<divisor)
        cases+=values.size
        records.append(dict(divisor=divisor,magic=magic,shift=shift,cases=int(values.size),
            input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),output_sha256=hashlib.sha256(remainder.tobytes()).hexdigest()))
    proof='For L=ceil(log2(d)), M=ceil(2^(32+L)/d): M/2^(32+L) >= 1/d and the positive error times any uint32 x is strictly less than 2^-L <= 1/d. Every noninteger x/d is at least 1/d below the next integer, so floor(x*M/2^(32+L))=floor(x/d). With M=2^32+m, high=umulhi(x,m), ((x-high)>>1)+high avoids overflow and yields the same quotient after shifting L-1. q*d <= x, so x-q*d is the original exact uint32 remainder.'
    out=dict(schema_version=1,cases=int(cases),divisors=len(records),all_exact=True,
        proof=proof,records=records,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=root/'dist/pf_magic_mod_cpu_reproduction.json')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(f'PASS:{cases} exact uint32 quotients/remainders across {len(records)} divisors, including every quotient boundary of each divisor')


if __name__ == '__main__':
    main()
