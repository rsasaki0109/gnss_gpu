#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace gnss_gpu {

struct PFDQuotientBounds {
    double denominator;
    double lower;
    double upper;
    bool valid;
};

// Enclose the exact positive reciprocal with adjacent FP64 values. The sign of
// fma(-d, r, 1) identifies which side the RN reciprocal is on: its nonzero
// residual near one cannot underflow. Other denominators use native division.
__device__ __forceinline__ PFDQuotientBounds pfd_quotient_bounds(double d) {
    PFDQuotientBounds b{d, 0.0, 0.0, false};
    if (!(d > 0.0) || !isfinite(d)) return b;
    double r = __drcp_rn(d);
    if (!(r > 0.0) || !isfinite(r)) return b;
    double e = __fma_rn(-d, r, 1.0);
    long long bits = __double_as_longlong(r);
    b.lower = e < 0.0 ? __longlong_as_double(bits - 1) : r;
    b.upper = e > 0.0 ? __longlong_as_double(bits + 1) : r;
    b.valid = b.lower > 0.0 && isfinite(b.upper);
    return b;
}

// q + (x - q*d)/d is exactly x/d for any finite q. Directed FMA bounds the
// residual. Select reciprocal endpoints according to each residual's sign.
// RN is monotone: identical finite RN endpoint bits certify the native RN
// quotient. Ambiguous endpoints, special values and overflow use __ddiv_rn.
__device__ __forceinline__ double pfd_quotient_rn(double x, const PFDQuotientBounds& b) {
    if (!b.valid || !isfinite(x)) return __ddiv_rn(x, b.denominator);
    double q = __dmul_rn(x, b.lower);
    if (b.lower == b.upper) return q;
    if (!isfinite(q)) return __ddiv_rn(x, b.denominator);
    double lo = __fma_rd(-q, b.denominator, x);
    double hi = __fma_ru(-q, b.denominator, x);
    double a = __fma_rn(lo, lo < 0.0 ? b.upper : b.lower, q);
    double z = __fma_rn(hi, hi < 0.0 ? b.lower : b.upper, q);
    if (isfinite(a) && __double_as_longlong(a) == __double_as_longlong(z)) return a;
    return __ddiv_rn(x, b.denominator);
}

}  // namespace gnss_gpu
