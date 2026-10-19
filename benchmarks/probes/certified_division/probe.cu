// Experimental component probe only; never linked into the production PF.
// CUDA12.8 intrinsic rounding contracts:
// https://docs.nvidia.com/cuda/archive/12.8.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

__device__ __forceinline__ double adjacent_down(double x) {
    if (x == 0.0) return -__longlong_as_double(1);
    unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(x));
    return __longlong_as_double(bits + (x > 0.0 ? -1LL : 1LL));
}

__device__ __forceinline__ double adjacent_up(double x) {
    if (x == 0.0) return __longlong_as_double(1);
    unsigned long long bits = static_cast<unsigned long long>(__double_as_longlong(x));
    return __longlong_as_double(bits + (x > 0.0 ? 1LL : -1LL));
}

struct ReciprocalBounds {
    double inverse, lower, upper;
    bool safe;
    __device__ explicit ReciprocalBounds(double b) {
        safe = b >= 0x1p-500 && b <= 0x1p500;
        if (safe) {
            inverse = __drcp_rn(b);
            lower = adjacent_down(inverse);
            upper = adjacent_up(inverse);
        }
    }
};

template<int MODE, bool COUNT>
__device__ __forceinline__ double certified_quotient(
    double a, double b, const ReciprocalBounds& inv, unsigned int& accepted) {
    // These bounds keep the provisional quotient finite and normal. Zero,
    // infinities/NaNs, signed/exceptional denominators and extreme scales keep
    // the original RN division, including its signed-zero/NaN behaviour.
    const double magnitude = fabs(a);
    if (inv.safe && magnitude >= 0x1p-500 && magnitude <= 0x1p500) {
        const double q = __dmul_rn(a, inv.inverse);
        double residual_lower, residual_upper;
        if constexpr (MODE == 1) {
            residual_lower = __fma_rd(-q, b, a);
            residual_upper = __fma_ru(-q, b, a);
        } else {
            const double residual = __fma_rn(-q, b, a);
            residual_lower = adjacent_down(residual);
            residual_upper = adjacent_up(residual);
        }
        // Exact a/b = q + (a-q*b)/b. Each outward product encloses this
        // correction; the reciprocal is positive. RN addition is monotone.
        // If both endpoint RN results have the same bits, the correctly
        // rounded exact quotient must have those bits, including tie cases.
        const double correction_lower = __dmul_rd(residual_lower,
            residual_lower < 0.0 ? inv.upper : inv.lower);
        const double correction_upper = __dmul_ru(residual_upper,
            residual_upper < 0.0 ? inv.lower : inv.upper);
        const double lower = __dadd_rn(q, correction_lower);
        const double upper = __dadd_rn(q, correction_upper);
        if (__double_as_longlong(lower) == __double_as_longlong(upper)) {
            if constexpr (COUNT) ++accepted;
            return lower;
        }
    }
    return __ddiv_rn(a, b);
}

template<int MODE, bool COUNT>
__global__ void quotient_triples(const double* input, double* output,
    unsigned int* accepted, int count, int steps) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double a0 = input[4*i], a1 = input[4*i+1], a2 = input[4*i+2];
    const double b = input[4*i+3];
    double x = 0.0, y = 0.0, z = 0.0;
    unsigned int fast = 0;
    #pragma unroll 1
    for (int s = 0; s < steps; ++s) {
        const double denominator = steps == 1 ? b : __dadd_rn(b, __dmul_rn(double(s), 0.25));
        const double p = steps == 1 ? a0 : __dadd_rn(a0, __dmul_rn(double(s), 0.125));
        const double q = steps == 1 ? a1 : __dadd_rn(a1, __dmul_rn(double(s), 0.125));
        const double r = steps == 1 ? a2 : __dadd_rn(a2, __dmul_rn(double(s), 0.125));
        double dx, dy, dz;
        if constexpr (MODE == 0) {
            dx = __ddiv_rn(p, denominator);
            dy = __ddiv_rn(q, denominator);
            dz = __ddiv_rn(r, denominator);
        } else {
            const ReciprocalBounds inverse(denominator);
            dx = certified_quotient<MODE,COUNT>(p, denominator, inverse, fast);
            dy = certified_quotient<MODE,COUNT>(q, denominator, inverse, fast);
            dz = certified_quotient<MODE,COUNT>(r, denominator, inverse, fast);
        }
        // One-step validation exposes each division directly, preserving -0.
        if (steps == 1) {
            x = dx; y = dy; z = dz;
        } else {
            x = __dadd_rn(x, dx); y = __dadd_rn(y, dy); z = __dadd_rn(z, dz);
        }
    }
    output[3*i] = x; output[3*i+1] = y; output[3*i+2] = z;
    accepted[i] = fast;
}

static void launch(int mode, bool count_fast, const double* in, double* out,
    unsigned int* fast, int n, int steps) {
    const int blocks = (n+127)/128;
    if (mode == 0) quotient_triples<0,false><<<blocks,128>>>(in,out,fast,n,steps);
    else if (mode == 1 && count_fast) quotient_triples<1,true><<<blocks,128>>>(in,out,fast,n,steps);
    else if (mode == 1) quotient_triples<1,false><<<blocks,128>>>(in,out,fast,n,steps);
    else if (count_fast) quotient_triples<2,true><<<blocks,128>>>(in,out,fast,n,steps);
    else quotient_triples<2,false><<<blocks,128>>>(in,out,fast,n,steps);
}

extern "C" __declspec(dllexport) int run_probe(const double* input,
    double* output, unsigned int* accepted, int n, int steps,
    int mode, int count_fast, int repeats, float* milliseconds) {
    if (n < 1 || steps < 1 || mode < 0 || mode > 2 || repeats < 1) return -1;
    double *in = nullptr, *out = nullptr;
    unsigned int* fast = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaError_t status = cudaSuccess;
    #define CHECK(call) do { status = (call); if (status != cudaSuccess) goto done; } while (0)
    CHECK(cudaMalloc(&in, size_t(n)*4*sizeof(double)));
    CHECK(cudaMalloc(&out, size_t(n)*3*sizeof(double)));
    CHECK(cudaMalloc(&fast, size_t(n)*sizeof(unsigned int)));
    CHECK(cudaMemcpy(in,input,size_t(n)*4*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    launch(mode,count_fast,in,out,fast,n,steps);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    for (int repeat = 0; repeat < repeats; ++repeat) launch(mode,count_fast,in,out,fast,n,steps);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(milliseconds,start,stop));
    CHECK(cudaMemcpy(output,out,size_t(n)*3*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(accepted,fast,size_t(n)*sizeof(unsigned int),cudaMemcpyDeviceToHost));
done:
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (in) cudaFree(in);
    if (out) cudaFree(out);
    if (fast) cudaFree(fast);
    #undef CHECK
    return int(status);
}
