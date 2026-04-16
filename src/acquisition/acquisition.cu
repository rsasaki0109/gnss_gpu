// NOTE: Link with cuFFT (-lcufft). In CMakeLists.txt add:
//   target_link_libraries(gnss_gpu_core CUDA::cudart CUDA::cufft)
// and add this source file to the gnss_gpu_core library.

#include "gnss_gpu/acquisition.h"
#include "gnss_gpu/cuda_check.h"
#include <cufft.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cfloat>

namespace gnss_gpu {

constexpr double kPi = 3.14159265358979323846;

// GPS C/A code chip rate
static constexpr double CA_CHIP_RATE = 1.023e6;
static constexpr int CA_CODE_LEN = 1023;

// G2 delay taps for PRN 1-32 (1-indexed tap pairs)
static const int G2_TAPS[32][2] = {
    {2, 6},   // PRN 1
    {3, 7},   // PRN 2
    {4, 8},   // PRN 3
    {5, 9},   // PRN 4
    {1, 9},   // PRN 5
    {2, 10},  // PRN 6
    {1, 8},   // PRN 7
    {2, 9},   // PRN 8
    {3, 10},  // PRN 9
    {2, 3},   // PRN 10
    {3, 4},   // PRN 11
    {5, 6},   // PRN 12
    {6, 7},   // PRN 13
    {7, 8},   // PRN 14
    {8, 9},   // PRN 15
    {9, 10},  // PRN 16
    {1, 4},   // PRN 17
    {2, 5},   // PRN 18
    {3, 6},   // PRN 19
    {4, 7},   // PRN 20
    {5, 8},   // PRN 21
    {6, 9},   // PRN 22
    {1, 3},   // PRN 23
    {4, 6},   // PRN 24
    {5, 7},   // PRN 25
    {6, 8},   // PRN 26
    {7, 9},   // PRN 27
    {8, 10},  // PRN 28
    {1, 6},   // PRN 29
    {2, 7},   // PRN 30
    {3, 8},   // PRN 31
    {4, 9},   // PRN 32
};

// CPU: Generate 1023-chip GPS C/A code for given PRN (1-32)
void generate_ca_code(int prn, int* code_out) {
    // G1 LFSR: x^10 + x^3 + 1, feedback taps 3,10 (1-indexed)
    // G2 LFSR: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1, feedback taps 2,3,6,8,9,10
    int g1[10], g2[10];
    for (int i = 0; i < 10; i++) {
        g1[i] = 1;
        g2[i] = 1;
    }

    int tap1 = G2_TAPS[prn - 1][0] - 1;  // convert to 0-indexed
    int tap2 = G2_TAPS[prn - 1][1] - 1;

    for (int i = 0; i < CA_CODE_LEN; i++) {
        // Output: G1[10] XOR G2[tap1] XOR G2[tap2] (all 1-indexed, so [9], [tap1], [tap2] in 0-indexed)
        int g1_out = g1[9];
        int g2_delayed = g2[tap1] ^ g2[tap2];
        int ca_bit = g1_out ^ g2_delayed;

        // Map {0,1} -> {-1,+1}  (GPS ICD convention: XOR=1 means +1 chip)
        code_out[i] = 2 * ca_bit - 1;

        // G1 feedback: taps 3,10 (0-indexed: 2,9)
        int g1_fb = g1[2] ^ g1[9];
        // G2 feedback: taps 2,3,6,8,9,10 (0-indexed: 1,2,5,7,8,9)
        int g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9];

        // Shift registers
        for (int j = 9; j > 0; j--) {
            g1[j] = g1[j - 1];
            g2[j] = g2[j - 1];
        }
        g1[0] = g1_fb;
        g2[0] = g2_fb;
    }
}

// Kernel: Resample 1023-chip code to n_samples at given sampling frequency
__global__ void resample_code_kernel(const int* code_1023, float* code_sampled,
                                      int n_samples, double sampling_freq,
                                      double chip_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    double t = (double)idx / sampling_freq;
    int chip_idx = ((int)(t * chip_rate)) % CA_CODE_LEN;
    code_sampled[idx] = (float)code_1023[chip_idx];
}

// Kernel: Carrier wipe-off (remove carrier by mixing with local replica)
__global__ void carrier_wipeoff_kernel(const float* signal, float* out_i,
                                        float* out_q, int n, double freq,
                                        double fs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double phase = 2.0 * kPi * freq * (double)idx / fs;
    float cos_val = cosf((float)phase);
    float sin_val = sinf((float)phase);
    out_i[idx] = signal[idx] * cos_val;
    out_q[idx] = signal[idx] * (-sin_val);
}

// Kernel: Compute magnitude squared of complex array
__global__ void mag_squared_kernel(cufftComplex* data, float* mag, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float re = data[idx].x;
    float im = data[idx].y;
    mag[idx] = re * re + im * im;
}

// Kernel: Complex multiply A * conj(B)
__global__ void multiply_conj_kernel(cufftComplex* a, cufftComplex* b,
                                      cufftComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float ar = a[idx].x, ai = a[idx].y;
    float br = b[idx].x, bi = b[idx].y;
    // A * conj(B) = (ar + j*ai)(br - j*bi) = (ar*br + ai*bi) + j*(ai*br - ar*bi)
    out[idx].x = ar * br + ai * bi;
    out[idx].y = ai * br - ar * bi;
}

// Kernel: Parallel reduction to find peak value, index, and sum
__global__ void find_peak_reduce_kernel(const float* data, float* max_val,
                                         int* max_idx, float* sum, int n) {
    extern __shared__ char smem[];
    float* s_val = (float*)smem;
    int* s_idx = (int*)(s_val + blockDim.x);
    float* s_sum = (float*)(s_idx + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    if (gid < n) {
        s_val[tid] = data[gid];
        s_idx[tid] = gid;
        s_sum[tid] = data[gid];
    } else {
        s_val[tid] = -FLT_MAX;
        s_idx[tid] = 0;
        s_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Reduce for max and sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_val[blockIdx.x] = s_val[0];
        max_idx[blockIdx.x] = s_idx[0];
        sum[blockIdx.x] = s_sum[0];
    }
}

// Host helper: find peak and compute SNR using second-peak ratio with guard band
// Returns the peak value, integer peak index, refined peak code phase, and
// SNR (peak / second_peak_excluding_guard).
static void find_peak_and_snr(float* d_mag, int n, float& peak_val,
                               int& peak_idx, double& peak_code_phase, float& snr) {
    int block = 256;
    int grid = (n + block - 1) / block;
    size_t smem_size = block * (sizeof(float) + sizeof(int) + sizeof(float));

    float* d_max_val;
    int* d_max_idx;
    float* d_sum;
    CUDA_CHECK(cudaMalloc(&d_max_val, grid * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_idx, grid * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, grid * sizeof(float)));

    find_peak_reduce_kernel<<<grid, block, smem_size>>>(d_mag, d_max_val,
                                                         d_max_idx, d_sum, n);
    CUDA_CHECK_LAST();

    // Copy partial results to host and finalize
    float* h_max_val = new float[grid];
    int* h_max_idx = new int[grid];
    float* h_sum = new float[grid];

    CUDA_CHECK(cudaMemcpy(h_max_val, d_max_val, grid * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_max_idx, d_max_idx, grid * sizeof(int), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_sum, d_sum, grid * sizeof(float), cudaMemcpyDeviceToHost));

    peak_val = h_max_val[0];
    peak_idx = h_max_idx[0];

    for (int i = 1; i < grid; i++) {
        if (h_max_val[i] > peak_val) {
            peak_val = h_max_val[i];
            peak_idx = h_max_idx[i];
        }
    }

    delete[] h_max_val;
    delete[] h_max_idx;
    delete[] h_sum;
    CUDA_CHECK(cudaFree(d_max_val));
    CUDA_CHECK(cudaFree(d_max_idx));
    CUDA_CHECK(cudaFree(d_sum));

    // Copy full magnitude array to host to find second peak with guard band
    float* h_mag = new float[n];
    CUDA_CHECK(cudaMemcpy(h_mag, d_mag, n * sizeof(float), cudaMemcpyDeviceToHost));

    peak_code_phase = static_cast<double>(peak_idx);
    if (n >= 3) {
        int left = (peak_idx - 1 + n) % n;
        int right = (peak_idx + 1) % n;
        double y_left = h_mag[left];
        double y_center = h_mag[peak_idx];
        double y_right = h_mag[right];
        double denom = y_left - 2.0 * y_center + y_right;

        if (y_center >= y_left && y_center >= y_right && fabs(denom) > 1e-12) {
            double delta = 0.5 * (y_left - y_right) / denom;
            if (delta > 0.5) delta = 0.5;
            if (delta < -0.5) delta = -0.5;
            peak_code_phase += delta;
            if (peak_code_phase < 0.0) peak_code_phase += n;
            if (peak_code_phase >= n) peak_code_phase -= n;
        }
    }

    // Find second peak excluding a guard band of +-16 samples around the first peak
    // This ensures the "second peak" is a genuinely different correlation peak,
    // not just a neighbor of the main peak on the autocorrelation triangle.
    const int guard = 16;
    float second_peak = 0.0f;
    for (int i = 0; i < n; i++) {
        int dist = abs(i - peak_idx);
        // Handle circular wrap-around
        if (dist > n / 2) dist = n - dist;
        if (dist > guard && h_mag[i] > second_peak) {
            second_peak = h_mag[i];
        }
    }

    snr = (second_peak > 0) ? peak_val / second_peak : 0;

    delete[] h_mag;
}

// Main acquisition function
void acquire_parallel(
    const float* signal, int n_samples,
    double sampling_freq, double intermediate_freq,
    const int* prn_list, int n_prn,
    double doppler_range, double doppler_step,
    float threshold,
    AcquisitionResult* results) {

    int block = 256;
    int grid = (n_samples + block - 1) / block;

    // Allocate device memory for input signal
    float* d_signal;
    CUDA_CHECK(cudaMalloc(&d_signal, n_samples * sizeof(float))); CUDA_CHECK(cudaMemcpy(d_signal, signal, n_samples * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate working buffers
    float* d_out_i;
    float* d_out_q;
    float* d_mag;
    cufftComplex* d_signal_fft;
    cufftComplex* d_code_fft;
    cufftComplex* d_corr_fft;

    CUDA_CHECK(cudaMalloc(&d_out_i, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_q, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mag, n_samples * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_signal_fft, n_samples * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_code_fft, n_samples * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_corr_fft, n_samples * sizeof(cufftComplex)));

    // Code generation buffers
    int* d_code_1023;
    float* d_code_sampled;
    CUDA_CHECK(cudaMalloc(&d_code_1023, CA_CODE_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_code_sampled, n_samples * sizeof(float)));

    // cuFFT plans
    cufftHandle plan_fwd, plan_inv;
    cufftPlan1d(&plan_fwd, n_samples, CUFFT_C2C, 1);
    cufftPlan1d(&plan_inv, n_samples, CUFFT_C2C, 1);

    // Number of Doppler bins
    int n_doppler = (int)(2.0 * doppler_range / doppler_step) + 1;

    for (int p = 0; p < n_prn; p++) {
        int prn = prn_list[p];
        results[p].prn = prn;
        results[p].acquired = false;
        results[p].code_phase = 0;
        results[p].doppler_hz = 0;
        results[p].snr = 0;

        if (prn < 1 || prn > 32) continue;

        // Generate C/A code on host, copy to device
        int h_code[CA_CODE_LEN];
        generate_ca_code(prn, h_code);
        CUDA_CHECK(cudaMemcpy(d_code_1023, h_code, CA_CODE_LEN * sizeof(int), cudaMemcpyHostToDevice));

        // Resample code to IF sampling rate
        resample_code_kernel<<<grid, block>>>(d_code_1023, d_code_sampled,
                                               n_samples, sampling_freq,
                                               CA_CHIP_RATE);
        CUDA_CHECK_LAST();

        // Pack resampled code into complex (real part only) and FFT
        // We reuse d_code_fft: set real = code_sampled, imag = 0
        CUDA_CHECK(cudaMemset(d_code_fft, 0, n_samples * sizeof(cufftComplex)));
        // Copy real part using a simple kernel approach: reuse carrier_wipeoff trick
        // Actually, let's use a small kernel to pack float -> cufftComplex
        // For simplicity, do it on host side with a device copy
        {
            // Pack code_sampled into complex on device
            float* h_code_sampled = new float[n_samples];
            CUDA_CHECK(cudaMemcpy(h_code_sampled, d_code_sampled, n_samples * sizeof(float), cudaMemcpyDeviceToHost));
            cufftComplex* h_code_complex = new cufftComplex[n_samples];
            for (int i = 0; i < n_samples; i++) {
                h_code_complex[i].x = h_code_sampled[i];
                h_code_complex[i].y = 0.0f;
            }
            CUDA_CHECK(cudaMemcpy(d_code_fft, h_code_complex, n_samples * sizeof(cufftComplex), cudaMemcpyHostToDevice));
            delete[] h_code_sampled;
            delete[] h_code_complex;
        }

        // FFT of code
        cufftExecC2C(plan_fwd, d_code_fft, d_code_fft, CUFFT_FORWARD);

        float best_peak = 0;
        int best_peak_idx = 0;
        double best_code_phase = 0.0;
        double best_doppler = 0;
        float best_snr = 0;

        // Search over Doppler bins
        for (int d = 0; d < n_doppler; d++) {
            double doppler = -doppler_range + d * doppler_step;
            double carrier_freq = intermediate_freq + doppler;

            // Carrier wipe-off
            carrier_wipeoff_kernel<<<grid, block>>>(d_signal, d_out_i, d_out_q,
                                                     n_samples, carrier_freq,
                                                     sampling_freq);
            CUDA_CHECK_LAST();

            // Pack I/Q into complex signal
            // d_signal_fft.x = d_out_i, d_signal_fft.y = d_out_q
            {
                float* h_i = new float[n_samples];
                float* h_q = new float[n_samples];
                CUDA_CHECK(cudaMemcpy(h_i, d_out_i, n_samples * sizeof(float), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(h_q, d_out_q, n_samples * sizeof(float), cudaMemcpyDeviceToHost));
                cufftComplex* h_sig = new cufftComplex[n_samples];
                for (int i = 0; i < n_samples; i++) {
                    h_sig[i].x = h_i[i];
                    h_sig[i].y = h_q[i];
                }
                CUDA_CHECK(cudaMemcpy(d_signal_fft, h_sig, n_samples * sizeof(cufftComplex), cudaMemcpyHostToDevice));
                delete[] h_i;
                delete[] h_q;
                delete[] h_sig;
            }

            // FFT of wiped-off signal
            cufftExecC2C(plan_fwd, d_signal_fft, d_signal_fft, CUFFT_FORWARD);

            // Multiply signal_FFT * conj(code_FFT)
            multiply_conj_kernel<<<grid, block>>>(d_signal_fft, d_code_fft,
                                                   d_corr_fft, n_samples);
            CUDA_CHECK_LAST();

            // IFFT
            cufftExecC2C(plan_inv, d_corr_fft, d_corr_fft, CUFFT_INVERSE);

            // Magnitude squared
            mag_squared_kernel<<<grid, block>>>(d_corr_fft, d_mag, n_samples);
            CUDA_CHECK_LAST();

            // Find peak and compute SNR (peak / second_peak with guard band)
            float peak_val;
            int peak_idx;
            double peak_code_phase;
            float snr;
            find_peak_and_snr(d_mag, n_samples, peak_val, peak_idx, peak_code_phase, snr);

            if (snr > best_snr) {
                best_snr = snr;
                best_peak = peak_val;
                best_peak_idx = peak_idx;
                best_code_phase = peak_code_phase;
                best_doppler = doppler;
            }
        }

        results[p].snr = best_snr;
        results[p].code_phase = best_code_phase;
        // Real zero-IF input is symmetric in +/- Doppler, so report magnitude
        // instead of the arbitrary sign selected by the search order.
        results[p].doppler_hz = (fabs(intermediate_freq) < 1e-12)
            ? fabs(best_doppler)
            : best_doppler;
        results[p].acquired = (best_snr > threshold);
    }

    // Cleanup
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_out_i));
    CUDA_CHECK(cudaFree(d_out_q));
    CUDA_CHECK(cudaFree(d_mag));
    CUDA_CHECK(cudaFree(d_signal_fft));
    CUDA_CHECK(cudaFree(d_code_fft));
    CUDA_CHECK(cudaFree(d_corr_fft));
    CUDA_CHECK(cudaFree(d_code_1023));
    CUDA_CHECK(cudaFree(d_code_sampled));
}

}  // namespace gnss_gpu
