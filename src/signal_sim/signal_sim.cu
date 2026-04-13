#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>

#include <curand_kernel.h>

#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/signal_sim.h"

namespace {

constexpr int kNumSatellites = 32;
constexpr int kCaCodeLength = 1023;
constexpr int kBlockSize = 256;
constexpr double kCaChipRate = 1.023e6;
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// G2 delay taps for PRN 1-32 (1-indexed tap pairs) — matches IS-GPS-200
static const int G2_TAPS[kNumSatellites][2] = {
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

// C/A codes stored in global device memory (constant memory is only 64KB,
// 32*1023*4 = ~128KB exceeds the limit and other modules also use constant memory)
int* d_ca_codes = nullptr;
std::once_flag g_ca_init_flag;
std::atomic<unsigned long long> g_noise_seed_counter{0};

unsigned long long resolve_noise_seed(unsigned long long requested_seed) {
    if (requested_seed != 0ULL) {
        return requested_seed;
    }

    const auto now = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto counter =
        g_noise_seed_counter.fetch_add(0x9e3779b97f4a7c15ULL, std::memory_order_relaxed);
    return now ^ counter ^ 0xa0761d6478bd642fULL;
}

void build_ca_codes_host() {
    int h_ca_codes[kNumSatellites][kCaCodeLength];

    for (int prn = 1; prn <= kNumSatellites; ++prn) {
        int g1[10], g2[10];
        for (int i = 0; i < 10; ++i) {
            g1[i] = 1;
            g2[i] = 1;
        }

        int tap1 = G2_TAPS[prn - 1][0] - 1;
        int tap2 = G2_TAPS[prn - 1][1] - 1;

        for (int chip = 0; chip < kCaCodeLength; ++chip) {
            int g1_out = g1[9];
            int g2_delayed = g2[tap1] ^ g2[tap2];
            int ca_bit = g1_out ^ g2_delayed;
            h_ca_codes[prn - 1][chip] = 2 * ca_bit - 1;  // {0,1} -> {-1,+1}

            int g1_fb = g1[2] ^ g1[9];
            int g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9];

            for (int i = 9; i > 0; --i) {
                g1[i] = g1[i - 1];
                g2[i] = g2[i - 1];
            }
            g1[0] = g1_fb;
            g2[0] = g2_fb;
        }
    }

    CUDA_CHECK(cudaMalloc(&d_ca_codes,
                           sizeof(int) * kNumSatellites * kCaCodeLength));
    CUDA_CHECK(cudaMemcpy(d_ca_codes, h_ca_codes,
                           sizeof(int) * kNumSatellites * kCaCodeLength,
                           cudaMemcpyHostToDevice));
}

__global__ void generate_channel_kernel(float* iq_out,
                                        const int* ca_codes,
                                        gnss_gpu::SatChannel ch,
                                        int n_samples,
                                        double sampling_freq,
                                        double intermediate_freq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;
    if (ch.prn < 1 || ch.prn > kNumSatellites) return;

    double t = (double)idx / sampling_freq;
    int chip_idx = ((int)(t * kCaChipRate + ch.code_phase)) % kCaCodeLength;
    if (chip_idx < 0) chip_idx += kCaCodeLength;

    float ca_code = (float)ca_codes[(ch.prn - 1) * kCaCodeLength + chip_idx];

    double phase = kTwoPi * (intermediate_freq + ch.doppler_hz) * t + ch.carrier_phase;
    float cos_val, sin_val;
    __sincosf((float)phase, &sin_val, &cos_val);

    float signal = ch.amplitude * (float)ch.nav_bit * ca_code;
    iq_out[2 * idx]     = signal * cos_val;
    iq_out[2 * idx + 1] = signal * (-sin_val);
}

__global__ void composite_kernel(const float* channel_iq,
                                 float* iq_out,
                                 int n_samples,
                                 int n_channels,
                                 float noise_std,
                                 unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    int base = 2 * idx;
    float sum_i = 0.0f, sum_q = 0.0f;

    size_t stride = (size_t)n_samples * 2;
    for (int ch = 0; ch < n_channels; ++ch) {
        size_t offset = stride * (size_t)ch + (size_t)base;
        sum_i += channel_iq[offset];
        sum_q += channel_iq[offset + 1];
    }

    if (noise_std > 0.0f) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, (unsigned long long)idx, 0, &state);
        sum_i += curand_normal(&state) * noise_std;
        sum_q += curand_normal(&state) * noise_std;
    }

    iq_out[base] = sum_i;
    iq_out[base + 1] = sum_q;
}

}  // namespace

namespace gnss_gpu {

void init_ca_codes() {
    std::call_once(g_ca_init_flag, build_ca_codes_host);
}

void simulate_epoch(const SignalSimConfig& config,
                    const SatChannel* channels,
                    int n_channels,
                    float* iq_output,
                    int n_samples) {
    if (n_samples <= 0 || iq_output == nullptr) return;

    init_ca_codes();

    int blocks = (n_samples + kBlockSize - 1) / kBlockSize;
    size_t num_floats = (size_t)n_samples * 2;
    size_t output_bytes = num_floats * sizeof(float);

    float* d_channel = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    if (n_channels > 0 && channels != nullptr) {
        CUDA_CHECK(cudaMalloc(&d_channel, output_bytes * (size_t)n_channels));

        for (int ch = 0; ch < n_channels; ++ch) {
            float* ch_out = d_channel + (size_t)ch * num_floats;
            generate_channel_kernel<<<blocks, kBlockSize>>>(
                ch_out, d_ca_codes, channels[ch], n_samples,
                config.sampling_freq, config.intermediate_freq);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    float noise_std = std::pow(10.0f, (float)config.noise_floor_db / 20.0f);
    composite_kernel<<<blocks, kBlockSize>>>(
        d_channel, d_output, n_samples,
        (n_channels > 0) ? n_channels : 0, noise_std,
        resolve_noise_seed(config.noise_seed));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(iq_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    if (d_channel) CUDA_CHECK(cudaFree(d_channel));
    CUDA_CHECK(cudaFree(d_output));
}

void generate_single_channel(const SatChannel& ch,
                             float* iq_out,
                             int n_samples,
                             double sampling_freq,
                             double intermediate_freq) {
    if (n_samples <= 0 || iq_out == nullptr) return;

    init_ca_codes();

    int blocks = (n_samples + kBlockSize - 1) / kBlockSize;
    size_t output_bytes = (size_t)n_samples * 2 * sizeof(float);

    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    generate_channel_kernel<<<blocks, kBlockSize>>>(
        d_output, d_ca_codes, ch, n_samples, sampling_freq, intermediate_freq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(iq_out, d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
}

}  // namespace gnss_gpu
