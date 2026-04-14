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

constexpr int kBlockSize = 256;
constexpr double kTwoPi = 2.0 * 3.14159265358979323846;

// Per-constellation parameters
constexpr int kGpsNumPrn = 32;
constexpr int kGpsCodeLen = 1023;
constexpr double kGpsChipRate = 1.023e6;

constexpr int kGloNumPrn = 24;
constexpr int kGloCodeLen = 511;
constexpr double kGloChipRate = 0.511e6;

constexpr int kGalNumPrn = 36;
constexpr int kGalCodeLen = 4092;
constexpr double kGalChipRate = 1.023e6;

constexpr int kBdsNumPrn = 37;
constexpr int kBdsCodeLen = 2046;
constexpr double kBdsChipRate = 2.046e6;

// Maximum code length across all systems (for kernel generalization)
constexpr int kMaxCodeLen = 4092;

// G2 delay taps for PRN 1-32 (1-indexed tap pairs) — matches IS-GPS-200
static const int G2_TAPS[kGpsNumPrn][2] = {
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

// Device code tables: one pointer per constellation
int* d_gps_codes = nullptr;   // [32 * 1023]
int* d_glo_codes = nullptr;   // [24 * 511]
int* d_gal_codes = nullptr;   // [36 * 4092]
int* d_bds_codes = nullptr;   // [37 * 2046]

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
    int h_ca_codes[kGpsNumPrn][kGpsCodeLen];

    for (int prn = 1; prn <= kGpsNumPrn; ++prn) {
        int g1[10], g2[10];
        for (int i = 0; i < 10; ++i) {
            g1[i] = 1;
            g2[i] = 1;
        }

        int tap1 = G2_TAPS[prn - 1][0] - 1;
        int tap2 = G2_TAPS[prn - 1][1] - 1;

        for (int chip = 0; chip < kGpsCodeLen; ++chip) {
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

    CUDA_CHECK(cudaMalloc(&d_gps_codes,
                           sizeof(int) * kGpsNumPrn * kGpsCodeLen));
    CUDA_CHECK(cudaMemcpy(d_gps_codes, h_ca_codes,
                           sizeof(int) * kGpsNumPrn * kGpsCodeLen,
                           cudaMemcpyHostToDevice));

    // --- GLONASS L1 C/A: all satellites share the same 511-chip code ---
    // 9-stage maximal-length LFSR, taps at stages 5 and 9
    {
        int glo_code[kGloCodeLen];
        int reg[9];
        for (int i = 0; i < 9; ++i) reg[i] = 1;
        for (int chip = 0; chip < kGloCodeLen; ++chip) {
            glo_code[chip] = 2 * reg[8] - 1;  // {0,1} -> {-1,+1}
            int fb = reg[4] ^ reg[8];
            for (int i = 8; i > 0; --i) reg[i] = reg[i - 1];
            reg[0] = fb;
        }
        // All 24 GLONASS satellites use the same code (FDMA distinguishes them)
        int h_glo[kGloNumPrn][kGloCodeLen];
        for (int s = 0; s < kGloNumPrn; ++s)
            for (int c = 0; c < kGloCodeLen; ++c)
                h_glo[s][c] = glo_code[c];

        CUDA_CHECK(cudaMalloc(&d_glo_codes,
                               sizeof(int) * kGloNumPrn * kGloCodeLen));
        CUDA_CHECK(cudaMemcpy(d_glo_codes, h_glo,
                               sizeof(int) * kGloNumPrn * kGloCodeLen,
                               cudaMemcpyHostToDevice));
    }

    // --- Galileo E1-B: memory codes (use GPS-like Gold codes as placeholder,
    //     actual Galileo uses stored PRN sequences from ICD) ---
    // For simulation purposes, generate pseudo-random 4092-chip codes per PRN
    {
        int* h_gal = new int[kGalNumPrn * kGalCodeLen];
        for (int prn = 0; prn < kGalNumPrn; ++prn) {
            // Deterministic PRNG seeded by PRN for reproducibility
            unsigned int state = 0xCAFE0000u + prn * 0x9e3779b9u;
            for (int chip = 0; chip < kGalCodeLen; ++chip) {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                h_gal[prn * kGalCodeLen + chip] = (state & 1) ? 1 : -1;
            }
        }
        CUDA_CHECK(cudaMalloc(&d_gal_codes,
                               sizeof(int) * kGalNumPrn * kGalCodeLen));
        CUDA_CHECK(cudaMemcpy(d_gal_codes, h_gal,
                               sizeof(int) * kGalNumPrn * kGalCodeLen,
                               cudaMemcpyHostToDevice));
        delete[] h_gal;
    }

    // --- BeiDou B1I: Gold codes, 2046 chips ---
    // Simplified: generate pseudo-random codes (actual B1I uses specific LFSR config)
    {
        int* h_bds = new int[kBdsNumPrn * kBdsCodeLen];
        for (int prn = 0; prn < kBdsNumPrn; ++prn) {
            unsigned int state = 0xBD510000u + prn * 0x6c62272eu;
            for (int chip = 0; chip < kBdsCodeLen; ++chip) {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                h_bds[prn * kBdsCodeLen + chip] = (state & 1) ? 1 : -1;
            }
        }
        CUDA_CHECK(cudaMalloc(&d_bds_codes,
                               sizeof(int) * kBdsNumPrn * kBdsCodeLen));
        CUDA_CHECK(cudaMemcpy(d_bds_codes, h_bds,
                               sizeof(int) * kBdsNumPrn * kBdsCodeLen,
                               cudaMemcpyHostToDevice));
        delete[] h_bds;
    }
}

// Generalized channel kernel: supports any constellation via code table + parameters
__global__ void generate_channel_kernel(float* iq_out,
                                        const int* code_table,
                                        int code_length,
                                        double chip_rate,
                                        int prn_index,  // 0-based
                                        double code_phase,
                                        double carrier_phase,
                                        double doppler_hz,
                                        float amplitude,
                                        int static_nav_bit,
                                        const int* nav_bits,
                                        int nav_bit_count,
                                        double nav_bit_rate,
                                        int n_samples,
                                        double sampling_freq,
                                        double intermediate_freq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    double t = (double)idx / sampling_freq;
    int chip_idx = ((int)(t * chip_rate + code_phase)) % code_length;
    if (chip_idx < 0) chip_idx += code_length;

    float code_val = (float)code_table[prn_index * code_length + chip_idx];

    // Navigation bit modulation
    int nav_bit;
    if (nav_bits != nullptr && nav_bit_count > 0) {
        int bit_idx = (int)(t * nav_bit_rate);
        if (bit_idx < 0) bit_idx = 0;
        if (bit_idx >= nav_bit_count) bit_idx = nav_bit_count - 1;
        nav_bit = nav_bits[bit_idx];
    } else {
        nav_bit = static_nav_bit;
    }

    double phase = kTwoPi * (intermediate_freq + doppler_hz) * t + carrier_phase;
    float cos_val, sin_val;
    __sincosf((float)phase, &sin_val, &cos_val);

    float signal = amplitude * (float)nav_bit * code_val;
    iq_out[2 * idx]     = signal * cos_val;
    iq_out[2 * idx + 1] = signal * (-sin_val);
}

// Helper: resolve code table, length, chip rate, and PRN index for a channel
struct ChannelCodeParams {
    const int* code_table;
    int code_length;
    double chip_rate;
    int prn_index;  // 0-based
    bool valid;
};

ChannelCodeParams resolve_code_params(const gnss_gpu::SatChannel& ch) {
    ChannelCodeParams p = {nullptr, 0, 0.0, 0, false};
    switch (ch.system) {
    case gnss_gpu::GNSS_GPS:
    case gnss_gpu::GNSS_QZSS:
        if (ch.prn < 1 || ch.prn > kGpsNumPrn) return p;
        p = {d_gps_codes, kGpsCodeLen, kGpsChipRate, ch.prn - 1, true};
        break;
    case gnss_gpu::GNSS_GLONASS:
        if (ch.prn < 1 || ch.prn > kGloNumPrn) return p;
        p = {d_glo_codes, kGloCodeLen, kGloChipRate, ch.prn - 1, true};
        break;
    case gnss_gpu::GNSS_GALILEO:
        if (ch.prn < 1 || ch.prn > kGalNumPrn) return p;
        p = {d_gal_codes, kGalCodeLen, kGalChipRate, ch.prn - 1, true};
        break;
    case gnss_gpu::GNSS_BEIDOU:
        if (ch.prn < 1 || ch.prn > kBdsNumPrn) return p;
        p = {d_bds_codes, kBdsCodeLen, kBdsChipRate, ch.prn - 1, true};
        break;
    default:
        return p;
    }
    return p;
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
            auto cp = resolve_code_params(channels[ch]);
            if (!cp.valid) continue;

            // Copy nav bits to device if provided
            int* d_nav_bits = nullptr;
            if (channels[ch].nav_bits != nullptr && channels[ch].nav_bit_count > 0) {
                size_t nb_bytes = (size_t)channels[ch].nav_bit_count * sizeof(int);
                CUDA_CHECK(cudaMalloc(&d_nav_bits, nb_bytes));
                CUDA_CHECK(cudaMemcpy(d_nav_bits, channels[ch].nav_bits,
                                       nb_bytes, cudaMemcpyHostToDevice));
            }

            float* ch_out = d_channel + (size_t)ch * num_floats;
            generate_channel_kernel<<<blocks, kBlockSize>>>(
                ch_out, cp.code_table, cp.code_length, cp.chip_rate,
                cp.prn_index, channels[ch].code_phase,
                channels[ch].carrier_phase, channels[ch].doppler_hz,
                channels[ch].amplitude, channels[ch].nav_bit,
                d_nav_bits, channels[ch].nav_bit_count,
                channels[ch].nav_bit_rate,
                n_samples, config.sampling_freq, config.intermediate_freq);
            CUDA_CHECK(cudaGetLastError());

            if (d_nav_bits) CUDA_CHECK(cudaFree(d_nav_bits));
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

    auto cp = resolve_code_params(ch);
    if (!cp.valid) { CUDA_CHECK(cudaFree(d_output)); return; }

    generate_channel_kernel<<<blocks, kBlockSize>>>(
        d_output, cp.code_table, cp.code_length, cp.chip_rate,
        cp.prn_index, ch.code_phase, ch.carrier_phase, ch.doppler_hz,
        ch.amplitude, ch.nav_bit, nullptr, 0, 50.0,
        n_samples, sampling_freq, intermediate_freq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(iq_out, d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));
}

}  // namespace gnss_gpu
