#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

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

struct DeviceChannel {
    const int* code_table;
    double chip_rate;
    double code_phase;
    double carrier_phase;
    double doppler_hz;
    double nav_bit_rate;
    size_t nav_offset;
    int code_length;
    int prn_index;
    int static_nav_bit;
    int nav_bit_count;
    float amplitude;
};

struct EpochDescriptor {
    size_t channel_offset;
    int channel_count;
    unsigned long long seed;
};
struct BatchChannels {
    const DeviceChannel* data;
    const EpochDescriptor* epochs;
    __device__ const DeviceChannel& operator[](int index) const {
        return data[epochs[blockIdx.y].channel_offset + (size_t)index];
    }
};

// A normal epoch fits within CUDA's 4 KiB kernel-parameter limit. Passing this
// batch by value avoids a separate descriptor allocation and host/device copy.
constexpr int kInlineChannels = 32;
constexpr int kCachedGlobalChannels = 256;
struct InlineChannels {
    DeviceChannel items[kInlineChannels];

    __device__ const DeviceChannel& operator[](int index) const {
        return items[index];
    }
};
static_assert(sizeof(InlineChannels) + 128 <= 4096);

// Evaluate channels in input order, retaining the two rounding steps of the
// former channel-output store followed by the composite-kernel addition.
template <typename ChannelStorage, bool CACHE_FREQUENCY = false,
          int CACHE_CHANNELS = kInlineChannels, bool BATCHED = false>
__global__ void fused_signal_kernel(ChannelStorage channels,
                                    const int* nav_bits,
                                    float* iq_out,
                                    int n_samples,
                                    int n_channels,
                                    double sampling_freq,
                                    double intermediate_freq,
                                    float noise_std,
                                    unsigned long long seed) {
    if constexpr (BATCHED) {
        int epoch = blockIdx.y;
        n_channels = channels.epochs[epoch].channel_count;
        seed = channels.epochs[epoch].seed;
        iq_out += (size_t)epoch * (size_t)n_samples * 2;
    }
    static_assert(!CACHE_FREQUENCY || CACHE_CHANNELS <= kBlockSize);
    __shared__ double angular_frequency[CACHE_CHANNELS];
    if constexpr (CACHE_FREQUENCY) {
        // Both cached batches fit within one 256-thread block. Use the
        // original GPU FP64 operations, then share across output samples.
        if (threadIdx.x < n_channels)
            angular_frequency[threadIdx.x] = kTwoPi * (
                intermediate_freq + channels[threadIdx.x].doppler_hz);
        __syncthreads();
    }
    // Partial blocks must participate in the coefficient barrier first.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    double t = (double)idx / sampling_freq;
    float sum_i = 0.0f, sum_q = 0.0f;
    for (int ch = 0; ch < n_channels; ++ch) {
        const DeviceChannel& channel = channels[ch];
        int chip_idx = ((int)(t * channel.chip_rate + channel.code_phase))
                       % channel.code_length;
        if (chip_idx < 0) chip_idx += channel.code_length;
        float code_val = (float)channel.code_table[
            channel.prn_index * channel.code_length + chip_idx];

        int nav_bit = channel.static_nav_bit;
        if (channel.nav_bit_count > 0) {
            int bit_idx = (int)(t * channel.nav_bit_rate);
            if (bit_idx < 0) bit_idx = 0;
            if (bit_idx >= channel.nav_bit_count) bit_idx = channel.nav_bit_count - 1;
            nav_bit = nav_bits[channel.nav_offset + (size_t)bit_idx];
        }

        double phase;
        if constexpr (CACHE_FREQUENCY)
            phase = angular_frequency[ch] * t + channel.carrier_phase;
        else
            phase = kTwoPi * (intermediate_freq + channel.doppler_hz) * t
                    + channel.carrier_phase;
        float cos_val, sin_val;
        __sincosf((float)phase, &sin_val, &cos_val);
        float signal = channel.amplitude * (float)nav_bit * code_val;
        sum_i = __fadd_rn(sum_i, __fmul_rn(signal, cos_val));
        sum_q = __fadd_rn(sum_q, __fmul_rn(signal, -sin_val));
    }

    if (noise_std > 0.0f) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, (unsigned long long)idx, 0, &state);
        sum_i += curand_normal(&state) * noise_std;
        sum_q += curand_normal(&state) * noise_std;
    }
    iq_out[2 * idx] = sum_i;
    iq_out[2 * idx + 1] = sum_q;
}

// Host packing proves every valid channel has nav_bit_count==0 when the
// packed NAV array is empty. Keep the original producer for all other cases;
// this copy drops only the unreachable dynamic-NAV branch.
template <typename ChannelStorage, bool CACHE_FREQUENCY = false,
          int CACHE_CHANNELS = kInlineChannels, bool BATCHED = false>
__global__ void fused_static_nav_signal_kernel(ChannelStorage channels,
                                    const int* nav_bits,
                                    float* iq_out,
                                    int n_samples,
                                    int n_channels,
                                    double sampling_freq,
                                    double intermediate_freq,
                                    float noise_std,
                                    unsigned long long seed) {
    if constexpr (BATCHED) {
        int epoch = blockIdx.y;
        n_channels = channels.epochs[epoch].channel_count;
        seed = channels.epochs[epoch].seed;
        iq_out += (size_t)epoch * (size_t)n_samples * 2;
    }
    static_assert(!CACHE_FREQUENCY || CACHE_CHANNELS <= kBlockSize);
    __shared__ double angular_frequency[CACHE_CHANNELS];
    if constexpr (CACHE_FREQUENCY) {
        // Both cached batches fit within one 256-thread block. Use the
        // original GPU FP64 operations, then share across output samples.
        if (threadIdx.x < n_channels)
            angular_frequency[threadIdx.x] = kTwoPi * (
                intermediate_freq + channels[threadIdx.x].doppler_hz);
        __syncthreads();
    }
    // Partial blocks must participate in the coefficient barrier first.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    double t = (double)idx / sampling_freq;
    float sum_i = 0.0f, sum_q = 0.0f;
    for (int ch = 0; ch < n_channels; ++ch) {
        const DeviceChannel& channel = channels[ch];
        int chip_idx = ((int)(t * channel.chip_rate + channel.code_phase))
                       % channel.code_length;
        if (chip_idx < 0) chip_idx += channel.code_length;
        float code_val = (float)channel.code_table[
            channel.prn_index * channel.code_length + chip_idx];

        int nav_bit = channel.static_nav_bit;

        double phase;
        if constexpr (CACHE_FREQUENCY)
            phase = angular_frequency[ch] * t + channel.carrier_phase;
        else
            phase = kTwoPi * (intermediate_freq + channel.doppler_hz) * t
                    + channel.carrier_phase;
        float cos_val, sin_val;
        __sincosf((float)phase, &sin_val, &cos_val);
        float signal = channel.amplitude * (float)nav_bit * code_val;
        sum_i = __fadd_rn(sum_i, __fmul_rn(signal, cos_val));
        sum_q = __fadd_rn(sum_q, __fmul_rn(signal, -sin_val));
    }

    if (noise_std > 0.0f) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, (unsigned long long)idx, 0, &state);
        sum_i += curand_normal(&state) * noise_std;
        sum_q += curand_normal(&state) * noise_std;
    }
    iq_out[2 * idx] = sum_i;
    iq_out[2 * idx + 1] = sum_q;
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

template <typename T>
struct WorkBuffer {
    T* data = nullptr;
    size_t capacity = 0;

    void reserve(size_t count) {
        if (count <= capacity) return;
        T* replacement = nullptr;
        CUDA_CHECK(cudaMalloc(&replacement, count * sizeof(T)));
        T* previous = data;
        data = replacement;
        capacity = count;
        if (previous) CUDA_CHECK(cudaFree(previous));
    }
    void reset() noexcept {
        if (data) cudaFree(data);
        data = nullptr;
        capacity = 0;
    }
};

class SignalDeviceScope {
 public:
    explicit SignalDeviceScope(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        changed_ = previous_ != device;
        if (changed_) CUDA_CHECK(cudaSetDevice(device));
    }
    ~SignalDeviceScope() { if (changed_) cudaSetDevice(previous_); }
 private:
    int previous_ = 0;
    bool changed_ = false;
};

}  // namespace

namespace gnss_gpu {

void init_ca_codes() {
    std::call_once(g_ca_init_flag, build_ca_codes_host);
}

struct DeviceSignalWorkspace::Impl {
    int device = 0;
    WorkBuffer<DeviceChannel> channels;
    WorkBuffer<int> nav_bits;
    WorkBuffer<float> output;
    WorkBuffer<EpochDescriptor> epochs;
    std::vector<EpochDescriptor> packed_epochs;
    std::vector<DeviceChannel> packed_channels;
    std::vector<int> packed_nav_bits;

    ~Impl() {
        int previous = 0;
        bool known = cudaGetDevice(&previous) == cudaSuccess;
        if ((!known || previous != device) && cudaSetDevice(device) != cudaSuccess)
            return;
        channels.reset(); nav_bits.reset(); output.reset(); epochs.reset();
        if (known && previous != device) cudaSetDevice(previous);
    }
};

DeviceSignalWorkspace::DeviceSignalWorkspace() : impl_(std::make_unique<Impl>()) {
    CUDA_CHECK(cudaGetDevice(&impl_->device));
}

DeviceSignalWorkspace::~DeviceSignalWorkspace() = default;
void DeviceSignalWorkspace::close() noexcept { impl_.reset(); }

size_t DeviceSignalWorkspace::allocated_bytes() const noexcept {
    if (!impl_) return 0;
    return impl_->channels.capacity * sizeof(DeviceChannel)
         + impl_->nav_bits.capacity * sizeof(int)
         + impl_->output.capacity * sizeof(float)
         + impl_->epochs.capacity * sizeof(EpochDescriptor);
}

void simulate_epoch(const SignalSimConfig& config,
                    const SatChannel* channels,
                    int n_channels,
                    float* iq_output,
                    int n_samples) {
    if (n_samples <= 0 || iq_output == nullptr) return;
    DeviceSignalWorkspace workspace;
    workspace.simulate_epoch(config, channels, n_channels, iq_output, n_samples);
}

void DeviceSignalWorkspace::simulate_epoch(const SignalSimConfig& config,
                                          const SatChannel* channels,
                                          int n_channels,
                                          float* iq_output,
                                          int n_samples) {
    if (!impl_) throw std::runtime_error("DeviceSignalWorkspace is closed");
    if (n_samples <= 0 || iq_output == nullptr) return;
    SignalDeviceScope device(impl_->device);

    init_ca_codes();

    int blocks = (n_samples + kBlockSize - 1) / kBlockSize;
    size_t num_floats = (size_t)n_samples * 2;
    size_t output_bytes = num_floats * sizeof(float);

    auto& packed_channels = impl_->packed_channels;
    auto& packed_nav_bits = impl_->packed_nav_bits;
    packed_channels.clear();
    packed_nav_bits.clear();
    int* d_nav_bits = nullptr;

    // Pack only valid channels, preserving input order and nav-bit clamping.
    if (n_channels > 0 && channels != nullptr) {
        packed_channels.reserve((size_t)n_channels);
        for (int ch = 0; ch < n_channels; ++ch) {
            auto cp = resolve_code_params(channels[ch]);
            if (!cp.valid) continue;
            size_t nav_offset = packed_nav_bits.size();
            int nav_bit_count = 0;
            if (channels[ch].nav_bits != nullptr && channels[ch].nav_bit_count > 0) {
                nav_bit_count = channels[ch].nav_bit_count;
                packed_nav_bits.insert(packed_nav_bits.end(), channels[ch].nav_bits,
                                       channels[ch].nav_bits + nav_bit_count);
            }
            packed_channels.push_back({
                cp.code_table, cp.chip_rate, channels[ch].code_phase,
                channels[ch].carrier_phase, channels[ch].doppler_hz,
                channels[ch].nav_bit_rate, nav_offset, cp.code_length,
                cp.prn_index, channels[ch].nav_bit, nav_bit_count,
                channels[ch].amplitude});
        }
    }

    impl_->output.reserve(num_floats);
    float* d_output = impl_->output.data;
    float noise_std = std::pow(10.0f, (float)config.noise_floor_db / 20.0f);
    if (!packed_channels.empty()) {
        if (!packed_nav_bits.empty()) {
            size_t nav_bytes = packed_nav_bits.size() * sizeof(int);
            impl_->nav_bits.reserve(packed_nav_bits.size());
            d_nav_bits = impl_->nav_bits.data;
            CUDA_CHECK(cudaMemcpy(d_nav_bits, packed_nav_bits.data(), nav_bytes,
                                   cudaMemcpyHostToDevice));
        }
        auto seed = resolve_noise_seed(config.noise_seed);
        if (packed_channels.size() <= kInlineChannels) {
            InlineChannels batch{};
            for (size_t ch = 0; ch < packed_channels.size(); ++ch)
                batch.items[ch] = packed_channels[ch];
            fused_signal_kernel<InlineChannels, true><<<blocks, kBlockSize>>>(
                batch, d_nav_bits, d_output, n_samples, (int)packed_channels.size(),
                config.sampling_freq, config.intermediate_freq, noise_std, seed);
        } else {
            size_t channel_bytes = packed_channels.size() * sizeof(DeviceChannel);
            impl_->channels.reserve(packed_channels.size());
            DeviceChannel* d_channels = impl_->channels.data;
            CUDA_CHECK(cudaMemcpy(d_channels, packed_channels.data(), channel_bytes,
                                   cudaMemcpyHostToDevice));
            if (packed_channels.size() <= kCachedGlobalChannels) {
                if (packed_nav_bits.empty()) {
                    fused_static_nav_signal_kernel<DeviceChannel*, true, kCachedGlobalChannels>
                        <<<blocks, kBlockSize>>>(
                        d_channels, d_nav_bits, d_output, n_samples, (int)packed_channels.size(),
                        config.sampling_freq, config.intermediate_freq, noise_std, seed);
                } else {
                    fused_signal_kernel<DeviceChannel*, true, kCachedGlobalChannels>
                        <<<blocks, kBlockSize>>>(
                        d_channels, d_nav_bits, d_output, n_samples, (int)packed_channels.size(),
                        config.sampling_freq, config.intermediate_freq, noise_std, seed);
                }
            } else {
                if (packed_nav_bits.empty()) {
                    fused_static_nav_signal_kernel<DeviceChannel*, false><<<blocks, kBlockSize>>>(
                        d_channels, d_nav_bits, d_output, n_samples, (int)packed_channels.size(),
                        config.sampling_freq, config.intermediate_freq, noise_std, seed);
                } else {
                    fused_signal_kernel<<<blocks, kBlockSize>>>(
                        d_channels, d_nav_bits, d_output, n_samples, (int)packed_channels.size(),
                        config.sampling_freq, config.intermediate_freq, noise_std, seed);
                }
            }
        }
    } else {
        composite_kernel<<<blocks, kBlockSize>>>(
            nullptr, d_output, n_samples, 0, noise_std,
            resolve_noise_seed(config.noise_seed));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(iq_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

}

void DeviceSignalWorkspace::simulate_epochs(
    const SignalSimConfig& config, const SatChannel* const* epoch_channels,
    const int* channel_counts, const unsigned long long* noise_seeds,
    int n_epochs, float* iq_output, int n_samples) {
    if (!impl_) throw std::runtime_error("DeviceSignalWorkspace is closed");
    if (n_epochs < 0 || n_epochs > 65535 || n_samples < 1 || n_samples > std::numeric_limits<int>::max() / 2)
        throw std::invalid_argument("Invalid signal epoch batch dimensions");
    if (!n_epochs) return;
    if (!epoch_channels || !channel_counts || !iq_output)
        throw std::invalid_argument("Signal epoch batch buffers must not be null");
    SignalDeviceScope device(impl_->device);
    init_ca_codes();
    auto& packed = impl_->packed_channels;
    auto& nav = impl_->packed_nav_bits;
    auto& epochs = impl_->packed_epochs;
    packed.clear(); nav.clear(); epochs.clear();
    epochs.reserve(n_epochs);
    int max_channels = 0;
    for (int e = 0; e < n_epochs; ++e) {
        if (channel_counts[e] < 0 || (channel_counts[e] && !epoch_channels[e]))
            throw std::invalid_argument("Invalid channels in signal epoch batch");
        size_t begin = packed.size();
        for (int i = 0; i < channel_counts[e]; ++i) {
            const auto& channel = epoch_channels[e][i];
            auto cp = resolve_code_params(channel);
            if (!cp.valid) continue;
            size_t nav_offset = nav.size();
            int nav_count = 0;
            if (channel.nav_bits && channel.nav_bit_count > 0) {
                nav_count = channel.nav_bit_count;
                nav.insert(nav.end(), channel.nav_bits, channel.nav_bits + nav_count);
            }
            packed.push_back({cp.code_table, cp.chip_rate, channel.code_phase,
                channel.carrier_phase, channel.doppler_hz, channel.nav_bit_rate,
                nav_offset, cp.code_length, cp.prn_index, channel.nav_bit, nav_count, channel.amplitude});
        }
        if (packed.size() - begin > (size_t)std::numeric_limits<int>::max())
            throw std::overflow_error("Too many channels in an epoch");
        int count = (int)(packed.size() - begin);
        max_channels = std::max(max_channels, count);
        epochs.push_back({begin, count, resolve_noise_seed(noise_seeds ? noise_seeds[e] : config.noise_seed)});
    }
    size_t floats = (size_t)n_epochs * (size_t)n_samples * 2;
    if (floats > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::overflow_error("Signal batch output size overflow");
    impl_->output.reserve(floats);
    impl_->channels.reserve(packed.size());
    impl_->nav_bits.reserve(nav.size());
    impl_->epochs.reserve(epochs.size());
    if (!packed.empty()) CUDA_CHECK(cudaMemcpy(impl_->channels.data, packed.data(), packed.size() * sizeof(DeviceChannel), cudaMemcpyHostToDevice));
    if (!nav.empty()) CUDA_CHECK(cudaMemcpy(impl_->nav_bits.data, nav.data(), nav.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl_->epochs.data, epochs.data(), epochs.size() * sizeof(EpochDescriptor), cudaMemcpyHostToDevice));
    BatchChannels batch{impl_->channels.data, impl_->epochs.data};
    dim3 blocks((n_samples + kBlockSize - 1) / kBlockSize, n_epochs);
    float noise_std = std::pow(10.0f, (float)config.noise_floor_db / 20.0f);
    if (max_channels <= kInlineChannels) {
        if (nav.empty() && max_channels > 0) {
            fused_static_nav_signal_kernel<BatchChannels, true, kInlineChannels, true><<<blocks, kBlockSize>>>(
                batch, impl_->nav_bits.data, impl_->output.data, n_samples, max_channels,
                config.sampling_freq, config.intermediate_freq, noise_std, 0ULL);
        } else {
            fused_signal_kernel<BatchChannels, true, kInlineChannels, true><<<blocks, kBlockSize>>>(
                batch, impl_->nav_bits.data, impl_->output.data, n_samples, max_channels,
                config.sampling_freq, config.intermediate_freq, noise_std, 0ULL);
        }
    } else if (max_channels <= kCachedGlobalChannels) {
        if (nav.empty() && max_channels > 0) {
            fused_static_nav_signal_kernel<BatchChannels, true, kCachedGlobalChannels, true><<<blocks, kBlockSize>>>(
                batch, impl_->nav_bits.data, impl_->output.data, n_samples, max_channels,
                config.sampling_freq, config.intermediate_freq, noise_std, 0ULL);
        } else {
            fused_signal_kernel<BatchChannels, true, kCachedGlobalChannels, true><<<blocks, kBlockSize>>>(
                batch, impl_->nav_bits.data, impl_->output.data, n_samples, max_channels,
                config.sampling_freq, config.intermediate_freq, noise_std, 0ULL);
        }
    } else {
        if (nav.empty()) {
            fused_static_nav_signal_kernel<BatchChannels, false, kInlineChannels, true><<<blocks, kBlockSize>>>(
                batch, impl_->nav_bits.data, impl_->output.data, n_samples, max_channels,
                config.sampling_freq, config.intermediate_freq, noise_std, 0ULL);
        } else {
            fused_signal_kernel<BatchChannels, false, kInlineChannels, true><<<blocks, kBlockSize>>>(
                batch, impl_->nav_bits.data, impl_->output.data, n_samples, max_channels,
                config.sampling_freq, config.intermediate_freq, noise_std, 0ULL);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(iq_output, impl_->output.data, floats * sizeof(float), cudaMemcpyDeviceToHost));
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

    int* d_nav_bits = nullptr;
    if (ch.nav_bits != nullptr && ch.nav_bit_count > 0) {
        size_t nb_bytes = (size_t)ch.nav_bit_count * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_nav_bits, nb_bytes));
        CUDA_CHECK(cudaMemcpy(d_nav_bits, ch.nav_bits, nb_bytes, cudaMemcpyHostToDevice));
    }

    generate_channel_kernel<<<blocks, kBlockSize>>>(
        d_output, cp.code_table, cp.code_length, cp.chip_rate,
        cp.prn_index, ch.code_phase, ch.carrier_phase, ch.doppler_hz,
        ch.amplitude, ch.nav_bit, d_nav_bits, ch.nav_bit_count,
        ch.nav_bit_rate, n_samples, sampling_freq, intermediate_freq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(iq_out, d_output, output_bytes, cudaMemcpyDeviceToHost));
    if (d_nav_bits) CUDA_CHECK(cudaFree(d_nav_bits));
    CUDA_CHECK(cudaFree(d_output));
}

}  // namespace gnss_gpu
