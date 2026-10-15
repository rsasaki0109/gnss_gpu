#pragma once
#include <cstddef>
#include <memory>

namespace gnss_gpu {

struct SignalSimConfig {
    double sampling_freq = 2.6e6;
    double intermediate_freq = 0.0;
    double noise_floor_db = -20.0;
    unsigned long long noise_seed = 0;  // 0 => choose a fresh seed per epoch
};

// GNSS system identifiers
enum GnssSystem : int {
    GNSS_GPS     = 0,  // GPS L1 C/A  (1575.42 MHz, 1.023 Mchip/s, 1023 chips)
    GNSS_GLONASS = 1,  // GLONASS L1 C/A (0.511 Mchip/s, 511 chips, shared code)
                       // FDMA: caller must add slot-dependent offset to doppler_hz
                       //   offset = (slot_k * 562500) Hz relative to GPS L1 center
    GNSS_GALILEO = 2,  // Galileo E1-B/C (1575.42 MHz, 1.023 Mchip/s, 4092 chips)
    GNSS_BEIDOU  = 3,  // BeiDou B1I (1561.098 MHz, 2.046 Mchip/s, 2046 chips)
    GNSS_QZSS    = 4,  // QZSS L1 C/A (same as GPS)
};

struct SatChannel {
    int prn = 1;              // PRN/slot number
    int system = GNSS_GPS;    // GnssSystem enum
    double code_phase = 0.0;  // chips
    double carrier_phase = 0.0; // radians
    double doppler_hz = 0.0;
    float amplitude = 1.0f;
    int nav_bit = 1;          // +1 / -1 (static bit for single-epoch)
    double nav_bit_rate = 50.0;  // bps (GPS=50, Galileo E1-B=250, GLONASS=50)
    int nav_bit_count = 0;    // number of bits in nav_bits array (0 = use static nav_bit)
    const int* nav_bits = nullptr;  // +1/-1 array, nav_bit_rate determines timing
};

void init_ca_codes();

// Valid channels are summed in input order with float channel-output rounding.
// Unknown systems/out-of-range PRNs are skipped. IQ output is interleaved I/Q.
void simulate_epoch(const SignalSimConfig& config,
                    const SatChannel* channels,
                    int n_channels,
                    float* iq_output,
                    int n_samples);

void generate_single_channel(const SatChannel& ch,
                             float* iq_out,
                             int n_samples,
                             double sampling_freq,
                             double intermediate_freq);

// Synchronous IQ generation with reusable work buffers on the owning CUDA
// device. Retains the largest requested buffers until close/destruction; no
// channel or navigation-bit data are cached between calls. Not thread-safe.
class DeviceSignalWorkspace {
 public:
    DeviceSignalWorkspace();
    ~DeviceSignalWorkspace();
    DeviceSignalWorkspace(const DeviceSignalWorkspace&) = delete;
    DeviceSignalWorkspace& operator=(const DeviceSignalWorkspace&) = delete;

    void simulate_epoch(const SignalSimConfig& config, const SatChannel* channels,
                        int n_channels, float* iq_output, int n_samples);
    // Generate equal-length epochs without changing sample indices or seeds.
    // Channel arrays remain owned by the caller until this synchronous call returns.
    void simulate_epochs(const SignalSimConfig& config, const SatChannel* const* channels,
                         const int* channel_counts, const unsigned long long* noise_seeds,
                         int n_epochs, float* iq_output, int n_samples);
    void close() noexcept;
    std::size_t allocated_bytes() const noexcept;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gnss_gpu
