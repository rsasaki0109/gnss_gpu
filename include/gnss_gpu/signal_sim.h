#pragma once

namespace gnss_gpu {

struct SignalSimConfig {
    double sampling_freq = 2.6e6;
    double intermediate_freq = 0.0;
    double noise_floor_db = -20.0;
    unsigned long long noise_seed = 0;  // 0 => choose a fresh seed per epoch
};

struct SatChannel {
    int prn = 1;              // 1..32
    double code_phase = 0.0;  // chips
    double carrier_phase = 0.0; // radians
    double doppler_hz = 0.0;
    float amplitude = 1.0f;
    int nav_bit = 1;          // +1 / -1
};

void init_ca_codes();

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

}  // namespace gnss_gpu
