#pragma once

namespace gnss_gpu {

struct AcquisitionResult {
    int prn;
    bool acquired;
    double code_phase;    // samples (fractional after peak interpolation)
    double doppler_hz;
    double snr;           // peak/mean ratio
};

void generate_ca_code(int prn, int* code_out);  // 1023 chips, values {-1, +1}

void acquire_parallel(
    const float* signal, int n_samples,
    double sampling_freq, double intermediate_freq,
    const int* prn_list, int n_prn,
    double doppler_range, double doppler_step,
    float threshold,
    AcquisitionResult* results);

}  // namespace gnss_gpu
