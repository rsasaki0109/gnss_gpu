#pragma once

namespace gnss_gpu {

// Simulate multipath reflections for receiver-satellite pairs
// rx_ecef: [n_rx * 3] receiver ECEF positions
// sat_ecef: [n_sat * 3] satellite ECEF positions
// reflector_planes: [n_ref * 6] each plane = (point[3], normal[3])
// excess_delays: [n_rx * n_sat] output excess path delay [m]
// attenuations: [n_rx * n_sat] output composite attenuation factor
void simulate_multipath(
    const double* rx_ecef, const double* sat_ecef,
    const double* reflector_planes,
    double* excess_delays, double* attenuations,
    int n_rx, int n_sat, int n_ref,
    double carrier_freq_hz, double chip_rate);

// Apply multipath-induced DLL tracking error to clean pseudoranges
// clean_pr: [n_epoch * n_sat] clean pseudoranges [m]
// rx_ecef: [n_epoch * 3] receiver ECEF positions
// sat_ecef: [n_epoch * n_sat * 3] satellite ECEF positions per epoch
// reflector_planes: [n_ref * 6]
// corrupted_pr: [n_epoch * n_sat] output corrupted pseudoranges [m]
// mp_errors: [n_epoch * n_sat] output multipath errors [m]
void apply_multipath_error(
    const double* clean_pr, const double* rx_ecef, const double* sat_ecef,
    const double* reflector_planes,
    double* corrupted_pr, double* mp_errors,
    int n_epoch, int n_sat, int n_ref,
    double carrier_freq_hz, double chip_rate, double correlator_spacing);

}  // namespace gnss_gpu
