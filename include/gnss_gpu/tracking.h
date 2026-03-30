#pragma once

namespace gnss_gpu {

struct ChannelState {
    double code_phase;      // chips
    double code_freq;       // Hz
    double carrier_phase;   // cycles
    double carrier_freq;    // Hz
    double cn0;             // dB-Hz
    double dll_integrator;  // DLL loop filter integrator state
    double pll_integrator;  // PLL loop filter integrator state
    int prn;
    bool locked;
};

struct TrackingConfig {
    double sampling_freq;
    double intermediate_freq;
    double integration_time;   // s (1e-3)
    double dll_bandwidth;      // Hz (2.0)
    double pll_bandwidth;      // Hz (15.0)
    double correlator_spacing; // chips (0.5)
};

// Batch correlator: compute early/prompt/late I/Q correlations for all channels
// signal: [n_samples] raw IF samples (float, signal domain)
// channels: [n_channels] current channel states
// correlations: [n_channels * 6] output (EI, EQ, PI, PQ, LI, LQ per channel)
void batch_correlate(
    const float* signal, const ChannelState* channels,
    double* correlations,
    int n_channels, int n_samples, const TrackingConfig& config);

// Scalar tracking loop update: apply DLL/PLL discriminators and loop filters
// channels: [n_channels] channel states (updated in place)
// correlations: [n_channels * 6] correlator outputs
void scalar_tracking_update(
    ChannelState* channels, const double* correlations,
    int n_channels, const TrackingConfig& config);

// Vector tracking loop update: EKF-based navigation-aided tracking
// channels: [n_channels] channel states (updated in place)
// correlations: [n_channels * 6] correlator outputs
// sat_ecef: [n_channels * 3] satellite ECEF positions
// sat_vel: [n_channels * 3] satellite ECEF velocities
// nav_state: [8] navigation state (x,y,z,vx,vy,vz,cb,cd)
// nav_cov: [64] navigation covariance (8x8)
// dt: time step [s]
void vector_tracking_update(
    ChannelState* channels, const double* correlations,
    const double* sat_ecef, const double* sat_vel,
    double* nav_state, double* nav_cov,
    int n_channels, const TrackingConfig& config, double dt);

// CN0 estimation using Narrow-Wideband Power Ratio
// correlations_hist: [n_channels * n_hist * 6] history of correlations
// cn0: [n_channels] output CN0 in dB-Hz
void cn0_nwpr(const double* correlations_hist, double* cn0,
              int n_channels, int n_hist, double T);

}  // namespace gnss_gpu
