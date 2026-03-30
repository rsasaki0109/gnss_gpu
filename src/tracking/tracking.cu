#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/tracking.h"
#include "gnss_gpu/coordinates.h"
#include <cmath>
#include <cstring>

namespace gnss_gpu {

// GPS L1 C/A code parameters
static constexpr double GPS_L1_FREQ = 1575.42e6;       // Hz
static constexpr double CA_CODE_RATE = 1.023e6;        // chips/s
static constexpr int CA_CODE_LENGTH = 1023;             // chips
static constexpr double TWO_PI = 2.0 * M_PI;

// G2 delay taps for PRN 1-32 (1-indexed tap pairs)
static const int G2_TAPS[32][2] = {
    {2, 6}, {3, 7}, {4, 8}, {5, 9}, {1, 9}, {2, 10}, {1, 8}, {2, 9},
    {3, 10}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10},
    {1, 4}, {2, 5}, {3, 6}, {4, 7}, {5, 8}, {6, 9}, {1, 3}, {4, 6},
    {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 6}, {2, 7}, {3, 8}, {4, 9},
};

// Pre-computed C/A Gold codes for PRN 1-32, stored in __constant__ device memory.
// Each code is 1023 chips (+1/-1 values stored as int8_t). Total: 1023 * 32 = 32736 bytes.
__constant__ int8_t d_ca_codes[32][CA_CODE_LENGTH];

// Host-side flag to track whether constant memory has been initialized
static bool s_ca_codes_initialized = false;

// Host function: generate C/A code for a given PRN using LFSR Gold code algorithm
static void generate_ca_code_host(int prn, int8_t* code_out) {
    int g1[10], g2[10];
    for (int i = 0; i < 10; i++) { g1[i] = 1; g2[i] = 1; }

    int tap1 = G2_TAPS[prn - 1][0] - 1;
    int tap2 = G2_TAPS[prn - 1][1] - 1;

    for (int i = 0; i < CA_CODE_LENGTH; i++) {
        int g1_out = g1[9];
        int g2_delayed = g2[tap1] ^ g2[tap2];
        int ca_bit = g1_out ^ g2_delayed;
        code_out[i] = (int8_t)(2 * ca_bit - 1);  // 0 -> -1, 1 -> +1

        // G1 feedback: taps 3,10 (0-indexed: 2,9)
        int g1_fb = g1[2] ^ g1[9];
        // G2 feedback: taps 2,3,6,8,9,10 (0-indexed: 1,2,5,7,8,9)
        int g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9];

        for (int j = 9; j > 0; j--) { g1[j] = g1[j - 1]; g2[j] = g2[j - 1]; }
        g1[0] = g1_fb;
        g2[0] = g2_fb;
    }
}

// Initialize __constant__ C/A code table (called once before first use)
static void ensure_ca_codes_initialized() {
    if (s_ca_codes_initialized) return;

    int8_t h_ca_codes[32][CA_CODE_LENGTH];
    for (int prn = 1; prn <= 32; prn++) {
        generate_ca_code_host(prn, h_ca_codes[prn - 1]);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_ca_codes, h_ca_codes, sizeof(h_ca_codes)));
    s_ca_codes_initialized = true;
}

// ============================================================
// Correlator kernel: 1 thread per channel
// ============================================================
__global__ void correlate_kernel(const float* signal,
                                  const ChannelState* channels,
                                  double* correlations,
                                  int n_channels, int n_samples,
                                  double sampling_freq,
                                  double intermediate_freq,
                                  double correlator_spacing) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n_channels) return;

  const ChannelState& state = channels[ch];
  if (!state.locked) {
    for (int k = 0; k < 6; k++) correlations[ch * 6 + k] = 0.0;
    return;
  }

  double code_phase = state.code_phase;
  double code_freq = state.code_freq;
  double carrier_phase = state.carrier_phase;
  double carrier_freq = state.carrier_freq;

  double ts = 1.0 / sampling_freq;
  double code_step = code_freq * ts;
  double half_spacing = correlator_spacing * 0.5;

  // C/A code lookup from pre-computed __constant__ memory
  int prn_idx = state.prn - 1;  // 0-indexed into d_ca_codes

  double EI = 0.0, EQ = 0.0;
  double PI_v = 0.0, PQ = 0.0;
  double LI = 0.0, LQ = 0.0;

  for (int i = 0; i < n_samples; i++) {
    double s = (double)signal[i];

    // Carrier replica
    double phase_rad = TWO_PI * (carrier_phase + carrier_freq * i * ts);
    double cos_car = cos(phase_rad);
    double sin_car = sin(phase_rad);

    // Code phase at this sample
    double cp = code_phase + code_step * i;
    // Wrap to [0, CA_CODE_LENGTH)
    cp = fmod(cp, (double)CA_CODE_LENGTH);
    if (cp < 0.0) cp += (double)CA_CODE_LENGTH;

    // Early/prompt/late code phase indices
    double cp_early = cp + half_spacing;
    double cp_prompt = cp;
    double cp_late = cp - half_spacing;

    // Wrap
    if (cp_early >= CA_CODE_LENGTH) cp_early -= CA_CODE_LENGTH;
    if (cp_late < 0.0) cp_late += CA_CODE_LENGTH;

    int chip_e = (int)cp_early % CA_CODE_LENGTH;
    int chip_p = (int)cp_prompt % CA_CODE_LENGTH;
    int chip_l = (int)cp_late % CA_CODE_LENGTH;

    double code_e = (double)d_ca_codes[prn_idx][chip_e];
    double code_p = (double)d_ca_codes[prn_idx][chip_p];
    double code_l = (double)d_ca_codes[prn_idx][chip_l];

    // Multiply-accumulate
    double i_mix = s * cos_car;
    double q_mix = s * sin_car;

    EI += i_mix * code_e;
    EQ += q_mix * code_e;
    PI_v += i_mix * code_p;
    PQ += q_mix * code_p;
    LI += i_mix * code_l;
    LQ += q_mix * code_l;
  }

  correlations[ch * 6 + 0] = EI;
  correlations[ch * 6 + 1] = EQ;
  correlations[ch * 6 + 2] = PI_v;
  correlations[ch * 6 + 3] = PQ;
  correlations[ch * 6 + 4] = LI;
  correlations[ch * 6 + 5] = LQ;
}

// ============================================================
// Discriminators and loop filter (device functions)
// ============================================================
__device__ double dll_discriminator(double EI, double EQ, double LI, double LQ) {
  double E_pow = EI * EI + EQ * EQ;
  double L_pow = LI * LI + LQ * LQ;
  double denom = E_pow + L_pow;
  if (denom < 1e-20) return 0.0;
  return (E_pow - L_pow) / denom;
}

__device__ double pll_discriminator(double PI_v, double PQ) {
  return atan2(PQ, PI_v);
}

__device__ void loop_filter_2nd(double disc, double* nco_freq, double* integrator,
                                 double bandwidth, double dt) {
  double zeta = 0.707;
  double omega_n = bandwidth * 8.0 * zeta / (4.0 * zeta * zeta + 1.0);
  *integrator += omega_n * omega_n * disc * dt;
  *nco_freq = omega_n * 2.0 * zeta * disc + *integrator;
}

// ============================================================
// Scalar tracking update kernel: 1 thread per channel
// ============================================================
__global__ void scalar_update_kernel(ChannelState* channels,
                                      const double* correlations,
                                      int n_channels,
                                      double integration_time,
                                      double dll_bandwidth,
                                      double pll_bandwidth) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n_channels) return;

  ChannelState& state = channels[ch];
  if (!state.locked) return;

  double EI = correlations[ch * 6 + 0];
  double EQ = correlations[ch * 6 + 1];
  double PI_v = correlations[ch * 6 + 2];
  double PQ = correlations[ch * 6 + 3];
  double LI = correlations[ch * 6 + 4];
  double LQ = correlations[ch * 6 + 5];

  double dt = integration_time;

  // DLL discriminator and filter
  double dll_disc = dll_discriminator(EI, EQ, LI, LQ);
  double dll_integrator = state.dll_integrator;
  double dll_nco = 0.0;
  loop_filter_2nd(dll_disc, &dll_nco, &dll_integrator, dll_bandwidth, dt);
  state.dll_integrator = dll_integrator;
  state.code_freq = CA_CODE_RATE + dll_nco;

  // PLL discriminator and filter
  double pll_disc = pll_discriminator(PI_v, PQ);
  double pll_integrator = state.pll_integrator;
  double pll_nco = 0.0;
  loop_filter_2nd(pll_disc, &pll_nco, &pll_integrator, pll_bandwidth, dt);
  state.pll_integrator = pll_integrator;
  state.carrier_freq += pll_nco;

  // Update code and carrier phases
  state.code_phase += state.code_freq * dt;
  state.code_phase = fmod(state.code_phase, (double)CA_CODE_LENGTH);
  if (state.code_phase < 0.0) state.code_phase += (double)CA_CODE_LENGTH;

  state.carrier_phase += state.carrier_freq * dt;
  state.carrier_phase = fmod(state.carrier_phase, 1.0);
  if (state.carrier_phase < 0.0) state.carrier_phase += 1.0;

  // Check lock: if prompt power is too low, mark as unlocked
  double prompt_power = PI_v * PI_v + PQ * PQ;
  if (prompt_power < 1e-20) {
    state.locked = false;
  }
}

// ============================================================
// CN0 estimation kernel: Narrow-Wideband Power Ratio
// ============================================================
__global__ void cn0_nwpr_kernel(const double* correlations_hist,
                                 double* cn0,
                                 int n_channels, int n_hist, double T) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n_channels) return;

  double sum_PI = 0.0, sum_PQ = 0.0;
  double sum_pow = 0.0;

  for (int m = 0; m < n_hist; m++) {
    double pi_val = correlations_hist[(ch * n_hist + m) * 6 + 2];
    double pq_val = correlations_hist[(ch * n_hist + m) * 6 + 3];
    sum_PI += pi_val;
    sum_PQ += pq_val;
    sum_pow += pi_val * pi_val + pq_val * pq_val;
  }

  // Narrow-band power
  double NP = sum_PI * sum_PI + sum_PQ * sum_PQ;
  // Wide-band power
  double WP = sum_pow;

  double M = (double)n_hist;
  double ratio = NP / WP;

  // Guard against invalid values
  if (ratio <= 1.0 || ratio >= M) {
    cn0[ch] = 0.0;
    return;
  }

  double cn0_linear = (1.0 / T) * (ratio - 1.0) / (M - ratio);
  cn0[ch] = 10.0 * log10(cn0_linear);
}

// ============================================================
// Host wrapper: batch_correlate
// ============================================================
void batch_correlate(const float* signal, const ChannelState* channels,
                     double* correlations,
                     int n_channels, int n_samples, const TrackingConfig& config) {
  ensure_ca_codes_initialized();

  float* d_signal;
  ChannelState* d_channels;
  double* d_corr;

  size_t sz_sig = (size_t)n_samples * sizeof(float);
  size_t sz_ch = (size_t)n_channels * sizeof(ChannelState);
  size_t sz_corr = (size_t)n_channels * 6 * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_signal, sz_sig));
  CUDA_CHECK(cudaMalloc(&d_channels, sz_ch));
  CUDA_CHECK(cudaMalloc(&d_corr, sz_corr));

  CUDA_CHECK(cudaMemcpy(d_signal, signal, sz_sig, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_channels, channels, sz_ch, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_channels + block - 1) / block;
  correlate_kernel<<<grid, block>>>(d_signal, d_channels, d_corr,
                                     n_channels, n_samples,
                                     config.sampling_freq,
                                     config.intermediate_freq,
                                     config.correlator_spacing);

  CUDA_CHECK(cudaMemcpy(correlations, d_corr, sz_corr, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_signal));
  CUDA_CHECK(cudaFree(d_channels));
  CUDA_CHECK(cudaFree(d_corr));
}

// ============================================================
// Host wrapper: scalar_tracking_update
// ============================================================
void scalar_tracking_update(ChannelState* channels, const double* correlations,
                            int n_channels, const TrackingConfig& config) {
  ChannelState* d_channels;
  double* d_corr;

  size_t sz_ch = (size_t)n_channels * sizeof(ChannelState);
  size_t sz_corr = (size_t)n_channels * 6 * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_channels, sz_ch));
  CUDA_CHECK(cudaMalloc(&d_corr, sz_corr));

  CUDA_CHECK(cudaMemcpy(d_channels, channels, sz_ch, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_corr, correlations, sz_corr, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_channels + block - 1) / block;
  scalar_update_kernel<<<grid, block>>>(d_channels, d_corr, n_channels,
                                         config.integration_time,
                                         config.dll_bandwidth,
                                         config.pll_bandwidth);

  CUDA_CHECK(cudaMemcpy(channels, d_channels, sz_ch, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_channels));
  CUDA_CHECK(cudaFree(d_corr));
}

// ============================================================
// VTL EKF helper functions (CPU with device memory transfers)
// ============================================================
static void ekf_predict(double* state, double* P, double dt) {
  // State transition: constant velocity + clock drift
  // x += vx*dt, y += vy*dt, z += vz*dt, cb += cd*dt
  state[0] += state[3] * dt;
  state[1] += state[4] * dt;
  state[2] += state[5] * dt;
  state[6] += state[7] * dt;

  // F matrix (8x8 identity + off-diagonals)
  // F = I + dt * [[0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], ...]
  // P = F*P*F' + Q
  double F[64];
  memset(F, 0, sizeof(F));
  for (int i = 0; i < 8; i++) F[i * 8 + i] = 1.0;
  F[0 * 8 + 3] = dt;  // x <- vx
  F[1 * 8 + 4] = dt;  // y <- vy
  F[2 * 8 + 5] = dt;  // z <- vz
  F[6 * 8 + 7] = dt;  // cb <- cd

  // FP = F * P
  double FP[64];
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      double sum = 0.0;
      for (int k = 0; k < 8; k++) sum += F[i * 8 + k] * P[k * 8 + j];
      FP[i * 8 + j] = sum;
    }
  }

  // P = FP * F' + Q
  double Q_pos = 1.0;        // m^2 position process noise
  double Q_vel = 0.1;        // (m/s)^2 velocity process noise
  double Q_cb = 100.0;       // m^2 clock bias process noise
  double Q_cd = 10.0;        // (m/s)^2 clock drift process noise

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      double sum = 0.0;
      for (int k = 0; k < 8; k++) sum += FP[i * 8 + k] * F[j * 8 + k];
      P[i * 8 + j] = sum;
    }
  }
  // Add process noise
  P[0 * 8 + 0] += Q_pos * dt;
  P[1 * 8 + 1] += Q_pos * dt;
  P[2 * 8 + 2] += Q_pos * dt;
  P[3 * 8 + 3] += Q_vel * dt;
  P[4 * 8 + 4] += Q_vel * dt;
  P[5 * 8 + 5] += Q_vel * dt;
  P[6 * 8 + 6] += Q_cb * dt;
  P[7 * 8 + 7] += Q_cd * dt;
}

// Invert NxN matrix in-place using Gauss-Jordan (small N only)
static bool invert_matrix(double* A, int N) {
  double aug[16 * 32];  // max 16x32 augmented
  if (N > 16) return false;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) aug[i * 2 * N + j] = A[i * N + j];
    for (int j = 0; j < N; j++) aug[i * 2 * N + N + j] = (i == j) ? 1.0 : 0.0;
  }

  for (int col = 0; col < N; col++) {
    int max_row = col;
    for (int row = col + 1; row < N; row++) {
      if (fabs(aug[row * 2 * N + col]) > fabs(aug[max_row * 2 * N + col])) max_row = row;
    }
    if (max_row != col) {
      for (int k = 0; k < 2 * N; k++) {
        double tmp = aug[col * 2 * N + k];
        aug[col * 2 * N + k] = aug[max_row * 2 * N + k];
        aug[max_row * 2 * N + k] = tmp;
      }
    }
    double pivot = aug[col * 2 * N + col];
    if (fabs(pivot) < 1e-15) return false;
    for (int k = 0; k < 2 * N; k++) aug[col * 2 * N + k] /= pivot;
    for (int row = 0; row < N; row++) {
      if (row == col) continue;
      double factor = aug[row * 2 * N + col];
      for (int k = 0; k < 2 * N; k++) aug[row * 2 * N + k] -= factor * aug[col * 2 * N + k];
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) A[i * N + j] = aug[i * 2 * N + N + j];
  }
  return true;
}

static void ekf_update(double* state, double* P,
                        const double* z_meas, const double* z_pred,
                        const double* H, const double* R_diag,
                        int n_meas) {
  // Innovation: y = z_meas - z_pred
  double y[32];  // max 32 measurements
  for (int i = 0; i < n_meas; i++) y[i] = z_meas[i] - z_pred[i];

  // S = H*P*H' + R
  int n = 8;
  double HP[256];  // n_meas x 8
  for (int i = 0; i < n_meas; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) sum += H[i * n + k] * P[k * n + j];
      HP[i * n + j] = sum;
    }
  }

  double S[1024];  // n_meas x n_meas
  for (int i = 0; i < n_meas; i++) {
    for (int j = 0; j < n_meas; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) sum += HP[i * n + k] * H[j * n + k];
      S[i * n_meas + j] = sum + ((i == j) ? R_diag[i] : 0.0);
    }
  }

  // S_inv = S^-1
  double S_inv[1024];
  memcpy(S_inv, S, n_meas * n_meas * sizeof(double));
  if (!invert_matrix(S_inv, n_meas)) return;

  // K = P * H' * S_inv
  double PHt[256];  // 8 x n_meas
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_meas; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) sum += P[i * n + k] * H[j * n + k];
      PHt[i * n_meas + j] = sum;
    }
  }

  double K[256];  // 8 x n_meas
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n_meas; j++) {
      double sum = 0.0;
      for (int k = 0; k < n_meas; k++) sum += PHt[i * n_meas + k] * S_inv[k * n_meas + j];
      K[i * n_meas + j] = sum;
    }
  }

  // state += K * y
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n_meas; j++) sum += K[i * n_meas + j] * y[j];
    state[i] += sum;
  }

  // P = (I - K*H) * P
  double KH[64];  // 8x8
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n_meas; k++) sum += K[i * n_meas + k] * H[k * n + j];
      KH[i * n + j] = sum;
    }
  }

  double P_new[64];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double IKH = ((i == j) ? 1.0 : 0.0) - KH[i * n + j];
      double sum = 0.0;
      for (int k = 0; k < n; k++) sum += IKH * P[k * n + j];  // This is wrong row-wise
      P_new[i * n + j] = sum;
    }
  }

  // Correct P = (I-KH)*P computation
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        double IKH_ik = ((i == k) ? 1.0 : 0.0) - KH[i * n + k];
        sum += IKH_ik * P[k * n + j];
      }
      P_new[i * n + j] = sum;
    }
  }
  memcpy(P, P_new, 64 * sizeof(double));
}

// ============================================================
// Host wrapper: vector_tracking_update
// ============================================================
void vector_tracking_update(ChannelState* channels, const double* correlations,
                            const double* sat_ecef, const double* sat_vel,
                            double* nav_state, double* nav_cov,
                            int n_channels, const TrackingConfig& config, double dt) {
  // 1. EKF predict
  ekf_predict(nav_state, nav_cov, dt);

  // 2. Build measurements: pseudorange and Doppler from correlator outputs
  int n_locked = 0;
  int locked_idx[32];
  for (int ch = 0; ch < n_channels && ch < 32; ch++) {
    if (channels[ch].locked) {
      locked_idx[n_locked++] = ch;
    }
  }

  if (n_locked < 4) return;  // Not enough satellites

  int n_meas = n_locked * 2;  // pseudorange + doppler per satellite
  double z_meas[64], z_pred[64], R_diag[64];
  double H[512];  // n_meas x 8
  memset(H, 0, sizeof(H));

  double rx = nav_state[0], ry = nav_state[1], rz = nav_state[2];
  double vx = nav_state[3], vy = nav_state[4], vz = nav_state[5];
  double cb = nav_state[6], cd = nav_state[7];

  for (int i = 0; i < n_locked; i++) {
    int ch = locked_idx[i];

    double sx = sat_ecef[ch * 3 + 0];
    double sy = sat_ecef[ch * 3 + 1];
    double sz = sat_ecef[ch * 3 + 2];

    double dx = rx - sx, dy = ry - sy, dz = rz - sz;
    double range = sqrt(dx * dx + dy * dy + dz * dz);

    // Unit vector from receiver to satellite
    double ux = dx / range, uy = dy / range, uz = dz / range;

    // Pseudorange measurement from code phase
    double pr_meas = range + cb;  // Simplified: use predicted range as proxy
    // In a real implementation, pseudorange comes from code phase measurement
    // Here we use PLL output to refine
    double PI_v = correlations[ch * 6 + 2];
    double PQ = correlations[ch * 6 + 3];
    double phase_err = atan2(PQ, PI_v);

    // Pseudorange observation
    z_meas[i] = range + cb + phase_err * C_LIGHT / (TWO_PI * GPS_L1_FREQ) * CA_CODE_RATE;
    z_pred[i] = range + cb;

    // Doppler measurement from carrier frequency offset
    double svx = sat_vel[ch * 3 + 0];
    double svy = sat_vel[ch * 3 + 1];
    double svz = sat_vel[ch * 3 + 2];

    double range_rate = ux * (vx - svx) + uy * (vy - svy) + uz * (vz - svz);
    double doppler_meas = channels[ch].carrier_freq - config.intermediate_freq;
    double doppler_pred = -range_rate * GPS_L1_FREQ / C_LIGHT;

    z_meas[n_locked + i] = doppler_meas;
    z_pred[n_locked + i] = doppler_pred;

    // Jacobian for pseudorange: [ux, uy, uz, 0, 0, 0, 1, 0]
    H[i * 8 + 0] = ux;
    H[i * 8 + 1] = uy;
    H[i * 8 + 2] = uz;
    H[i * 8 + 6] = 1.0;

    // Jacobian for Doppler: [0, 0, 0, ux, uy, uz, 0, 1] * (-f/c)
    // d(doppler)/d(vx) = -ux * f/c, etc.
    double fc = GPS_L1_FREQ / C_LIGHT;
    H[(n_locked + i) * 8 + 3] = -ux * fc;
    H[(n_locked + i) * 8 + 4] = -uy * fc;
    H[(n_locked + i) * 8 + 5] = -uz * fc;
    H[(n_locked + i) * 8 + 7] = 1.0;

    // Measurement noise
    R_diag[i] = 25.0;           // pseudorange noise variance (5m)^2
    R_diag[n_locked + i] = 1.0; // Doppler noise variance (1 Hz)^2
  }

  // 3. EKF update
  ekf_update(nav_state, nav_cov, z_meas, z_pred, H, R_diag, n_meas);

  // 4. Update channel NCOs from navigation solution
  rx = nav_state[0]; ry = nav_state[1]; rz = nav_state[2];
  vx = nav_state[3]; vy = nav_state[4]; vz = nav_state[5];
  cb = nav_state[6]; cd = nav_state[7];

  for (int i = 0; i < n_locked; i++) {
    int ch = locked_idx[i];

    double sx = sat_ecef[ch * 3 + 0];
    double sy = sat_ecef[ch * 3 + 1];
    double sz = sat_ecef[ch * 3 + 2];

    double dx = rx - sx, dy = ry - sy, dz = rz - sz;
    double range = sqrt(dx * dx + dy * dy + dz * dz);
    double ux = dx / range, uy = dy / range, uz = dz / range;

    double svx = sat_vel[ch * 3 + 0];
    double svy = sat_vel[ch * 3 + 1];
    double svz = sat_vel[ch * 3 + 2];

    double range_rate = ux * (vx - svx) + uy * (vy - svy) + uz * (vz - svz);

    // Update carrier frequency from nav solution
    channels[ch].carrier_freq = config.intermediate_freq - range_rate * GPS_L1_FREQ / C_LIGHT;

    // Update code frequency from nav solution
    channels[ch].code_freq = CA_CODE_RATE * (1.0 - range_rate / C_LIGHT);
  }
}

// ============================================================
// Host wrapper: cn0_nwpr
// ============================================================
void cn0_nwpr(const double* correlations_hist, double* cn0,
              int n_channels, int n_hist, double T) {
  double* d_hist;
  double* d_cn0;

  size_t sz_hist = (size_t)n_channels * n_hist * 6 * sizeof(double);
  size_t sz_cn0 = (size_t)n_channels * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_hist, sz_hist));
  CUDA_CHECK(cudaMalloc(&d_cn0, sz_cn0));

  CUDA_CHECK(cudaMemcpy(d_hist, correlations_hist, sz_hist, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_channels + block - 1) / block;
  cn0_nwpr_kernel<<<grid, block>>>(d_hist, d_cn0, n_channels, n_hist, T);

  CUDA_CHECK(cudaMemcpy(cn0, d_cn0, sz_cn0, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_hist));
  CUDA_CHECK(cudaFree(d_cn0));
}

}  // namespace gnss_gpu
