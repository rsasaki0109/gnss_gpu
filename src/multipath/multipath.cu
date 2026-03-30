#include "gnss_gpu/multipath.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cstring>

namespace gnss_gpu {

static constexpr double SPEED_OF_LIGHT = 299792458.0;
static constexpr double PI = 3.14159265358979323846;
static constexpr double FRESNEL_CONCRETE = 0.5;  // approximate reflection coefficient

// Mirror a point across a plane defined by (plane_pt, plane_normal)
__device__ void reflect_point(const double pt[3], const double plane_pt[3],
                              const double plane_normal[3], double reflected[3]) {
  // d = dot(pt - plane_pt, normal)
  double d = 0.0;
  for (int i = 0; i < 3; i++) {
    d += (pt[i] - plane_pt[i]) * plane_normal[i];
  }
  // reflected = pt - 2*d*normal
  for (int i = 0; i < 3; i++) {
    reflected[i] = pt[i] - 2.0 * d * plane_normal[i];
  }
}

// Compute excess path delay [m] for a single reflection
// excess = |rx - refl_pt| + |refl_pt - sat| - |rx - sat|
// where refl_pt is the specular reflection point on the plane
__device__ double compute_excess_delay(const double rx[3], const double sat[3],
                                       const double plane_pt[3], const double plane_n[3]) {
  // Reflect receiver across the plane
  double rx_reflected[3];
  reflect_point(rx, plane_pt, plane_n, rx_reflected);

  // Direct path length
  double direct = 0.0;
  for (int i = 0; i < 3; i++) {
    double d = rx[i] - sat[i];
    direct += d * d;
  }
  direct = sqrt(direct);

  // Reflected path length = |rx_reflected - sat| (by image method)
  double reflected_path = 0.0;
  for (int i = 0; i < 3; i++) {
    double d = rx_reflected[i] - sat[i];
    reflected_path += d * d;
  }
  reflected_path = sqrt(reflected_path);

  double excess = reflected_path - direct;
  return (excess > 0.0) ? excess : 0.0;
}

// C/A code correlation triangle envelope
// Returns (1 - |delay_chips|) for |delay_chips| < 1, else 0
__device__ double multipath_envelope(double delay_chips) {
  double abs_delay = fabs(delay_chips);
  if (abs_delay < 1.0) {
    return 1.0 - abs_delay;
  }
  return 0.0;
}

// DLL discriminator multipath error (early-minus-late)
// Returns pseudorange error in meters
__device__ double dll_multipath_error(double delay_m, double attenuation, double phase_rad,
                                      double carrier_freq_hz, double chip_rate,
                                      double spacing) {
  double chip_length = SPEED_OF_LIGHT / chip_rate;  // ~293.05 m for C/A
  double delay_chips = delay_m / chip_length;

  // Early and late multipath correlation values
  double half_spacing = spacing / 2.0;
  double early_mp = multipath_envelope(delay_chips - half_spacing);
  double late_mp = multipath_envelope(delay_chips + half_spacing);

  // Direct signal early-minus-late
  double early_direct = multipath_envelope(-half_spacing);  // = 1 - d/2
  double late_direct = multipath_envelope(half_spacing);    // = 1 - d/2
  double eml_direct = early_direct - late_direct;           // = 0 for symmetric

  // Multipath component early-minus-late with phase
  double eml_mp = attenuation * cos(phase_rad) * (early_mp - late_mp);

  // The discriminator slope (normalizing factor)
  // For triangle correlator: slope = 2/chip_length at zero offset
  double slope = 2.0 / chip_length;

  // DLL tracking error = multipath EML perturbation / discriminator slope
  double error_m = 0.0;
  if (fabs(slope) > 1e-15) {
    error_m = eml_mp / slope;
  }

  return error_m;
}

__global__ void simulate_kernel(const double* rx_ecef, const double* sat_ecef,
                                 const double* reflector_planes,
                                 double* excess_delays, double* attenuations,
                                 int n_rx, int n_sat, int n_ref,
                                 double carrier_freq_hz, double chip_rate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_rx * n_sat;
  if (idx >= total) return;

  int rx_idx = idx / n_sat;
  int sat_idx = idx % n_sat;

  const double* rx = rx_ecef + rx_idx * 3;
  const double* sat = sat_ecef + sat_idx * 3;

  double wavelength = SPEED_OF_LIGHT / carrier_freq_hz;
  double chip_length = SPEED_OF_LIGHT / chip_rate;

  double total_delay = 0.0;
  double total_atten = 0.0;

  // Sum contributions from all reflectors
  for (int r = 0; r < n_ref; r++) {
    const double* plane_pt = reflector_planes + r * 6;
    const double* plane_n = reflector_planes + r * 6 + 3;

    double excess = compute_excess_delay(rx, sat, plane_pt, plane_n);
    double delay_chips = excess / chip_length;

    // Only contributes if within correlation triangle
    double envelope = multipath_envelope(delay_chips);
    if (envelope <= 0.0) continue;

    double atten = FRESNEL_CONCRETE * envelope;

    // Phase: pi for reflection + 2*pi*excess/wavelength
    double phase = PI + 2.0 * PI * excess / wavelength;

    // Weighted accumulation
    total_delay += excess * atten;
    total_atten += atten;
  }

  // Store composite values
  if (total_atten > 0.0) {
    excess_delays[idx] = total_delay / total_atten;  // weighted mean delay
  } else {
    excess_delays[idx] = 0.0;
  }
  attenuations[idx] = total_atten;
}

__global__ void apply_error_kernel(const double* clean_pr, const double* rx_ecef,
                                    const double* sat_ecef,
                                    const double* reflector_planes,
                                    double* corrupted_pr, double* mp_errors,
                                    int n_epoch, int n_sat, int n_ref,
                                    double carrier_freq_hz, double chip_rate,
                                    double correlator_spacing) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (idx >= total) return;

  int epoch_idx = idx / n_sat;
  int sat_idx = idx % n_sat;

  const double* rx = rx_ecef + epoch_idx * 3;
  const double* sat = sat_ecef + (epoch_idx * n_sat + sat_idx) * 3;

  double wavelength = SPEED_OF_LIGHT / carrier_freq_hz;

  double total_error = 0.0;

  // Sum DLL error contributions from all reflectors
  for (int r = 0; r < n_ref; r++) {
    const double* plane_pt = reflector_planes + r * 6;
    const double* plane_n = reflector_planes + r * 6 + 3;

    double excess = compute_excess_delay(rx, sat, plane_pt, plane_n);
    if (excess <= 0.0) continue;

    double atten = FRESNEL_CONCRETE;

    // Phase: pi for reflection + 2*pi*excess/wavelength
    double phase = PI + 2.0 * PI * excess / wavelength;

    total_error += dll_multipath_error(excess, atten, phase,
                                       carrier_freq_hz, chip_rate,
                                       correlator_spacing);
  }

  mp_errors[idx] = total_error;
  corrupted_pr[idx] = clean_pr[idx] + total_error;
}

void simulate_multipath(const double* rx_ecef, const double* sat_ecef,
                         const double* reflector_planes,
                         double* excess_delays, double* attenuations,
                         int n_rx, int n_sat, int n_ref,
                         double carrier_freq_hz, double chip_rate) {
  int total = n_rx * n_sat;
  size_t sz_rx = (size_t)n_rx * 3 * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_ref = (size_t)n_ref * 6 * sizeof(double);
  size_t sz_out = (size_t)total * sizeof(double);

  double *d_rx, *d_sat, *d_ref, *d_delays, *d_atten;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_ref, sz_ref));
  CUDA_CHECK(cudaMalloc(&d_delays, sz_out));
  CUDA_CHECK(cudaMalloc(&d_atten, sz_out));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref, reflector_planes, sz_ref, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (total + block - 1) / block;
  simulate_kernel<<<grid, block>>>(d_rx, d_sat, d_ref, d_delays, d_atten,
                                    n_rx, n_sat, n_ref,
                                    carrier_freq_hz, chip_rate);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(excess_delays, d_delays, sz_out, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(attenuations, d_atten, sz_out, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx)); CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_delays)); CUDA_CHECK(cudaFree(d_atten));
}

void apply_multipath_error(const double* clean_pr, const double* rx_ecef,
                            const double* sat_ecef,
                            const double* reflector_planes,
                            double* corrupted_pr, double* mp_errors,
                            int n_epoch, int n_sat, int n_ref,
                            double carrier_freq_hz, double chip_rate,
                            double correlator_spacing) {
  int total = n_epoch * n_sat;
  size_t sz_pr = (size_t)total * sizeof(double);
  size_t sz_rx = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_ref = (size_t)n_ref * 6 * sizeof(double);

  double *d_pr, *d_rx, *d_sat, *d_ref, *d_corrupted, *d_errors;

  CUDA_CHECK(cudaMalloc(&d_pr, sz_pr));
  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_ref, sz_ref));
  CUDA_CHECK(cudaMalloc(&d_corrupted, sz_pr));
  CUDA_CHECK(cudaMalloc(&d_errors, sz_pr));

  CUDA_CHECK(cudaMemcpy(d_pr, clean_pr, sz_pr, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ref, reflector_planes, sz_ref, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (total + block - 1) / block;
  apply_error_kernel<<<grid, block>>>(d_pr, d_rx, d_sat, d_ref,
                                       d_corrupted, d_errors,
                                       n_epoch, n_sat, n_ref,
                                       carrier_freq_hz, chip_rate,
                                       correlator_spacing);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(corrupted_pr, d_corrupted, sz_pr, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mp_errors, d_errors, sz_pr, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_pr)); CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_corrupted)); CUDA_CHECK(cudaFree(d_errors));
}

}  // namespace gnss_gpu
