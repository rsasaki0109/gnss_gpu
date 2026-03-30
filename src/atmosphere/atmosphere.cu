#include "gnss_gpu/atmosphere.h"
#include "gnss_gpu/coordinates.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>

namespace gnss_gpu {

// ============================================================
// CPU implementations
// ============================================================

// Saastamoinen tropospheric delay model
// Uses standard atmosphere: P, T, e from altitude
// Mapping function: 1/sin(sqrt(el^2 + 6.25*(pi/180)^2))
double tropo_saastamoinen(double lat, double alt, double el) {
  // Standard atmosphere model
  double P = 1013.25 * pow(1.0 - 2.2557e-5 * alt, 5.2568);  // pressure [mbar]
  double T = 15.0 - 6.5e-3 * alt + 273.15;                    // temperature [K]
  double e = 6.108 * exp((17.15 * (T - 273.15)) / (T - 273.15 + 234.7));  // water vapor [mbar]
  // Relative humidity ~50% at standard conditions
  e *= 0.5;

  // Zenith tropospheric delay (simplified Saastamoinen)
  double z = M_PI / 2.0 - el;  // zenith angle
  double cos_z = cos(z);

  // Correction factor B (latitude/altitude dependent, simplified)
  // Use simplified version: B ~ 0.00266*cos(2*lat) + 0.00028*alt_km
  double alt_km = alt / 1000.0;

  // Zenith delay components
  double tropo_zenith = 0.002277 * (P + (1255.0 / T + 0.05) * e) /
                        (1.0 - 0.00266 * cos(2.0 * lat) - 0.00028 * alt_km);

  // Mapping function (Hopfield-style, avoids singularity at low elevation)
  double el_min = 2.0 * M_PI / 180.0;  // minimum 2 degrees
  double el_eff = (el > el_min) ? el : el_min;
  double sin_el = sin(sqrt(el_eff * el_eff + 6.25 * (M_PI / 180.0) * (M_PI / 180.0)));
  double mapped = tropo_zenith / sin_el;

  return mapped;
}

// Klobuchar ionospheric delay model (IS-GPS-200)
double iono_klobuchar(const double alpha[4], const double beta[4],
                      double lat, double lon, double az, double el,
                      double gps_time) {
  const double PI = M_PI;

  // Semi-circles conversion
  double lat_sc = lat / PI;  // latitude in semi-circles
  double lon_sc = lon / PI;  // longitude in semi-circles
  double el_sc = el / PI;    // elevation in semi-circles

  // 1. Earth-centered angle
  double psi = 0.0137 / (el_sc + 0.11) - 0.022;

  // 2. Ionospheric pierce point latitude
  double phi_i = lat_sc + psi * cos(az);
  if (phi_i > 0.416) phi_i = 0.416;
  if (phi_i < -0.416) phi_i = -0.416;

  // 3. Ionospheric pierce point longitude
  double lambda_i = lon_sc + psi * sin(az) / cos(phi_i * PI);

  // 4. Geomagnetic latitude
  double phi_m = phi_i + 0.064 * cos((lambda_i - 1.617) * PI);

  // 5. Local time
  double t = 4.32e4 * lambda_i + gps_time;
  // Normalize to [0, 86400)
  t = fmod(t, 86400.0);
  if (t < 0.0) t += 86400.0;

  // 6. Obliquity factor
  double F = 1.0 + 16.0 * pow(0.53 - el_sc, 3);

  // 7. Period
  double phi_m_pow = 1.0;
  double PER = 0.0;
  for (int n = 0; n < 4; n++) {
    PER += beta[n] * phi_m_pow;
    phi_m_pow *= phi_m;
  }
  if (PER < 72000.0) PER = 72000.0;

  // 8. Amplitude
  phi_m_pow = 1.0;
  double AMP = 0.0;
  for (int n = 0; n < 4; n++) {
    AMP += alpha[n] * phi_m_pow;
    phi_m_pow *= phi_m;
  }
  if (AMP < 0.0) AMP = 0.0;

  // 9. Phase
  double x = 2.0 * PI * (t - 50400.0) / PER;

  // 10. Ionospheric delay in seconds
  double Tiono;
  if (fabs(x) < 1.57) {
    Tiono = F * (5.0e-9 + AMP * (1.0 - x * x / 2.0 + x * x * x * x / 24.0));
  } else {
    Tiono = F * 5.0e-9;
  }

  // 11. Convert to meters
  return Tiono * C_LIGHT;
}

// ============================================================
// GPU device functions
// ============================================================

__device__ double d_tropo_saastamoinen(double lat, double alt, double el) {
  double P = 1013.25 * pow(1.0 - 2.2557e-5 * alt, 5.2568);
  double T = 15.0 - 6.5e-3 * alt + 273.15;
  double e_wv = 6.108 * exp((17.15 * (T - 273.15)) / (T - 273.15 + 234.7));
  e_wv *= 0.5;

  double alt_km = alt / 1000.0;
  double tropo_zenith = 0.002277 * (P + (1255.0 / T + 0.05) * e_wv) /
                        (1.0 - 0.00266 * cos(2.0 * lat) - 0.00028 * alt_km);

  double el_min = 2.0 * M_PI / 180.0;
  double el_eff = (el > el_min) ? el : el_min;
  double sin_el = sin(sqrt(el_eff * el_eff + 6.25 * (M_PI / 180.0) * (M_PI / 180.0)));
  return tropo_zenith / sin_el;
}

__device__ double d_iono_klobuchar(const double* alpha, const double* beta,
                                    double lat, double lon, double az, double el,
                                    double gps_time) {
  const double PI = M_PI;
  double lat_sc = lat / PI;
  double lon_sc = lon / PI;
  double el_sc = el / PI;

  double psi = 0.0137 / (el_sc + 0.11) - 0.022;

  double phi_i = lat_sc + psi * cos(az);
  if (phi_i > 0.416) phi_i = 0.416;
  if (phi_i < -0.416) phi_i = -0.416;

  double lambda_i = lon_sc + psi * sin(az) / cos(phi_i * PI);

  double phi_m = phi_i + 0.064 * cos((lambda_i - 1.617) * PI);

  double t = 4.32e4 * lambda_i + gps_time;
  t = fmod(t, 86400.0);
  if (t < 0.0) t += 86400.0;

  double F = 1.0 + 16.0 * pow(0.53 - el_sc, 3);

  double phi_m_pow = 1.0;
  double PER = 0.0;
  for (int n = 0; n < 4; n++) {
    PER += beta[n] * phi_m_pow;
    phi_m_pow *= phi_m;
  }
  if (PER < 72000.0) PER = 72000.0;

  phi_m_pow = 1.0;
  double AMP = 0.0;
  for (int n = 0; n < 4; n++) {
    AMP += alpha[n] * phi_m_pow;
    phi_m_pow *= phi_m;
  }
  if (AMP < 0.0) AMP = 0.0;

  double x = 2.0 * PI * (t - 50400.0) / PER;

  double Tiono;
  if (fabs(x) < 1.57) {
    Tiono = F * (5.0e-9 + AMP * (1.0 - x * x / 2.0 + x * x * x * x / 24.0));
  } else {
    Tiono = F * 5.0e-9;
  }

  return Tiono * 299792458.0;
}

// ============================================================
// GPU kernels
// ============================================================

__global__ void tropo_batch_kernel(const double* rx_lla, const double* sat_el,
                                    double* corrections,
                                    int n_epoch, int n_sat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (idx >= total) return;

  int epoch = idx / n_sat;
  int sat = idx % n_sat;

  double lat = rx_lla[epoch * 3 + 0];
  double alt = rx_lla[epoch * 3 + 2];
  double el = sat_el[epoch * n_sat + sat];

  corrections[epoch * n_sat + sat] = d_tropo_saastamoinen(lat, alt, el);
}

__global__ void iono_batch_kernel(const double* rx_lla,
                                   const double* sat_az, const double* sat_el,
                                   const double* alpha, const double* beta,
                                   const double* gps_times,
                                   double* corrections,
                                   int n_epoch, int n_sat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (idx >= total) return;

  int epoch = idx / n_sat;
  int sat = idx % n_sat;

  double lat = rx_lla[epoch * 3 + 0];
  double lon = rx_lla[epoch * 3 + 1];
  double az = sat_az[epoch * n_sat + sat];
  double el = sat_el[epoch * n_sat + sat];
  double gps_time = gps_times[epoch];

  corrections[epoch * n_sat + sat] = d_iono_klobuchar(alpha, beta,
                                                       lat, lon, az, el,
                                                       gps_time);
}

// ============================================================
// Host batch functions
// ============================================================

void tropo_correction_batch(const double* rx_lla, const double* sat_el,
                            double* corrections,
                            int n_epoch, int n_sat) {
  int total = n_epoch * n_sat;
  size_t sz_lla = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_el = (size_t)total * sizeof(double);

  double *d_lla, *d_el, *d_corr;
  CUDA_CHECK(cudaMalloc(&d_lla, sz_lla));
  CUDA_CHECK(cudaMalloc(&d_el, sz_el));
  CUDA_CHECK(cudaMalloc(&d_corr, sz_el));

  CUDA_CHECK(cudaMemcpy(d_lla, rx_lla, sz_lla, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_el, sat_el, sz_el, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (total + block - 1) / block;
  tropo_batch_kernel<<<grid, block>>>(d_lla, d_el, d_corr, n_epoch, n_sat);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(corrections, d_corr, sz_el, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_lla));
  CUDA_CHECK(cudaFree(d_el));
  CUDA_CHECK(cudaFree(d_corr));
}

void iono_correction_batch(const double* rx_lla,
                           const double* sat_az, const double* sat_el,
                           const double alpha[4], const double beta[4],
                           const double* gps_times,
                           double* corrections,
                           int n_epoch, int n_sat) {
  int total = n_epoch * n_sat;
  size_t sz_lla = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_sat = (size_t)total * sizeof(double);
  size_t sz_time = (size_t)n_epoch * sizeof(double);
  size_t sz_param = 4 * sizeof(double);

  double *d_lla, *d_az, *d_el, *d_alpha, *d_beta, *d_times, *d_corr;
  CUDA_CHECK(cudaMalloc(&d_lla, sz_lla));
  CUDA_CHECK(cudaMalloc(&d_az, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_el, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_alpha, sz_param));
  CUDA_CHECK(cudaMalloc(&d_beta, sz_param));
  CUDA_CHECK(cudaMalloc(&d_times, sz_time));
  CUDA_CHECK(cudaMalloc(&d_corr, sz_sat));

  CUDA_CHECK(cudaMemcpy(d_lla, rx_lla, sz_lla, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_az, sat_az, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_el, sat_el, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_alpha, alpha, sz_param, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_beta, beta, sz_param, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_times, gps_times, sz_time, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (total + block - 1) / block;
  iono_batch_kernel<<<grid, block>>>(d_lla, d_az, d_el, d_alpha, d_beta,
                                      d_times, d_corr, n_epoch, n_sat);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(corrections, d_corr, sz_sat, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_lla));
  CUDA_CHECK(cudaFree(d_az));
  CUDA_CHECK(cudaFree(d_el));
  CUDA_CHECK(cudaFree(d_alpha));
  CUDA_CHECK(cudaFree(d_beta));
  CUDA_CHECK(cudaFree(d_times));
  CUDA_CHECK(cudaFree(d_corr));
}

}  // namespace gnss_gpu
