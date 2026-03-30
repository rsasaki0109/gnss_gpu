#include "gnss_gpu/coordinates.h"
#include <cmath>

namespace gnss_gpu {

__global__ void ecef_to_lla_kernel(const double* ex, const double* ey, const double* ez,
                                   double* lat, double* lon, double* alt, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double x = ex[i], y = ey[i], z = ez[i];
  double p = sqrt(x * x + y * y);
  double theta = atan2(z * WGS84_A, p * WGS84_B);
  double st = sin(theta), ct = cos(theta);

  double lat_rad = atan2(z + WGS84_E2 / (1.0 - WGS84_E2) * WGS84_B * st * st * st,
                         p - WGS84_E2 * WGS84_A * ct * ct * ct);
  double lon_rad = atan2(y, x);
  double sin_lat = sin(lat_rad);
  double N_val = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat);
  double alt_val = p / cos(lat_rad) - N_val;

  lat[i] = lat_rad;
  lon[i] = lon_rad;
  alt[i] = alt_val;
}

void ecef_to_lla(const double* ecef_x, const double* ecef_y, const double* ecef_z,
                 double* lat, double* lon, double* alt, int n) {
  double *d_ex, *d_ey, *d_ez, *d_lat, *d_lon, *d_alt;
  size_t sz = n * sizeof(double);

  cudaMalloc(&d_ex, sz); cudaMalloc(&d_ey, sz); cudaMalloc(&d_ez, sz);
  cudaMalloc(&d_lat, sz); cudaMalloc(&d_lon, sz); cudaMalloc(&d_alt, sz);

  cudaMemcpy(d_ex, ecef_x, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ey, ecef_y, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ez, ecef_z, sz, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (n + block - 1) / block;
  ecef_to_lla_kernel<<<grid, block>>>(d_ex, d_ey, d_ez, d_lat, d_lon, d_alt, n);

  cudaMemcpy(lat, d_lat, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(lon, d_lon, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(alt, d_alt, sz, cudaMemcpyDeviceToHost);

  cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_ez);
  cudaFree(d_lat); cudaFree(d_lon); cudaFree(d_alt);
}

__global__ void lla_to_ecef_kernel(const double* lat, const double* lon, const double* alt,
                                   double* ex, double* ey, double* ez, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double la = lat[i], lo = lon[i], h = alt[i];
  double sin_la = sin(la), cos_la = cos(la);
  double sin_lo = sin(lo), cos_lo = cos(lo);
  double N_val = WGS84_A / sqrt(1.0 - WGS84_E2 * sin_la * sin_la);

  ex[i] = (N_val + h) * cos_la * cos_lo;
  ey[i] = (N_val + h) * cos_la * sin_lo;
  ez[i] = (N_val * (1.0 - WGS84_E2) + h) * sin_la;
}

void lla_to_ecef(const double* lat, const double* lon, const double* alt,
                 double* ecef_x, double* ecef_y, double* ecef_z, int n) {
  double *d_lat, *d_lon, *d_alt, *d_ex, *d_ey, *d_ez;
  size_t sz = n * sizeof(double);

  cudaMalloc(&d_lat, sz); cudaMalloc(&d_lon, sz); cudaMalloc(&d_alt, sz);
  cudaMalloc(&d_ex, sz); cudaMalloc(&d_ey, sz); cudaMalloc(&d_ez, sz);

  cudaMemcpy(d_lat, lat, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_lon, lon, sz, cudaMemcpyHostToDevice);
  cudaMemcpy(d_alt, alt, sz, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (n + block - 1) / block;
  lla_to_ecef_kernel<<<grid, block>>>(d_lat, d_lon, d_alt, d_ex, d_ey, d_ez, n);

  cudaMemcpy(ecef_x, d_ex, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(ecef_y, d_ey, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(ecef_z, d_ez, sz, cudaMemcpyDeviceToHost);

  cudaFree(d_lat); cudaFree(d_lon); cudaFree(d_alt);
  cudaFree(d_ex); cudaFree(d_ey); cudaFree(d_ez);
}

__global__ void satellite_azel_kernel(double rx, double ry, double rz,
                                      double sin_lat, double cos_lat,
                                      double sin_lon, double cos_lon,
                                      const double* sat_ecef,
                                      double* az, double* el, int n_sat) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_sat) return;

  double dx = sat_ecef[i * 3 + 0] - rx;
  double dy = sat_ecef[i * 3 + 1] - ry;
  double dz = sat_ecef[i * 3 + 2] - rz;

  // ENU rotation
  double e = -sin_lon * dx + cos_lon * dy;
  double n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
  double u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;

  double hz = sqrt(e * e + n * n);
  el[i] = atan2(u, hz);
  az[i] = atan2(e, n);
}

void satellite_azel(double rx, double ry, double rz,
                    const double* sat_ecef, double* az, double* el, int n_sat) {
  // Compute receiver LLA for ENU rotation
  double p = sqrt(rx * rx + ry * ry);
  double theta = atan2(rz * WGS84_A, p * WGS84_B);
  double st = sin(theta), ct = cos(theta);
  double lat_r = atan2(rz + WGS84_E2 / (1.0 - WGS84_E2) * WGS84_B * st * st * st,
                       p - WGS84_E2 * WGS84_A * ct * ct * ct);
  double lon_r = atan2(ry, rx);

  double sin_lat = sin(lat_r), cos_lat = cos(lat_r);
  double sin_lon = sin(lon_r), cos_lon = cos(lon_r);

  double *d_sat, *d_az, *d_el;
  size_t sz3 = n_sat * 3 * sizeof(double);
  size_t sz = n_sat * sizeof(double);

  cudaMalloc(&d_sat, sz3); cudaMalloc(&d_az, sz); cudaMalloc(&d_el, sz);
  cudaMemcpy(d_sat, sat_ecef, sz3, cudaMemcpyHostToDevice);

  int block = 256;
  int grid = (n_sat + block - 1) / block;
  satellite_azel_kernel<<<grid, block>>>(rx, ry, rz, sin_lat, cos_lat, sin_lon, cos_lon,
                                         d_sat, d_az, d_el, n_sat);

  cudaMemcpy(az, d_az, sz, cudaMemcpyDeviceToHost);
  cudaMemcpy(el, d_el, sz, cudaMemcpyDeviceToHost);

  cudaFree(d_sat); cudaFree(d_az); cudaFree(d_el);
}

}  // namespace gnss_gpu
