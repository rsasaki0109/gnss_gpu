#include "gnss_gpu/diffraction.h"
#include "gnss_gpu/cuda_check.h"

#include <cuda_runtime.h>
#include <math.h>

namespace gnss_gpu {
namespace {

__device__ inline void set_invalid(
    int idx,
    int* valid,
    double* excess,
    double* amplitude,
    double* fresnel_v,
    double* atten_db,
    double* point) {
  valid[idx] = 0;
  excess[idx] = 0.0;
  amplitude[idx] = 0.0;
  fresnel_v[idx] = 0.0;
  atten_db[idx] = 0.0;

  const int p = idx * 3;
  point[p + 0] = 0.0;
  point[p + 1] = 0.0;
  point[p + 2] = 0.0;
}

__global__ void diffraction_candidates_kernel(
    const double* rx_ecef,
    const double* sat_ecef,
    const double* edge_start,
    const double* edge_end,
    const double* edge_mid,
    int n_sat,
    int n_edge,
    double max_edge_range_m,
    double max_ray_edge_distance_m,
    double max_excess_path_m,
    double wavelength_m,
    int* valid,
    double* excess,
    double* amplitude,
    double* fresnel_v,
    double* atten_db,
    double* point) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_sat * n_edge;
  if (idx >= total) {
    return;
  }

  set_invalid(idx, valid, excess, amplitude, fresnel_v, atten_db, point);

  const int s = idx / n_edge;
  const int e = idx % n_edge;

  const double rx0 = rx_ecef[0];
  const double rx1 = rx_ecef[1];
  const double rx2 = rx_ecef[2];

  const int s3 = s * 3;
  const double sat0 = sat_ecef[s3 + 0];
  const double sat1 = sat_ecef[s3 + 1];
  const double sat2 = sat_ecef[s3 + 2];

  const double direct0 = sat0 - rx0;
  const double direct1 = sat1 - rx1;
  const double direct2 = sat2 - rx2;
  const double direct_dist =
      sqrt(direct0 * direct0 + direct1 * direct1 + direct2 * direct2);

  if (direct_dist <= 1.0e-9) {
    return;
  }

  const double dir0 = direct0 / direct_dist;
  const double dir1 = direct1 / direct_dist;
  const double dir2 = direct2 / direct_dist;

  const int e3 = e * 3;

  const double mid0 = edge_mid[e3 + 0];
  const double mid1 = edge_mid[e3 + 1];
  const double mid2 = edge_mid[e3 + 2];

  const double mid_rx0 = mid0 - rx0;
  const double mid_rx1 = mid1 - rx1;
  const double mid_rx2 = mid2 - rx2;
  const double midpoint_range =
      sqrt(mid_rx0 * mid_rx0 + mid_rx1 * mid_rx1 + mid_rx2 * mid_rx2);

  if (midpoint_range > max_edge_range_m) {
    return;
  }

  const double start0 = edge_start[e3 + 0];
  const double start1 = edge_start[e3 + 1];
  const double start2 = edge_start[e3 + 2];

  const double end0 = edge_end[e3 + 0];
  const double end1 = edge_end[e3 + 1];
  const double end2 = edge_end[e3 + 2];

  const double ev0 = end0 - start0;
  const double ev1 = end1 - start1;
  const double ev2 = end2 - start2;

  const double w0 = rx0 - start0;
  const double w1 = rx1 - start1;
  const double w2 = rx2 - start2;

  const double b = ev0 * dir0 + ev1 * dir1 + ev2 * dir2;
  const double c = ev0 * ev0 + ev1 * ev1 + ev2 * ev2;
  const double d = w0 * dir0 + w1 * dir1 + w2 * dir2;
  const double ee = ev0 * w0 + ev1 * w1 + ev2 * w2;
  const double denom = c - b * b;

  double t_edge = (denom > 1.0e-9) ? (ee - b * d) / denom : 0.0;
  if (t_edge < 0.0) {
    t_edge = 0.0;
  } else if (t_edge > 1.0) {
    t_edge = 1.0;
  }

  const double p0 = start0 + t_edge * ev0;
  const double p1 = start1 + t_edge * ev1;
  const double p2 = start2 + t_edge * ev2;

  const double rel0 = p0 - rx0;
  const double rel1 = p1 - rx1;
  const double rel2 = p2 - rx2;

  const double along_ray = rel0 * dir0 + rel1 * dir1 + rel2 * dir2;

  const double closest0 = rx0 + along_ray * dir0;
  const double closest1 = rx1 + along_ray * dir1;
  const double closest2 = rx2 + along_ray * dir2;

  const double clear0 = p0 - closest0;
  const double clear1 = p1 - closest1;
  const double clear2 = p2 - closest2;
  const double h = sqrt(clear0 * clear0 + clear1 * clear1 + clear2 * clear2);

  const double range_limit =
      (max_edge_range_m < direct_dist) ? max_edge_range_m : direct_dist;

  if (!(along_ray > 0.0 && along_ray < range_limit &&
        h <= max_ray_edge_distance_m)) {
    return;
  }

  const double d1 = sqrt(rel0 * rel0 + rel1 * rel1 + rel2 * rel2);

  const double sat_p0 = sat0 - p0;
  const double sat_p1 = sat1 - p1;
  const double sat_p2 = sat2 - p2;
  const double d2 = sqrt(sat_p0 * sat_p0 + sat_p1 * sat_p1 + sat_p2 * sat_p2);

  double excess_path = d1 + d2 - direct_dist;
  if (excess_path < 0.0) {
    excess_path = 0.0;
  }

  if (excess_path > max_excess_path_m) {
    return;
  }

  const double d1g = (d1 > 1.0e-6) ? d1 : 1.0e-6;
  const double d2g = (d2 > 1.0e-6) ? d2 : 1.0e-6;

  const double v =
      h * sqrt((2.0 / wavelength_m) * (d1g + d2g) / (d1g * d2g));

  const double x = v - 0.1;
  const double j =
      (v <= -0.78) ? 0.0 : 6.9 + 20.0 * log10(sqrt(x * x + 1.0) + x);

  double amp = pow(10.0, -j / 20.0);
  if (amp < 0.0) {
    amp = 0.0;
  } else if (amp > 1.0) {
    amp = 1.0;
  }

  valid[idx] = 1;
  excess[idx] = excess_path;
  amplitude[idx] = amp;
  fresnel_v[idx] = v;
  atten_db[idx] = j;

  const int out3 = idx * 3;
  point[out3 + 0] = p0;
  point[out3 + 1] = p1;
  point[out3 + 2] = p2;
}

}  // namespace

void compute_diffraction_candidates(
    const double* rx_ecef,
    const double* sat_ecef,
    const double* edge_start,
    const double* edge_end,
    const double* edge_mid,
    int n_sat,
    int n_edge,
    double max_edge_range_m,
    double max_ray_edge_distance_m,
    double max_excess_path_m,
    double wavelength_m,
    int* valid,
    double* excess,
    double* amplitude,
    double* fresnel_v,
    double* atten_db,
    double* point) {
  if (n_sat == 0 || n_edge == 0) {
    return;
  }

  const int total = n_sat * n_edge;

  const size_t rx_bytes = 3 * sizeof(double);
  const size_t sat_bytes = static_cast<size_t>(n_sat) * 3 * sizeof(double);
  const size_t edge_bytes = static_cast<size_t>(n_edge) * 3 * sizeof(double);
  const size_t valid_bytes = static_cast<size_t>(total) * sizeof(int);
  const size_t scalar_out_bytes = static_cast<size_t>(total) * sizeof(double);
  const size_t point_bytes = static_cast<size_t>(total) * 3 * sizeof(double);

  double* d_rx = nullptr;
  double* d_sat = nullptr;
  double* d_edge_start = nullptr;
  double* d_edge_end = nullptr;
  double* d_edge_mid = nullptr;
  int* d_valid = nullptr;
  double* d_excess = nullptr;
  double* d_amplitude = nullptr;
  double* d_fresnel_v = nullptr;
  double* d_atten_db = nullptr;
  double* d_point = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rx), rx_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sat), sat_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_start), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_end), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_mid), edge_bytes));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_valid), valid_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_excess), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_amplitude), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fresnel_v), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_atten_db), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_point), point_bytes));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, rx_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sat_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_start, edge_start, edge_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_end, edge_end, edge_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_mid, edge_mid, edge_bytes, cudaMemcpyHostToDevice));

  const int block = 256;
  const int grid = (total + block - 1) / block;

  diffraction_candidates_kernel<<<grid, block>>>(
      d_rx,
      d_sat,
      d_edge_start,
      d_edge_end,
      d_edge_mid,
      n_sat,
      n_edge,
      max_edge_range_m,
      max_ray_edge_distance_m,
      max_excess_path_m,
      wavelength_m,
      d_valid,
      d_excess,
      d_amplitude,
      d_fresnel_v,
      d_atten_db,
      d_point);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(valid, d_valid, valid_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(excess, d_excess, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(amplitude, d_amplitude, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(fresnel_v, d_fresnel_v, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(atten_db, d_atten_db, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(point, d_point, point_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_edge_start));
  CUDA_CHECK(cudaFree(d_edge_end));
  CUDA_CHECK(cudaFree(d_edge_mid));
  CUDA_CHECK(cudaFree(d_valid));
  CUDA_CHECK(cudaFree(d_excess));
  CUDA_CHECK(cudaFree(d_amplitude));
  CUDA_CHECK(cudaFree(d_fresnel_v));
  CUDA_CHECK(cudaFree(d_atten_db));
  CUDA_CHECK(cudaFree(d_point));
}

}  // namespace gnss_gpu
