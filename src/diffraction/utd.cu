// UTD (Kouyoumjian-Pathak) wedge diffraction candidate search.
//
// This mirrors python/gnss_gpu/utd_diffraction.py::compute_utd_diffraction_paths
// term for term (edge/ray geometry, wedge basis, phi/phi', the UTD
// coefficient D, and the resulting amplitude/attenuation). Numerical parity
// with that Python model is the acceptance bar for this file -- see
// tests/test_utd_gpu.py.
//
// The one deliberate deviation from the Python reference is the Fresnel
// C(x)/S(x) evaluation for the "numeric" branch (argument <= 8.0): the CPU
// implementation integrates a batched trapezoidal rule via numpy, which is
// only well defined per-call for the *scalar* invocation pattern used by
// fresnel_transition(). This file reproduces that exact scalar trapezoidal
// rule (same step count n = max(16, ceil(v/1e-3)), same summation order) so
// device results agree with the host to near machine precision -- it is not
// an "improved" approximation, just the same algorithm evaluated per thread
// instead of per batched numpy call. The large-argument branch (> 8.0) uses
// the same leading-order asymptotic closed form the CPU uses.
#include "gnss_gpu/diffraction.h"
#include "gnss_gpu/cuda_check.h"

#include <cuda_runtime.h>
#include <math.h>

namespace gnss_gpu {
namespace {

static constexpr double PI = 3.14159265358979323846;
static constexpr double FRESNEL_STEP = 1.0e-3;
static constexpr double FRESNEL_ASYMPTOTIC_START = 8.0;

struct Vec3 {
  double x, y, z;

  __device__ Vec3 operator-(const Vec3& o) const {
    return Vec3{x - o.x, y - o.y, z - o.z};
  }
  __device__ Vec3 operator+(const Vec3& o) const {
    return Vec3{x + o.x, y + o.y, z + o.z};
  }
  __device__ Vec3 operator*(double s) const {
    return Vec3{x * s, y * s, z * s};
  }
};

__device__ inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
}

__device__ inline double vnorm(const Vec3& a) { return sqrt(dot(a, a)); }

__device__ inline Vec3 load3(const double* p) {
  return Vec3{p[0], p[1], p[2]};
}

// Fresnel C(v), S(v) for v >= 0, matching gnss_gpu.diffraction.fresnel_integral
// evaluated on a scalar input (the only calling convention used by
// utd_diffraction.fresnel_transition).
__device__ inline void fresnel_cs(double v, double* c_out, double* s_out) {
  if (!(v > 0.0)) {
    *c_out = 0.0;
    *s_out = 0.0;
    return;
  }

  if (v > FRESNEL_ASYMPTOTIC_START) {
    const double phase = 0.5 * PI * v * v;
    *c_out = 0.5 + sin(phase) / (PI * v);
    *s_out = 0.5 - cos(phase) / (PI * v);
    return;
  }

  int n = static_cast<int>(ceil(v / FRESNEL_STEP));
  if (n < 16) {
    n = 16;
  }
  const double dx = v / static_cast<double>(n);

  double c = 0.0;
  double s = 0.0;
  double prev_cos = 1.0;  // cos(phase(0)) == 1
  double prev_sin = 0.0;  // sin(phase(0)) == 0
  for (int i = 1; i <= n; ++i) {
    const double xi = static_cast<double>(i) * dx;
    const double phase = 0.5 * PI * xi * xi;
    const double cy = cos(phase);
    const double sy = sin(phase);
    c += 0.5 * dx * (prev_cos + cy);
    s += 0.5 * dx * (prev_sin + sy);
    prev_cos = cy;
    prev_sin = sy;
  }
  *c_out = c;
  *s_out = s;
}

// UTD transition function F(x), x >= 0 (utd_diffraction.fresnel_transition).
__device__ inline void fresnel_transition(double x, double* re, double* im) {
  if (!(x > 0.0)) {
    *re = 0.0;
    *im = 0.0;
    return;
  }

  const double u0 = sqrt(2.0 * x / PI);
  double c, s;
  fresnel_cs(u0, &c, &s);

  const double root_half_pi = sqrt(PI / 2.0);
  const double tail_re = root_half_pi * (0.5 - c);
  const double tail_im = -root_half_pi * (0.5 - s);

  const double sq = 2.0 * sqrt(x);
  const double ex_re = cos(x);
  const double ex_im = sin(x);
  // (2j * sqrt(x)) * exp(i x) == (0, sq) * (ex_re, ex_im)
  const double a_re = -sq * ex_im;
  const double a_im = sq * ex_re;

  *re = a_re * tail_re - a_im * tail_im;
  *im = a_re * tail_im + a_im * tail_re;
}

// cot(arg) * F(klA), with the small-sin(arg) analytic limit
// (utd_diffraction._cot_times_F).
__device__ inline void cot_times_f(
    double arg, double klA, double k, double L, double n,
    double* re, double* im) {
  const double sin_arg = sin(arg);
  if (fabs(sin_arg) < 1.0e-7) {
    const double m = round(arg / PI);
    const double eps = (arg - m * PI) * n;
    const double sgn = (eps >= 0.0) ? 1.0 : -1.0;

    const double phase_re = cos(PI / 4.0);
    const double phase_im = sin(PI / 4.0);
    const double root = sqrt(fmax(0.0, 2.0 * PI * k * L));

    const double t1_re = phase_re * root * sgn;
    const double t1_im = phase_im * root * sgn;

    // phase * phase == exp(i*pi/2)
    const double phase2_re = phase_re * phase_re - phase_im * phase_im;
    const double phase2_im = 2.0 * phase_re * phase_im;
    const double t2_re = 2.0 * k * L * eps * phase2_re;
    const double t2_im = 2.0 * k * L * eps * phase2_im;

    *re = n * (t1_re - t2_re);
    *im = n * (t1_im - t2_im);
    return;
  }

  const double klA_clamped = fmax(0.0, klA);
  double f_re, f_im;
  fresnel_transition(klA_clamped, &f_re, &f_im);
  const double cot = cos(arg) / sin_arg;
  *re = cot * f_re;
  *im = cot * f_im;
}

// term_plus(beta) + term_minus(beta) (utd_diffraction.utd_coefficient).
__device__ inline void utd_d_term(
    double beta, double k, double L, double n, double* re, double* im) {
  const double kL = k * L;

  const double n_plus = round((PI + beta) / (2.0 * PI * n));
  const double aplus_cos = cos((2.0 * PI * n * n_plus - beta) / 2.0);
  const double aplus = 2.0 * aplus_cos * aplus_cos;
  const double argp = (PI + beta) / (2.0 * n);
  double tp_re, tp_im;
  cot_times_f(argp, kL * aplus, k, L, n, &tp_re, &tp_im);

  const double n_minus = round((-PI + beta) / (2.0 * PI * n));
  const double aminus_cos = cos((2.0 * PI * n * n_minus - beta) / 2.0);
  const double aminus = 2.0 * aminus_cos * aminus_cos;
  const double argm = (PI - beta) / (2.0 * n);
  double tm_re, tm_im;
  cot_times_f(argm, kL * aminus, k, L, n, &tm_re, &tm_im);

  *re = tp_re + tm_re;
  *im = tp_im + tm_im;
}

// mode: 0 = absorbing, 1 = soft, 2 = hard.
__device__ inline void utd_coefficient(
    double phi, double phi_p, double beta0, double n, double k, double L,
    int mode, double* out_re, double* out_im) {
  const double sin_beta0 = sin(beta0);
  if (fabs(sin_beta0) < 1.0e-15) {
    *out_re = 0.0;
    *out_im = 0.0;
    return;
  }

  const double denom = 2.0 * n * sqrt(2.0 * PI * k) * sin_beta0;
  const double exp_re = cos(-PI / 4.0);
  const double exp_im = sin(-PI / 4.0);
  const double prefac_re = -exp_re / denom;
  const double prefac_im = -exp_im / denom;

  const double beta_minus = phi - phi_p;
  const double beta_plus = phi + phi_p;

  double d1_re, d1_im, d2_re, d2_im;
  utd_d_term(beta_minus, k, L, n, &d1_re, &d1_im);
  utd_d_term(beta_plus, k, L, n, &d2_re, &d2_im);

  double sum_re, sum_im;
  if (mode == 1) {          // soft: D1 - D2
    sum_re = d1_re - d2_re;
    sum_im = d1_im - d2_im;
  } else if (mode == 2) {   // hard: D1 + D2
    sum_re = d1_re + d2_re;
    sum_im = d1_im + d2_im;
  } else {                  // absorbing: D1
    sum_re = d1_re;
    sum_im = d1_im;
  }

  *out_re = prefac_re * sum_re - prefac_im * sum_im;
  *out_im = prefac_re * sum_im + prefac_im * sum_re;
}

// utd_diffraction._exterior_angle. Returns false when the projection is
// degenerate (mirrors the Python function returning None).
__device__ inline bool exterior_angle(
    const Vec3& vec, const Vec3& ehat, const Vec3& uhat, const Vec3& vhat,
    double* out_angle) {
  const double along_e = dot(vec, ehat);
  Vec3 p = vec - ehat * along_e;
  const double plen = vnorm(p);
  if (!(plen >= 1.0e-12)) {
    return false;
  }
  p = p * (1.0 / plen);
  double ang = atan2(dot(p, vhat), dot(p, uhat));
  if (!isfinite(ang)) {
    return false;
  }
  if (ang < 0.0) {
    ang += 2.0 * PI;
  }
  *out_angle = ang;
  return true;
}

// utd_diffraction._wedge_n_at.
__device__ inline double wedge_n_at(const double* wedge_n, int n_wedge_n, int e) {
  if (n_wedge_n == 0) {
    return 2.0;
  }
  double val;
  if (n_wedge_n == 1) {
    val = wedge_n[0];
  } else if (e < n_wedge_n) {
    val = wedge_n[e];
  } else {
    return 2.0;
  }
  if (!isfinite(val) || !(val > 0.0)) {
    return 2.0;
  }
  return val;
}

__device__ inline void set_invalid(
    int idx, int* valid, double* excess, double* amplitude, double* beta0_out,
    double* phi_out, double* phi_p_out, double* wedge_n_out, double* atten_db,
    double* point) {
  valid[idx] = 0;
  excess[idx] = 0.0;
  amplitude[idx] = 0.0;
  beta0_out[idx] = 0.0;
  phi_out[idx] = 0.0;
  phi_p_out[idx] = 0.0;
  wedge_n_out[idx] = 0.0;
  atten_db[idx] = 0.0;

  const int p = idx * 3;
  point[p + 0] = 0.0;
  point[p + 1] = 0.0;
  point[p + 2] = 0.0;
}

__global__ void utd_diffraction_kernel(
    const double* rx_ecef,
    const double* sat_ecef,
    const double* edge_start,
    const double* edge_end,
    const double* edge_mid,
    const double* face_dir_a,
    const double* face_dir_b,
    const double* wedge_n,
    int n_wedge_n,
    int n_sat,
    int n_edge,
    double max_edge_range_m,
    double max_ray_edge_distance_m,
    double max_excess_path_m,
    double wavelength_m,
    int mode,
    int* valid,
    double* excess,
    double* amplitude,
    double* beta0_out,
    double* phi_out,
    double* phi_p_out,
    double* wedge_n_out,
    double* atten_db,
    double* point) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_sat * n_edge;
  if (idx >= total) {
    return;
  }

  set_invalid(idx, valid, excess, amplitude, beta0_out, phi_out, phi_p_out,
              wedge_n_out, atten_db, point);

  const int s = idx / n_edge;
  const int e = idx % n_edge;

  const Vec3 rx = load3(rx_ecef);
  const Vec3 sat = load3(sat_ecef + s * 3);

  const Vec3 direct = sat - rx;
  const double direct_dist = vnorm(direct);
  if (!(direct_dist > 0.0)) {
    return;
  }
  const Vec3 direction = direct * (1.0 / direct_dist);

  const Vec3 mid = load3(edge_mid + e * 3);
  const double midpoint_range = vnorm(mid - rx);
  if (!(midpoint_range <= max_edge_range_m)) {
    return;
  }

  const Vec3 p0 = load3(edge_start + e * 3);
  const Vec3 p1 = load3(edge_end + e * 3);
  const Vec3 edge_vec = p1 - p0;
  const double c = dot(edge_vec, edge_vec);
  if (!(c > 1.0e-18)) {
    return;
  }

  const Vec3 w = rx - p0;
  const double b = dot(edge_vec, direction);
  const double d = dot(w, direction);
  const double ee = dot(edge_vec, w);
  const double denom = c - b * b;
  if (!(denom > 1.0e-9)) {
    return;
  }

  double t = (ee - b * d) / denom;
  t = fmin(fmax(t, 0.0), 1.0);
  const Vec3 Q = p0 + edge_vec * t;

  const Vec3 rel = Q - rx;
  const double along_ray = dot(rel, direction);
  const double along_limit = fmin(max_edge_range_m, direct_dist);
  if (!(along_ray > 0.0 && along_ray < along_limit)) {
    return;
  }

  const Vec3 closest = rx + direction * along_ray;
  const Vec3 clear = Q - closest;
  const double h = vnorm(clear);
  if (!(h <= max_ray_edge_distance_m)) {
    return;
  }

  const double d1 = vnorm(rel);
  const Vec3 sat_q = sat - Q;
  const double d2 = vnorm(sat_q);
  if (!(d1 > 0.0 && d2 > 0.0)) {
    return;
  }

  const double excess_path = d1 + d2 - direct_dist;
  if (!(excess_path > 1.0e-6 && excess_path <= max_excess_path_m)) {
    return;
  }

  const double edge_len = sqrt(c);
  const Vec3 ehat = edge_vec * (1.0 / edge_len);
  double edge_dot = fabs(dot(direction, ehat));
  if (!isfinite(edge_dot)) {
    return;
  }
  edge_dot = fmin(fmax(edge_dot, 0.0), 1.0);
  const double beta0 = acos(edge_dot);
  const double sin_beta0 = sin(beta0);
  if (!(sin_beta0 >= 1.0e-6)) {
    return;
  }

  const Vec3 fa = load3(face_dir_a + e * 3);
  const Vec3 fb = load3(face_dir_b + e * 3);

  Vec3 uhat = fa - ehat * dot(fa, ehat);
  const double ulen = vnorm(uhat);
  if (!(ulen >= 1.0e-12)) {
    return;
  }
  uhat = uhat * (1.0 / ulen);

  Vec3 v0 = cross(ehat, uhat);
  const double v0_len = vnorm(v0);
  if (!(v0_len >= 1.0e-12)) {
    return;
  }
  v0 = v0 * (1.0 / v0_len);

  Vec3 fb_proj = fb - ehat * dot(fb, ehat);
  const double fb_len = vnorm(fb_proj);
  double s_sign = 1.0;
  if (fb_len > 1.0e-9) {
    fb_proj = fb_proj * (1.0 / fb_len);
    const double sign_value = dot(fb_proj, v0);
    s_sign = (sign_value >= 0.0) ? 1.0 : -1.0;
  }
  const Vec3 vhat = v0 * (-s_sign);

  double phi_p_raw, phi_raw;
  if (!exterior_angle(sat_q, ehat, uhat, vhat, &phi_p_raw)) {
    return;
  }
  const Vec3 rx_q = rx - Q;
  if (!exterior_angle(rx_q, ehat, uhat, vhat, &phi_raw)) {
    return;
  }

  const double nn = wedge_n_at(wedge_n, n_wedge_n, e);
  const double upper = nn * PI - 1.0e-6;
  if (!(upper > 1.0e-6)) {
    return;
  }

  const double phi_p = fmin(fmax(phi_p_raw, 1.0e-6), upper);
  const double phi = fmin(fmax(phi_raw, 1.0e-6), upper);

  const double L = (d1 * d2 * sin_beta0 * sin_beta0) / (d1 + d2);
  const double k = 2.0 * PI / wavelength_m;

  double D_re, D_im;
  utd_coefficient(phi, phi_p, beta0, nn, k, L, mode, &D_re, &D_im);
  const double D_abs = sqrt(D_re * D_re + D_im * D_im);
  const double raw_amplitude = D_abs * sqrt(1.0 / d1 + 1.0 / d2);
  if (!isfinite(raw_amplitude)) {
    return;
  }

  const double amp = fmin(fmax(raw_amplitude, 0.0), 1.0);
  const double amp_floor = fmax(amp, 1.0e-12);
  const double atten = fmax(0.0, -20.0 * log10(amp_floor));

  valid[idx] = 1;
  excess[idx] = excess_path;
  amplitude[idx] = amp;
  beta0_out[idx] = beta0;
  phi_out[idx] = phi;
  phi_p_out[idx] = phi_p;
  wedge_n_out[idx] = nn;
  atten_db[idx] = atten;

  const int out3 = idx * 3;
  point[out3 + 0] = Q.x;
  point[out3 + 1] = Q.y;
  point[out3 + 2] = Q.z;
}

}  // namespace

void compute_utd_diffraction_candidates(
    const double* rx_ecef,
    const double* sat_ecef,
    const double* edge_start,
    const double* edge_end,
    const double* edge_mid,
    const double* face_dir_a,
    const double* face_dir_b,
    const double* wedge_n,
    int n_wedge_n,
    int n_sat,
    int n_edge,
    double max_edge_range_m,
    double max_ray_edge_distance_m,
    double max_excess_path_m,
    double wavelength_m,
    int mode,
    int* valid,
    double* excess,
    double* amplitude,
    double* beta0_out,
    double* phi_out,
    double* phi_p_out,
    double* wedge_n_out,
    double* atten_db,
    double* point) {
  if (n_sat == 0 || n_edge == 0) {
    return;
  }

  const int total = n_sat * n_edge;

  const size_t rx_bytes = 3 * sizeof(double);
  const size_t sat_bytes = static_cast<size_t>(n_sat) * 3 * sizeof(double);
  const size_t edge_bytes = static_cast<size_t>(n_edge) * 3 * sizeof(double);
  const size_t wedge_n_bytes =
      static_cast<size_t>(n_wedge_n > 0 ? n_wedge_n : 1) * sizeof(double);
  const size_t valid_bytes = static_cast<size_t>(total) * sizeof(int);
  const size_t scalar_out_bytes = static_cast<size_t>(total) * sizeof(double);
  const size_t point_bytes = static_cast<size_t>(total) * 3 * sizeof(double);

  double* d_rx = nullptr;
  double* d_sat = nullptr;
  double* d_edge_start = nullptr;
  double* d_edge_end = nullptr;
  double* d_edge_mid = nullptr;
  double* d_face_dir_a = nullptr;
  double* d_face_dir_b = nullptr;
  double* d_wedge_n = nullptr;
  int* d_valid = nullptr;
  double* d_excess = nullptr;
  double* d_amplitude = nullptr;
  double* d_beta0 = nullptr;
  double* d_phi = nullptr;
  double* d_phi_p = nullptr;
  double* d_wedge_n_out = nullptr;
  double* d_atten_db = nullptr;
  double* d_point = nullptr;

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rx), rx_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sat), sat_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_start), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_end), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_mid), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_dir_a), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_face_dir_b), edge_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_wedge_n), wedge_n_bytes));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_valid), valid_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_excess), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_amplitude), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_beta0), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_phi), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_phi_p), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_wedge_n_out), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_atten_db), scalar_out_bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_point), point_bytes));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, rx_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sat_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_start, edge_start, edge_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_end, edge_end, edge_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_edge_mid, edge_mid, edge_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_face_dir_a, face_dir_a, edge_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_face_dir_b, face_dir_b, edge_bytes, cudaMemcpyHostToDevice));
  if (n_wedge_n > 0) {
    CUDA_CHECK(cudaMemcpy(d_wedge_n, wedge_n, wedge_n_bytes, cudaMemcpyHostToDevice));
  }

  const int block = 256;
  const int grid = (total + block - 1) / block;

  utd_diffraction_kernel<<<grid, block>>>(
      d_rx,
      d_sat,
      d_edge_start,
      d_edge_end,
      d_edge_mid,
      d_face_dir_a,
      d_face_dir_b,
      d_wedge_n,
      n_wedge_n,
      n_sat,
      n_edge,
      max_edge_range_m,
      max_ray_edge_distance_m,
      max_excess_path_m,
      wavelength_m,
      mode,
      d_valid,
      d_excess,
      d_amplitude,
      d_beta0,
      d_phi,
      d_phi_p,
      d_wedge_n_out,
      d_atten_db,
      d_point);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(valid, d_valid, valid_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(excess, d_excess, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(amplitude, d_amplitude, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(beta0_out, d_beta0, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(phi_out, d_phi, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(phi_p_out, d_phi_p, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(wedge_n_out, d_wedge_n_out, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(atten_db, d_atten_db, scalar_out_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(point, d_point, point_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_edge_start));
  CUDA_CHECK(cudaFree(d_edge_end));
  CUDA_CHECK(cudaFree(d_edge_mid));
  CUDA_CHECK(cudaFree(d_face_dir_a));
  CUDA_CHECK(cudaFree(d_face_dir_b));
  CUDA_CHECK(cudaFree(d_wedge_n));
  CUDA_CHECK(cudaFree(d_valid));
  CUDA_CHECK(cudaFree(d_excess));
  CUDA_CHECK(cudaFree(d_amplitude));
  CUDA_CHECK(cudaFree(d_beta0));
  CUDA_CHECK(cudaFree(d_phi));
  CUDA_CHECK(cudaFree(d_phi_p));
  CUDA_CHECK(cudaFree(d_wedge_n_out));
  CUDA_CHECK(cudaFree(d_atten_db));
  CUDA_CHECK(cudaFree(d_point));
}

}  // namespace gnss_gpu
