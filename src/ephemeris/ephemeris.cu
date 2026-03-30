#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/ephemeris.h"
#include <cmath>

namespace gnss_gpu {

// WGS84 / GPS constants (IS-GPS-200)
constexpr double GPS_MU = 3.986005e14;          // Earth gravitational parameter [m^3/s^2]
constexpr double GPS_OMEGA_E = 7.2921151467e-5; // Earth rotation rate [rad/s]
constexpr double GPS_F = -4.442807633e-10;       // relativistic correction constant [s/m^0.5]
constexpr double GPS_WEEK_SEC = 604800.0;        // seconds per GPS week

// Solve Kepler's equation: M = E - e*sin(E) via Newton-Raphson
__device__ double kepler_equation(double M, double e) {
    double E = M;  // initial guess
    for (int i = 0; i < 10; i++) {
        double dE = (M - E + e * sin(E)) / (1.0 - e * cos(E));
        E += dE;
        if (fabs(dE) < 1e-15) break;
    }
    return E;
}

__global__ void sat_pos_kernel(const EphemerisParams* params, double gps_time,
                                double* sat_pos, double* sat_clk, int n_sat) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= n_sat) return;

    const EphemerisParams& eph = params[sid];

    // Semi-major axis
    double a = eph.sqrt_a * eph.sqrt_a;

    // Computed mean motion [rad/s]
    double n0 = sqrt(GPS_MU / (a * a * a));
    double n = n0 + eph.delta_n;

    // Time from ephemeris reference epoch
    double tk = gps_time - eph.toe;
    // Account for GPS week crossover
    if (tk > GPS_WEEK_SEC / 2.0) tk -= GPS_WEEK_SEC;
    if (tk < -GPS_WEEK_SEC / 2.0) tk += GPS_WEEK_SEC;

    // Mean anomaly
    double M = eph.M0 + n * tk;

    // Solve Kepler's equation for eccentric anomaly
    double E = kepler_equation(M, eph.e);

    // True anomaly
    double sinE = sin(E);
    double cosE = cos(E);
    double denom = 1.0 - eph.e * cosE;
    double sinv = sqrt(1.0 - eph.e * eph.e) * sinE / denom;
    double cosv = (cosE - eph.e) / denom;
    double v = atan2(sinv, cosv);

    // Argument of latitude (uncorrected)
    double phi = v + eph.omega;

    // Second harmonic corrections
    double sin2phi = sin(2.0 * phi);
    double cos2phi = cos(2.0 * phi);

    double du = eph.cuc * cos2phi + eph.cus * sin2phi;  // latitude correction
    double dr = eph.crc * cos2phi + eph.crs * sin2phi;  // radius correction
    double di = eph.cic * cos2phi + eph.cis * sin2phi;  // inclination correction

    // Corrected argument of latitude
    double u = phi + du;

    // Corrected radius
    double r = a * (1.0 - eph.e * cosE) + dr;

    // Corrected inclination
    double i = eph.i0 + eph.idot * tk + di;

    // Positions in orbital plane
    double xp = r * cos(u);
    double yp = r * sin(u);

    // Corrected longitude of ascending node
    double Omega = eph.omega0 + (eph.omega_dot - GPS_OMEGA_E) * tk - GPS_OMEGA_E * eph.toe;

    // ECEF coordinates
    double cosO = cos(Omega);
    double sinO = sin(Omega);
    double cosi = cos(i);
    double sini = sin(i);

    sat_pos[sid * 3 + 0] = xp * cosO - yp * cosi * sinO;
    sat_pos[sid * 3 + 1] = xp * sinO + yp * cosi * cosO;
    sat_pos[sid * 3 + 2] = yp * sini;

    // Clock correction
    double dt = gps_time - eph.toc;
    if (dt > GPS_WEEK_SEC / 2.0) dt -= GPS_WEEK_SEC;
    if (dt < -GPS_WEEK_SEC / 2.0) dt += GPS_WEEK_SEC;

    // Relativistic correction
    double dtr = GPS_F * eph.e * eph.sqrt_a * sinE;

    sat_clk[sid] = eph.af0 + eph.af1 * dt + eph.af2 * dt * dt + dtr - eph.tgd;
}

__global__ void sat_pos_batch_kernel(const EphemerisParams* params, const double* gps_times,
                                      double* sat_pos, double* sat_clk,
                                      int n_epoch, int n_sat) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_epoch * n_sat;
    if (tid >= total) return;

    int epoch = tid / n_sat;
    int sid = tid % n_sat;

    const EphemerisParams& eph = params[sid];
    double gps_time = gps_times[epoch];

    // Semi-major axis
    double a = eph.sqrt_a * eph.sqrt_a;

    // Computed mean motion
    double n0 = sqrt(GPS_MU / (a * a * a));
    double n = n0 + eph.delta_n;

    // Time from ephemeris reference epoch
    double tk = gps_time - eph.toe;
    if (tk > GPS_WEEK_SEC / 2.0) tk -= GPS_WEEK_SEC;
    if (tk < -GPS_WEEK_SEC / 2.0) tk += GPS_WEEK_SEC;

    // Mean anomaly
    double M = eph.M0 + n * tk;

    // Solve Kepler's equation
    double E = kepler_equation(M, eph.e);

    // True anomaly
    double sinE = sin(E);
    double cosE = cos(E);
    double denom = 1.0 - eph.e * cosE;
    double sinv = sqrt(1.0 - eph.e * eph.e) * sinE / denom;
    double cosv = (cosE - eph.e) / denom;
    double v = atan2(sinv, cosv);

    // Argument of latitude
    double phi = v + eph.omega;
    double sin2phi = sin(2.0 * phi);
    double cos2phi = cos(2.0 * phi);

    double du = eph.cuc * cos2phi + eph.cus * sin2phi;
    double dr = eph.crc * cos2phi + eph.crs * sin2phi;
    double di = eph.cic * cos2phi + eph.cis * sin2phi;

    double u = phi + du;
    double r = a * (1.0 - eph.e * cosE) + dr;
    double i = eph.i0 + eph.idot * tk + di;

    double xp = r * cos(u);
    double yp = r * sin(u);

    double Omega = eph.omega0 + (eph.omega_dot - GPS_OMEGA_E) * tk - GPS_OMEGA_E * eph.toe;

    double cosO = cos(Omega);
    double sinO = sin(Omega);
    double cosi = cos(i);
    double sini = sin(i);

    int out_idx = epoch * n_sat + sid;
    sat_pos[out_idx * 3 + 0] = xp * cosO - yp * cosi * sinO;
    sat_pos[out_idx * 3 + 1] = xp * sinO + yp * cosi * cosO;
    sat_pos[out_idx * 3 + 2] = yp * sini;

    // Clock correction
    double dt = gps_time - eph.toc;
    if (dt > GPS_WEEK_SEC / 2.0) dt -= GPS_WEEK_SEC;
    if (dt < -GPS_WEEK_SEC / 2.0) dt += GPS_WEEK_SEC;

    double dtr = GPS_F * eph.e * eph.sqrt_a * sinE;
    sat_clk[out_idx] = eph.af0 + eph.af1 * dt + eph.af2 * dt * dt + dtr - eph.tgd;
}

void compute_satellite_position(
    const EphemerisParams* params, double gps_time,
    double* sat_pos, double* sat_clk,
    int n_sat) {

    EphemerisParams* d_params;
    double *d_pos, *d_clk;

    size_t sz_params = (size_t)n_sat * sizeof(EphemerisParams);
    size_t sz_pos = (size_t)n_sat * 3 * sizeof(double);
    size_t sz_clk = (size_t)n_sat * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_params, sz_params));
    CUDA_CHECK(cudaMalloc(&d_pos, sz_pos));
    CUDA_CHECK(cudaMalloc(&d_clk, sz_clk));

    CUDA_CHECK(cudaMemcpy(d_params, params, sz_params, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n_sat + block - 1) / block;
    sat_pos_kernel<<<grid, block>>>(d_params, gps_time, d_pos, d_clk, n_sat);

    CUDA_CHECK(cudaMemcpy(sat_pos, d_pos, sz_pos, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sat_clk, d_clk, sz_clk, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_clk));
}

void compute_satellite_position_batch(
    const EphemerisParams* params, const double* gps_times,
    double* sat_pos, double* sat_clk,
    int n_epoch, int n_sat) {

    EphemerisParams* d_params;
    double *d_times, *d_pos, *d_clk;

    size_t sz_params = (size_t)n_sat * sizeof(EphemerisParams);
    size_t sz_times = (size_t)n_epoch * sizeof(double);
    size_t sz_pos = (size_t)n_epoch * n_sat * 3 * sizeof(double);
    size_t sz_clk = (size_t)n_epoch * n_sat * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_params, sz_params));
    CUDA_CHECK(cudaMalloc(&d_times, sz_times));
    CUDA_CHECK(cudaMalloc(&d_pos, sz_pos));
    CUDA_CHECK(cudaMalloc(&d_clk, sz_clk));

    CUDA_CHECK(cudaMemcpy(d_params, params, sz_params, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_times, gps_times, sz_times, cudaMemcpyHostToDevice));

    int total = n_epoch * n_sat;
    int block = 256;
    int grid = (total + block - 1) / block;
    sat_pos_batch_kernel<<<grid, block>>>(d_params, d_times, d_pos, d_clk, n_epoch, n_sat);

    CUDA_CHECK(cudaMemcpy(sat_pos, d_pos, sz_pos, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sat_clk, d_clk, sz_clk, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_times));
    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_clk));
}

}  // namespace gnss_gpu
