#include "gnss_gpu/skyplot.h"
#include "gnss_gpu/coordinates.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>

namespace gnss_gpu {

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Device-accessible WGS84 constants (constexpr namespace-scope doubles
// may not be accessible in __device__ code on all CUDA toolkits)
// ---------------------------------------------------------------------------

#define D_WGS84_A  6378137.0
#define D_WGS84_B  6356752.314245179
#define D_WGS84_E2 6.6943799901413165e-3

// ---------------------------------------------------------------------------
// Inline device helpers (avoid cross-module linking)
// ---------------------------------------------------------------------------

__device__ void ecef_to_lla_inline(double x, double y, double z,
                                   double& lat, double& lon, double& alt) {
    const double a = 6378137.0;
    const double b = 6356752.314245179;
    const double e2 = 6.6943799901413165e-3;

    double p = sqrt(x * x + y * y);
    double theta = atan2(z * a, p * b);
    double st = sin(theta), ct = cos(theta);

    lat = atan2(z + e2 / (1.0 - e2) * b * st * st * st,
                p - e2 * a * ct * ct * ct);
    lon = atan2(y, x);
    double sin_lat = sin(lat);
    double N_val = a / sqrt(1.0 - e2 * sin_lat * sin_lat);
    alt = p / cos(lat) - N_val;
}

__device__ void satellite_azel_inline(double rx, double ry, double rz,
                                      double sin_lat, double cos_lat,
                                      double sin_lon, double cos_lon,
                                      double sx, double sy, double sz,
                                      double& az, double& el) {
    double dx = sx - rx;
    double dy = sy - ry;
    double dz = sz - rz;

    // ENU rotation
    double e = -sin_lon * dx + cos_lon * dy;
    double n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
    double u =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;

    double hz = sqrt(e * e + n * n);
    el = atan2(u, hz);
    az = atan2(e, n);
}

// ---------------------------------------------------------------------------
// DOP computation from azimuth/elevation arrays
// ---------------------------------------------------------------------------

__device__ void compute_dop_from_azel(const double* az, const double* el, int n_vis,
                                      double& pdop, double& hdop, double& vdop, double& gdop) {
    pdop = hdop = vdop = gdop = 999.0;
    if (n_vis < 4) return;

    // Build G = H^T H  (4x4)
    double G[4][4] = {};
    for (int i = 0; i < n_vis; i++) {
        double ce = cos(el[i]);
        double se = sin(el[i]);
        double sa = sin(az[i]);
        double ca = cos(az[i]);
        double H[4] = {-ce * sa, -ce * ca, -se, 1.0};
        for (int a = 0; a < 4; a++) {
            for (int b = 0; b < 4; b++) {
                G[a][b] += H[a] * H[b];
            }
        }
    }

    // Invert 4x4 by Gaussian elimination with partial pivoting
    // Augmented matrix [G | I]
    double A[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            A[i][j] = G[i][j];
            A[i][j + 4] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < 4; col++) {
        // Partial pivoting
        int max_row = col;
        for (int row = col + 1; row < 4; row++) {
            if (fabs(A[row][col]) > fabs(A[max_row][col])) max_row = row;
        }
        if (max_row != col) {
            for (int k = 0; k < 8; k++) {
                double tmp = A[col][k];
                A[col][k] = A[max_row][k];
                A[max_row][k] = tmp;
            }
        }
        if (fabs(A[col][col]) < 1e-15) return;  // singular

        double pivot = A[col][col];
        for (int k = 0; k < 8; k++) A[col][k] /= pivot;

        for (int row = 0; row < 4; row++) {
            if (row == col) continue;
            double factor = A[row][col];
            for (int k = 0; k < 8; k++) A[row][k] -= factor * A[col][k];
        }
    }

    // Extract G^-1 (right half of augmented matrix)
    double Ginv[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            Ginv[i][j] = A[i][j + 4];

    double tr_xyz = Ginv[0][0] + Ginv[1][1] + Ginv[2][2];
    double tr_all = tr_xyz + Ginv[3][3];

    if (tr_all > 0) gdop = sqrt(tr_all);
    if (tr_xyz > 0) pdop = sqrt(tr_xyz);
    double tr_h = Ginv[0][0] + Ginv[1][1];
    if (tr_h > 0) hdop = sqrt(tr_h);
    if (Ginv[2][2] > 0) vdop = sqrt(Ginv[2][2]);
}

// ---------------------------------------------------------------------------
// Grid quality kernel: 1 thread per grid point
// ---------------------------------------------------------------------------

__global__ void grid_quality_kernel(const double* grid_ecef, const double* sat_ecef,
                                    double* pdop_out, double* hdop_out,
                                    double* vdop_out, double* gdop_out,
                                    int* n_visible_out,
                                    int n_grid, int n_sat, double el_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_grid) return;

    double rx = grid_ecef[idx * 3 + 0];
    double ry = grid_ecef[idx * 3 + 1];
    double rz = grid_ecef[idx * 3 + 2];

    // Compute receiver LLA for ENU rotation
    double lat, lon, alt;
    ecef_to_lla_inline(rx, ry, rz, lat, lon, alt);
    double sin_lat = sin(lat), cos_lat = cos(lat);
    double sin_lon = sin(lon), cos_lon = cos(lon);

    // Compute az/el for each satellite, filter by elevation mask
    // Max 64 satellites supported per grid point
    double az_vis[64], el_vis[64];
    int n_vis = 0;

    for (int s = 0; s < n_sat && n_vis < 64; s++) {
        double az, el;
        satellite_azel_inline(rx, ry, rz, sin_lat, cos_lat, sin_lon, cos_lon,
                              sat_ecef[s * 3 + 0], sat_ecef[s * 3 + 1], sat_ecef[s * 3 + 2],
                              az, el);
        if (el >= el_mask) {
            az_vis[n_vis] = az;
            el_vis[n_vis] = el;
            n_vis++;
        }
    }

    double pdop, hdop, vdop, gdop;
    compute_dop_from_azel(az_vis, el_vis, n_vis, pdop, hdop, vdop, gdop);

    pdop_out[idx] = pdop;
    hdop_out[idx] = hdop;
    vdop_out[idx] = vdop;
    gdop_out[idx] = gdop;
    n_visible_out[idx] = n_vis;
}

// ---------------------------------------------------------------------------
// Moeller-Trumbore ray-triangle intersection
// ---------------------------------------------------------------------------

__device__ bool ray_tri_hit(const double* origin, const double* dir,
                            const double* v0, const double* v1, const double* v2,
                            double& t) {
    const double EPSILON = 1e-9;
    double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

    // h = dir x e2
    double h[3] = {dir[1] * e2[2] - dir[2] * e2[1],
                   dir[2] * e2[0] - dir[0] * e2[2],
                   dir[0] * e2[1] - dir[1] * e2[0]};

    double a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    if (fabs(a) < EPSILON) return false;

    double f = 1.0 / a;
    double s[3] = {origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]};
    double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0 || u > 1.0) return false;

    // q = s x e1
    double q[3] = {s[1] * e1[2] - s[2] * e1[1],
                   s[2] * e1[0] - s[0] * e1[2],
                   s[0] * e1[1] - s[1] * e1[0]};
    double v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if (v < 0.0 || u + v > 1.0) return false;

    t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    return (t > EPSILON);
}

// ---------------------------------------------------------------------------
// Sky visibility kernel: 1 thread per (grid_point x az_bin)
// ---------------------------------------------------------------------------

__global__ void sky_visibility_kernel(const double* grid_ecef, const double* triangles,
                                      float* sky_mask,
                                      int n_grid, int n_tri, int n_az, int n_el) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_grid * n_az;
    if (tid >= total) return;

    int grid_idx = tid / n_az;
    int az_idx = tid % n_az;

    double rx = grid_ecef[grid_idx * 3 + 0];
    double ry = grid_ecef[grid_idx * 3 + 1];
    double rz = grid_ecef[grid_idx * 3 + 2];

    // Compute receiver LLA for ENU frame
    double lat, lon, alt;
    ecef_to_lla_inline(rx, ry, rz, lat, lon, alt);
    double sin_lat = sin(lat), cos_lat = cos(lat);
    double sin_lon = sin(lon), cos_lon = cos(lon);

    double az = (2.0 * kPi * az_idx) / n_az;

    // Loop over elevation bins
    for (int el_idx = 0; el_idx < n_el; el_idx++) {
        double el = (kPi / 2.0 * (el_idx + 0.5)) / n_el;

        // Direction in ENU
        double ce = cos(el), se = sin(el);
        double sa = sin(az), ca = cos(az);
        double e_dir = ce * sa;
        double n_dir = ce * ca;
        double u_dir = se;

        // ENU to ECEF direction
        double dir[3];
        dir[0] = -sin_lon * e_dir - sin_lat * cos_lon * n_dir + cos_lat * cos_lon * u_dir;
        dir[1] =  cos_lon * e_dir - sin_lat * sin_lon * n_dir + cos_lat * sin_lon * u_dir;
        dir[2] =                      cos_lat * n_dir          + sin_lat * u_dir;

        double origin[3] = {rx, ry, rz};
        bool blocked = false;

        for (int tri = 0; tri < n_tri; tri++) {
            const double* v0 = &triangles[tri * 9 + 0];
            const double* v1 = &triangles[tri * 9 + 3];
            const double* v2 = &triangles[tri * 9 + 6];
            double t;
            if (ray_tri_hit(origin, dir, v0, v1, v2, t)) {
                blocked = true;
                break;
            }
        }

        int mask_idx = grid_idx * n_az * n_el + az_idx * n_el + el_idx;
        sky_mask[mask_idx] = blocked ? 0.0f : 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void compute_grid_quality(const double* grid_ecef, const double* sat_ecef,
                          double* pdop, double* hdop, double* vdop, double* gdop,
                          int* n_visible,
                          int n_grid, int n_sat, double elevation_mask_rad) {
    double *d_grid, *d_sat;
    double *d_pdop, *d_hdop, *d_vdop, *d_gdop;
    int *d_nvis;

    size_t sz_grid = (size_t)n_grid * 3 * sizeof(double);
    size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
    size_t sz_out = (size_t)n_grid * sizeof(double);
    size_t sz_int = (size_t)n_grid * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_grid, sz_grid));
    CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
    CUDA_CHECK(cudaMalloc(&d_pdop, sz_out));
    CUDA_CHECK(cudaMalloc(&d_hdop, sz_out));
    CUDA_CHECK(cudaMalloc(&d_vdop, sz_out));
    CUDA_CHECK(cudaMalloc(&d_gdop, sz_out));
    CUDA_CHECK(cudaMalloc(&d_nvis, sz_int));

    CUDA_CHECK(cudaMemcpy(d_grid, grid_ecef, sz_grid, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n_grid + block - 1) / block;
    grid_quality_kernel<<<grid, block>>>(d_grid, d_sat,
                                         d_pdop, d_hdop, d_vdop, d_gdop, d_nvis,
                                         n_grid, n_sat, elevation_mask_rad);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(pdop, d_pdop, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdop, d_hdop, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vdop, d_vdop, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gdop, d_gdop, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_visible, d_nvis, sz_int, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grid)); CUDA_CHECK(cudaFree(d_sat));
    CUDA_CHECK(cudaFree(d_pdop)); CUDA_CHECK(cudaFree(d_hdop));
    CUDA_CHECK(cudaFree(d_nvis));
}

void compute_sky_visibility(const double* grid_ecef, const double* triangles,
                            float* sky_mask,
                            int n_grid, int n_tri, int n_az, int n_el) {
    double *d_grid, *d_tri;
    float *d_mask;

    size_t sz_grid = (size_t)n_grid * 3 * sizeof(double);
    size_t sz_tri = (size_t)n_tri * 9 * sizeof(double);
    size_t sz_mask = (size_t)n_grid * n_az * n_el * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_grid, sz_grid));
    CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
    CUDA_CHECK(cudaMalloc(&d_mask, sz_mask));

    CUDA_CHECK(cudaMemcpy(d_grid, grid_ecef, sz_grid, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tri, triangles, sz_tri, cudaMemcpyHostToDevice));

    int total_threads = n_grid * n_az;
    int block = 256;
    int grid = (total_threads + block - 1) / block;
    sky_visibility_kernel<<<grid, block>>>(d_grid, d_tri, d_mask,
                                           n_grid, n_tri, n_az, n_el);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(sky_mask, d_mask, sz_mask, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_grid)); CUDA_CHECK(cudaFree(d_tri));
}

}  // namespace gnss_gpu
